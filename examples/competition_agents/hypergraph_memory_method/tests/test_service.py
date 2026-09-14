from __future__ import annotations

from dataclasses import replace

import pytest

from hypergraph_memory_method import (
    BatchPolicy,
    ExperienceReview,
    MemoryRelationUpdate,
    MemoryService,
    RetrievalRequest,
    TaskFeedback,
    TaskMemoryCommit,
    compact_reuse_projection,
)
from hypergraph_memory_method.indexed_retrieval import personalized_rank


def begin(service, batch_id, **kwargs):
    kwargs.setdefault("task_family", "continuous_qa")
    kwargs.setdefault("dataset_id", "fixture-one")
    return service.begin_batch(batch_id, **kwargs)


def unit(name: str = "Orion", **fields) -> dict:
    return {
        "memory_id": name,
        "source_agent": "retriever",
        "task_topic": f"{name} project location",
        "summary": f"{name} is based in Paris.",
        "evidence": [f"Project registry: {name} is based in Paris."],
        "entities": [name],
        **fields,
    }


def episode(service, batch="batch-a", task="task-1", name="episode", **fields):
    return service.write_episode(
        batch, task, unit(name, **fields), operation_id=f"write:{task}:{name}"
    )["records"][0]["memory_id"]


def experience(
    service, source, batch="batch-a", task="task-1", name="experience", **fields
):
    return service.propose_experience(
        batch,
        task,
        unit(
            name,
            problem_pattern="resolve place",
            solution_pattern="Use the project registry",
            trigger_conditions=["place question"],
            source_memory_ids=[source],
            **fields,
        ),
        operation_id=f"experience:{name}",
    )["records"][0]["memory_id"]


def approve(service, memory_id, batch="batch-a", task="task-1"):
    return service.record_feedback(
        TaskFeedback(
            batch,
            task,
            answer_correct=True,
            memory_verdicts={memory_id: True},
            validation_source="fixture-verifier",
        )
    )


def test_retry_scope_and_strict_layer_contract(tmp_path):
    with MemoryService(tmp_path / "memory.db") as service:
        opened = begin(service, "batch-a")
        assert begin(service, "batch-a") == opened
        with pytest.raises(ValueError, match="different configuration"):
            begin(service, "batch-a", group_id="other")
        source = unit()
        first = service.write_knowledge("batch-a", [source], operation_id="k1")
        assert service.write_knowledge("batch-a", [source], operation_id="k1") == first
        with pytest.raises(ValueError, match="different inputs"):
            service.write_knowledge("batch-a", [unit("changed")], operation_id="k1")
        assert service.store.count() == 1
        begin(service, "batch-b")
        assert not service.retrieve(
            RetrievalRequest("batch-b", "Orion", layers=("knowledge",))
        )["hits"]
        with pytest.raises(KeyError):
            service.get_memory("batch-b", first["records"][0]["memory_id"])
        with pytest.raises(ValueError, match="valid memory layers"):
            RetrievalRequest("batch-a", "Orion", layers=("typo",))


def test_ten_tasks_finalize_and_reopen_keep_experience_and_provenance(tmp_path):
    path = tmp_path / "lifecycle.db"
    with MemoryService(path) as service:
        begin(service, "batch-a")
        knowledge = service.write_knowledge(
            "batch-a", [unit("registry")], operation_id="registry"
        )["records"][0]["memory_id"]
        episodes = []
        for i in range(10):
            episodes.append(
                episode(
                    service,
                    task=f"task-{i}",
                    name=f"Orion-{i}",
                    knowledge_refs=[knowledge],
                )
            )
            service.record_feedback(
                TaskFeedback(
                    "batch-a",
                    f"task-{i}",
                    answer_correct=True,
                    validation_source="fixture-verifier",
                )
            )
        candidate = experience(service, episodes[0], task="review")
        approve(service, candidate, task="review")
        assert service.get_memory("batch-a", candidate)["review_status"] == "candidate"
        preview = service.preview_maintenance("batch-a")
        assert service.store.count() == 12
        assert len(preview["cleared_ids"]) == 11
        report = service.finalize_batch("batch-a")
        assert service.finalize_batch("batch-a") == report
        assert service.store.count() == 1
        assert report["promoted_ids"] == [candidate]
    with MemoryService(path) as service:
        begin(service, "batch-b")
        hits = service.retrieve(
            RetrievalRequest("batch-b", "project registry", layers=("experience",))
        )["hits"]
        assert hits[0]["memory_id"] == candidate
        chain = service.get_evidence_path("batch-b", candidate)
        assert {item["global_memory_id"] for item in chain["memories"]} == {
            candidate,
            episodes[0],
            knowledge,
        }
        assert chain["unresolved_ids"] == []
        assert chain["structurally_complete"] is True
        assert chain["factual_correctness"] == "requires_external_validation"
        assert not service.retrieve(
            RetrievalRequest("batch-b", "registry", layers=("knowledge",))
        )["hits"]


def test_correct_answer_without_memory_verdict_does_not_promote():
    with MemoryService(":memory:") as service:
        begin(service, "batch-a")
        source = episode(service)
        candidate = experience(service, source)
        service.record_feedback(
            TaskFeedback(
                "batch-a",
                "task-1",
                retrieved_ids=(source, candidate),
                adopted_ids=(source,),
                answer_correct=True,
                validation_source="fixture-verifier",
            )
        )
        record = service.get_memory("batch-a", candidate)
        assert record["reuse_count"] == record["verified_failure_count"] == 0
        used = service.get_memory("batch-a", source)
        assert used["reuse_count"] == 1
        assert used["reuse_success_count"] == 0
        report = service.finalize_batch("batch-a")
        assert candidate in report["pending_ids"]
        begin(service, "batch-b")
        assert not service.retrieve(
            RetrievalRequest("batch-b", "registry", layers=("experience",))
        )["hits"]


def test_feedback_retry_and_failure_attribution(tmp_path):
    path = tmp_path / "feedback.db"
    with MemoryService(path) as service:
        begin(service, "batch-a")
        source = episode(service)
        candidate = experience(service, source)
        feedback = TaskFeedback(
            "batch-a",
            "task-2",
            (candidate,),
            (candidate,),
            False,
            validation_source="fixture-verifier",
        )
        receipt = service.record_feedback(feedback)
    with MemoryService(path) as service:
        assert service.record_feedback(feedback) == receipt
        record = service.get_memory("batch-a", candidate)
        assert record["reuse_count"] == 1
        assert record["verified_failure_count"] == 0
        assert record["review_status"] == "pending"
        with pytest.raises(ValueError, match="different inputs"):
            service.record_feedback(replace(feedback, answer_correct=True))
        for task in ("check-1", "check-2"):
            service.record_feedback(
                TaskFeedback(
                    "batch-a",
                    task,
                    memory_verdicts={candidate: False},
                    validation_source="manual-review",
                )
            )
        assert candidate in service.finalize_batch("batch-a")["rejected_ids"]


def test_explicit_source_link_changes_recall_and_generic_source_does_not():
    with MemoryService(":memory:") as service:
        begin(service, "batch-a")
        source = episode(
            service, name="Orion", summary="Orion measurement", entities=["Orion"]
        )
        related = experience(
            service,
            source,
            name="linked",
            task_topic="Unrelated procedure",
            summary="Use a calibrated sensor",
            entities=["sensor"],
        )
        unrelated_source = episode(
            service,
            task="other",
            name="other",
            summary="Gardening calendar",
            entities=["garden"],
        )
        other = experience(
            service,
            unrelated_source,
            name="unlinked",
            task_topic="Gardening procedure",
            summary="Follow a watering calendar",
            entities=["garden"],
        )
        request = RetrievalRequest(
            "batch-a",
            "",
            layers=("experience",),
            bridge_layers=("episodic",),
            seed_memory_ids=(source,),
            include_candidates=True,
        )
        result = service.retrieve(request)
        assert [hit["memory_id"] for hit in result["hits"]] == [related]
        assert any(
            path["relation"] == "source_reference"
            for path in result["hits"][0]["relation_paths"]
        )
        assert other not in {hit["memory_id"] for hit in result["hits"]}
        assert not service.retrieve(replace(request, graph_hops=0))["hits"]


def test_merge_keeps_all_sources_and_old_ids_resolve(tmp_path):
    with MemoryService(tmp_path / "merge.db") as service:
        begin(service, "batch-a")
        source1 = episode(service, name="source1")
        source2 = episode(service, task="task-2", name="source2")
        first = experience(service, source1, name="first")
        second = experience(service, source2, task="task-2", name="second")
        approve(service, first)
        approve(service, second, task="task-2")
        report = service.finalize_batch("batch-a")
        assert len(report["merged"]) == 1
        assert report["active_experience_count"] == 1
        begin(service, "batch-b")
        canonical = service.get_memory("batch-b", second)
        assert (
            canonical["global_memory_id"]
            == service.get_memory("batch-b", first)["global_memory_id"]
        )
        assert set(canonical["source_memory_ids"]) == {source1, source2}
        assert not service.get_evidence_path("batch-b", second)["unresolved_ids"]


def test_write_failure_rolls_back_records_indexes_and_receipt(monkeypatch):
    with MemoryService(":memory:") as service:
        begin(service, "batch-a")
        original = service.store.put
        calls = 0

        def failing(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = original(*args, **kwargs)
            if calls == 2:
                raise RuntimeError("injected write failure")
            return result

        monkeypatch.setattr(service.store, "put", failing)
        with pytest.raises(RuntimeError, match="injected"):
            service.write_knowledge(
                "batch-a", [unit("a"), unit("b")], operation_id="bulk"
            )
        assert service.store.count() == 0
        assert service.store.read_rows("SELECT count(*) FROM memory_search")[0][0] == 0
        assert (
            service.store.read_rows("SELECT count(*) FROM memory_operations")[0][0] == 0
        )
        monkeypatch.setattr(service.store, "put", original)
        service.write_knowledge("batch-a", [unit("a"), unit("b")], operation_id="bulk")
        assert service.store.count() == 2


def test_finalize_failure_restores_deleted_records(monkeypatch):
    with MemoryService(":memory:") as service:
        begin(service, "batch-a")
        source = episode(service)
        candidate = experience(service, source)
        approve(service, candidate)
        original = service.store.delete

        def failing(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected finalize failure")

        monkeypatch.setattr(service.store, "delete", failing)
        with pytest.raises(RuntimeError, match="injected"):
            service.finalize_batch("batch-a")
        assert service.store.count() == 2
        assert service.store.batch(service.namespace, "batch-a")["status"] == "open"
        assert service.get_memory("batch-a", candidate)["batch_id"] == "batch-a"
        monkeypatch.setattr(service.store, "delete", original)
        assert service.finalize_batch("batch-a")["promoted_ids"] == [candidate]


def test_number_and_chinese_recall_without_context_scan():
    with MemoryService(":memory:") as service:
        begin(service, "batch-a")
        service.write_knowledge(
            "batch-a",
            [
                unit(
                    f"item{i}",
                    summary=f"Registry entry item{i}",
                    task_topic=f"item{i}",
                    entities=[],
                )
                for i in range(200)
            ],
            operation_id="bulk",
        )
        chinese = service.write_knowledge(
            "batch-a",
            [
                unit(
                    "任务",
                    task_topic="探测器发射",
                    summary="天问一号于2020年发射",
                    entities=["天问一号"],
                )
            ],
            operation_id="cn",
        )["records"][0]["memory_id"]
        result = service.retrieve(
            RetrievalRequest(
                "batch-a", "item167", layers=("knowledge",), mode="keyword"
            )
        )
        assert len(result["hits"]) == 1
        assert result["diagnostics"]["records_loaded"] == 1
        assert (
            service.retrieve(
                RetrievalRequest(
                    "batch-a", "天问2020", layers=("knowledge",), mode="keyword"
                )
            )["hits"][0]["memory_id"]
            == chinese
        )


class TinyEncoder:
    model_id = "test-only-encoder-v1"

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.extend(texts)
        return [
            [1.0, 0.1] if "Paris" in text or "capital" in text else [0.1, 1.0]
            for text in texts
        ]


def test_persisted_vectors_and_query_cache_no_document_reencoding(tmp_path):
    path = tmp_path / "vectors.db"
    encoder = TinyEncoder()
    with MemoryService(path, encoder=encoder) as service:
        begin(service, "batch-a")
        episode(service)
        request = RetrievalRequest("batch-a", "capital", mode="fusion")
        assert len(encoder.calls) == 1
        first = service.retrieve(request)
        second = service.retrieve(request)
        assert len(encoder.calls) == 2
        assert second["diagnostics"]["query_embedding_cache_hit"] is True
        assert second["diagnostics"]["vector_cache_hit"] is True
        assert first["hits"] == second["hits"]
    encoder = TinyEncoder()
    with MemoryService(path, encoder=encoder) as service:
        assert service.retrieve(request)["hits"]
        assert encoder.calls == ["capital"]


def test_semantic_requires_encoder_and_matching_dimensions():
    with MemoryService(":memory:") as service:
        begin(service, "batch-a")
        episode(service)
        with pytest.raises(RuntimeError, match="configured memory encoder"):
            service.retrieve(RetrievalRequest("batch-a", "capital", mode="semantic"))
    encoder = TinyEncoder()
    with MemoryService(":memory:", encoder=encoder) as service:
        begin(service, "batch-a")
        episode(service)
        encoder.encode = lambda texts: [[1.0, 2.0, 3.0] for _ in texts]
        with pytest.raises(ValueError, match="dimensions differ"):
            service.retrieve(RetrievalRequest("batch-a", "capital", mode="semantic"))


def test_projection_preserves_conclusion_and_reports_missing_dependencies():
    projection = compact_reuse_projection(
        unit(evidence=[f"fact {i}" for i in range(8)]), max_steps=3
    )
    assert any(
        step["kind"] == "final_claim" for step in projection["evidence_chain_steps"]
    )
    assert projection["omitted_dependency_ids"]
    assert projection["projection_complete"] is False


def test_return_budget_does_not_truncate_facts():
    with MemoryService(":memory:") as service:
        begin(service, "batch-a")
        source = episode(service, summary="Orion " * 1000)
        result = service.retrieve(
            RetrievalRequest("batch-a", "Orion", max_payload_bytes=256)
        )
        assert result["hits"] == []
        assert source in result["diagnostics"]["budget_omitted_ids"]
        assert result["diagnostics"]["payload_bytes"] <= 256


def test_abort_and_other_namespace_are_isolated(tmp_path):
    path = tmp_path / "isolation.db"
    with MemoryService(path, namespace="a") as service:
        begin(service, "batch-a")
        source = episode(service)
    with MemoryService(path, namespace="b") as other:
        begin(other, "batch-a")
        assert not other.retrieve(RetrievalRequest("batch-a", "registry"))["hits"]
        with pytest.raises(KeyError):
            other.get_memory("batch-a", source)
    with MemoryService(path, namespace="a") as service:
        report = service.abort_batch("batch-a", reason="cancelled")
        assert service.abort_batch("batch-a", reason="cancelled") == report
        assert service.store.count() == 0
        with pytest.raises(ValueError, match="aborted"):
            service.finalize_batch("batch-a")


def test_local_random_walk_conserves_probability_and_ignores_disconnected_nodes():
    ranks = personalized_rank({"a": 1}, {("a", "b"): 3.0, ("a", "c"): 1.0})
    assert sum(ranks.values()) == pytest.approx(1.0)
    assert ranks["b"] > ranks["c"]
    assert personalized_rank({"a": 1}, {}) == {"a": 1}


def test_snapshot_restores_scopes_vectors_receipts_and_evidence():
    with MemoryService(":memory:", encoder=TinyEncoder()) as service:
        begin(service, "batch-a")
        source = episode(service)
        candidate = experience(service, source)
        receipt = approve(service, candidate)
        service.finalize_batch("batch-a")
        begin(service, "batch-b")
        retrieval = service.retrieve(
            RetrievalRequest(
                "batch-b", "registry", layers=("experience",), mode="keyword"
            )
        )
        snapshot = service.export_snapshot()
    encoder = TinyEncoder()
    with MemoryService(":memory:", encoder=encoder) as restored:
        restored.restore_snapshot(snapshot)
        assert restored.get_memory("batch-b", candidate)["review_status"] == "active"
        assert not restored.get_evidence_path("batch-b", candidate)["unresolved_ids"]
        assert approve(restored, candidate) == receipt
        restored.record_feedback(
            TaskFeedback(
                "batch-b",
                "task-2",
                retrieval_ids=(retrieval["retrieval_id"],),
                adopted_ids=(candidate,),
            )
        )
        assert restored.get_batch_status("batch-b")["retrieval_count"] == 1
        assert restored.retrieve(
            RetrievalRequest(
                "batch-b", "capital", layers=("experience",), mode="semantic"
            )
        )["hits"]
        assert encoder.calls == ["capital"]
        with pytest.raises(ValueError, match="empty namespace"):
            restored.restore_snapshot(snapshot)


def test_reindex_is_explicit_when_changing_encoder(tmp_path):
    path = tmp_path / "change-model.db"
    with MemoryService(path) as service:
        begin(service, "batch-a")
        source = episode(service)
    with MemoryService(path, encoder=TinyEncoder()) as service:
        request = RetrievalRequest("batch-a", "capital", mode="semantic")
        with pytest.raises(RuntimeError, match="reindex_embeddings"):
            service.retrieve(request)
        assert service.reindex_embeddings("batch-a")["record_count"] == 1
        assert service.retrieve(request)["hits"][0]["memory_id"] == source


def test_different_local_ids_do_not_overwrite_across_tasks():
    with MemoryService(":memory:") as service:
        begin(service, "batch-a")
        first = episode(service, task="task-1", name="same")
        second = episode(service, task="task-2", name="same")
        assert first != second
        assert (
            len(
                service.retrieve(RetrievalRequest("batch-a", "same", mode="keyword"))[
                    "hits"
                ]
            )
            == 2
        )
        with pytest.raises(ValueError, match="Ambiguous"):
            service.get_memory("batch-a", "same")


def test_same_task_duplicate_experiences_do_not_double_validation():
    with MemoryService(":memory:") as service:
        begin(service, "batch-a", policy=BatchPolicy(min_verified_successes=2))
        source = episode(service)
        one = experience(service, source, name="one")
        two = experience(service, source, name="two")
        service.record_feedback(
            TaskFeedback(
                "batch-a",
                "task-1",
                answer_correct=True,
                memory_verdicts={one: True, two: True},
                validation_source="fixture-verifier",
            )
        )
        report = service.finalize_batch("batch-a")
        assert report["active_experience_count"] == 0
        assert set(report["pending_ids"]) == {one, two}


def test_semantic_similarity_requires_explicit_equivalence_review():
    with MemoryService(":memory:", encoder=TinyEncoder()) as service:
        begin(service, "batch-a")
        source = episode(service)
        first_record = unit(
            "first",
            problem_pattern="place",
            solution_pattern="look up the location",
            trigger_conditions=["place"],
            source_memory_ids=[source],
        )
        second_record = unit(
            "second",
            problem_pattern="place",
            solution_pattern="consult a geographic register",
            trigger_conditions=["place"],
            source_memory_ids=[source],
        )
        ids = []
        for i, record in enumerate((first_record, second_record)):
            candidate = service.propose_experience(
                "batch-a", f"task-{i}", record, operation_id=f"e{i}"
            )["records"][0]["memory_id"]
            ids.append(candidate)
            approve(service, candidate, task=f"task-{i}")
        report = service.finalize_batch("batch-a")
        assert report["merged"] == []
        assert len(report["merge_suggestions"]) == 1
        assert report["active_experience_count"] == 2


def test_verified_equivalent_procedures_can_merge():
    with MemoryService(
        ":memory:", encoder=TinyEncoder(), merge_verifier=lambda left, right: True
    ) as service:
        begin(service, "batch-a")
        source = episode(service)
        for i, solution in enumerate(("consult the registry", "use the directory")):
            candidate = service.propose_experience(
                "batch-a",
                f"task-{i}",
                unit(
                    str(i),
                    problem_pattern="place",
                    solution_pattern=solution,
                    trigger_conditions=["place"],
                    source_memory_ids=[source],
                ),
                operation_id=f"e{i}",
            )["records"][0]["memory_id"]
            approve(service, candidate, task=f"task-{i}")
        assert len(service.finalize_batch("batch-a")["merged"]) == 1


def test_capacity_limit_archives_without_destroying_sources():
    with MemoryService(":memory:") as service:
        begin(service, "batch-a", policy=BatchPolicy(max_active_experiences=1))
        source = episode(service)
        for i in range(2):
            candidate = service.propose_experience(
                "batch-a",
                f"task-{i}",
                unit(
                    str(i),
                    problem_pattern=f"pattern {i}",
                    solution_pattern=f"procedure {i}",
                    source_memory_ids=[source],
                ),
                operation_id=f"e{i}",
            )["records"][0]["memory_id"]
            approve(service, candidate, task=f"task-{i}")
        report = service.finalize_batch("batch-a")
        assert report["active_experience_count"] == 1
        assert len(report["capacity_pruned_ids"]) == 1
        assert service.store.count() == 2


def test_high_degree_entity_cannot_fill_candidate_budget():
    with MemoryService(":memory:") as service:
        begin(service, "batch-a")
        records = [unit(f"record{i}", entities=["shared-hub"]) for i in range(30)]
        receipts = service.write_knowledge("batch-a", records, operation_id="bulk")
        source = receipts["records"][0]["memory_id"]
        result = service.retrieve(
            RetrievalRequest(
                "batch-a",
                "",
                layers=("knowledge",),
                seed_memory_ids=(source,),
                candidate_limit=10,
                max_node_degree=5,
            )
        )
        assert [hit["memory_id"] for hit in result["hits"]] == [source]
        assert result["diagnostics"]["records_loaded"] == 1


def test_conditional_reranker_receives_only_compact_candidates():
    calls = []

    def rerank(query, candidates):
        calls.append(candidates)
        assert all(
            "trace" not in candidate and "embedding" not in candidate
            for candidate in candidates
        )
        return list(range(len(candidates)))

    with MemoryService(":memory:", reranker=rerank) as service:
        begin(service, "batch-a")
        episode(service, name="item1")
        episode(service, name="item2", task="task-2")
        result = service.retrieve(RetrievalRequest("batch-a", "project location"))
        assert result["diagnostics"]["reranked"] is True
        assert len(calls) == 1


def test_task_metadata_and_explicit_entity_aliases_are_preserved():
    with MemoryService(":memory:") as service:
        begin(service, "batch-a", group_id="group-one")
        source = episode(
            service,
            name="map",
            entities=["NYC", "New York", " New York "],
            entity_aliases={"NYC": "New York"},
        )
        record = service.get_memory("batch-a", source)
        assert record["entities"].count("new york") == 1
        assert "nyc" not in record["entities"]
        assert all(
            step["source"]["task_id"] == "task-1"
            for step in record["evidence_chain"]["steps"]
        )


def test_offline_labels_are_never_added_to_query(monkeypatch):
    from hypergraph_memory_method.run_memory_retrieval_benchmark import _query_for_turn
    from hypergraph_memory_method.benchmark import evaluate_retrieval_modes

    query = _query_for_turn(
        "group",
        {
            "question": "Where did she work?",
            "expected_answer": "secret",
            "expected_reuse": ["turn-999.answer"],
        },
    )
    assert "turn-999" not in query and "secret" not in query
    seen = []
    monkeypatch.setattr(
        "hypergraph_memory_method.benchmark.search_memory_by_mode",
        lambda records, query, **kwargs: seen.append(records) or [],
    )
    evaluate_retrieval_modes(
        [unit()],
        [
            {
                "query": "cold start",
                "available_memory_ids": [],
                "expected_memory_ids": [],
            }
        ],
        modes=("keyword",),
    )
    assert seen == [[]]


def test_service_evaluation_reports_retrieval_not_answer_accuracy():
    from hypergraph_memory_method.evaluate_service import evaluate_service

    with MemoryService(":memory:") as service:
        begin(service, "batch-a")
        source = episode(service)
        report = evaluate_service(
            service,
            [
                {
                    "case_id": "one",
                    "request": {"batch_id": "batch-a", "query": "episode", "top_k": 2},
                    "relevant_memory_ids": [source],
                }
            ],
        )
        assert report["answer_accuracy"] is None
        assert report["modes"]["keyword"]["summary"]["precision_at_k"] == 0.5
        assert report["modes"]["keyword"]["summary"]["recall_at_k"] == 1.0


def test_local_encoder_requires_local_files(tmp_path):
    from hypergraph_memory_method import SentenceTransformerEncoder

    with pytest.raises(FileNotFoundError):
        SentenceTransformerEncoder(tmp_path / "not-downloaded", model_id="local-model")


def test_pending_long_term_experience_waits_for_finalize_after_review():
    with MemoryService(":memory:") as service:
        begin(service, "batch-a")
        source = episode(service)
        candidate = experience(service, source)
        service.finalize_batch("batch-a")
        begin(service, "batch-b")
        approve(service, candidate, batch="batch-b", task="review")
        assert service.get_memory("batch-b", candidate)["review_status"] == "candidate"
        assert not service.retrieve(
            RetrievalRequest("batch-b", "registry", layers=("experience",))
        )["hits"]
        service.finalize_batch("batch-b")
        begin(service, "batch-c")
        assert service.retrieve(
            RetrievalRequest("batch-c", "registry", layers=("experience",))
        )["hits"]


def test_cancellation_rolls_back_and_connection_remains_usable():
    with MemoryService(":memory:") as service:
        begin(service, "batch-a")
        with pytest.raises(KeyboardInterrupt):
            with service.store.transaction():
                episode(service)
                raise KeyboardInterrupt()
        assert service.store.count() == 0
        assert episode(service)


def test_vector_validation_rejects_strings_and_handles_large_finite_values():
    from hypergraph_memory_method.indexed_retrieval import checked_vector

    with pytest.raises(ValueError, match="sequence of numbers"):
        checked_vector("123")
    with pytest.raises(ValueError, match="finite"):
        checked_vector([float("nan"), 1])
    assert checked_vector([1e300, 1e300]) == pytest.approx([2**-0.5, 2**-0.5])


def test_failed_task_output_is_not_reused_as_a_verified_answer():
    with MemoryService(":memory:") as service:
        begin(service, "batch-a")
        source = episode(service)
        service.record_feedback(
            TaskFeedback(
                "batch-a",
                "task-1",
                answer_correct=False,
                validation_source="test-assertion",
            )
        )
        assert service.get_memory("batch-a", source)["review_status"] == "pending"
        assert not service.retrieve(RetrievalRequest("batch-a", "episode"))["hits"]


def test_experience_crosses_datasets_only_within_the_same_task_family():
    with MemoryService(":memory:") as service:
        service.begin_batch(
            "qa-a",
            task_family="continuous_qa",
            dataset_id="topiocqa-1",
        )
        source = episode(service, batch="qa-a")
        candidate = experience(service, source, batch="qa-a")
        approve(service, candidate, batch="qa-a")
        service.finalize_batch("qa-a")

        service.begin_batch(
            "qa-b",
            task_family="continuous_qa",
            dataset_id="topiocqa-2",
        )
        qa_hits = service.retrieve(
            RetrievalRequest("qa-b", "registry", layers=("experience",))
        )["hits"]
        assert qa_hits[0]["memory_id"] == candidate

        service.begin_batch(
            "code-a",
            task_family="code_execution",
            dataset_id="codeact-1",
        )
        assert not service.retrieve(
            RetrievalRequest("code-a", "registry", layers=("experience",))
        )["hits"]
        with pytest.raises(KeyError, match="outside batch scope"):
            service.get_memory("code-a", candidate)


def test_dataset_scoped_experience_does_not_cross_dataset_boundaries():
    with MemoryService(":memory:") as service:
        service.begin_batch(
            "qa-a",
            task_family="continuous_qa",
            dataset_id="private-qa",
            policy=BatchPolicy(experience_scope="dataset"),
        )
        source = episode(service, batch="qa-a")
        candidate = experience(service, source, batch="qa-a")
        approve(service, candidate, batch="qa-a")
        service.finalize_batch("qa-a")

        service.begin_batch(
            "qa-other",
            task_family="continuous_qa",
            dataset_id="other-qa",
        )
        assert not service.retrieve(
            RetrievalRequest("qa-other", "registry", layers=("experience",))
        )["hits"]

        service.begin_batch(
            "qa-same",
            task_family="continuous_qa",
            dataset_id="private-qa",
        )
        assert (
            service.retrieve(
                RetrievalRequest("qa-same", "registry", layers=("experience",))
            )["hits"][0]["memory_id"]
            == candidate
        )


def test_retrieval_receipt_binds_feedback_to_actual_hits():
    with MemoryService(":memory:") as service:
        begin(service, "batch-a")
        source = episode(service)
        other = episode(service, task="task-2", name="unrelated")
        result = service.retrieve(RetrievalRequest("batch-a", "episode", top_k=1))
        assert result["retrieval_id"].startswith("ret-")
        assert result["hits"][0]["memory_id"] == source
        receipt = service.record_feedback(
            TaskFeedback(
                "batch-a",
                "task-3",
                retrieval_ids=(result["retrieval_id"],),
                adopted_ids=(source,),
            )
        )
        assert receipt["retrieved_count"] == receipt["adopted_count"] == 1
        with pytest.raises(ValueError, match="exactly match"):
            service.record_feedback(
                TaskFeedback(
                    "batch-a",
                    "task-4",
                    retrieval_ids=(result["retrieval_id"],),
                    retrieved_ids=(other,),
                )
            )


def test_empty_retrieval_receipt_cannot_be_replaced_with_forged_hits():
    with MemoryService(":memory:") as service:
        begin(service, "batch-a")
        source = episode(service)
        empty = service.retrieve(
            RetrievalRequest("batch-a", "not-present", mode="keyword")
        )
        assert empty["hits"] == []
        with pytest.raises(ValueError, match="exactly match"):
            service.record_feedback(
                TaskFeedback(
                    "batch-a",
                    "task-2",
                    retrieved_ids=(source,),
                    retrieval_ids=(empty["retrieval_id"],),
                )
            )


def test_task_memory_commit_is_atomic_and_retry_safe():
    with MemoryService(":memory:") as service:
        begin(service, "batch-a")
        invalid_experience = unit(
            "bad-experience",
            problem_pattern="place",
            solution_pattern="look up registry",
            source_memory_ids=["missing-source"],
        )
        commit = TaskMemoryCommit(
            "batch-a",
            "task-1",
            "commit:task-1",
            episode=unit("answer"),
            experiences=(invalid_experience,),
        )
        with pytest.raises(KeyError, match="missing or outside"):
            service.commit_task_memory(commit)
        assert service.store.count() == 0

        valid = TaskMemoryCommit(
            "batch-a",
            "task-1",
            "commit:task-1:valid",
            episode=unit("answer"),
        )
        first = service.commit_task_memory(valid)
        assert service.commit_task_memory(valid) == first
        assert service.store.count() == 1


def test_explicit_experience_review_controls_long_term_promotion():
    with MemoryService(":memory:") as service:
        begin(service, "batch-a")
        source = episode(service)
        candidate = experience(service, source)
        assert service.list_active_experiences("batch-a") == []
        assert (
            service.list_pending_reviews("batch-a")[0]["global_memory_id"] == candidate
        )
        review = service.review_experience(
            ExperienceReview(
                "batch-a",
                "manual-1",
                {candidate: True},
                "human-review",
            )
        )
        assert review["review_id"] == "manual-1"
        service.finalize_batch("batch-a")
        begin(service, "batch-b")
        assert (
            service.list_active_experiences("batch-b")[0]["global_memory_id"]
            == candidate
        )


def test_supersede_memory_keeps_audit_and_removes_old_from_retrieval():
    with MemoryService(":memory:") as service:
        begin(service, "batch-a")
        old_id = episode(service)
        receipt = service.supersede_memory(
            "batch-a",
            old_id,
            {"summary": "The corrected project location is London."},
            operation_id="correct:episode",
        )
        old = service.get_memory("batch-a", old_id)
        new = service.get_memory("batch-a", receipt["replacement_id"])
        assert old["review_status"] == "superseded"
        assert old["superseded_by"] == new["global_memory_id"]
        assert new["supersedes_memory_id"] == old["global_memory_id"]
        hits = service.retrieve(RetrievalRequest("batch-a", "London"))["hits"]
        assert hits[0]["memory_id"] == new["global_memory_id"]
        assert old["global_memory_id"] not in {hit["memory_id"] for hit in hits}


def test_relation_update_changes_evidence_path_and_is_retry_safe():
    with MemoryService(":memory:") as service:
        begin(service, "batch-a")
        knowledge = service.write_knowledge(
            "batch-a", [unit("registry")], operation_id="knowledge"
        )["records"][0]["memory_id"]
        target = episode(service)
        update = MemoryRelationUpdate(
            "batch-a",
            target,
            "relations:add",
            add_knowledge_refs=(knowledge,),
        )
        first = service.update_memory_relations(update)
        assert service.update_memory_relations(update) == first
        path = service.get_evidence_path("batch-a", target)
        assert knowledge in {item["global_memory_id"] for item in path["memories"]}

        service.update_memory_relations(
            MemoryRelationUpdate(
                "batch-a",
                target,
                "relations:remove",
                remove_knowledge_refs=(knowledge,),
            )
        )
        assert knowledge not in service.get_memory("batch-a", target)["knowledge_refs"]


def test_batch_status_and_memory_stats_use_persisted_receipts():
    with MemoryService(":memory:") as service:
        begin(service, "batch-a")
        source = episode(service)
        retrieval = service.retrieve(RetrievalRequest("batch-a", "episode", top_k=1))
        service.record_feedback(
            TaskFeedback(
                "batch-a",
                "task-2",
                retrieval_ids=(retrieval["retrieval_id"],),
                adopted_ids=(source,),
                answer_correct=True,
                memory_verdicts={source: True},
                validation_source="fixture-verifier",
            )
        )
        status = service.get_batch_status("batch-a")
        stats = service.get_batch_memory_stats("batch-a")
        assert status["retrieval_count"] == status["feedback_count"] == 1
        assert stats["retrieved_count"] == stats["adopted_count"] == 1
        assert stats["correct_adoption_count"] == 1
        assert stats["reuse_rate"] == stats["correct_reuse_rate"] == 1.0
        assert stats["retrieval_payload_bytes"] > 0
