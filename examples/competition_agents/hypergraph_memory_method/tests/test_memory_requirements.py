from __future__ import annotations

from pathlib import Path

from competition_agents.embedding import EmbeddingResult
from competition_agents.memory import add_memory
from hypergraph_memory_method import (
    SQLiteHypergraphMemoryStore,
    build_hypergraph_view,
    compact_reuse_projection,
    compact_trace,
    configured_retrieval_mode,
    evaluate_paired_memory_gain,
    evaluate_retrieval_modes,
    evaluate_task_reuse,
    load_persistent_memory,
    normalize_memory_record,
    memory_organization_for_mode,
    search_hypergraph_memory,
    search_memory_by_mode,
    validate_evidence_chain,
)
from hypergraph_memory_method.index import (
    clear_hypergraph_index_cache,
    get_cached_hypergraph_index,
)


def _evidence_record(memory_id: str = "mem-0002") -> dict:
    return {
        "memory_id": memory_id,
        "memory_type": "evidence",
        "source_agent": "summarizer",
        "created_at": "2026-01-01T00:00:00+00:00",
        "task_topic": "What government position did Shirley Temple hold?",
        "summary": "Shirley Temple served as Chief of Protocol of the United States.",
        "answer": "Chief of Protocol of the United States",
        "tags": ["biography", "government-position"],
        "confidence": 0.9,
        "trace": {
            "task_meta": {
                "task_id": "chain-001:turn-1",
                "group_id": "chain-001",
                "turn_id": "turn-1",
                "dataset_type": "continuous",
                "question": "What government position did Shirley Temple hold?",
                "supporting_facts": [
                    ["Kiss and Tell", "Kiss and Tell stars Shirley Temple."],
                    [
                        "Shirley Temple",
                        "She served as Chief of Protocol of the United States.",
                    ],
                ],
            },
            "evidence": ["Kiss and Tell -> Shirley Temple -> Chief of Protocol."],
            "execution_result": {
                "reasoning_summary": "The film identifies Shirley Temple, whose biography supplies the office."
            },
        },
    }


def test_normalized_record_covers_unified_memory_contract() -> None:
    record = normalize_memory_record(_evidence_record())

    assert record["schema_version"] == "1.0"
    assert record["memory_id"] == "mem-0002"
    assert record["source_agent"] == "summarizer"
    assert record["created_at"]
    assert record["updated_at"]
    assert {"fact", "evidence", "conclusion"} <= set(record["memory_kinds"])
    assert record["search_document"]
    assert record["evidence_chain_id"].startswith("chain-")


def test_evidence_chain_is_ordered_sourced_valid_and_reusable() -> None:
    record = normalize_memory_record(_evidence_record())
    chain = record["evidence_chain"]

    assert chain["valid"] is True
    assert chain["completeness"] == 1.0
    assert chain["evidence_count"] == 1
    assert [step["position"] for step in chain["steps"]] == list(
        range(1, len(chain["steps"]) + 1)
    )
    assert all(step["source"]["source_type"] for step in chain["steps"])
    assert chain["final_claim_id"]
    final_claim = next(
        step for step in chain["steps"] if step["step_id"] == chain["final_claim_id"]
    )
    assert final_claim["depends_on"]

    projection = compact_reuse_projection(record)
    assert projection["evidence_chain_valid"] is True
    assert projection["evidence_chain_steps"]
    assert all("source" in step for step in projection["evidence_chain_steps"])


def test_evidence_chain_validation_rejects_broken_dependency() -> None:
    validation = validate_evidence_chain(
        {
            "final_claim_id": "claim-final",
            "steps": [
                {
                    "step_id": "claim-final",
                    "kind": "final_claim",
                    "text": "unsupported",
                    "source": {"source_type": "agent", "source_id": "summarizer"},
                    "depends_on": ["missing-evidence"],
                }
            ],
        }
    )

    assert validation["valid"] is False
    assert {issue["code"] for issue in validation["issues"]} >= {"missing_dependency"}


def test_hypergraph_contains_chain_support_and_provenance_edges() -> None:
    view = build_hypergraph_view(_evidence_record())
    edge_types = {edge["type"] for edge in view["hyperedges"]}

    assert view["evidence_chain"]["valid"] is True
    assert view["evidence_chain_nodes"]
    assert "evidence_chain" in edge_types
    assert "claim_support" in edge_types
    assert "evidence_provenance" in edge_types


def test_compact_trace_removes_recursive_memory_and_evaluation_answers() -> None:
    trace = {
        "task_meta": {
            "task_id": "turn-2",
            "expected_answer": "secret-gold",
            "expected_reuse": ["turn-1.answer"],
        },
        "memory_hits": [
            {
                "memory_id": "mem-0001",
                "summary": "prior",
                "trace": {
                    "memory_hits": [{"memory_id": "mem-old", "trace": {"deep": True}}]
                },
                "embedding": [0.1, 0.2],
                "hypergraph": {"large": True},
            }
        ],
    }

    compacted = compact_trace(trace)

    assert "expected_answer" not in compacted["task_meta"]
    assert "expected_reuse" not in compacted["task_meta"]
    assert compacted["memory_hits"] == [{"memory_id": "mem-0001", "summary": "prior"}]


def test_memory_graph_uses_canonical_entities_without_evaluation_labels() -> None:
    source = _evidence_record()
    source["trace"]["task_meta"]["expected_reuse"] = ["turn-1.answer"]
    source["trace"]["task_meta"]["supporting_facts"] = [
        ["Shirley Temple", "She served as Chief of Protocol."]
    ]
    memory = add_memory(
        [],
        source_agent=source["source_agent"],
        task_topic=source["task_topic"],
        summary=source["summary"],
        tags=source["tags"],
        trace=source["trace"],
    )

    record = memory[0]
    graph_nodes = {
        node
        for edge in record["hypergraph"]["hyperedges"]
        for node in edge.get("nodes", [])
    }
    assert "shirley temple" in record["entities"]
    assert "entity:shirley temple" in graph_nodes
    assert not any(node.startswith("expected_reuse:") for node in graph_nodes)


def test_add_memory_remains_compatible_and_gains_chain_fields() -> None:
    source = _evidence_record()
    memory = add_memory(
        [],
        source_agent=source["source_agent"],
        task_topic=source["task_topic"],
        summary=source["summary"],
        tags=source["tags"],
        trace=source["trace"],
    )

    assert len(memory) == 1
    assert memory[0]["memory_type"] == "evidence"
    assert memory[0]["evidence_chain_valid"] is True
    assert memory[0]["hypergraph"]["evidence_chain"]["completeness"] == 1.0


def test_semantic_mode_recalls_without_keyword_overlap(monkeypatch) -> None:
    monkeypatch.setattr(
        "hypergraph_memory_method.retrieval.embed_text",
        lambda text: EmbeddingResult(vector=[1.0, 0.0], provider="local", model="test"),
    )
    memory = [
        {
            **_evidence_record("mem-0001"),
            "task_topic": "unrelated words",
            "summary": "different vocabulary",
            "embedding": [1.0, 0.0],
        },
        {
            **_evidence_record("mem-0002"),
            "task_topic": "other subject",
            "summary": "another topic",
            "embedding": [0.0, 1.0],
        },
    ]

    hits = search_memory_by_mode(
        memory, "meaning with no shared token", top_k=1, mode="semantic"
    )

    assert hits[0]["memory_id"] == "mem-0001"
    assert hits[0]["retrieval_mode"] == "semantic"


def test_hypergraph_ranks_complete_chain_above_text_tie() -> None:
    distractor = {
        "memory_id": "mem-0001",
        "source_agent": "summarizer",
        "task_topic": "What government position did Shirley Temple hold?",
        "summary": "Unverified government position note about Shirley Temple.",
        "tags": ["misc"],
    }
    correct = _evidence_record("mem-0002")
    query = "What government position did Shirley Temple hold?"

    keyword_hits = search_memory_by_mode(
        [distractor, correct], query, top_k=2, mode="keyword"
    )
    hypergraph_hits = search_memory_by_mode(
        [distractor, correct], query, top_k=2, mode="hypergraph"
    )
    fusion_hits = search_memory_by_mode(
        [distractor, correct], query, top_k=2, mode="fusion"
    )

    assert keyword_hits[0]["memory_id"] == "mem-0001"
    assert hypergraph_hits[0]["memory_id"] == "mem-0002"
    assert fusion_hits[0]["memory_id"] == "mem-0002"
    assert (
        hypergraph_hits[0]["evidence_chain_quality_score"]
        > hypergraph_hits[1]["evidence_chain_quality_score"]
    )


def test_persistent_store_survives_reopen_and_rebuilds_indexes(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    with SQLiteHypergraphMemoryStore(database) as store:
        stored = store.upsert(_evidence_record())
        assert store.count() == 1
        assert stored["evidence_chain_valid"] is True
        assert (
            store.search("biography", top_k=1, mode="tag")[0]["memory_id"] == "mem-0002"
        )

    with SQLiteHypergraphMemoryStore(database) as reopened:
        assert reopened.count() == 1
        assert reopened.get("mem-0002")["summary"].startswith("Shirley Temple")
        assert reopened.rebuild_indexes() == 1
        assert (
            reopened.search("Chief of Protocol", top_k=1, mode="hypergraph")[0][
                "memory_id"
            ]
            == "mem-0002"
        )


def test_persistent_store_keeps_same_legacy_id_from_different_tasks(
    tmp_path: Path,
) -> None:
    first = _evidence_record("mem-0001")
    second = _evidence_record("mem-0001")
    second["created_at"] = "2026-01-02T00:00:00+00:00"
    second["trace"]["task_meta"]["task_id"] = "chain-002:turn-1"
    second["trace"]["task_meta"]["group_id"] = "chain-002"

    with SQLiteHypergraphMemoryStore(tmp_path / "global.sqlite3") as store:
        stored = store.upsert_many([first, second])
        assert stored[0]["global_memory_id"] != stored[1]["global_memory_id"]
        assert store.count() == 2


def test_persistent_store_updates_only_explicit_reuse(tmp_path: Path) -> None:
    with SQLiteHypergraphMemoryStore(tmp_path / "reuse.sqlite3") as store:
        stored = store.upsert(_evidence_record())
        unchanged = store.record_reuse(
            stored["global_memory_id"], adopted=False, successful=True
        )
        assert unchanged["reuse_count"] == 0

        reused = store.record_reuse(
            stored["global_memory_id"], adopted=True, successful=True
        )
        assert reused["reuse_count"] == 1
        assert reused["reuse_success_count"] == 1
        assert reused["last_reused_at"]
        assert reused["record_version"] == 2


def test_existing_write_and_search_interfaces_support_opt_in_persistence(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "transparent.sqlite3"
    monkeypatch.setenv("HYPERGRAPH_MEMORY_DB", str(database))
    monkeypatch.setenv("HYPERGRAPH_MEMORY_AUTOPERSIST", "1")
    monkeypatch.setenv("HYPERGRAPH_MEMORY_AUTOLOAD", "1")

    source = _evidence_record("mem-0007")
    add_memory(
        [],
        source_agent=source["source_agent"],
        task_topic=source["task_topic"],
        summary=source["summary"],
        tags=source["tags"],
        trace=source["trace"],
    )

    with SQLiteHypergraphMemoryStore(database) as store:
        assert store.count() == 1
    assert load_persistent_memory()[0]["memory_id"] == "mem-0001"
    hits = search_hypergraph_memory([], "Chief of Protocol Shirley Temple", top_k=1)
    assert hits[0]["memory_id"] == "mem-0001"


def test_persistence_is_disabled_without_explicit_database(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "disabled.sqlite3"
    monkeypatch.delenv("HYPERGRAPH_MEMORY_DB", raising=False)
    monkeypatch.setenv("HYPERGRAPH_MEMORY_AUTOPERSIST", "1")

    build_hypergraph_view(_evidence_record())

    assert database.exists() is False


def test_reuse_evaluation_does_not_treat_candidates_as_adopted() -> None:
    hit = normalize_memory_record(_evidence_record("mem-0001"))
    task = {
        "task_id": "chain-001:turn-2",
        "turn_id": "turn-2",
        "expected_reuse": ["turn-1.answer", "turn-1.supporting_facts"],
        "memory_hits": [hit],
        "memory_context": [{"memory_id": "mem-0001"}],
        "execution_result": {},
        "exact_match": True,
    }

    row = evaluate_task_reuse(
        task,
        question_to_turn={
            "what government position did shirley temple hold?": "turn-1"
        },
    )

    assert row["memory_retrieval_correct"] is True
    assert row["used_memory_ids"] == []
    assert row["correct_reuse"] is False
    assert row["evidence_chain_reuse_f1"] > 0.0


def test_paired_memory_gain_reports_improvement_and_degradation() -> None:
    full = {
        "task_results": [
            {"task_id": "t1", "expected_reuse": ["turn-1.answer"], "exact_match": True},
            {
                "task_id": "t2",
                "expected_reuse": ["turn-1.answer"],
                "exact_match": False,
            },
        ]
    }
    no_memory = {
        "task_results": [
            {
                "task_id": "t1",
                "expected_reuse": ["turn-1.answer"],
                "exact_match": False,
            },
            {"task_id": "t2", "expected_reuse": ["turn-1.answer"], "exact_match": True},
        ]
    }

    report = evaluate_paired_memory_gain(full, no_memory)

    assert report["summary"]["memory_improvement_count"] == 1
    assert report["summary"]["memory_degradation_count"] == 1
    assert report["summary"]["memory_causal_net_gain_rate"] == 0.0


def test_benchmark_proves_hypergraph_is_best_single_mode() -> None:
    distractor = {
        "memory_id": "mem-0001",
        "source_agent": "summarizer",
        "task_topic": "What government position did Shirley Temple hold?",
        "summary": "Unverified government position note about Shirley Temple.",
        "tags": ["misc"],
    }
    correct = _evidence_record("mem-0002")

    report = evaluate_retrieval_modes(
        [distractor, correct],
        [
            {
                "case_id": "evidence-chain-case",
                "query": "What government position did Shirley Temple hold?",
                "expected_memory_ids": ["mem-0002"],
            }
        ],
        top_k=1,
    )

    assert report["comparison"]["hypergraph_is_best_single"] is True
    assert report["comparison"]["hypergraph_is_at_most_second_to_fusion"] is True
    assert report["modes"]["hypergraph"]["summary"]["evidence_chain_recall_at_k"] == 1.0


def test_hypergraph_ignores_high_degree_source_hub() -> None:
    memory = [
        {
            "memory_id": f"mem-{index:04d}",
            "source_agent": "summarizer",
            "created_at": "2026-01-01T00:00:00+00:00",
            "task_topic": (
                "target protocol memory"
                if index == 29
                else f"unrelated archive {index}"
            ),
            "summary": "target protocol memory" if index == 29 else "unrelated archive",
            "supporting_facts": [["shared generic source", f"fact {index}"]],
            "answer": "target" if index == 29 else "unrelated",
            "tags": [],
        }
        for index in range(30)
    ]

    hits = search_memory_by_mode(
        memory, "target protocol memory", top_k=3, mode="hypergraph"
    )

    assert hits[0]["memory_id"] == "mem-0029"
    assert all(hit["memory_id"] == "mem-0029" for hit in hits)


def test_hypergraph_index_is_reused_for_an_unchanged_memory_snapshot() -> None:
    clear_hypergraph_index_cache()
    memory = [_evidence_record("mem-0001")]

    first = get_cached_hypergraph_index(memory)
    second = get_cached_hypergraph_index(memory)
    changed = get_cached_hypergraph_index([*memory, _evidence_record("mem-0002")])

    assert first is second
    assert changed is not first


def test_retrieval_modes_use_the_declared_memory_organization(monkeypatch) -> None:
    monkeypatch.setattr(
        "hypergraph_memory_method.retrieval.embed_text",
        lambda text: EmbeddingResult(
            vector=[float("shirley" in text.lower()), 1.0],
            provider="test",
            model="test",
        ),
    )
    memory = [_evidence_record()]

    for mode in ("keyword", "semantic"):
        hits = search_memory_by_mode(memory, "Shirley biography", top_k=1, mode=mode)
        assert hits[0]["memory_organization"] == "text"
        assert "hypergraph" not in hits[0]
        assert "evidence_chain" not in hits[0]
        assert hits[0]["reusable_memory"]["memory_organization"] == "text"

    for mode in ("hypergraph", "fusion"):
        hits = search_memory_by_mode(memory, "Shirley biography", top_k=1, mode=mode)
        assert hits[0]["memory_organization"] == "hypergraph"
        assert hits[0]["reusable_memory"]["evidence_chain_steps"]


def test_environment_dispatches_all_five_retrieval_modes(monkeypatch) -> None:
    monkeypatch.setattr(
        "hypergraph_memory_method.retrieval.embed_text",
        lambda text: EmbeddingResult(
            vector=[float("government" in text.lower()), 1.0],
            provider="test",
            model="test",
        ),
    )
    memory = [_evidence_record()]

    for mode in ("keyword", "tag", "semantic", "hypergraph", "fusion"):
        monkeypatch.setenv("MEMORY_RETRIEVAL_MODE", mode)
        assert configured_retrieval_mode() == mode
        assert memory_organization_for_mode(mode) == (
            "hypergraph" if mode in {"hypergraph", "fusion"} else "text"
        )
        search_hypergraph_memory(memory, "biography government-position", top_k=1)
