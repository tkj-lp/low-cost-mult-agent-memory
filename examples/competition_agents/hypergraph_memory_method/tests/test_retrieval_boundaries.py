from copy import deepcopy

import pytest
from competition_agents.embedding import EmbeddingResult
from hypergraph_memory_method import (
    SQLiteHypergraphMemoryStore,
    clear_context_cache,
    compact_trace,
    context,
    deduplicate_memory,
    index,
    search_context,
    search_hypergraph_memory,
    search_memory_by_mode,
)
from hypergraph_memory_method.index import (
    _cosine_similarity,
    get_cached_hypergraph_index,
)


@pytest.fixture(autouse=True)
def isolated_memory(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.delenv("HYPERGRAPH_MEMORY_DB", raising=False)
    index.clear_hypergraph_index_cache()
    clear_context_cache()
    yield
    index.clear_hypergraph_index_cache()
    clear_context_cache()


def record(memory_id="mem-1", **values):
    return {
        "memory_id": memory_id,
        "source_agent": "test",
        "created_at": "2026-09-20T00:00:00+00:00",
        "task_topic": "Orion location",
        "summary": "Orion is in Paris",
        "tags": ["orion"],
        **values,
    }


@pytest.mark.parametrize("mode", ["keyword", "tag", "semantic", "hypergraph", "fusion"])
def test_layer_filter_precedes_top_k(mode):
    memories = [record(f"wrong-{i}", memory_type="evidence") for i in range(8)]
    memories.append(record("experience", memory_type="experience"))
    hits = search_memory_by_mode(
        memories, "Orion location", top_k=1, mode=mode, memory_types={"experience"}
    )
    assert [hit["memory_id"] for hit in hits] == ["experience"]


@pytest.mark.parametrize("mode", ["keyword", "tag", "semantic", "hypergraph", "fusion"])
def test_duplicate_ids_are_rejected_before_indexing(mode):
    with pytest.raises(ValueError, match="Duplicate memory_id"):
        search_memory_by_mode(
            [record(), record(summary="Orion is in London")],
            "Orion",
            top_k=2,
            mode=mode,
        )


@pytest.mark.parametrize("mode", ["keyword", "tag", "semantic", "hypergraph", "fusion"])
def test_review_filters_apply_to_every_retrieval_mode(mode):
    memories = [record("good")]
    memories += [
        record(status, review_status=status)
        for status in ("archived", "rejected", "superseded")
    ]
    memories += [
        record("bad", quality_status="failed"),
        record("disabled", use_as_positive=False),
    ]
    memories += [
        record("candidate", memory_type="experience", review_status="candidate")
    ]
    hits = search_memory_by_mode(memories, "Orion", top_k=20, mode=mode)
    assert [hit["memory_id"] for hit in hits] == ["good"]


def test_persistent_reload_does_not_resurrect_episodes_or_rejected_experience(
    tmp_path, monkeypatch
):
    path = tmp_path / "legacy.db"
    with SQLiteHypergraphMemoryStore(path) as store:
        store.upsert(record("old-episode", memory_type="evidence"))
        store.upsert(record("bad", memory_type="experience", quality_status="failed"))
        store.upsert(
            record("pending", memory_type="experience", quality_status="suspect")
        )
        store.upsert(
            record("valid", memory_type="experience", quality_status="verified")
        )
    monkeypatch.setenv("HYPERGRAPH_MEMORY_DB", str(path))
    monkeypatch.setenv("HYPERGRAPH_MEMORY_AUTOLOAD", "1")
    monkeypatch.setenv("MEMORY_RETRIEVAL_MODE", "keyword")
    assert [
        hit["memory_id"] for hit in search_hypergraph_memory([], "Orion", top_k=10)
    ] == ["valid"]


@pytest.mark.parametrize(
    "changed",
    [
        {"failure_pattern": "never reuse a stale source"},
        {"trigger_conditions": ["after verification"]},
        {"task_family": "code_execution"},
        {"experience_scope_key": "other-project"},
        {"review_status": "rejected"},
    ],
)
def test_experience_deduplication_preserves_different_conditions(changed):
    left = record(
        memory_type="experience",
        problem_pattern="resolve entity",
        solution_pattern="check source",
    )
    right = {**left, "memory_id": "mem-2", **changed}
    assert len(deduplicate_memory([left, right])) == 2


def test_experience_merge_preserves_all_provenance():
    left = record(
        memory_type="experience",
        problem_pattern="resolve entity",
        solution_pattern="check source",
        source_memory_ids=["qa-1"],
        knowledge_refs=["doc-1"],
    )
    right = {
        **left,
        "memory_id": "mem-2",
        "source_memory_ids": ["qa-2"],
        "knowledge_refs": ["doc-2"],
    }
    merged = deduplicate_memory([left, right])
    assert len(merged) == 1
    assert set(merged[0]["source_memory_ids"]) == {"qa-1", "qa-2"}
    assert set(merged[0]["knowledge_refs"]) == {"doc-1", "doc-2"}


def test_hash_vectors_do_not_contribute_semantic_scores():
    memories = [
        record(
            embedding=[1.0, 0.0], embedding_provider="hash", embedding_model="hash-v1"
        )
    ]
    hit = search_memory_by_mode(memories, "Orion", top_k=1, mode="hypergraph")[0]
    assert hit["embedding_score"] == 0
    assert "semantic" not in hit["recall_channels"]
    assert not search_memory_by_mode(
        memories, "unrelated zeppelin", top_k=1, mode="semantic"
    )


@pytest.mark.parametrize("right", [[1.0], [float("nan"), 0.0], [float("inf"), 0.0]])
def test_invalid_or_mismatched_vectors_have_no_similarity(right):
    assert _cosine_similarity([1.0, 0.0], right) == 0


def test_semantic_retrieval_reencodes_incompatible_vectors(monkeypatch):
    calls = []

    def encode(text):
        calls.append(text)
        return EmbeddingResult([1.0, 0.0], "local", "current")

    monkeypatch.setattr("hypergraph_memory_method.retrieval.embed_text", encode)
    memories = [
        record(
            embedding=[1.0, 0.0, 0.0], embedding_provider="local", embedding_model="old"
        )
    ]
    hits = search_memory_by_mode(memories, "Orion", top_k=1, mode="semantic")
    assert len(calls) == 2
    assert hits[0]["embedding_model"] == "current"
    assert memories[0]["embedding_model"] == "old"


def test_cache_invalidates_on_embedding_and_provenance_changes():
    memories = [record(embedding=[1.0, 0.0], source_memory_ids=["qa-1"])]
    first = get_cached_hypergraph_index(memories)
    assert get_cached_hypergraph_index(memories) is first
    memories[0]["embedding"] = [0.0, 1.0]
    second = get_cached_hypergraph_index(memories)
    assert second is not first
    memories[0]["source_memory_ids"] = ["qa-2"]
    assert get_cached_hypergraph_index(memories) is not second


def test_cached_hits_cannot_be_mutated_by_consumers():
    memories = [record()]
    cached = get_cached_hypergraph_index(memories)
    hit = cached.search("Orion", top_k=1)[0]
    hit["tags"].append("corrupted")
    assert "corrupted" not in cached.search("Orion", top_k=1)[0]["tags"]


def test_only_selected_hits_materialize_evidence_chains(monkeypatch):
    cached = get_cached_hypergraph_index([record(str(i)) for i in range(20)])
    projection = index.compact_reuse_projection
    calls = []

    def counted(value, **kwargs):
        calls.append(value["memory_id"])
        return projection(value, **kwargs)

    monkeypatch.setattr(index, "compact_reuse_projection", counted)
    hits = cached.search("Orion", top_k=2)
    assert len(calls) == len(hits) == 2


def test_cached_merged_experience_refreshes_all_source_counters():
    memories = [
        record(
            str(i),
            memory_type="experience",
            problem_pattern="resolve entity",
            solution_pattern="check source",
            reuse_count=1,
        )
        for i in range(2)
    ]
    cached = get_cached_hypergraph_index(memories)
    memories[1]["reuse_count"] = 4
    assert get_cached_hypergraph_index(memories) is cached
    assert next(iter(cached.memory_by_id.values()))["reuse_count"] == 5


def test_cached_hash_queries_do_not_retokenize_every_document(monkeypatch):
    cached = get_cached_hypergraph_index(
        [record(str(i), summary=f"entry {i}") for i in range(30)]
    )
    documents = {item["search_document"] for item in cached.memory_by_id.values()}
    tokenize = index.tokenize

    def checked(text):
        assert text not in documents
        return tokenize(text)

    monkeypatch.setattr(index, "tokenize", checked)
    cached.search("Orion", top_k=1)


def test_compact_trace_drops_recursive_objects_before_copying():
    class RecursivePayload:
        def __deepcopy__(self, memo):
            raise AssertionError("discarded history must not be traversed")

    trace = {
        "memory_hits": [{"memory_id": "old", "trace": RecursivePayload()}],
        "plan": ["lookup"],
    }
    result = compact_trace(trace)
    assert result["memory_hits"] == [{"memory_id": "old"}]
    result["plan"].append("changed")
    assert trace["plan"] == ["lookup"]


def context_meta():
    return {
        "task_id": "batch:turn-1",
        "question": "Where is Orion?",
        "topic": "Orion",
        "context": [{"title": "Orion", "sentences": ["Orion is in Paris."]}],
        "conversation_context": ["What is Orion?", "A project."],
    }


def test_context_cache_reuses_graph_and_does_not_mutate_filters(monkeypatch):
    meta = context_meta()
    original = deepcopy(meta)
    build = context._build_context_hypergraph
    builds = []

    def counted(*args):
        builds.append(1)
        return build(*args)

    monkeypatch.setattr(context, "_build_context_hypergraph", counted)
    first = search_context(
        meta,
        dataset_type="topiocqa",
        top_k=10,
        exclude_kinds={"dialogue_question", "dialogue_answer"},
    )
    first[0]["metadata"]["poison"] = True
    second = search_context(meta, dataset_type="topiocqa", top_k=10)
    assert len(builds) == 1
    assert any(hit["kind"].startswith("dialogue_") for hit in second)
    assert all("poison" not in hit["metadata"] for hit in second)
    meta["context"][0]["sentences"] = ["Orion is in London."]
    third = search_context(meta, dataset_type="topiocqa", top_k=10)
    assert len(builds) == 2
    assert any("London" in hit["text"] for hit in third)
    assert original["context"][0]["sentences"] == ["Orion is in Paris."]


def test_context_does_not_use_answer_annotations_as_evidence():
    meta = context_meta()
    meta["context"][0]["source"] = "official_answer_annotation"
    with pytest.raises(ValueError, match="Answer annotations"):
        search_context(meta, dataset_type="topiocqa")


def test_nonpositive_context_budget_returns_no_hits():
    assert search_context(context_meta(), top_k=0) == []
    assert search_context(context_meta(), top_k=-1) == []
