from __future__ import annotations

from hypergraph_memory_method import build_hypergraph_view, search_context
from hypergraph_memory_method.layers import (
    EPISODIC_LAYER,
    EXPERIENCE_LAYER,
    KNOWLEDGE_LAYER,
    memory_layer_for_record,
)
from hypergraph_memory_method.retrieval_batches import (
    KNOWLEDGE_FOLLOWUP_BATCH,
    MEMORY_ROUTE_BATCH,
    build_retrieval_batches,
    resolve_followup_decision,
)
from hypergraph_memory_method.store import SQLiteHypergraphMemoryStore


def test_three_primary_layers_are_explicit_and_backward_compatible() -> None:
    assert memory_layer_for_record({"memory_type": "evidence"}) == EPISODIC_LAYER
    assert memory_layer_for_record({"memory_type": "experience"}) == EXPERIENCE_LAYER
    assert memory_layer_for_record({"memory_layer": "knowledge"}) == KNOWLEDGE_LAYER

    context_hits = search_context(
        {
            "task_id": "task-1",
            "question": "Which protocol is used?",
            "context": [
                {
                    "title": "Protocol",
                    "sentences": ["The agents use a structured protocol."],
                }
            ],
        },
        top_k=1,
    )

    assert context_hits[0]["memory_layer"] == KNOWLEDGE_LAYER


def test_hypergraph_records_cross_layer_provenance() -> None:
    episodic = {
        "memory_id": "mem-0001",
        "memory_type": "evidence",
        "source_agent": "summarizer",
        "task_topic": "Which protocol is used?",
        "summary": "A structured protocol is used.",
        "trace": {
            "agent_outputs": [
                {
                    "agent": "RetrieverAgent",
                    "context_hit_ids": ["task-1:sentence:001"],
                }
            ]
        },
    }
    episodic_view = build_hypergraph_view(episodic, persist=False)

    assert episodic["memory_layer"] == EPISODIC_LAYER
    assert episodic_view["memory_layers"] == [EPISODIC_LAYER]
    assert any(
        edge["type"] == "knowledge_grounding" for edge in episodic_view["hyperedges"]
    )

    experience = {
        "memory_id": "mem-0002",
        "memory_type": "experience",
        "source_agent": "summarizer",
        "task_topic": "Protocol selection",
        "summary": "Prefer compact structured fields.",
        "source_memory_ids": ["mem-0001"],
    }
    experience_view = build_hypergraph_view(experience, persist=False)

    assert experience["memory_layer"] == EXPERIENCE_LAYER
    assert experience_view["memory_layers"] == [EXPERIENCE_LAYER]
    assert any(
        edge["type"] == "experience_abstraction"
        for edge in experience_view["hyperedges"]
    )


def test_retrieval_contract_has_parallel_memory_route_and_planner_followup() -> None:
    batches = build_retrieval_batches(need_memory=True, need_knowledge=True)

    assert batches[0]["batch_id"] == MEMORY_ROUTE_BATCH
    assert batches[0]["layers"] == [EPISODIC_LAYER, EXPERIENCE_LAYER]
    assert batches[0]["execution"] == "single_query_multi_layer"
    assert batches[1]["batch_id"] == KNOWLEDGE_FOLLOWUP_BATCH
    assert batches[1]["decision_owner"] == "planner"
    assert batches[1]["decision"] == "pending"

    assert resolve_followup_decision({"need_knowledge": True})[0] is True
    assert (
        resolve_followup_decision(
            {
                "need_knowledge": True,
                "next_batch_decision": {
                    "decision": "skip",
                    "reason": "episodic answer can be verified directly",
                },
            }
        )[0]
        is False
    )


def test_persistent_store_can_filter_primary_memory_layers(tmp_path) -> None:
    with SQLiteHypergraphMemoryStore(tmp_path / "memory.sqlite3") as store:
        store.upsert(
            {
                "memory_id": "mem-0001",
                "memory_type": "evidence",
                "source_agent": "summarizer",
                "task_topic": "Question",
                "summary": "Answer",
            }
        )
        store.upsert(
            {
                "memory_id": "mem-0002",
                "memory_type": "experience",
                "source_agent": "summarizer",
                "task_topic": "Task pattern",
                "summary": "Reusable strategy",
            }
        )

        episodic = store.list(memory_layers=[EPISODIC_LAYER])
        experience = store.list(memory_layers=[EXPERIENCE_LAYER])

    assert [record["memory_id"] for record in episodic] == ["mem-0001"]
    assert [record["memory_id"] for record in experience] == ["mem-0002"]
