from __future__ import annotations

from agents.retriever.agent import retriever_agent
from competition_agents.memory import add_memory
from competition_agents.state import initial_state


def test_retriever_forwards_compact_memory_context_without_reembedding_summary() -> None:
    """Retriever 应转发裁剪后的记忆上下文，但 evidence latent 不再包含记忆摘要正文。"""

    unique_summary = "UNIQUE_MEMORY_SUMMARY_SHOULD_NOT_BE_REEMBEDDED"
    memory = add_memory(
        [],
        source_agent="summarizer",
        task_topic="Protocol memory for openEuler",
        summary=unique_summary,
        tags=["protocol", "memory"],
    )
    state = initial_state("reuse protocol memory", "protocol", memory)

    result = retriever_agent(state)

    assert result["memory_context"]
    assert result["memory_context"][0]["summary"] == unique_summary
    evidence_ref = result["state_refs"]["evidence"]
    evidence_state = result["latent_states"][evidence_ref]
    assert unique_summary not in evidence_state["source_text"]

    payload = result["messages"][-1]["payload"]
    assert payload["m"] == ["mem-0001"]
    assert payload["r"]["memory_context"][0]["memory_id"] == "mem-0001"
