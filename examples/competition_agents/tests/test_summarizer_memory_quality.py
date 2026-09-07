from __future__ import annotations

from agents.summarizer.agent import summarizer_agent
from competition_agents.state import initial_state


def test_continuous_summarizer_writes_pending_qa_only(monkeypatch) -> None:
    monkeypatch.setenv("COMPETITION_SUMMARIZER_MODE", "direct")
    monkeypatch.setenv("COMPETITION_MEMORY_DISTILLATION_MODE", "local_fallback")

    state = initial_state(
        "Question: what is the film about?",
        "protocol",
        task_meta={
            "dataset_type": "continuous",
            "group_id": "chain-001",
            "turn_id": "turn-2",
            "question": "what is the film about?",
            "expected_answer": "an evacuation film",
        },
    )
    state["evidence"] = ["Dunkirk is an evacuation film."]
    state["execution_result"] = {
        "answer_candidate": "a love story",
        "reasoning_summary": "The model selected an unsupported answer.",
    }

    result = summarizer_agent(state)

    assert result["memory_write_count"] == 1
    assert result["experience_write_count"] == 0
    assert result["latest_evidence_write"]["memory_type"] == "evidence"
    assert result["latest_evidence_write"]["quality_status"] == "pending"
    assert result["latest_experience_write"] == {}
    assert len(result["memory"]) == 1
