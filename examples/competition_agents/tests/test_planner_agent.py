from __future__ import annotations

from agents.planner.agent import planner_agent
from competition_agents.state import initial_state


class BadDialoguePlannerClient:
    def plan_with_raw(self, task: str) -> tuple[list[str], str]:
        raw = (
            '1. Ask: "What\'s the main subject of the movie?"\n'
            '2. Receive response: "The movie is about a love story."\n'
            '3. Ask: "What does the protagonist do in the movie?"\n'
            '4. Receive response: "'
        )
        return [line for line in raw.splitlines() if line], raw


def test_planner_sanitizes_dialogue_script_output(monkeypatch) -> None:
    monkeypatch.setattr("agents.planner.agent.get_llm_client", lambda role: BadDialoguePlannerClient())

    state = initial_state("what is the film about?", "protocol", [])
    state["task_meta"] = {
        "dataset_type": "continuous",
        "turn_id": "turn-2",
        "question": "what is the film about?",
        "topic": "Dunkirk (2017 film)",
        "expected_reuse": ["turn-1.answer", "group.supporting_facts"],
        "supporting_passage": (
            "Dunkirk is a 2017 war film written, directed, and produced by Christopher Nolan "
            "that depicts the Dunkirk evacuation of World War II."
        ),
    }

    result = planner_agent(state)

    assert result["agent_outputs"][-1]["plan_sanitized"] is True
    assert result["retrieval_policy"]["need_memory"] is False
    assert result["retrieval_policy"]["next_batch_interface"] == {
        "batch_id": "knowledge_followup",
        "decision_owner": "planner",
        "allowed_decisions": ["run", "skip"],
        "decision_field": "next_batch_decision",
    }
    assert result["retrieval_batches"][0]["layers"] == [
        "episodic",
        "experience",
    ]
    wire_policy = result["messages"][-1]["payload"]["p"]["retrieval_policy"]
    assert "retrieval_batches" not in wire_policy
    assert "next_batch_interface" not in wire_policy
    assert result["plan"] == [
        "Identify the question target and requested answer type.",
        "Resolve cross-turn references against available prior memory before retrieval.",
        "Retrieve the most relevant supporting evidence for the target.",
        "Route evidence to reasoning and summarize a concise final answer.",
    ]
    assert not any("Receive response" in step or "love story" in step for step in result["plan"])
