from __future__ import annotations

from agents.planner.agent import _build_retrieval_policy, planner_agent
from agents.retriever.agent import retriever_agent
from competition_agents.memory import add_memory
from competition_agents.state import initial_state
from hypergraph_memory_method import memory_layer_for_record


class CapturingPlannerClient:
    def __init__(self) -> None:
        self.task = ""

    def plan_with_raw(self, task: str) -> tuple[list[str], str]:
        self.task = task
        plan = [
            "Identify the referenced subject.",
            "Retrieve facts about the referenced subject.",
            "Summarize the final answer.",
        ]
        return plan, "\n".join(plan)


def test_planner_retrieves_experience_before_planning(monkeypatch) -> None:
    experience = {
        "memory_id": "exp-001",
        "memory_type": "experience",
        "memory_layer": "experience",
        "summary": "Resolve pronouns from prior question-answer memory.",
        "problem_pattern": "cross_turn_coreference",
        "solution_pattern": "retrieve_prior_answer",
        "failure_pattern": "guess_referent",
        "trigger_conditions": ["pronoun", "cross_turn"],
    }
    captured: dict[str, object] = {}

    def fake_search(memory, query, **kwargs):
        captured["query"] = query
        captured["kwargs"] = kwargs
        return [experience]

    client = CapturingPlannerClient()
    monkeypatch.setattr("agents.planner.agent.search_memory", fake_search)
    monkeypatch.setattr("agents.planner.agent.get_llm_client", lambda role: client)

    state = initial_state(
        "Question: What is it about?",
        "protocol",
        [experience, {"memory_id": "qa-001", "memory_type": "evidence", "summary": "Dunkirk"}],
        task_meta={
            "dataset_type": "continuous",
            "group_id": "chain-002",
            "turn_id": "turn-2",
            "question": "What is it about?",
        },
    )
    result = planner_agent(state)

    assert captured["kwargs"]["memory_layers"] == ["experience"]
    assert result["experience_hits"] == [experience]
    assert "cross_turn_coreference" in client.task
    assert result["retrieval_policy"]["need_qa_memory"] is True
    assert result["retrieval_policy"]["knowledge_steps"]


def test_planner_qa_gate_uses_question_content_not_dataset_annotations() -> None:
    qa_memory = [{"memory_id": "qa-001", "memory_type": "evidence", "summary": "Dunkirk"}]
    self_contained = initial_state(
        "Question: When was Dunkirk released?",
        "protocol",
        qa_memory,
        task_meta={
            "dataset_type": "continuous",
            "turn_id": "turn-2",
            "question": "When was Dunkirk released?",
            "expected_reuse": ["turn-1.answer"],
        },
    )
    anaphoric = initial_state(
        "Question: What is the film about?",
        "protocol",
        qa_memory,
        task_meta={
            "dataset_type": "continuous",
            "turn_id": "turn-2",
            "question": "What is the film about?",
        },
    )

    self_contained_policy, _ = _build_retrieval_policy(
        self_contained, ["Retrieve the release date."], []
    )
    anaphoric_policy, _ = _build_retrieval_policy(
        anaphoric, ["Identify the referenced film."], []
    )

    assert self_contained_policy["need_qa_memory"] is False
    assert anaphoric_policy["need_qa_memory"] is True


def test_retriever_obeys_qa_gate_and_runs_planned_knowledge_steps(monkeypatch) -> None:
    memory_calls: list[dict] = []
    context_queries: list[str] = []

    def fake_memory_search(memory, query, **kwargs):
        memory_calls.append(kwargs)
        return []

    def fake_context_search(task_meta, *, query, dataset_type, planner_focus, top_k):
        context_queries.append(query)
        return [
            {
                "context_id": f"ctx-{len(context_queries)}",
                "kind": "sentence",
                "title": "Topic",
                "text": f"fact {len(context_queries)}",
                "context_score": 1.0,
            }
        ]

    monkeypatch.setattr("agents.retriever.agent.search_memory", fake_memory_search)
    monkeypatch.setattr("agents.retriever.agent.search_context", fake_context_search)

    state = initial_state(
        "Question: Which role did she hold?",
        "protocol",
        [],
        task_meta={
            "dataset_type": "continuous",
            "question": "Which role did she hold?",
            "context_source": "hypergraph",
        },
    )
    state["retrieval_policy"] = {
        "need_qa_memory": False,
        "need_memory": False,
        "need_knowledge": True,
        "knowledge_steps": [
            {"step": 1, "goal": "Identify the person", "query": "identify person"},
            {"step": 2, "goal": "Find the role", "query": "find role", "depends_on": 1},
        ],
    }
    result = retriever_agent(state)

    assert memory_calls == []
    assert len(context_queries) == 2
    assert "Prior step findings:" in context_queries[1]
    assert len(result["memory_layer_hits"]["knowledge"]) == 2
    assert result["metrics"]["knowledge_query_count"] == 2


def test_batch_memory_can_hold_qa_and_persistent_experience() -> None:
    memory = add_memory(
        [],
        source_agent="summarizer",
        task_topic="What is it about?",
        summary="It is about an evacuation.",
        tags=["qa-memory"],
        memory_type="evidence",
        archive_scope="batch:chain-001",
        quality_status="pending",
    )
    memory = add_memory(
        memory,
        source_agent="batch_summarizer",
        task_topic="Resolve a cross-turn reference",
        summary="Use the preceding answer to resolve the pronoun.",
        tags=["experience"],
        memory_type="experience",
        archive_scope="global",
        quality_status="verified",
        source_task_correct=True,
        source_memory_ids=[memory[0]["memory_id"]],
    )

    assert memory_layer_for_record(memory[0]) == "episodic"
    assert memory_layer_for_record(memory[1]) == "experience"
    assert memory[1]["source_memory_ids"] == [memory[0]["memory_id"]]
