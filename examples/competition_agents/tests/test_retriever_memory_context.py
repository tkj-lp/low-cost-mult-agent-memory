from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from agents.retriever.agent import (
    _augment_knowledge_query,
    _build_continuous_task_hints,
    _extract_hypergraph_context,
    _rerank_hypergraph_hits,
    retriever_agent,
)
from competition_agents.memory import _rerank_memory_hits, add_memory
from competition_agents.state import initial_state


def test_retriever_keeps_memory_context_in_state_and_sends_refs_only() -> None:
    """Retriever 应把记忆详情留在共享状态里，协议包只传记忆引用。"""

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
    assert "source_text" not in evidence_state

    payload = result["messages"][-1]["payload"]
    assert payload["m"] == ["mem-0001"]
    assert "memory_context" not in payload["r"]
    assert payload["r"]["hits"] == 1


def test_retriever_uses_planner_focus_in_memory_query(monkeypatch) -> None:
    """Retriever 应把 Planner 的计划拼进检索查询，而不是只使用原始 task。"""

    captured: dict[str, object] = {}

    def fake_search_memory(memory: list[dict], query: str, *, top_k: int, memory_types=None, **kwargs) -> list[dict]:
        captured["query"] = query
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr("agents.retriever.agent.search_memory", fake_search_memory)

    state = initial_state("Find the diplomatic role held by Shirley Temple.", "protocol", [])
    state["plan"] = [
        "Identify the actress who portrayed Corliss Archer.",
        "Retrieve biographical details about Shirley Temple.",
        "Extract the government position she later held.",
    ]
    state["retrieval_policy"] = {
        "need_qa_memory": True,
        "need_memory": True,
        "need_knowledge": True,
    }

    result = retriever_agent(state)

    assert "Planner retrieval focus:" in str(captured["query"])
    assert "Retrieve biographical details about Shirley Temple" in str(captured["query"])
    assert captured["top_k"] == 3
    assert result["agent_outputs"][-1]["planner_focus"] == [
        "Identify the actress who portrayed Corliss Archer",
        "Retrieve biographical details about Shirley Temple",
        "Extract the government position she later held",
    ]
    assert not any("Planner-approved" in item for item in result["evidence"])
    assert result["retrieval_batches"]
    assert not any("Planner-guided retrieval focus:" in item for item in result["evidence"])


def test_hypergraph_knowledge_channel_excludes_dialogue_nodes() -> None:
    hits = [
        {
            "context_id": "dialogue-1",
            "kind": "dialogue_answer",
            "title": "Prior answer",
            "text": "A previous answer that must stay in episodic memory.",
            "context_score": 99.0,
        },
        {
            "context_id": "knowledge-1",
            "kind": "knowledge",
            "title": "Current source",
            "text": "Current evidence for the requested attribute.",
            "context_score": 1.0,
        },
    ]
    with patch("agents.retriever.agent.search_context", return_value=hits):
        evidence, selected = _extract_hypergraph_context(
            {"question": "What is the requested attribute?"},
            [],
            limit=3,
        )

    assert evidence == ["Current source: Current evidence for the requested attribute."]
    assert [hit["context_id"] for hit in selected] == ["knowledge-1"]


def test_expected_reuse_metadata_does_not_override_retrieval_score() -> None:
    """连续任务排序应优先复用 expected_reuse 指向的历史轮次。"""

    hits = [
        {
            "memory_id": "mem-0006",
            "group_id": "chain-001",
            "turn_id": "turn-6",
            "artifact_types": ["answer", "evidence"],
            "retrieval_score": 0.23,
        },
        {
            "memory_id": "mem-0007",
            "group_id": "chain-001",
            "turn_id": "turn-7",
            "artifact_types": ["answer", "evidence"],
            "retrieval_score": 0.22,
        },
    ]
    query = "\n".join(
        [
            "Question: what kinds of fish are commonly found in this sea?",
            "Group id: chain-001",
            "Expected reuse: turn-7.answer, group.supporting_facts",
            "Required artifacts: answer, supporting_facts",
        ]
    )

    reranked = _rerank_memory_hits(hits, query)

    assert reranked[0]["memory_id"] == "mem-0006"
    assert reranked[0]["retrieval_score"] > reranked[1]["retrieval_score"]


def test_online_retrieval_hints_do_not_use_evaluation_annotations() -> None:
    hints = _build_continuous_task_hints(
        {
            "group_id": "group-1",
            "turn_id": "turn-3",
            "topic": "Portable topic",
            "topic_section": "Location",
            "expected_reuse": ["turn-2.answer", "group.supporting_facts"],
        }
    )

    joined = "\n".join(hints)
    assert "Portable topic" in joined
    assert "Location" in joined
    assert "expected_reuse" not in joined.lower()
    assert "turn-2.answer" not in joined


def test_knowledge_query_prefers_public_topic_over_unverified_qa_answer() -> None:
    query = _augment_knowledge_query(
        "Question: where is it found?",
        [{"memory_id": "mem-1", "summary": "The referenced animal is an oribi."}],
        {"reason": "cross-turn reference"},
        {"topic": "Oribi", "topic_section": "Distribution"},
    )

    assert "Current topic: Oribi" in query
    assert "Requested section or attribute: Distribution" in query
    assert "The referenced animal is an oribi" not in query


def test_knowledge_query_uses_unverified_qa_only_without_public_topic() -> None:
    query = _augment_knowledge_query(
        "Question: where is it found?",
        [{"memory_id": "mem-1", "summary": "The referenced animal is an oribi."}],
        {"reason": "cross-turn reference"},
        {},
    )

    assert "reference resolution only" in query
    assert "Unverified answer reference: The referenced animal is an oribi" in query


def test_retriever_separately_surfaces_experience_hits(monkeypatch) -> None:
    """Retriever 应把 experience memory 单独暴露出来，而不是全都混成一类。"""

    memory = [
        {
            "memory_id": "mem-0001",
            "memory_type": "evidence",
            "summary": "18 December 2017",
            "retrieval_score": 42.0,
        },
        {
            "memory_id": "mem-0002",
            "memory_type": "experience",
            "summary": "经验记忆：先锁定影片与地区，再选满足介质约束的日期槽位。",
            "problem_pattern": "temporal_slot_selection",
            "solution_pattern": "choose_date_matching_constraints",
            "failure_pattern": "wrong_date_slot",
            "trigger_conditions": ["question:when", "constraint:dvd_release"],
            "supporting_evidence": ["Dunkirk (2017 film): DVD on 18 December in the United Kingdom."],
            "confidence": 0.9,
            "retrieval_score": 51.0,
        },
    ]

    def fake_search_memory(memory_records: list[dict], query: str, *, top_k: int, memory_types=None, **kwargs) -> list[dict]:
        if memory_types == ["evidence"]:
            return [memory_records[0]]
        if memory_types == ["experience"]:
            return [memory_records[1]]
        return []

    monkeypatch.setattr("agents.retriever.agent.search_memory", fake_search_memory)

    state = initial_state("when was dunkirk released on dvd in the united kingdom?", "protocol", memory)
    state["experience_hits"] = [memory[1]]
    state["retrieval_policy"] = {
        "need_qa_memory": True,
        "need_memory": True,
        "need_knowledge": True,
    }
    state["plan"] = [
        "Retrieve the film release sentence.",
        "Check the United Kingdom constraint.",
        "Select the DVD release date.",
    ]

    result = retriever_agent(state)
    retriever_output = result["agent_outputs"][-1]

    assert result["experience_hits"]
    assert result["experience_hits"][0]["memory_type"] == "experience"
    assert retriever_output["experience_hit_ids"] == ["mem-0002"]
    assert retriever_output["experience_hits"][0]["problem_pattern"] == "temporal_slot_selection"
    assert retriever_output["experience_rules"][0]["solution_pattern"] == "choose_date_matching_constraints"
    assert any(item["memory_type"] == "experience" for item in result["memory_context"])


def test_retriever_extracts_context_evidence_from_planner_focus(monkeypatch) -> None:
    """Retriever 应按 Planner 规划从 Context 中抽 supporting facts 给 Reasoner。"""

    def fake_search_memory(memory: list[dict], query: str, *, top_k: int, memory_types=None, **kwargs) -> list[dict]:
        return []

    monkeypatch.setattr("agents.retriever.agent.search_memory", fake_search_memory)

    state = initial_state("Hotpot multi-hop task", "protocol", [])
    state["plan"] = [
        "Identify the actress who portrayed Corliss Archer.",
        "Locate the biography of Shirley Temple.",
        "Extract the government position she held.",
    ]
    state["task_meta"] = {
        "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
        "context": [
            {
                "title": "Kiss and Tell (1945 film)",
                "sentences": [
                    "Kiss and Tell is a 1945 American comedy film starring Shirley Temple as Corliss Archer.",
                ],
            },
            {
                "title": "Shirley Temple",
                "sentences": [
                    "As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.",
                ],
            },
        ],
        "supporting_facts": [
            ["Kiss and Tell (1945 film)", "Kiss and Tell is a 1945 American comedy film starring Shirley Temple as Corliss Archer."],
            ["Shirley Temple", "As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States."],
        ],
    }

    result = retriever_agent(state)

    assert any("Kiss and Tell (1945 film): Kiss and Tell is a 1945 American comedy film starring Shirley Temple as Corliss Archer." == item for item in result["evidence"])
    assert any("Shirley Temple: As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States." == item for item in result["evidence"])
    assert result["agent_outputs"][-1]["context_evidence"]


def test_retriever_prioritizes_answer_bearing_sentence_over_title_overlap(monkeypatch) -> None:
    """Retriever 应优先保留承载答案的职位句，而不是只和片名重合的干扰句。"""

    def fake_search_memory(memory: list[dict], query: str, *, top_k: int, memory_types=None, **kwargs) -> list[dict]:
        return []

    monkeypatch.setattr("agents.retriever.agent.search_memory", fake_search_memory)

    state = initial_state("Hotpot multi-hop task", "protocol", [])
    state["plan"] = [
        "Identify Shirley Temple as the actress who portrayed Corliss Archer.",
        "Retrieve Shirley Temple's biography.",
        "Extract the government position she held.",
    ]
    state["task_meta"] = {
        "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
        "context": [
            {
                "title": "Kiss and Tell (1945 film)",
                "sentences": [
                    "Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer.",
                ],
            },
            {
                "title": "Shirley Temple",
                "sentences": [
                    "Shirley Temple Black was an American actress, singer, dancer, businesswoman, and diplomat.",
                    "As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.",
                ],
            },
            {
                "title": "A Kiss for Corliss",
                "sentences": [
                    "A Kiss for Corliss is a sequel to the 1945 film Kiss and Tell.",
                    "A Kiss for Corliss was retitled Almost a Bride before release.",
                ],
            },
        ],
        "supporting_facts": [
            ["Kiss and Tell (1945 film)", "Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer."],
            ["Shirley Temple", "As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States."],
        ],
    }

    result = retriever_agent(state)
    context_evidence = result["agent_outputs"][-1]["context_evidence"]

    target = (
        "Shirley Temple: As an adult, she was named United States ambassador to Ghana "
        "and to Czechoslovakia and also served as Chief of Protocol of the United States."
    )
    distractor = "A Kiss for Corliss: A Kiss for Corliss is a sequel to the 1945 film Kiss and Tell."

    assert target in context_evidence
    assert distractor in context_evidence
    assert context_evidence.index(target) < context_evidence.index(distractor)


def test_hidden_plan_vector_reranks_knowledge_candidates(monkeypatch) -> None:
    def fake_hidden_state(text: str):
        vector = [1.0, 0.0] if text == "target fact" else [0.0, 1.0]
        return SimpleNamespace(pooled_vector=vector), None

    monkeypatch.setattr(
        "agents.retriever.agent.extract_last_hidden_state",
        fake_hidden_state,
    )
    hits = [
        {"context_id": "noise", "text": "noise", "context_score": 0.0},
        {"context_id": "target", "text": "target fact", "context_score": 0.0},
    ]

    ranked = _rerank_hypergraph_hits(
        {"question": "x"},
        [],
        hits,
        limit=2,
        plan_state_payload={
            "content_type": "latent.llm_hidden_state",
            "pooled_vector": [1.0, 0.0],
        },
    )

    assert ranked[0]["context_id"] == "target"
    assert ranked[0]["latent_similarity"] == 1.0


def test_local_embedding_plan_vector_reranks_in_the_same_vector_space(monkeypatch) -> None:
    def fake_embedding(text: str):
        vector = [1.0, 0.0] if text == "target fact" else [0.0, 1.0]
        return SimpleNamespace(vector=vector, provider="local", model="test")

    monkeypatch.setattr("agents.retriever.agent.embed_text", fake_embedding)
    hits = [
        {"context_id": "noise", "text": "noise", "context_score": 0.0},
        {"context_id": "target", "text": "target fact", "context_score": 0.0},
    ]

    ranked = _rerank_hypergraph_hits(
        {"question": "x"},
        [],
        hits,
        limit=2,
        plan_state_payload={
            "content_type": "latent.embedding",
            "embedding_provider": "local",
            "vector": [1.0, 0.0],
        },
    )

    assert ranked[0]["context_id"] == "target"
    assert ranked[0]["latent_similarity"] == 1.0


def test_hash_placeholder_does_not_rerank_knowledge_candidates(monkeypatch) -> None:
    def unexpected_embedding(_text: str):
        raise AssertionError("hash placeholder must not trigger semantic reranking")

    monkeypatch.setattr("agents.retriever.agent.embed_text", unexpected_embedding)
    hits = [{"context_id": "first", "text": "first", "context_score": 1.0}]

    ranked = _rerank_hypergraph_hits(
        {"question": "x"},
        [],
        hits,
        limit=1,
        plan_state_payload={
            "content_type": "latent.embedding",
            "embedding_provider": "hash",
            "vector": [1.0, 0.0],
        },
    )

    assert ranked[0]["context_id"] == "first"
    assert "latent_similarity" not in ranked[0]


def test_nested_knowledge_snippets_leave_slots_for_distinct_evidence() -> None:
    hits = [
        {"context_id": "short", "kind": "sentence", "title": "Example", "section": "Introduction",
         "text": "The example is a sample.", "context_score": 10.0},
        {"context_id": "long", "kind": "sentence", "title": "Example", "section": "Introduction",
         "text": "The example is a sample. Its construction uses titanium.", "context_score": 9.0},
        {"context_id": "other", "kind": "sentence", "title": "Example", "section": "Use",
         "text": "The example is used in research.", "context_score": 8.0},
    ]
    with patch("agents.retriever.agent.search_context", return_value=hits):
        evidence, selected = _extract_hypergraph_context({"question": "What is the example?"}, [], limit=2)
    assert {hit["context_id"] for hit in selected} == {"long", "other"}
    assert len(evidence) == 2
