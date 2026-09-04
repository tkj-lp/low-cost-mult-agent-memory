from __future__ import annotations

from agents.retriever.agent import (
    _build_retrieval_query,
    _extract_context_evidence,
    retriever_agent,
)
from competition_agents.memory import _rerank_memory_hits, add_memory
from competition_agents.state import initial_state
from competition_agents.state_store import clear_state_store, resolve_state_payload


def test_retriever_keeps_memory_context_in_state_and_sends_refs_only() -> None:
    """协议模式应把检索结果放入状态池，下游只接收可解引用标识。"""

    clear_state_store()
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

    assert result["memory_context"] == []
    retrieval_ref = result["state_refs"]["retrieval"]
    retrieval_bundle = resolve_state_payload(retrieval_ref)
    assert retrieval_bundle is not None
    assert retrieval_bundle["memory_context"][0]["summary"] == unique_summary
    evidence_ref = result["state_refs"]["evidence"]
    evidence_state = result["latent_states"][evidence_ref]
    assert unique_summary not in evidence_state["summary"]

    payload = result["messages"][-1]["payload"]
    assert payload["m"] == ["mem-0001"]
    assert retrieval_ref in payload["s"]
    assert "memory_context" not in payload["r"]
    assert payload["r"]["hits"] == 1


def test_retriever_uses_planner_focus_in_memory_query(monkeypatch) -> None:
    """Retriever 应把 Planner 的计划拼进检索查询，而不是只使用原始 task。"""

    captured: dict[str, object] = {}

    def fake_search_memory(
        memory: list[dict],
        query: str,
        *,
        top_k: int,
        memory_types=None,
        type_limits=None,
        memory_layers=None,
        layer_limits=None,
    ) -> list[dict]:
        captured["query"] = query
        captured["top_k"] = top_k
        captured["type_limits"] = type_limits
        captured["layer_limits"] = layer_limits
        return []

    monkeypatch.setattr("agents.retriever.agent.search_memory", fake_search_memory)

    state = initial_state(
        "Find the diplomatic role held by Shirley Temple.",
        "protocol",
        [{"memory_id": "mem-0001", "memory_type": "evidence", "summary": "prior"}],
    )
    state["retrieval_policy"] = {"need_memory": True, "need_knowledge": True}
    state["plan"] = [
        "Identify the actress who portrayed Corliss Archer.",
        "Retrieve biographical details about Shirley Temple.",
        "Extract the government position she later held.",
    ]

    result = retriever_agent(state)

    assert "Planner retrieval focus:" in str(captured["query"])
    assert "Retrieve biographical details about Shirley Temple" in str(captured["query"])
    assert captured["top_k"] == 6
    assert captured["type_limits"] is None
    assert captured["layer_limits"] == {"episodic": 2, "experience": 4}
    assert result["agent_outputs"][-1]["planner_focus"] == [
        "Identify the actress who portrayed Corliss Archer",
        "Retrieve biographical details about Shirley Temple",
        "Extract the government position she later held",
    ]
    retrieval_bundle = resolve_state_payload(result["state_refs"]["retrieval"])
    assert retrieval_bundle is not None
    assert any(
        "hypergraph structural matching plus embedding similarity fusion ranking" in item
        for item in retrieval_bundle["evidence"]
    )
    assert not any(
        "Planner-guided retrieval focus:" in item
        for item in result["agent_outputs"][-1]["context_evidence"]
    )


def test_expected_reuse_labels_do_not_change_online_query_or_ranking() -> None:
    """评测标注只能用于离线评分，不能影响在线查询和排序。"""

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
    task = "Question: what kinds of fish are commonly found in this sea?"
    plan = ["Resolve the sea named by the previous turn", "Retrieve relevant fish evidence"]
    plain_query, _ = _build_retrieval_query(
        task,
        plan,
        {"group_id": "chain-001", "turn_id": "turn-8", "task_type": "continuous"},
    )
    labelled_query, _ = _build_retrieval_query(
        task,
        plan,
        {
            "group_id": "chain-001",
            "turn_id": "turn-8",
            "task_type": "continuous",
            "expected_reuse": ["turn-7.answer", "group.supporting_facts"],
        },
    )

    assert labelled_query == plain_query
    reranked = _rerank_memory_hits(
        hits,
        plain_query + "\nExpected reuse: turn-7.answer\nRequired artifacts: answer",
    )

    assert [hit["memory_id"] for hit in reranked] == ["mem-0006", "mem-0007"]


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

    calls = {"count": 0}

    def fake_search_memory(
        memory_records: list[dict],
        query: str,
        *,
        top_k: int,
        memory_types=None,
        type_limits=None,
        memory_layers=None,
        layer_limits=None,
    ) -> list[dict]:
        calls["count"] += 1
        assert type_limits is None
        assert layer_limits == {"episodic": 2, "experience": 4}
        return memory_records

    monkeypatch.setattr("agents.retriever.agent.search_memory", fake_search_memory)

    state = initial_state("when was dunkirk released on dvd in the united kingdom?", "protocol", memory)
    state["retrieval_policy"] = {"need_memory": True, "need_knowledge": True}
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
    retrieval_bundle = resolve_state_payload(result["state_refs"]["retrieval"])
    assert retrieval_bundle is not None
    assert any(item["memory_type"] == "experience" for item in retrieval_bundle["memory_context"])
    assert calls["count"] == 1


def test_gold_supporting_facts_do_not_change_context_retrieval() -> None:
    """更换金标准证据标签不应改变同一上下文的在线检索结果。"""

    base_meta = {
        "question": "Which basket item is mentioned?",
        "context": [
            {"title": "Basket", "sentences": ["The basket item is apples."]},
            {"title": "Basket", "sentences": ["The basket item is pears."]},
        ],
    }
    apples_gold = {
        **base_meta,
        "supporting_facts": [["Basket", "The basket item is apples."]],
    }
    pears_gold = {
        **base_meta,
        "supporting_facts": [["Basket", "The basket item is pears."]],
    }

    assert _extract_context_evidence(apples_gold, []) == _extract_context_evidence(pears_gold, [])


def test_retriever_extracts_context_evidence_from_planner_focus(monkeypatch) -> None:
    """Retriever 应按 Planner 规划从 Context 中抽 supporting facts 给 Reasoner。"""

    def fake_search_memory(memory: list[dict], query: str, *, top_k: int, memory_types=None) -> list[dict]:
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

    context_evidence = result["agent_outputs"][-1]["context_evidence"]
    assert any("Kiss and Tell (1945 film): Kiss and Tell is a 1945 American comedy film starring Shirley Temple as Corliss Archer." == item for item in context_evidence)
    assert any("Shirley Temple: As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States." == item for item in context_evidence)
    assert context_evidence


def test_retriever_prioritizes_answer_bearing_sentence_over_title_overlap(monkeypatch) -> None:
    """Retriever 应优先保留承载答案的职位句，而不是只和片名重合的干扰句。"""

    def fake_search_memory(memory: list[dict], query: str, *, top_k: int, memory_types=None) -> list[dict]:
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
