from __future__ import annotations

"""超图记忆模块的单元测试。

这些测试只覆盖 memory.py 和 hypergraph_memory.py 的边界，不触碰任何 Agent。
目标是保证超图记忆可以独立演进，同时不破坏队友已经依赖的记忆接口。
"""

from competition_agents.hypergraph_memory import build_hypergraph_view, search_hypergraph_memory
from competition_agents.memory import (
    _rule_card_summary,
    add_memory,
    apply_memory_outcome,
    increment_reuse_counts,
    record_memory_usage,
    refine_memory,
    search_memory,
)
from hypergraph_memory_method.distill import distill_experience
from hypergraph_memory_method.evolution import maintain_memory


def _record(
    memory_id: str = "mem-0001",
    *,
    task_topic: str = "Design LangGraph protocol memory",
    summary: str = "Protocol memory stores reusable agent summaries.",
    tags: list[str] | None = None,
) -> dict:
    """构造最小 memory record，用于直接测试超图视图构建。"""

    return {
        "memory_id": memory_id,
        "source_agent": "summarizer",
        "task_topic": task_topic,
        "summary": summary,
        "tags": tags or ["langgraph", "protocol", "memory"],
    }


def test_maintenance_merges_equivalent_experience_and_recomputes_lifecycle() -> None:
    records = [
        {
            **_record("mem-0001"),
            "memory_type": "experience",
            "problem_pattern": "temporal_constraint",
            "solution_pattern": "filter_then_select",
            "trigger_conditions": ["temporal-query"],
            "retrieval_count": 2,
            "reuse_count": 1,
            "reuse_success_count": 1,
            "reuse_failure_count": 0,
            "lifecycle_stage": "needs-validation",
        },
        {
            **_record("mem-0002"),
            "memory_type": "experience",
            "problem_pattern": "temporal_constraint",
            "solution_pattern": "filter_then_select",
            "trigger_conditions": ["temporal-query"],
            "retrieval_count": 3,
            "reuse_count": 1,
            "reuse_success_count": 1,
            "reuse_failure_count": 0,
            "lifecycle_stage": "needs-validation",
        },
    ]

    maintained = maintain_memory(records)

    assert len(maintained) == 1
    assert maintained[0]["retrieval_count"] == 5
    assert maintained[0]["reuse_count"] == 2
    assert maintained[0]["reuse_success_count"] == 2
    assert maintained[0]["lifecycle_stage"] == "validated"


def test_add_memory_adds_hypergraph_and_default_fields() -> None:
    """add_memory 应同时保持旧字段兼容，并补齐后续超图检索会用的默认字段。"""

    memory = add_memory(
        [],
        source_agent="summarizer",
        task_topic="Design LangGraph protocol memory",
        summary="Protocol memory stores reusable agent summaries.",
        tags=["langgraph", "protocol", "memory"],
    )

    record = memory[0]
    # hypergraph 是网页展示和后续索引构建的入口。
    assert "hypergraph" in record
    # 这些字段现在先给默认值，避免后续队友接入时再改 record schema。
    assert "protocol" in record["entities"]
    assert record["supporting_facts"] == []
    assert record["evidence"] == []
    assert record["reasoning_path"] == []
    assert record["state_refs"] == []
    assert record["answer"] == record["summary"]
    assert record["reuse_count"] == 0
    assert record["reuse_success_count"] == 0


def test_add_memory_keeps_optional_trace() -> None:
    """add_memory 新增 trace 后仍应兼容旧调用，并在传入时保存 state 快照。"""

    trace = {
        "task_meta": {"task_id": "chain-001:turn-1"},
        "messages": [{"from": "planner", "to": "retriever"}],
        "memory_hits": [],
        "metrics": {"message_count": 1},
    }
    memory = add_memory(
        [],
        source_agent="summarizer",
        task_topic="Trace memory",
        summary="Trace should be stored for hypergraph reconstruction.",
        tags=["trace"],
        trace=trace,
    )

    assert memory[0]["trace"] == trace


def test_trace_memory_builds_structured_hyperedges() -> None:
    """trace 写入后应生成 task/message/state/gap/evidence 等结构化超边。"""

    trace = {
        "task_meta": {
            "task_id": "chain-001:turn-2",
            "group_id": "chain-001",
            "turn_id": "turn-2",
            "dataset_type": "continuous",
            "expected_reuse": ["turn-1.answer", "turn-1.supporting_facts"],
            "question": "Reuse the previous evidence chain answer.",
            "supporting_facts": [["Kiss and Tell (1945 film)", "Shirley Temple played Corliss Archer."]],
        },
        "messages": [
            {
                "from": "retriever",
                "to": "reasoner",
                "action": "reason",
                "payload": {"s": ["state-0002"], "m": ["mem-0001"], "cap": ["reason"]},
            }
        ],
        "memory_hits": [{"memory_id": "mem-0001"}],
        "state_refs": {"plan": "state-0001", "evidence": "state-0002", "answer": "state-0003"},
        "evidence": ["Reusable evidence chain is available."],
        "cognitive_gaps": [{"agent": "reasoner", "gap_type": "memory_grounded_reasoning"}],
        "metrics": {"message_count": 1, "text_chars": 42},
        "execution_result": {"retrieval_mode": "hypergraph_memory", "answer_ref": "state-0003"},
    }
    memory = add_memory(
        [],
        source_agent="summarizer",
        task_topic="Continuous task with long Context: 1928 1945 17-year-old",
        summary="Answer uses state refs and evidence.",
        tags=["protocol"],
        trace=trace,
    )

    hypergraph = memory[0]["hypergraph"]
    edge_types = {edge["type"] for edge in hypergraph["hyperedges"]}

    assert "memory_event" in edge_types
    assert "task_context" in edge_types
    assert "agent_message" in edge_types
    assert "state_flow" in edge_types
    assert "memory_reuse" in edge_types
    assert "cognitive_gap" in edge_types
    assert "supporting_fact" not in edge_types
    assert "evidence" in edge_types
    assert "strategy" in edge_types
    assert "experience" in edge_types
    assert "reuse_benefit" in edge_types
    assert "archive_scope" in edge_types
    assert "keyword:1928" not in hypergraph["keyword_nodes"]
    assert "keyword:1945" not in hypergraph["keyword_nodes"]
    assert hypergraph["distilled_experience"]["archive_scope"] == "cross-turn-reuse"
    assert "state-ref-passing" in hypergraph["distilled_experience"]["strategies"]
    assert "memory-grounded-reasoning" in hypergraph["distilled_experience"]["experiences"]


def test_distill_experience_extracts_strategy_and_reuse_benefit() -> None:
    """缁忛獙钂搁搴斾粠 trace 鎻愬彇绛栫暐銆佺粡楠屽拰澶嶇敤鏀剁泭銆?"""

    record = {
        "memory_id": "mem-0001",
        "summary": "Reusable answer.",
        "trace": {
            "task_meta": {
                "group_id": "chain-001",
                "expected_reuse": ["turn-1.answer", "turn-1.supporting_facts"],
                "supporting_facts": [["A", "B"]],
            },
            "messages": [
                {
                    "from": "retriever",
                    "to": "reasoner",
                    "action": "reason",
                    "payload": {"s": ["state-0002"], "m": ["mem-0001"]},
                }
            ],
            "memory_hits": [{"memory_id": "mem-0001"}],
            "evidence": ["A: B"],
            "state_refs": {"evidence": "state-0002", "answer": "state-0003"},
            "metrics": {"latent_transfer_count": 3, "text_chars": 120},
            "execution_result": {"retrieval_mode": "hypergraph_memory"},
        },
    }

    distilled = distill_experience(record)

    assert distilled["archive_scope"] == "cross-turn-reuse"
    assert distilled["lifecycle_stage"] == "promoted-candidate"
    assert "memory-ref-passing" in distilled["strategies"]
    assert "supporting-fact-chain" in distilled["experiences"]
    assert not any(
        value.startswith("expected-reuse:")
        for value in distilled["reuse_benefits"]
    )
    assert "memory-hit-count:1" in distilled["reuse_benefits"]


def test_refine_memory_generates_experience_fields() -> None:
    """memory_refine 第一版应产出经验记忆所需的核心字段。"""

    trace = {
        "task_meta": {
            "dataset_type": "continuous",
            "group_id": "chain-001",
            "question": "when was the 2017 film dunkirk released on dvd in the united kingdom?",
            "expected_reuse": ["turn-1.answer"],
        },
        "messages": [{"from": "planner", "to": "retriever", "action": "retrieve", "payload": {"s": ["state-0001"]}}],
        "memory_hits": [{"memory_id": "mem-0001"}],
        "evidence": [
            "Dunkirk (2017 film): DVD on 18 December in the United Kingdom.",
            "North Sea: what is the film about?",
            "Planner-guided retrieval focus: verify the medium and region before selecting the date.",
        ],
        "state_refs": {"plan": "state-0001", "evidence": "state-0002", "answer": "state-0003"},
    }

    refined = refine_memory(
        task_topic="Question: when was the 2017 film dunkirk released on dvd in the united kingdom?",
        summary="18 December 2017",
        trace=trace,
    )

    assert refined["memory_type"] == "experience"
    assert refined["problem_pattern"] == "multiple_temporal_candidates_with_constraints"
    assert refined["solution_pattern"] == "choose_date_matching_all_constraints"
    assert refined["failure_pattern"] == "confuse_digital_release_with_disc_release"
    assert refined["trigger_conditions"] == ["temporal_query", "format_constraint", "region_constraint"]
    assert refined["supporting_evidence"] == ["Dunkirk (2017 film): DVD on 18 December in the United Kingdom."]
    assert refined["experience_source"] in {"success-derived", "failure-derived", "outcome-unknown"}
    assert refined["distillation_case"]["question"]
    assert refined["experience_rule_card"]["problem_pattern"] == "multiple_temporal_candidates_with_constraints"
    assert refined["strategies"]


def test_increment_reuse_counts_tracks_successful_reuse() -> None:
    """reuse 计数应区分“被命中”与“命中后成功帮助答对”两层信号。"""

    memory = add_memory(
        [],
        source_agent="summarizer",
        task_topic="Temporal constraint experience",
        summary="Reusable experience card.",
        tags=["experience"],
        memory_type="experience",
        problem_pattern="multiple_temporal_candidates_with_constraints",
        solution_pattern="choose_date_matching_all_constraints",
        failure_pattern="confuse_competing_date_candidates",
        trigger_conditions=["question_when", "medium_constraint"],
    )

    updated = increment_reuse_counts(
        memory,
        ["mem-0001"],
        successful_memory_ids=["mem-0001"],
    )

    assert updated[0]["reuse_count"] == 1
    assert updated[0]["reuse_success_count"] == 1


def test_add_memory_merges_similar_experience_records() -> None:
    """相同策略的经验记忆应优先合并，而不是无限追加重复记录。"""

    memory = add_memory(
        [],
        source_agent="summarizer",
        task_topic="Question: what does the economy of this place depend on?",
        summary="experience_rule_card: problem=attribute_selection_among_related_facts; solution=pick_fact_matching_requested_attribute; avoid=return_related_fact_instead_of_target_attribute; triggers=question_targets_single_attribute",
        tags=["experience"],
        memory_type="experience",
        problem_pattern="attribute_selection_among_related_facts",
        solution_pattern="pick_fact_matching_requested_attribute",
        failure_pattern="return_related_fact_instead_of_target_attribute",
        trigger_conditions=["attribute_lookup_query"],
        supporting_evidence=["Urk: The important economic pillar of the village is the fishery."],
        confidence=0.7,
        abstraction_note="先解析目标属性，再从相关事实中只选匹配该属性的答案短语。",
    )

    merged = add_memory(
        memory,
        source_agent="summarizer",
        task_topic="Question: what natural hazard affects this sea and its coasts?",
        summary="experience_rule_card: problem=attribute_selection_among_related_facts; solution=pick_fact_matching_requested_attribute; avoid=return_related_fact_instead_of_target_attribute; triggers=question_targets_single_attribute, supporting_fact_contains_attribute_value",
        tags=["experience", "continuous"],
        memory_type="experience",
        problem_pattern="attribute_selection_among_related_facts",
        solution_pattern="pick_fact_matching_requested_attribute",
        failure_pattern="return_related_fact_instead_of_target_attribute",
        trigger_conditions=["attribute_lookup_query", "reference_resolution_query"],
        supporting_evidence=["North Sea: Storm surges threaten the coasts around the North Sea."],
        confidence=0.8,
        abstraction_note="遇到隐式指代后的属性题，先锁定实体，再只返回被问属性的事实短语。",
    )

    assert len(merged) == 1
    assert merged[0]["merge_count"] == 1
    assert set(merged[0]["trigger_conditions"]) == {
        "attribute_lookup_query",
        "reference_resolution_query",
    }
    assert len(merged[0]["supporting_evidence"]) == 2
    assert merged[0]["confidence"] == 0.8


def test_refine_memory_skips_simple_direct_copy_case() -> None:
    """单条直接证据即可完成的简单事实题，不应稳定产出泛化价值很弱的经验。"""

    trace = {
        "task_meta": {
            "dataset_type": "continuous",
            "group_id": "chain-simple",
            "question": "when was dunkirk released in the united kingdom?",
            "expected_answer": "18 December 2017",
        },
        "memory_hits": [],
        "evidence": [
            "Dunkirk (2017 film): DVD on 18 December in the United Kingdom.",
        ],
        "execution_result": {
            "answer_candidate": "18 December 2017",
            "reasoning_summary": "The evidence directly states the date.",
        },
    }

    refined = refine_memory(
        task_topic="Question: when was dunkirk released in the united kingdom?",
        summary="18 December 2017",
        trace=trace,
    )

    assert refined["should_write"] is False


def test_rule_card_summary_does_not_fall_back_to_general() -> None:
    """经验摘要里不应再出现过泛的 general 触发条件占位。"""

    summary = _rule_card_summary(
        {
            "problem_pattern": "attribute_selection_among_related_facts",
            "solution_pattern": "pick_fact_matching_requested_attribute",
            "failure_pattern": "return_related_fact_instead_of_target_attribute",
            "trigger_conditions": [],
        }
    )

    assert "general" not in summary
    assert "triggers=none" in summary


def test_add_memory_merges_same_problem_and_solution_with_different_failure_wording() -> None:
    """相近经验即使 failure_pattern 写法略有差异，也应优先合并而不是继续膨胀。"""

    memory = add_memory(
        [],
        source_agent="summarizer",
        task_topic="Question: what does the economy of this place depend on?",
        summary="experience_rule_card: problem=attribute_selection_among_related_facts; solution=pick_fact_matching_requested_attribute; avoid=return_related_fact_instead_of_target_attribute; triggers=attribute_lookup_query",
        tags=["experience"],
        memory_type="experience",
        problem_pattern="attribute_selection_among_related_facts",
        solution_pattern="pick_fact_matching_requested_attribute",
        failure_pattern="return_related_fact_instead_of_target_attribute",
        trigger_conditions=["attribute_lookup_query"],
    )

    merged = add_memory(
        memory,
        source_agent="summarizer",
        task_topic="Question: what natural hazard affects this sea and its coasts?",
        summary="experience_rule_card: problem=attribute_selection_among_related_facts; solution=pick_fact_matching_requested_attribute; avoid=mix_multiple_attributes_into_one_answer; triggers=attribute_lookup_query, reference_resolution_query",
        tags=["experience", "continuous"],
        memory_type="experience",
        problem_pattern="attribute_selection_among_related_facts",
        solution_pattern="pick_fact_matching_requested_attribute",
        failure_pattern="mix_multiple_attributes_into_one_answer",
        trigger_conditions=["attribute_lookup_query", "reference_resolution_query"],
    )

    assert len(merged) == 1
    assert merged[0]["merge_count"] == 1


def test_search_memory_can_filter_by_memory_type() -> None:
    """search_memory 应支持 evidence / experience 分流检索。"""

    memory = []
    memory = add_memory(
        memory,
        source_agent="summarizer",
        task_topic="Question: when was dunkirk released on dvd in the united kingdom?",
        summary="18 December 2017",
        tags=["protocol"],
        memory_type="evidence",
    )
    memory = add_memory(
        memory,
        source_agent="summarizer",
        task_topic="Question: when was dunkirk released on dvd in the united kingdom?",
        summary="经验记忆：先锁定影片与地区，再选满足介质约束的日期槽位。",
        tags=["experience"],
        memory_type="experience",
        problem_pattern="multiple_temporal_candidates_with_constraints",
        solution_pattern="choose_date_matching_all_constraints",
        failure_pattern="confuse_digital_release_with_disc_release",
        trigger_conditions=["temporal_query", "format_constraint", "region_constraint"],
    )

    evidence_hits = search_memory(memory, "dvd release date in united kingdom", top_k=2, memory_types=["evidence"])
    experience_hits = search_memory(
        memory,
        "temporal_query format_constraint region_constraint",
        top_k=2,
        memory_types=["experience"],
    )

    assert evidence_hits
    assert experience_hits
    assert all(hit["memory_type"] == "evidence" for hit in evidence_hits)
    assert all(hit["memory_type"] == "experience" for hit in experience_hits)


def test_build_hypergraph_view_adds_nodes_and_event_hyperedge() -> None:
    """一条记忆应被映射成多个类型节点和一条 memory_event 超边。"""

    view = build_hypergraph_view(_record())

    # 节点 id 带类型前缀，避免 tag:memory 和 keyword:memory 混在一起。
    assert view["memory_node"] == "memory:mem-0001"
    assert view["task_node"].startswith("task:")
    assert "task_label" in view
    assert view["agent_node"] == "agent:summarizer"
    assert "tag:protocol" in view["tag_nodes"]
    assert "keyword:protocol" in view["keyword_nodes"]
    # 事件超边把本条记忆、任务、Agent、标签和关键词连成一次协作事件。
    assert view["event_hyperedge"]["type"] == "memory_event"
    assert view["event_hyperedge"]["hyperedge_id"] == "hyperedge:mem-0001"
    assert view["memory_node"] in view["event_hyperedge"]["nodes"]


def test_search_hypergraph_memory_hits_related_memory() -> None:
    """相关 query 应通过 tag / keyword 节点命中对应历史记忆。"""

    memory = []
    memory = add_memory(
        memory,
        source_agent="summarizer",
        task_topic="Design LangGraph protocol memory",
        summary="Protocol memory stores reusable agent summaries.",
        tags=["langgraph", "protocol", "memory"],
    )

    hits = search_hypergraph_memory(memory, "reuse protocol memory", top_k=2)

    # 返回对象仍然是 memory record，Agent 侧继续读取 memory_id / summary 即可。
    assert [hit["memory_id"] for hit in hits] == ["mem-0001"]


def test_trace_search_prefers_structured_nodes_over_numeric_keywords() -> None:
    """连续任务复用时，解释应优先来自结构化节点而不是年份等噪声 keyword。"""

    trace = {
        "task_meta": {
            "task_id": "chain-001:turn-1",
            "group_id": "chain-001",
            "turn_id": "turn-1",
            "dataset_type": "continuous",
            "question": "What government position was held by the woman in Kiss and Tell?",
            "supporting_facts": [
                ["Kiss and Tell (1945 film)", "Kiss and Tell stars Shirley Temple."],
                ["Shirley Temple", "She served as Chief of Protocol of the United States."],
            ],
        },
        "messages": [
            {
                "from": "reasoner",
                "to": "summarizer",
                "action": "summarize",
                "payload": {"s": ["state-0001", "state-0003"], "m": ["mem-0001"]},
            }
        ],
        "state_refs": {"plan": "state-0001", "answer": "state-0003"},
        "evidence": ["Kiss and Tell -> Shirley Temple -> Chief of Protocol."],
        "cognitive_gaps": [{"agent": "reasoner", "gap_type": "memory_grounded_reasoning"}],
        "metrics": {"message_count": 3},
    }
    memory = add_memory(
        [],
        source_agent="summarizer",
        task_topic="Question: What government position was held? Context: 1928 1945 1938",
        summary="Chief of Protocol was supported by the evidence chain.",
        tags=["protocol"],
        trace=trace,
    )

    hits = search_hypergraph_memory(
        memory,
        "Reuse the previous evidence chain. Which answer was supported by Kiss and Tell 1945 and Shirley Temple?",
        top_k=1,
    )

    assert hits
    assert hits[0]["matched_nodes"][0].startswith("artifact:")
    assert not any(node.startswith("keyword:1945") for node in hits[0]["matched_nodes"])
    assert hits[0]["evidence_score"] > 0
    assert hits[0]["state_ref_score"] > 0


def test_experience_nodes_can_drive_search() -> None:
    """妫€绱㈠彲浠ラ€氳繃缁忛獙鍜岀瓥鐣ヨ妭鐐瑰懡涓紝鑰屼笉渚濊禆鏅€?keyword銆?"""

    trace = {
        "task_meta": {
            "task_id": "chain-001:turn-2",
            "group_id": "chain-001",
            "turn_id": "turn-2",
            "dataset_type": "continuous",
            "expected_reuse": ["turn-1.supporting_facts"],
            "question": "Reuse prior evidence.",
            "supporting_facts": [["Shirley Temple", "Chief of Protocol"]],
        },
        "messages": [
            {
                "from": "retriever",
                "to": "reasoner",
                "action": "reason",
                "payload": {"s": ["state-0002"], "m": ["mem-0001"]},
            }
        ],
        "memory_hits": [{"memory_id": "mem-0001"}],
        "state_refs": {"evidence": "state-0002", "answer": "state-0003"},
        "evidence": ["Prior supporting facts were reused."],
        "metrics": {"latent_transfer_count": 3},
        "execution_result": {"retrieval_mode": "hypergraph_memory"},
    }
    memory = add_memory(
        [],
        source_agent="summarizer",
        task_topic="Prior evidence reuse",
        summary="Evidence was reused through memory-grounded reasoning.",
        tags=["protocol"],
        trace=trace,
    )

    hits = search_hypergraph_memory(memory, "experience distill insight for previous evidence reuse", top_k=1)

    assert hits
    assert any(node.startswith("experience:") for node in hits[0]["matched_nodes"])
    assert hits[0]["experience_score"] > 0
    assert hits[0]["strategy_score"] > 0


def test_search_hypergraph_memory_ignores_unrelated_query() -> None:
    """无共享节点的 query 不应误返回无关记忆。"""

    memory = add_memory(
        [],
        source_agent="summarizer",
        task_topic="Design LangGraph protocol memory",
        summary="Protocol memory stores reusable agent summaries.",
        tags=["langgraph", "protocol", "memory"],
    )

    assert search_hypergraph_memory(memory, "database transaction rollback", top_k=2) == []


def test_tag_match_scores_higher_than_keyword_only_match() -> None:
    """tag 是人工或上游模块给出的强信号，权重应高于普通 keyword 命中。"""

    memory = []
    # 第一条通过 tag:openeuler 命中。
    memory = add_memory(
        memory,
        source_agent="summarizer",
        task_topic="Unrelated deployment note",
        summary="Deployment note without the topic term.",
        tags=["openeuler"],
    )
    # 第二条只通过 keyword:openeuler 命中。
    memory = add_memory(
        memory,
        source_agent="summarizer",
        task_topic="Keyword only openeuler note",
        summary="This summary mentions openeuler once.",
        tags=["deployment"],
    )

    hits = search_hypergraph_memory(memory, "openeuler", top_k=2)

    # 如果这条断言失败，说明强结构信号和弱文本信号的区分被破坏了。
    assert hits[0]["memory_id"] == "mem-0001"
    assert hits[0]["hypergraph_score"] > hits[1]["hypergraph_score"]


def test_memory_usage_separates_retrieval_adoption_and_success() -> None:
    """召回、进入推理和经结果验证是三个不同的维护信号。"""

    memory = [_record("mem-0001"), _record("mem-0002")]
    memory = record_memory_usage(
        memory,
        retrieved_memory_ids=["mem-0001", "mem-0002"],
        adopted_memory_ids=["mem-0002"],
    )

    first, second = memory
    assert first["retrieval_count"] == 1
    assert first.get("reuse_count", 0) == 0
    assert second["retrieval_count"] == 1
    assert second["reuse_count"] == 1
    assert second.get("reuse_success_count", 0) == 0

    memory = apply_memory_outcome(memory, ["mem-0002"], successful=True)
    second = memory[1]
    assert second["reuse_success_count"] == 1
    assert second["lifecycle_stage"] == "promoted-candidate"


def test_failed_adopted_memory_is_not_counted_as_success() -> None:
    memory = record_memory_usage(
        [_record("mem-0001")],
        retrieved_memory_ids=["mem-0001"],
        adopted_memory_ids=["mem-0001"],
    )

    memory = apply_memory_outcome(memory, ["mem-0001"], successful=False)

    assert memory[0]["reuse_success_count"] == 0
    assert memory[0]["reuse_failure_count"] == 1
    assert memory[0]["lifecycle_stage"] == "needs-validation"


def test_search_result_contains_explanation_fields() -> None:
    """检索结果必须带解释字段，方便网页展示和后续调试召回路径。"""

    memory = add_memory(
        [],
        source_agent="summarizer",
        task_topic="Reuse LangGraph protocol memory for openEuler",
        summary="Deployment can reuse previous protocol memory.",
        tags=["langgraph", "protocol", "memory", "openeuler"],
    )

    hit = search_hypergraph_memory(memory, "reuse protocol memory for openeuler", top_k=1)[0]

    # retrieval_score 预留给未来融合模型分数，hypergraph_score 表示结构检索分。
    assert "retrieval_score" in hit
    assert "hypergraph_score" in hit
    assert "embedding_score" in hit
    assert hit["retrieval_score"] >= hit["hypergraph_score"]
    # matched_* 和 reasoning_path 说明 query 是怎么走到这条 memory 的。
    assert "matched_nodes" in hit
    assert "matched_hyperedges" in hit
    assert "reasoning_path" in hit
    assert isinstance(hit["reasoning_path"], list)
    assert hit["reasoning_path"]
    assert {"query_node", "hyperedge_id", "memory_id"} <= set(hit["reasoning_path"][0])


def test_embedding_score_is_fused_into_retrieval_score() -> None:
    """embedding_score 搴斿弬涓庢渶缁堟绱㈠垎锛岃€屼笉鍙槸灞曠ず瀛楁銆?"""

    memory = add_memory(
        [],
        source_agent="summarizer",
        task_topic="Protocol memory for openEuler deployment",
        summary="Protocol memory helps openEuler deployment.",
        tags=["protocol", "memory", "openeuler"],
    )

    hit = search_hypergraph_memory(memory, "protocol memory openeuler", top_k=1)[0]

    assert hit["embedding_score"] > 0
    assert hit["retrieval_score"] > hit["hypergraph_score"]


def test_top_k_and_score_order_are_stable() -> None:
    """top_k 应限制返回数量，并按检索分稳定排序。"""

    memory = []
    memory = add_memory(
        memory,
        source_agent="summarizer",
        task_topic="Protocol memory for openEuler deployment",
        summary="Protocol memory helps openEuler deployment.",
        tags=["protocol", "memory", "openeuler"],
    )
    memory = add_memory(
        memory,
        source_agent="summarizer",
        task_topic="Protocol memory",
        summary="Protocol memory is reusable.",
        tags=["protocol"],
    )
    memory = add_memory(
        memory,
        source_agent="summarizer",
        task_topic="Protocol note",
        summary="Protocol note.",
        tags=["protocol"],
    )

    hits = search_hypergraph_memory(memory, "protocol memory openeuler", top_k=2)

    # 第一条同时命中 protocol / memory / openeuler，应该稳定排在最前。
    assert len(hits) == 2
    assert hits[0]["memory_id"] == "mem-0001"
    assert hits[0]["retrieval_score"] >= hits[1]["retrieval_score"]
