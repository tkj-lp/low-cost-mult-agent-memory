from __future__ import annotations

"""超图记忆模块的单元测试。

这些测试只覆盖 memory.py 和 hypergraph_memory.py 的边界，不触碰任何 Agent。
目标是保证超图记忆可以独立演进，同时不破坏队友已经依赖的记忆接口。
"""

from competition_agents.hypergraph_memory import build_hypergraph_view, search_hypergraph_memory
from competition_agents.memory import add_memory


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
    assert record["entities"] == []
    assert record["supporting_facts"] == []
    assert record["evidence"] == []
    assert record["reasoning_path"] == []
    assert record["state_refs"] == []
    assert record["answer"] == record["summary"]
    assert record["reuse_count"] == 0


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
    # matched_* 和 reasoning_path 说明 query 是怎么走到这条 memory 的。
    assert "matched_nodes" in hit
    assert "matched_hyperedges" in hit
    assert "reasoning_path" in hit
    assert isinstance(hit["reasoning_path"], list)
    assert hit["reasoning_path"]
    assert {"query_node", "hyperedge_id", "memory_id"} <= set(hit["reasoning_path"][0])


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
