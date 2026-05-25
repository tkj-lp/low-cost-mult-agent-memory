from __future__ import annotations

from datetime import datetime, timezone

from competition_agents.hypergraph_memory import build_hypergraph_view, search_hypergraph_memory
from competition_agents.state import simple_embedding


def add_memory(
    memory: list[dict],
    *,
    source_agent: str,
    task_topic: str,
    summary: str,
    tags: list[str],
) -> list[dict]:
    """向当前内存存储追加一条可复用记忆。"""

    # 元数据字段保持稳定，方便网页和超图模块对接。
    record = {
        "memory_id": f"mem-{len(memory) + 1:04d}",
        "source_agent": source_agent,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_topic": task_topic,
        "summary": summary,
        "tags": tags,
        "embedding": simple_embedding(task_topic + " " + summary),
        "entities": [],
        "supporting_facts": [],
        "evidence": [],
        "reasoning_path": [],
        "state_refs": [],
        "answer": summary,
        "reuse_count": 0,
    }
    record["hypergraph"] = build_hypergraph_view(record)
    return [*memory, record]


def search_memory(memory: list[dict], query: str, *, top_k: int) -> list[dict]:
    """用最小超图索引检索记忆。

    返回对象仍然是 memory record，保证 RetrieverAgent 不需要改。额外添加
    hypergraph_score / matched_nodes / matched_hyperedges 字段，方便网页同学展示
    “为什么命中这条记忆”。
    """

    if not memory:
        return []

    return search_hypergraph_memory(memory, query, top_k=top_k)
