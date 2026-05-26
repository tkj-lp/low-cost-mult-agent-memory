from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from competition_agents.embedding import embed_text
from hypergraph_memory_method import build_hypergraph_view, search_hypergraph_memory


def add_memory(
    memory: list[dict],
    *,
    source_agent: str,
    task_topic: str,
    summary: str,
    tags: list[str],
    trace: dict[str, Any] | None = None,
) -> list[dict]:
    """向当前内存存储追加一条可复用记忆。"""

    # 保留可插拔 embedding 元数据，同时补齐超图检索需要的结构字段。
    embedding = embed_text(task_topic + " " + summary)
    record = {
        "memory_id": f"mem-{len(memory) + 1:04d}",
        "source_agent": source_agent,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_topic": task_topic,
        "summary": summary,
        "tags": tags,
        "embedding": embedding.vector,
        "embedding_provider": embedding.provider,
        "embedding_model": embedding.model,
        "entities": [],
        "supporting_facts": [],
        "evidence": [],
        "reasoning_path": [],
        "state_refs": [],
        "answer": summary,
        "reuse_count": 0,
    }
    if trace is not None:
        record["trace"] = trace
        state_refs = trace.get("state_refs", {})
        if isinstance(state_refs, dict):
            record["state_refs"] = [ref for ref in state_refs.values() if ref]
        record["evidence"] = trace.get("evidence", [])
        task_meta = trace.get("task_meta", {})
        if isinstance(task_meta, dict):
            record["supporting_facts"] = task_meta.get("supporting_facts", [])
    # hypergraph 字段用于解释检索命中路径，不改变 Agent 读取 memory record 的方式。
    record["hypergraph"] = build_hypergraph_view(record)
    return [*memory, record]


def search_memory(memory: list[dict], query: str, *, top_k: int) -> list[dict]:
    """用超图索引检索记忆，并保持返回对象仍是 memory record。"""

    if not memory:
        return []

    return search_hypergraph_memory(memory, query, top_k=top_k)
