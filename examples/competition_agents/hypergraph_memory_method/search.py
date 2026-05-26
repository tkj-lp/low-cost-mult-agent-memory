from __future__ import annotations

from typing import Any

from .index import HypergraphIndex


def search_hypergraph_memory(
    memory: list[dict[str, Any]], query: str, *, top_k: int
) -> list[dict[str, Any]]:
    """从超图索引中检索相关记忆，并返回可解释命中结果。"""

    index = HypergraphIndex.from_memory_records(memory)
    return index.search(query, top_k=top_k)
