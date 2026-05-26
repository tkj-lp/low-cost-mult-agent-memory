from __future__ import annotations

from collections import defaultdict
from typing import Any

from .builder import ensure_hypergraph_view
from .scoring import score_record
from .text import node_id, tokenize


def query_nodes(query: str) -> set[str]:
    """查询文本同时尝试匹配 tag 节点和 keyword 节点。"""

    nodes: set[str] = set()
    for token in tokenize(query):
        nodes.add(node_id("keyword", token))
        nodes.add(node_id("tag", token))
    return nodes


class HypergraphIndex:
    """内存版超图索引。

    第一版只维护检索所需的倒排关系。后续接数据库时，可以把这个类替换成
    数据库-backed index，但继续保留 search_hypergraph_memory() 的外部接口。
    """

    def __init__(
        self,
        *,
        memory_by_id: dict[str, dict[str, Any]],
        node_to_memory_ids: dict[str, set[str]],
        node_to_hyperedge_ids: dict[str, set[str]],
        hyperedge_to_memory_id: dict[str, str],
    ) -> None:
        self.memory_by_id = memory_by_id
        self.node_to_memory_ids = node_to_memory_ids
        self.node_to_hyperedge_ids = node_to_hyperedge_ids
        self.hyperedge_to_memory_id = hyperedge_to_memory_id

    @classmethod
    def from_memory_records(cls, memory: list[dict[str, Any]]) -> "HypergraphIndex":
        """从 memory record 列表构建全局索引。"""

        memory_by_id: dict[str, dict[str, Any]] = {}
        node_to_memory_ids: dict[str, set[str]] = defaultdict(set)
        node_to_hyperedge_ids: dict[str, set[str]] = defaultdict(set)
        hyperedge_to_memory_id: dict[str, str] = {}

        for record in memory:
            memory_id = record["memory_id"]
            hypergraph = ensure_hypergraph_view(record)
            hyperedge = hypergraph["event_hyperedge"]
            hyperedge_id = hyperedge["hyperedge_id"]

            memory_by_id[memory_id] = record
            hyperedge_to_memory_id[hyperedge_id] = memory_id
            for node in hyperedge["nodes"]:
                node_to_memory_ids[node].add(memory_id)
                node_to_hyperedge_ids[node].add(hyperedge_id)

        return cls(
            memory_by_id=memory_by_id,
            node_to_memory_ids=dict(node_to_memory_ids),
            node_to_hyperedge_ids=dict(node_to_hyperedge_ids),
            hyperedge_to_memory_id=hyperedge_to_memory_id,
        )

    def search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        """按 query 激活节点，聚合到 memory，再返回 top_k 可解释结果。"""

        matched_by_memory: dict[str, dict[str, Any]] = {}
        for node in query_nodes(query):
            for memory_id in self.node_to_memory_ids.get(node, set()):
                entry = matched_by_memory.setdefault(
                    memory_id,
                    {
                        "matched_nodes": set(),
                        "matched_hyperedges": set(),
                        "reasoning_path": [],
                    },
                )
                entry["matched_nodes"].add(node)

                hyperedge_ids = self.node_to_hyperedge_ids.get(node, set())
                for hyperedge_id in hyperedge_ids:
                    if self.hyperedge_to_memory_id.get(hyperedge_id) != memory_id:
                        continue
                    entry["matched_hyperedges"].add(hyperedge_id)
                    entry["reasoning_path"].append(
                        {
                            "query_node": node,
                            "hyperedge_id": hyperedge_id,
                            "memory_id": memory_id,
                        }
                    )

        ranked_hits = []
        for memory_id, entry in matched_by_memory.items():
            record = self.memory_by_id[memory_id]
            matched_nodes = entry["matched_nodes"]
            score = score_record(record, matched_nodes)
            if score <= 0:
                continue

            reasoning_path = sorted(
                entry["reasoning_path"],
                key=lambda path: (path["query_node"], path["hyperedge_id"], path["memory_id"]),
            )
            ranked_hits.append(
                {
                    **record,
                    "retrieval_score": score,
                    "hypergraph_score": score,
                    "matched_nodes": sorted(matched_nodes),
                    "matched_hyperedges": sorted(entry["matched_hyperedges"]),
                    "reasoning_path": reasoning_path,
                }
            )

        return sorted(
            ranked_hits,
            key=lambda hit: (-hit["retrieval_score"], hit["memory_id"]),
        )[:top_k]
