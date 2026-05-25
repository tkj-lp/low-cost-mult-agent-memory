from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any


# 这个模块只负责“怎么把记忆组织成超图、怎么基于超图检索”。
# memory.py 继续负责对 Agent 暴露 add_memory() / search_memory() 接口。
# 后续如果换成数据库或真正的超图神经网络，优先替换本文件。
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

NODE_WEIGHTS = {
    "tag": 3.0,
    "keyword": 1.0,
}


def _normalize_text(text: str) -> str:
    return text.lower().replace("_", "-").strip()


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9-]+", "-", _normalize_text(text)).strip("-")
    return slug[:48] or "empty"


def _stable_id(text: str, *, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _tokenize(text: str) -> list[str]:
    """把任务和总结文本转成可复用的关键词节点。"""

    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9-]*", _normalize_text(text))
    return sorted({token for token in tokens if len(token) > 1 and token not in STOPWORDS})


def _node_id(kind: str, value: str) -> str:
    """统一节点 id 格式，避免 tag 和 keyword 的同名值混在一起。"""

    return f"{kind}:{_normalize_text(value)}"


def task_node_id(task_topic: str) -> str:
    """用任务内容生成稳定任务节点，而不是复用 memory_id。"""

    return _node_id("task", f"{_stable_id(task_topic)}-{_slug(task_topic)}")


def build_hypergraph_view(record: dict[str, Any]) -> dict[str, Any]:
    """为单条记忆构建可解释超图视图。

    一条记忆对应一个事件超边；tag / keyword / agent 等节点按具体取值复用。
    多条记忆通过这些共享节点相连，检索时再聚合回 memory 节点。
    """

    memory_id = record["memory_id"]
    task_topic = record.get("task_topic", "")
    tags = sorted({_normalize_text(tag) for tag in record.get("tags", [])})
    keywords = _tokenize(
        " ".join(
            [
                task_topic,
                record.get("summary", ""),
            ]
        )
    )

    memory_node = _node_id("memory", memory_id)
    task_node = task_node_id(task_topic)
    agent_node = _node_id("agent", record.get("source_agent", "unknown"))
    tag_nodes = [_node_id("tag", tag) for tag in tags]
    keyword_nodes = [_node_id("keyword", keyword) for keyword in keywords]
    hyperedge_id = _node_id("hyperedge", memory_id)
    hyperedge_nodes = [
        memory_node,
        task_node,
        agent_node,
        *tag_nodes,
        *keyword_nodes,
    ]

    return {
        "memory_node": memory_node,
        "task_node": task_node,
        "task_label": task_topic,
        "agent_node": agent_node,
        "tag_nodes": tag_nodes,
        "keyword_nodes": keyword_nodes,
        "event_hyperedge": {
            "hyperedge_id": hyperedge_id,
            "type": "memory_event",
            "nodes": hyperedge_nodes,
        },
    }


def ensure_hypergraph_view(record: dict[str, Any]) -> dict[str, Any]:
    """兼容旧记录：没有超图字段时按当前字段即时补建。"""

    return record.get("hypergraph") or build_hypergraph_view(record)


def _query_nodes(query: str) -> set[str]:
    """查询文本同时尝试匹配 tag 节点和 keyword 节点。"""

    tokens = _tokenize(query)
    nodes: set[str] = set()
    for token in tokens:
        nodes.add(_node_id("keyword", token))
        nodes.add(_node_id("tag", token))
    return nodes


def _score_record(record: dict[str, Any], matched_nodes: set[str]) -> float:
    """按命中的节点类型给记忆打分，避免常见词命中后返回所有记忆。"""

    score = 0.0
    for node in matched_nodes:
        kind, _, _ = node.partition(":")
        score += NODE_WEIGHTS.get(kind, 0.5)
    # 同分时让信息更丰富的总结略微靠前。
    score += min(len(record.get("summary", "")) / 1000.0, 0.25)
    return round(score, 4)


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
        for node in _query_nodes(query):
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
            score = _score_record(record, matched_nodes)
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


def search_hypergraph_memory(memory: list[dict[str, Any]], query: str, *, top_k: int) -> list[dict[str, Any]]:
    """从超图索引中检索相关记忆，并返回可解释命中结果。"""

    index = HypergraphIndex.from_memory_records(memory)
    return index.search(query, top_k=top_k)
