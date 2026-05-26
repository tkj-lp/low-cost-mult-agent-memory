from __future__ import annotations

from typing import Any

from .text import node_id, slug, stable_id, tokenize


def task_node_id(task_topic: str) -> str:
    """用任务内容生成稳定任务节点，而不是复用 memory_id。"""

    return node_id("task", f"{stable_id(task_topic)}-{slug(task_topic)}")


def build_hypergraph_view(record: dict[str, Any]) -> dict[str, Any]:
    """为单条记忆构建可解释超图视图。

    一条记忆对应一个事件超边；tag / keyword / agent 等节点按具体取值复用。
    多条记忆通过这些共享节点相连，检索时再聚合回 memory 节点。
    """

    memory_id = record["memory_id"]
    task_topic = record.get("task_topic", "")
    tags = sorted({tag.lower().replace("_", "-").strip() for tag in record.get("tags", [])})
    keywords = tokenize(
        " ".join(
            [
                task_topic,
                record.get("summary", ""),
            ]
        )
    )

    memory_node = node_id("memory", memory_id)
    task_node = task_node_id(task_topic)
    agent_node = node_id("agent", record.get("source_agent", "unknown"))
    tag_nodes = [node_id("tag", tag) for tag in tags]
    keyword_nodes = [node_id("keyword", keyword) for keyword in keywords]
    hyperedge_id = node_id("hyperedge", memory_id)
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
