from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, TypedDict


class Metrics(TypedDict):
    """用于比较 text 模式和 protocol 模式的统计指标。"""

    message_count: int
    text_chars: int
    latent_transfer_count: int
    latent_bytes: int


class CompetitionState(TypedDict):
    """在所有 LangGraph 节点之间传递的共享状态。"""

    task: str
    mode: Literal["text", "protocol"]
    plan: list[str]
    evidence: list[str]
    execution_result: dict[str, Any]
    final_answer: str
    messages: list[dict[str, Any]]
    latent_states: dict[str, dict[str, Any]]
    memory: list[dict[str, Any]]
    memory_hits: list[dict[str, Any]]
    metrics: Metrics


def initial_state(task: str, mode: Literal["text", "protocol"], memory: list[dict] | None = None) -> CompetitionState:
    """创建单个任务状态，同时保留跨任务共享记忆。"""

    return {
        "task": task,
        "mode": mode,
        "plan": [],
        "evidence": [],
        "execution_result": {},
        "final_answer": "",
        "messages": [],
        "latent_states": {},
        "memory": memory or [],
        "memory_hits": [],
        "metrics": {
            "message_count": 0,
            "text_chars": 0,
            "latent_transfer_count": 0,
            "latent_bytes": 0,
        },
    }


def simple_embedding(text: str, dimensions: int = 16) -> list[float]:
    # 原型阶段使用轻量、确定性的 embedding。
    # 环境准备好后，可替换成真实 embedding 模型或 API。
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    for index in range(dimensions):
        raw = digest[index % len(digest)]
        values.append(round((raw / 255.0) * 2.0 - 1.0, 4))
    return values


def put_latent_state(
    state: CompetitionState,
    *,
    producer: str,
    topic: str,
    text: str,
) -> tuple[str, dict[str, dict[str, Any]], Metrics]:
    """创建非文本 latent state，并更新状态传递指标。"""

    vector = simple_embedding(text)
    state_id = f"state-{len(state['latent_states']) + 1:04d}"
    latent_state = {
        "state_id": state_id,
        "type": "embedding",
        "producer": producer,
        "topic": topic,
        "shape": [len(vector)],
        "vector": vector,
    }
    metrics = dict(state["metrics"])
    # 简化估算：把向量按 32 位浮点数统计数据规模。
    metrics["latent_transfer_count"] += 1
    metrics["latent_bytes"] += len(vector) * 4
    return state_id, {**state["latent_states"], state_id: latent_state}, metrics


def record_message(
    state: CompetitionState,
    *,
    sender: str,
    receiver: str,
    action: str,
    payload: str | dict[str, Any],
    metrics: Metrics | None = None,
) -> tuple[list[dict[str, Any]], Metrics]:
    """记录一条 Agent 间消息，并更新通信指标。"""

    # 结构化 payload 使用紧凑 JSON 统计传输大小。
    serialized = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    next_metrics = dict(metrics or state["metrics"])
    next_metrics["message_count"] += 1
    next_metrics["text_chars"] += len(serialized)

    # 网页端可以把这条 message 画成通信图中的一条边。
    message = {
        "from": sender,
        "to": receiver,
        "action": action,
        "payload": payload,
        "char_cost": len(serialized),
    }
    return [*state["messages"], message], next_metrics
