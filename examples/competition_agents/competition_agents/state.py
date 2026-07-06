from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from competition_agents.capabilities import build_capability_registry
from competition_agents.embedding import embed_text, hash_embedding
from competition_agents.latent_hidden import (
    extract_last_hidden_state,
    hidden_storage_mode,
    should_store_full_hidden_tensor,
    stored_hidden_bytes,
    stored_hidden_shape,
)
from competition_agents.runtime_events import emit_message_event, emit_state_transfer_event
from competition_agents.state_store import put_state_payload


class Metrics(TypedDict):
    """用于比较 text 模式和 protocol 模式的统计指标。"""

    message_count: int
    text_chars: int
    protocol_bytes: int
    communication_bytes: int
    latent_transfer_count: int
    latent_bytes: int
    transfer_latency_ms: float
    state_generation_ms: float


AGENT_CODES = {
    "planner": "p",
    "retriever": "r",
    "reasoner": "n",
    "executor": "x",
    "summarizer": "s",
    "memory": "m",
    "runtime": "u",
}

ACTION_CODES = {
    "retrieve": "ret",
    "reason": "rea",
    "summarize": "sum",
    "execute": "exe",
    "start_task": "sta",
    "open_session": "opn",
    "discover_capability": "cap",
    "map_protocol": "map",
}


class CompetitionState(TypedDict):
    """在所有 LangGraph 节点之间传递的共享状态。"""

    task: str
    task_meta: dict[str, Any]
    mode: Literal["text", "protocol", "hidden"]
    plan: list[str]
    evidence: list[str]
    execution_result: dict[str, Any]
    final_answer: str
    messages: list[dict[str, Any]]
    agent_outputs: list[dict[str, Any]]
    capability_registry: dict[str, dict[str, Any]]
    cognitive_gaps: list[dict[str, Any]]
    latent_states: dict[str, dict[str, Any]]
    state_refs: dict[str, str]
    memory: list[dict[str, Any]]
    memory_hits: list[dict[str, Any]]
    experience_hits: list[dict[str, Any]]
    experience_rules: list[dict[str, Any]]
    memory_context: list[dict[str, Any]]
    memory_refine_case: dict[str, Any]
    memory_refine_debug: dict[str, Any]
    latest_evidence_write: dict[str, Any]
    latest_experience_write: dict[str, Any]
    memory_write_count: int
    experience_write_count: int
    metrics: Metrics


def initial_state(
    task: str,
    mode: Literal["text", "protocol", "hidden"],
    memory: list[dict] | None = None,
    task_meta: dict[str, Any] | None = None,
) -> CompetitionState:
    """创建单个任务状态，同时保留跨任务共享记忆。"""

    return {
        "task": task,
        "task_meta": task_meta or {},
        "mode": mode,
        "plan": [],
        "evidence": [],
        "execution_result": {},
        "final_answer": "",
        "messages": [],
        "agent_outputs": [],
        "capability_registry": build_capability_registry(),
        "cognitive_gaps": [],
        "latent_states": {},
        "state_refs": {},
        "memory": memory or [],
        "memory_hits": [],
        "experience_hits": [],
        "experience_rules": [],
        "memory_context": [],
        "memory_refine_case": {},
        "memory_refine_debug": {},
        "latest_evidence_write": {},
        "latest_experience_write": {},
        "memory_write_count": 0,
        "experience_write_count": 0,
        "metrics": {
            "message_count": 0,
            "text_chars": 0,
            "protocol_bytes": 0,
            "communication_bytes": 0,
            "latent_transfer_count": 0,
            "latent_bytes": 0,
            "transfer_latency_ms": 0.0,
            "state_generation_ms": 0.0,
        },
    }


def simple_embedding(text: str, dimensions: int = 16) -> list[float]:
    # 保留旧接口，避免历史实验脚本需要同步大改。
    return hash_embedding(text, dimensions)


def put_latent_state(
    state: CompetitionState,
    *,
    producer: str,
    topic: str,
    text: str,
    consumers: list[str] | None = None,
    scope: str = "task",
    ttl: int = 3,
) -> tuple[str, dict[str, dict[str, Any]], Metrics]:
    """创建非文本 latent state，并更新状态传递指标。"""

    started = time.perf_counter()
    embedding = embed_text(text)
    generation_elapsed_ms = (time.perf_counter() - started) * 1000
    vector = embedding.vector
    state_id = f"state-{len(state['latent_states']) + 1:04d}"
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    store_info = put_state_payload(
        state_id,
        {
            "content_type": "latent.embedding",
            "producer": producer,
            "topic": topic,
            "source_text": text,
            "embedding_provider": embedding.provider,
            "embedding_model": embedding.model,
            "vector": vector,
        },
    )
    latent_state = {
        "state_id": state_id,
        "type": "latent_state_ref",
        "content_type": "latent.embedding",
        "producer": producer,
        "consumers": consumers or [],
        "topic": topic,
        "summary": text[:160],
        "source_text": text,
        "embedding_provider": embedding.provider,
        "embedding_model": embedding.model,
        "shape": [len(vector)],
        "store_ref": store_info["state_id"],
        "storage": store_info["storage"],
        "stored_payload_bytes": store_info["bytes"],
        "scope": scope,
        "ttl": ttl,
        "checksum": checksum,
        "source_text_chars": len(text),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics = dict(state["metrics"])
    # 简化估算：把向量按 32 位浮点数统计数据规模。
    metrics["latent_transfer_count"] += 1
    metrics["latent_bytes"] += len(vector) * 4
    metrics["state_generation_ms"] = metrics.get("state_generation_ms", 0.0) + generation_elapsed_ms
    emit_state_transfer_event(
        state,
        state_id=state_id,
        producer=producer,
        consumers=consumers or [],
        content_type=latent_state["content_type"],
        summary=latent_state["summary"],
        shape=latent_state["shape"],
        bytes_size=len(vector) * 4,
    )
    return state_id, {**state["latent_states"], state_id: latent_state}, metrics


def put_hidden_state(
    state: CompetitionState,
    *,
    producer: str,
    topic: str,
    text: str,
    consumers: list[str] | None = None,
    scope: str = "task",
    ttl: int = 3,
    hidden_result: Any | None = None,
    hidden_tensor: Any | None = None,
) -> tuple[str, dict[str, dict[str, Any]], Metrics]:
    """创建本地 LLM hidden state，并复用现有 latent_state 容器输出 state_ref。"""

    if hidden_result is None:
        started = time.perf_counter()
        result, tensor = extract_last_hidden_state(text)
        generation_elapsed_ms = (time.perf_counter() - started) * 1000
    else:
        result = hidden_result
        tensor = hidden_tensor
        generation_elapsed_ms = 0.0
    state_id = f"hidden-{len(state['latent_states']) + 1:04d}"
    latent_state = {
        "state_id": state_id,
        "type": "latent_hidden_state",
        "content_type": "latent.llm_hidden_state",
        "producer": producer,
        "consumers": consumers or [],
        "topic": topic,
        "summary": text[:160],
        "source_text": text,
        "model": result.model_name,
        "layer": result.layer,
        "shape": result.shape,
        "stored_shape": stored_hidden_shape(result),
        "pooling": result.pooling,
        "pooled_shape": [len(result.pooled_vector)],
        "token_count": result.token_count,
        "dtype": result.dtype,
        "device": result.device,
        "checksum": result.checksum,
        "scope": scope,
        "ttl": ttl,
        "storage": "process_state_store",
        "storage_mode": hidden_storage_mode(),
        "stored_hidden_bytes": stored_hidden_bytes(result),
        "reused_generation_hidden": hidden_result is not None,
        "source_text_chars": len(text),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    store_info = put_state_payload(
        state_id,
        {
            "content_type": "latent.llm_hidden_state",
            "producer": producer,
            "topic": topic,
            "source_text": text,
            "pooled_vector": result.pooled_vector,
            "tensor": tensor if should_store_full_hidden_tensor() else None,
            "model": result.model_name,
            "shape": result.shape,
            "stored_shape": stored_hidden_shape(result),
        },
    )
    latent_state["store_ref"] = store_info["state_id"]
    latent_state["stored_payload_bytes"] = store_info["bytes"]
    metrics = dict(state["metrics"])
    metrics["latent_transfer_count"] += 1
    metrics["latent_bytes"] += stored_hidden_bytes(result)
    metrics["state_generation_ms"] = metrics.get("state_generation_ms", 0.0) + generation_elapsed_ms
    emit_state_transfer_event(
        state,
        state_id=state_id,
        producer=producer,
        consumers=consumers or [],
        content_type=latent_state["content_type"],
        summary=latent_state["summary"],
        shape=latent_state["stored_shape"],
        bytes_size=stored_hidden_bytes(result),
    )
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

    started = time.perf_counter()
    # 结构化 payload 使用紧凑 JSON 统计传输大小。
    serialized = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    protocol_bytes = len(serialized.encode("utf-8"))
    minimal_payload = minimal_wire_payload(state, sender=sender, receiver=receiver, action=action, payload=payload)
    minimal_serialized = minimal_payload if isinstance(minimal_payload, str) else json.dumps(minimal_payload, ensure_ascii=False, separators=(",", ":"))
    communication_bytes = len(minimal_serialized.encode("utf-8"))
    transfer_elapsed_ms = (time.perf_counter() - started) * 1000
    next_metrics = dict(metrics or state["metrics"])
    next_metrics["message_count"] += 1
    next_metrics["text_chars"] += len(serialized)
    next_metrics["protocol_bytes"] = next_metrics.get("protocol_bytes", 0) + protocol_bytes
    next_metrics["communication_bytes"] = next_metrics.get("communication_bytes", 0) + communication_bytes
    next_metrics["transfer_latency_ms"] += transfer_elapsed_ms

    # 网页端可以把这条 message 画成通信图中的一条边。
    message = {
        "from": sender,
        "to": receiver,
        "action": action,
        "payload": payload,
        "char_cost": len(serialized),
        "protocol_bytes": protocol_bytes,
        "communication_bytes": communication_bytes,
        "transfer_latency_ms": transfer_elapsed_ms,
    }
    emit_message_event(
        state,
        sender=sender,
        receiver=receiver,
        action=action,
        payload=payload,
        char_cost=len(serialized),
        transfer_elapsed_ms=transfer_elapsed_ms,
    )
    return [*state["messages"], message], next_metrics


def minimal_wire_payload(
    state: CompetitionState,
    *,
    sender: str,
    receiver: str,
    action: str,
    payload: str | dict[str, Any],
) -> str | dict[str, Any]:
    """返回用于指标统计的最小线缆载荷，不影响可读协议包和运行日志。"""

    if isinstance(payload, str):
        return payload

    mode = state.get("mode", "protocol")
    state_refs = _payload_list(payload, "s", "state_refs")
    memory_refs = _payload_list(payload, "m", "memory_refs")
    base: dict[str, Any] = {
        "f": _agent_code(sender),
        "t": _agent_code(receiver),
        "a": ACTION_CODES.get(action, action[:3] or "msg"),
    }
    if state_refs:
        base["s"] = [_short_ref(ref) for ref in state_refs]
    if memory_refs:
        base["m"] = [_short_ref(ref) for ref in memory_refs]
        base["n"] = len(memory_refs)

    if mode == "hidden":
        base["z"] = "h"
        return base

    embedding_provider = _state_ref_provider(state, state_refs)
    if embedding_provider not in {"", "hash", "hash-v1"}:
        base["z"] = "e"
        return base

    params = _compact_value(_payload_value(payload, "p", "params"))
    result = _compact_value(_payload_value(payload, "r", "result"))
    capability = _payload_list(payload, "cap", "capability_required")
    if params:
        base["p"] = params
    if result:
        base["r"] = result
    if capability:
        base["c"] = [str(item)[:3] for item in capability]
    return base


def _payload_value(payload: dict[str, Any], compact_key: str, verbose_key: str) -> Any:
    return payload.get(compact_key) if compact_key in payload else payload.get(verbose_key)


def _payload_list(payload: dict[str, Any], compact_key: str, verbose_key: str) -> list[str]:
    value = _payload_value(payload, compact_key, verbose_key)
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if value:
        return [str(value)]
    return []


def _agent_code(name: str) -> str:
    value = str(name or "").lower()
    return AGENT_CODES.get(value, value[:1] or "-")


def _short_ref(value: str) -> str:
    text = str(value or "")
    prefix_map = {"hidden-": "h", "state-": "s", "mem-": "m"}
    for prefix, short in prefix_map.items():
        if text.startswith(prefix):
            return f"{short}{text[len(prefix):].lstrip('0') or '0'}"
    return text[:10]


def _state_ref_provider(state: CompetitionState, refs: list[str]) -> str:
    latent_states = state.get("latent_states", {})
    for ref in refs:
        latent_state = latent_states.get(ref, {})
        provider = str(latent_state.get("embedding_provider") or latent_state.get("content_type") or "")
        if provider:
            return provider
    return ""


def _compact_value(value: Any) -> Any:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            if item in (None, "", [], {}):
                continue
            short_key = str(key)[:2]
            if isinstance(item, (list, tuple)):
                compact[short_key] = len(item)
            elif isinstance(item, dict):
                compact[short_key] = len(item)
            elif isinstance(item, bool):
                compact[short_key] = int(item)
            elif isinstance(item, (int, float)):
                compact[short_key] = item
            else:
                compact[short_key] = str(item)[:18]
        return compact or None
    if isinstance(value, list):
        return len(value)
    return str(value)[:18]
