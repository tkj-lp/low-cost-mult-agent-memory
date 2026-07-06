from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


EVENT_LOG_ENV = "AGENT_EVENT_LOG_PATH"
DEFAULT_EVENT_LOG_FILE = "examples/competition_agents/frontend/public/runtime/run_events.jsonl"
SCHEMA_VERSION = 1


def configure_event_log(path: str, run_meta: dict[str, Any]) -> None:
    """初始化运行事件文件；失败时不影响实验主流程。"""

    if not path:
        os.environ.pop(EVENT_LOG_ENV, None)
        return
    os.environ[EVENT_LOG_ENV] = path
    try:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
    except OSError:
        return
    emit_runtime_event({"event_type": "run_start", **run_meta})


def emit_runtime_event(event: dict[str, Any]) -> None:
    """追加一条 JSONL 运行事件；这是旁路观测，不参与状态计算。"""

    path = os.environ.get(EVENT_LOG_ENV, "").strip()
    if not path:
        return
    record = {
        "schema_version": SCHEMA_VERSION,
        "event_id": event.get("event_id") or f"evt-{uuid4().hex[:10]}",
        "timestamp": event.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        **event,
    }
    try:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_frontend_safe(record), ensure_ascii=False, default=str, separators=(",", ":")))
            handle.write("\n")
    except OSError:
        return


def emit_message_event(
    state: dict[str, Any],
    *,
    sender: str,
    receiver: str,
    action: str,
    payload: str | dict[str, Any],
    char_cost: int,
    transfer_elapsed_ms: float,
) -> None:
    """把已有 Agent 间消息复制成前端可消费的事件。"""

    task_meta = _task_meta(state)
    state_refs, memory_refs = _payload_refs(payload)
    emit_runtime_event(
        {
            **_base_context(state, task_meta),
            "event_type": "agent_message",
            "from_agent": sender,
            "to_agent": receiver,
            "action_type": action,
            "input_params": _payload_value(payload, "params", "p"),
            "return_result": _payload_value(payload, "result", "r"),
            "capability": _payload_value(payload, "capability_required", "cap"),
            "receiver_capability": _payload_value(payload, "receiver_capability", "receiver_capability"),
            "state_refs": state_refs,
            "memory_refs": memory_refs,
            "text_chars": char_cost,
            "elapsed_ms": transfer_elapsed_ms,
            "payload": payload,
        }
    )


def emit_state_transfer_event(
    state: dict[str, Any],
    *,
    state_id: str,
    producer: str,
    consumers: list[str],
    content_type: str,
    summary: str,
    shape: list[int] | None,
    bytes_size: int,
) -> None:
    """记录非文本中间状态引用的生成和传递。"""

    task_meta = _task_meta(state)
    emit_runtime_event(
        {
            **_base_context(state, task_meta),
            "event_type": "state_transfer",
            "from_agent": producer,
            "to_agent": ",".join(consumers) if consumers else "unknown",
            "action_type": "transfer_state",
            "state_refs": [state_id],
            "memory_refs": [],
            "text_chars": 0,
            "elapsed_ms": 0,
            "state": {
                "state_ref": state_id,
                "content_type": content_type,
                "summary": summary,
                "shape": shape or [],
                "bytes": bytes_size,
            },
        }
    )


def emit_memory_retrieval_events(
    state: dict[str, Any],
    *,
    query: str,
    hits: list[dict[str, Any]],
) -> None:
    """在 Retriever 实际检索完成时写事件，避免前端只能在任务结束时看到记忆命中。"""

    task_meta = _task_meta(state)
    base = _base_context(state, task_meta)
    memory_refs = [hit.get("memory_id") for hit in hits if hit.get("memory_id")]
    emit_runtime_event(
        {
            **base,
            "event_type": "memory_query",
            "from_agent": "retriever",
            "to_agent": "memory",
            "action_type": "query_memory",
            "input_params": {"query": query[:500], "top_k": len(hits)},
            "return_result": {"hit_count": len(memory_refs)},
            "state_refs": [],
            "memory_refs": memory_refs,
            "text_chars": 0,
            "elapsed_ms": 0,
        }
    )
    for hit in hits:
        emit_memory_hit_event(base, hit)


def emit_memory_hit_event(base: dict[str, Any], hit: dict[str, Any]) -> None:
    emit_runtime_event(
        {
            **base,
            "event_type": "memory_hit",
            "from_agent": "memory",
            "to_agent": "retriever",
            "action_type": "return_memory",
            "state_refs": [],
            "memory_refs": [hit.get("memory_id")] if hit.get("memory_id") else [],
            "text_chars": 0,
            "elapsed_ms": 0,
            "return_result": {
                "memory_id": hit.get("memory_id"),
                "summary": hit.get("summary"),
                "retrieval_score": hit.get("retrieval_score"),
                "memory_type": hit.get("memory_type"),
                "task_topic": hit.get("task_topic"),
                "tags": hit.get("tags"),
                "matched_terms": hit.get("matched_terms"),
                "matched_nodes": hit.get("matched_nodes"),
                "reasoning_path": hit.get("reasoning_path"),
                "score_components": _memory_score_components(hit),
                "selection_reason": _memory_selection_reason(hit),
            },
        }
    )


def emit_task_result_events(
    *,
    mode: str,
    task_index: int,
    task_item: dict[str, Any],
    result: dict[str, Any],
    elapsed_ms: float,
    exact_match: bool | None,
) -> None:
    """把单任务完成后的命中、写回和汇总指标写成事件。"""

    base = _base_context({"mode": mode, "task_meta": _task_meta_from_item(task_index, task_item)}, _task_meta_from_item(task_index, task_item))
    memory_hits = result.get("memory_hits", []) or []

    latest_memory = result.get("latest_memory_write") or result.get("latest_evidence_write") or {}
    if latest_memory.get("memory_id"):
        emit_runtime_event(
            {
                **base,
                "event_type": "memory_write",
                "from_agent": latest_memory.get("source_agent") or "summarizer",
                "to_agent": "memory",
                "action_type": "write_memory",
                "state_refs": latest_memory.get("state_refs") or [],
                "memory_refs": [latest_memory.get("memory_id")],
                "text_chars": 0,
                "elapsed_ms": 0,
                "return_result": {
                    "memory_id": latest_memory.get("memory_id"),
                    "summary": latest_memory.get("summary"),
                    "memory_type": latest_memory.get("memory_type"),
                },
            }
        )

    metrics = result.get("metrics", {}) or {}
    emit_runtime_event(
        {
            **base,
            "event_type": "task_done",
            "from_agent": "runtime",
            "to_agent": "dashboard",
            "action_type": "finish_task",
            "state_refs": list((result.get("state_refs") or {}).values()),
            "memory_refs": [hit.get("memory_id") for hit in memory_hits if hit.get("memory_id")],
            "text_chars": metrics.get("text_chars", 0),
            "elapsed_ms": elapsed_ms,
            "return_result": {
                "message_count": metrics.get("message_count", 0),
                "state_transfer_count": metrics.get("latent_transfer_count", 0),
                "state_generation_ms": metrics.get("state_generation_ms", 0),
                "adjusted_elapsed_ms": max(0.0, elapsed_ms - float(metrics.get("state_generation_ms", 0) or 0)),
                "memory_hit_count": len(memory_hits),
                "exact_match": exact_match,
                "final_answer": result.get("final_answer", ""),
            },
        }
    )


def _task_meta(state: dict[str, Any]) -> dict[str, Any]:
    meta = state.get("task_meta", {})
    return meta if isinstance(meta, dict) else {}


def _task_meta_from_item(task_index: int, task_item: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_index": task_index,
        "task_id": task_item.get("task_id") or task_item.get("id") or str(task_index),
        "group_id": task_item.get("group_id"),
        "turn_id": task_item.get("turn_id"),
        "dataset_type": task_item.get("dataset_type"),
        "question": task_item.get("question", ""),
        "task_type": task_item.get("task_type"),
    }


def _base_context(state: dict[str, Any], task_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": state.get("mode"),
        "task_index": task_meta.get("task_index"),
        "task_id": task_meta.get("task_id"),
        "group_id": task_meta.get("group_id"),
        "turn_id": task_meta.get("turn_id"),
        "dataset_type": task_meta.get("dataset_type"),
        "task_title": task_meta.get("question") or task_meta.get("task_type") or task_meta.get("task_id"),
    }


def _payload_refs(payload: str | dict[str, Any]) -> tuple[list[str], list[str]]:
    if not isinstance(payload, dict):
        return [], []
    state_refs = payload.get("state_refs") or payload.get("refs") or payload.get("s") or []
    memory_refs = payload.get("memory_refs") or payload.get("m") or []
    if isinstance(state_refs, str):
        state_refs = [state_refs]
    if isinstance(memory_refs, str):
        memory_refs = [memory_refs]
    return list(state_refs), list(memory_refs)


def _payload_value(payload: str | dict[str, Any], full_key: str, compact_key: str) -> Any:
    if not isinstance(payload, dict):
        return {}
    return payload.get(full_key, payload.get(compact_key, {}))


def _memory_score_components(hit: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "retrieval_score",
        "text_score",
        "semantic_score",
        "embedding_score",
        "hypergraph_score",
        "fusion_score",
        "lexical_overlap_score",
        "tag_overlap_score",
        "graph_expansion_score",
        "evidence_chain_query_score",
        "gap_coverage_score",
        "structural_score",
        "reuse_score",
    ]
    return {key: hit.get(key) for key in keys if hit.get(key) is not None}


def _memory_selection_reason(hit: dict[str, Any]) -> str:
    if hit.get("selection_reason"):
        return str(hit["selection_reason"])
    matched_terms = [str(item) for item in (hit.get("matched_terms") or [])[:5] if str(item).strip()]
    if matched_terms:
        return "匹配查询词：" + "、".join(matched_terms)
    reasoning_path = hit.get("reasoning_path") or []
    if reasoning_path:
        first = reasoning_path[0]
        if isinstance(first, dict):
            query_node = first.get("query_node") or first.get("source") or first.get("term")
            target = first.get("memory_id") or first.get("target_id") or first.get("context_id")
            relation = first.get("relation") or first.get("hyperedge_id") or first.get("reason")
            parts = [str(item) for item in (query_node, relation, target) if item]
            if parts:
                return "检索路径：" + " → ".join(parts)
        return "存在可解释检索路径，排序靠前"
    score = hit.get("retrieval_score")
    if score is not None:
        return f"综合检索分数较高：{score}"
    return "命中结果由当前检索器排序返回"


def _frontend_safe(value: Any, *, max_string: int = 1200) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _frontend_safe(item, max_string=max_string)
            for key, item in value.items()
            if str(key) not in {"embedding", "hypergraph", "pooled_vector", "source_text", "tensor", "vector"}
        }
    if isinstance(value, list):
        return [_frontend_safe(item, max_string=max_string) for item in value]
    if isinstance(value, tuple):
        return [_frontend_safe(item, max_string=max_string) for item in value]
    if isinstance(value, str) and len(value) > max_string:
        return value[:max_string] + "..."
    return value
