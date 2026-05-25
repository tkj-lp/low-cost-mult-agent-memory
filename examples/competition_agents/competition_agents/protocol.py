from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

PROTOCOL_VERSION = "0.2"


def make_cognitive_gap(
    *,
    agent: str,
    known: list[str],
    unknown: list[str],
    needed: list[str],
    gap_type: str,
    confidence: float,
) -> dict[str, Any]:
    """创建轻量 CKM 认知差距描述，表达 Agent 当前知道和缺少什么。"""

    return {
        "agent": agent,
        "known": known,
        "unknown": unknown,
        "needed": needed,
        "gap_type": gap_type,
        "confidence": round(confidence, 2),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def make_packet(
    *,
    sender: str,
    receiver: str,
    action: str,
    content_type: str = "protocol.command",
    schema_ref: str | None = None,
    params: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    state_refs: list[str] | None = None,
    memory_refs: list[str] | None = None,
    capability_required: list[str] | None = None,
    receiver_capability: str | None = None,
    cognitive_gap: dict[str, Any] | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    """创建一条 Agent 间结构化通信包。"""

    packet = {
        "protocol_version": PROTOCOL_VERSION,
        # message_id 方便网页和日志跨模块追踪同一条消息。
        "msg_id": f"msg-{uuid4().hex[:8]}",
        "from": sender,
        "to": receiver,
        "action": action,
        "content_type": content_type,
        "params": params or {},
        "capability_required": capability_required or [],
        "sender_capability_ref": f"capability://{sender}",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    optional_fields = {
        "schema_ref": schema_ref,
        "result": result,
        # state_refs 传递非文本状态 id，避免重复传大段内容。
        "state_refs": state_refs,
        "memory_refs": memory_refs,
        "receiver_capability": receiver_capability,
        "cognitive_gap": cognitive_gap,
        "confidence": confidence,
    }
    for key, value in optional_fields.items():
        if value not in (None, [], {}):
            packet[key] = value
    return packet



def make_compact_packet(
    *,
    sender: str,
    receiver: str,
    action: str,
    content_type: str,
    params: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    state_refs: list[str] | None = None,
    memory_refs: list[str] | None = None,
    capability_required: list[str] | None = None,
    cognitive_gap: dict[str, Any] | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    """????????????

    ?? protocol ???????????????????state_ref ?
    memory_ref????????? schema??? cognitive_gap ?????
    state ????????????
    """

    packet: dict[str, Any] = {
        "v": PROTOCOL_VERSION,
        "id": f"m-{uuid4().hex[:6]}",
        "from": sender,
        "to": receiver,
        "action": action,
        "type": content_type,
    }
    optional_fields = {
        "p": params,
        "r": result,
        "s": state_refs,
        "m": memory_refs,
        "cap": capability_required,
        "conf": round(confidence, 2) if confidence is not None else None,
    }
    if cognitive_gap:
        optional_fields["gap"] = {
            "agent": cognitive_gap.get("agent"),
            "type": cognitive_gap.get("gap_type"),
            "conf": cognitive_gap.get("confidence"),
        }
    for key, value in optional_fields.items():
        if value not in (None, [], {}):
            packet[key] = value
    return packet
