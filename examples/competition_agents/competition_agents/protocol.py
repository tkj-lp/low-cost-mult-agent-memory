from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def make_packet(
    *,
    sender: str,
    receiver: str,
    action: str,
    params: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    state_refs: list[str] | None = None,
    capability_required: list[str] | None = None,
) -> dict[str, Any]:
    """创建一条 Agent 间结构化通信包。"""

    return {
        # msg_id 方便网页和日志跨模块追踪同一条消息。
        "msg_id": f"msg-{uuid4().hex[:8]}",
        "from": sender,
        "to": receiver,
        "action": action,
        "params": params or {},
        "result": result,
        # state_refs 传递非文本状态 id，避免重复传大段内容。
        "state_refs": state_refs or [],
        "capability_required": capability_required or [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
