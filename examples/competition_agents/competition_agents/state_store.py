from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from typing import Any


def estimate_bytes(value: Any) -> int:
    """估算进程内状态对象大小，用于展示非文本状态规模。"""

    if value is None:
        return 0
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, (int, float, bool)):
        return sys.getsizeof(value)
    if isinstance(value, dict):
        return sum(estimate_bytes(key) + estimate_bytes(item) for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return sum(estimate_bytes(item) for item in value)
    shape = getattr(value, "shape", None)
    element_size = getattr(value, "element_size", None)
    nelement = getattr(value, "nelement", None)
    if shape is not None and callable(element_size) and callable(nelement):
        try:
            return int(element_size()) * int(nelement())
        except Exception:
            return sys.getsizeof(value)
    return sys.getsizeof(value)


@dataclass
class StateStore:
    """进程内非文本状态池。

    Agent 间消息只传 `state_id`，向量或 hidden 本体放在这里，避免把大对象塞进协议包。
    """

    _items: dict[str, dict[str, Any]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def put(self, state_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        size = estimate_bytes(payload)
        record = {
            "state_id": state_id,
            "bytes": size,
            "payload": payload,
        }
        with self._lock:
            self._items[state_id] = record
        return {"state_id": state_id, "bytes": size, "storage": "process_state_store"}

    def get(self, state_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._items.get(state_id)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


GLOBAL_STATE_STORE = StateStore()


def put_state_payload(state_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return GLOBAL_STATE_STORE.put(state_id, payload)


def get_state_payload(state_id: str) -> dict[str, Any] | None:
    return GLOBAL_STATE_STORE.get(state_id)


def clear_state_store() -> None:
    GLOBAL_STATE_STORE.clear()
