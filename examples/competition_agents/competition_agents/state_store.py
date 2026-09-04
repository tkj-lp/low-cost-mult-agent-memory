from __future__ import annotations

import sys
import threading
import time
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

    def put(
        self,
        state_id: str,
        payload: dict[str, Any],
        *,
        scope: str = "task",
        ttl_seconds: float | None = None,
    ) -> dict[str, Any]:
        size = estimate_bytes(payload)
        created_at = time.monotonic()
        record = {
            "state_id": state_id,
            "bytes": size,
            "payload": payload,
            "scope": scope,
            "created_at_monotonic": created_at,
            "expires_at_monotonic": (
                created_at + max(0.0, float(ttl_seconds))
                if ttl_seconds is not None
                else None
            ),
        }
        with self._lock:
            self._items[state_id] = record
        return {
            "state_id": state_id,
            "bytes": size,
            "storage": "process_state_store",
            "scope": scope,
        }

    def get(self, state_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._items.get(state_id)
            if record is None:
                return None
            expires_at = record.get("expires_at_monotonic")
            if expires_at is not None and time.monotonic() >= float(expires_at):
                self._items.pop(state_id, None)
                return None
            return record

    def resolve(
        self,
        state_id: str,
        *,
        expected_content_type: str | None = None,
    ) -> dict[str, Any] | None:
        record = self.get(state_id)
        if record is None:
            return None
        payload = record.get("payload")
        if not isinstance(payload, dict):
            return None
        if expected_content_type and payload.get("content_type") != expected_content_type:
            return None
        return payload

    def release(self, state_id: str) -> bool:
        with self._lock:
            return self._items.pop(state_id, None) is not None

    def clear_scope(self, scope: str) -> int:
        with self._lock:
            state_ids = [
                state_id
                for state_id, record in self._items.items()
                if str(record.get("scope") or "task") == scope
            ]
            for state_id in state_ids:
                self._items.pop(state_id, None)
            return len(state_ids)

    def clear_expired(self) -> int:
        now = time.monotonic()
        with self._lock:
            state_ids = [
                state_id
                for state_id, record in self._items.items()
                if record.get("expires_at_monotonic") is not None
                and now >= float(record["expires_at_monotonic"])
            ]
            for state_id in state_ids:
                self._items.pop(state_id, None)
            return len(state_ids)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


GLOBAL_STATE_STORE = StateStore()


def put_state_payload(
    state_id: str,
    payload: dict[str, Any],
    *,
    scope: str = "task",
    ttl_seconds: float | None = None,
) -> dict[str, Any]:
    return GLOBAL_STATE_STORE.put(
        state_id,
        payload,
        scope=scope,
        ttl_seconds=ttl_seconds,
    )


def get_state_payload(state_id: str) -> dict[str, Any] | None:
    return GLOBAL_STATE_STORE.get(state_id)


def resolve_state_payload(
    state_id: str,
    *,
    expected_content_type: str | None = None,
) -> dict[str, Any] | None:
    return GLOBAL_STATE_STORE.resolve(
        state_id,
        expected_content_type=expected_content_type,
    )


def release_state_payload(state_id: str) -> bool:
    return GLOBAL_STATE_STORE.release(state_id)


def clear_state_scope(scope: str) -> int:
    return GLOBAL_STATE_STORE.clear_scope(scope)


def clear_expired_states() -> int:
    return GLOBAL_STATE_STORE.clear_expired()


def clear_state_store() -> None:
    GLOBAL_STATE_STORE.clear()
