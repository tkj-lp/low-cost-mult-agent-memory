from __future__ import annotations

import os
from typing import Any

from .retrieval import RETRIEVAL_MODE_VALUES, RetrievalMode, search_memory_by_mode


DEFAULT_RETRIEVAL_MODE: RetrievalMode = "hypergraph"


def configured_retrieval_mode(value: str | None = None) -> RetrievalMode:
    """读取并校验当前共享记忆检索方式。"""

    mode = str(value or os.environ.get("MEMORY_RETRIEVAL_MODE") or DEFAULT_RETRIEVAL_MODE)
    mode = mode.strip().lower()
    if mode not in RETRIEVAL_MODE_VALUES:
        supported = ", ".join(RETRIEVAL_MODE_VALUES)
        raise ValueError(f"Unsupported memory retrieval mode: {mode!r}. Expected one of: {supported}.")
    return mode  # type: ignore[return-value]


def memory_organization_for_mode(mode: str) -> str:
    """普通检索使用文本记忆，超图与融合检索使用超图记忆。"""

    return "hypergraph" if mode in {"hypergraph", "fusion"} else "text"


def load_persistent_memory(database_path: str | None = None) -> list[dict[str, Any]]:
    """加载持久化记忆，供冷启动调用侧直接传入原有状态接口。"""

    database_path = (
        database_path or os.environ.get("HYPERGRAPH_MEMORY_DB", "")
    ).strip()
    enabled = os.environ.get("HYPERGRAPH_MEMORY_AUTOLOAD", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if not database_path or not enabled:
        return []

    from .store import SQLiteHypergraphMemoryStore

    with SQLiteHypergraphMemoryStore(database_path) as store:
        return store.list()


def _with_persistent_memory(memory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    persisted = load_persistent_memory()
    if not persisted:
        return memory
    merged: dict[str, dict[str, Any]] = {}
    for record in [*persisted, *memory]:
        key = str(
            record.get("global_memory_id") or record.get("memory_id") or id(record)
        )
        merged[key] = record
    return list(merged.values())


def search_hypergraph_memory(
    memory: list[dict[str, Any]],
    query: str,
    *,
    top_k: int,
    memory_types: set[str] | None = None,
    type_limits: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """保持原接口不变，并按配置分派五种共享记忆检索方式。"""

    mode = configured_retrieval_mode()
    return search_memory_by_mode(
        _with_persistent_memory(memory),
        query,
        top_k=top_k,
        mode=mode,
        memory_types=memory_types,
        type_limits=type_limits,
    )
