from __future__ import annotations

from typing import Any, Final, Literal


MemoryLayer = Literal["knowledge", "episodic", "experience"]

KNOWLEDGE_LAYER: Final[MemoryLayer] = "knowledge"
EPISODIC_LAYER: Final[MemoryLayer] = "episodic"
EXPERIENCE_LAYER: Final[MemoryLayer] = "experience"
MEMORY_LAYERS: Final[tuple[MemoryLayer, ...]] = (
    KNOWLEDGE_LAYER,
    EPISODIC_LAYER,
    EXPERIENCE_LAYER,
)

_LAYER_ALIASES: Final[dict[str, MemoryLayer]] = {
    "knowledge": KNOWLEDGE_LAYER,
    "context": KNOWLEDGE_LAYER,
    "fact": KNOWLEDGE_LAYER,
    "facts": KNOWLEDGE_LAYER,
    "episodic": EPISODIC_LAYER,
    "episode": EPISODIC_LAYER,
    "qa": EPISODIC_LAYER,
    "question-answer": EPISODIC_LAYER,
    "evidence": EPISODIC_LAYER,
    "answer": EPISODIC_LAYER,
    "experience": EXPERIENCE_LAYER,
    "strategy": EXPERIENCE_LAYER,
    "procedural": EXPERIENCE_LAYER,
}


def normalize_memory_layer(
    value: Any,
    *,
    memory_type: Any = None,
) -> MemoryLayer:
    """Return one of the three primary memory layers.

    Older records only contain `memory_type`. They remain readable by mapping
    evidence and answer records to the episodic layer and strategy records to
    the experience layer.
    """

    normalized = str(value or "").strip().lower().replace("_", "-")
    if normalized in _LAYER_ALIASES:
        return _LAYER_ALIASES[normalized]

    normalized_type = str(memory_type or "").strip().lower().replace("_", "-")
    return _LAYER_ALIASES.get(normalized_type, EPISODIC_LAYER)


def memory_layer_for_record(record: dict[str, Any]) -> MemoryLayer:
    """Resolve the primary layer without mutating a memory record."""

    return normalize_memory_layer(
        record.get("memory_layer"),
        memory_type=record.get("memory_type"),
    )


def partition_memory_by_layer(
    records: list[dict[str, Any]],
) -> dict[MemoryLayer, list[dict[str, Any]]]:
    """Partition records while keeping the original order within each layer."""

    partitions: dict[MemoryLayer, list[dict[str, Any]]] = {
        layer: [] for layer in MEMORY_LAYERS
    }
    for record in records:
        partitions[memory_layer_for_record(record)].append(record)
    return partitions
