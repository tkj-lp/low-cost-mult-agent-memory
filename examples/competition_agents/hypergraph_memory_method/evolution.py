from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import re
from typing import Any, Iterable

from .gating import record_primary_trigger_families
from .distill import lifecycle_stage, reuse_success_rate
from .text import normalize_text


def _normalized_pattern(value: Any) -> str:
    text = normalize_text(str(value or ""))
    return re.sub(r"[^a-z0-9-]+", "-", text).strip("-")


def experience_identity(record: dict[str, Any]) -> str:
    """Return a conservative identity for safely mergeable experience cards."""

    if str(record.get("memory_type", "evidence") or "evidence") != "experience":
        return ""
    problem = _normalized_pattern(record.get("problem_pattern"))
    solution = _normalized_pattern(record.get("solution_pattern"))
    if not problem or not solution:
        return ""
    primary_families = ",".join(sorted(record_primary_trigger_families(record))) or "unspecified"
    return f"{problem}|{solution}|{primary_families}"


def _unique_values(values: Iterable[Any]) -> list[Any]:
    unique: list[Any] = []
    seen: set[str] = set()
    for value in values:
        if value in (None, "", [], {}):
            continue
        marker = repr(value)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(deepcopy(value))
    return unique


def _as_list(value: Any) -> list[Any]:
    if value in (None, "", [], {}):
        return []
    return value if isinstance(value, list) else [value]


def _quality_key(record: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        -float(record.get("reuse_success_count", 0) or 0),
        -float(record.get("reuse_count", 0) or 0),
        -float(record.get("confidence", 0.0) or 0.0),
        str(record.get("memory_id", "")),
    )


def merge_experience_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge one compatible experience cluster without mutating its inputs."""

    if not records:
        raise ValueError("records must not be empty")
    identities = {experience_identity(record) for record in records}
    if "" in identities or len(identities) != 1:
        raise ValueError("experience records do not share a safe merge identity")

    canonical_source = min(records, key=_quality_key)
    merged = deepcopy(canonical_source)
    merged_ids = _unique_values(
        value
        for record in records
        for value in [record.get("memory_id"), *_as_list(record.get("merged_memory_ids"))]
        if value and value != merged.get("memory_id")
    )

    list_fields = (
        "tags",
        "trigger_conditions",
        "supporting_evidence",
        "supporting_facts",
        "evidence",
        "strategies",
        "experiences",
        "reuse_benefits",
        "failure_patterns",
        "source_cases",
    )
    for field in list_fields:
        values: list[Any] = []
        for record in records:
            values.extend(_as_list(record.get(field)))
            if field == "failure_patterns":
                values.extend(_as_list(record.get("failure_pattern")))
            if field == "source_cases":
                values.extend(_as_list(record.get("source_case")))
        combined = _unique_values(values)
        if combined:
            merged[field] = combined

    merged["reuse_count"] = sum(int(record.get("reuse_count", 0) or 0) for record in records)
    merged["reuse_success_count"] = sum(
        int(record.get("reuse_success_count", 0) or 0) for record in records
    )
    merged["confidence"] = max(float(record.get("confidence", 0.0) or 0.0) for record in records)
    merged["merge_count"] = (
        sum(int(record.get("merge_count", 0) or 0) for record in records) + len(records) - 1
    )
    merged["merged_memory_ids"] = merged_ids
    merged["deduplication_identity"] = next(iter(identities))
    created_times = [
        (parsed, str(record.get("created_at")))
        for record in records
        if (parsed := _parse_timestamp(record.get("created_at"))) is not None
    ]
    if created_times:
        merged["created_at"] = min(created_times, key=lambda item: item[0])[1]
    for timestamp_field in ("updated_at", "last_reused_at"):
        timestamp_values = [
            (parsed, str(record.get(timestamp_field)))
            for record in records
            if (parsed := _parse_timestamp(record.get(timestamp_field))) is not None
        ]
        if timestamp_values:
            merged[timestamp_field] = max(timestamp_values, key=lambda item: item[0])[1]
    merged.pop("hypergraph", None)
    return merged


def deduplicate_memory(memory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a deterministic, non-destructive search view with experience deduplication."""

    output: list[dict[str, Any]] = []
    cluster_positions: dict[str, int] = {}
    clusters: dict[str, list[dict[str, Any]]] = {}

    for record in memory:
        identity = experience_identity(record)
        if not identity:
            output.append(deepcopy(record))
            continue
        if identity not in clusters:
            cluster_positions[identity] = len(output)
            clusters[identity] = []
            output.append({})
        clusters[identity].append(record)

    for identity, records in clusters.items():
        output[cluster_positions[identity]] = (
            merge_experience_records(records) if len(records) > 1 else deepcopy(records[0])
        )
    return output


LIFECYCLE_RETENTION = {
    "generalized": 1.0,
    "validated": 0.9,
    "promoted-candidate": 0.75,
    "episodic": 0.6,
    "stored": 0.6,
    "needs-validation": 0.45,
    "deprecated": 0.0,
}


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def retention_profile(
    record: dict[str, Any],
    *,
    now: datetime | None = None,
    half_life_days: float | None = None,
) -> dict[str, Any]:
    """Compute an explainable retention score without mutating the record."""

    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    memory_type = str(record.get("memory_type", "evidence") or "evidence")
    if half_life_days is None:
        half_life_days = 180.0 if memory_type == "experience" else 90.0
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")

    timestamp = next(
        (
            parsed
            for parsed in (
                _parse_timestamp(record.get("last_reused_at")),
                _parse_timestamp(record.get("updated_at")),
                _parse_timestamp(record.get("created_at")),
            )
            if parsed is not None
        ),
        None,
    )
    age_days = max(0.0, (now - timestamp).total_seconds() / 86400.0) if timestamp else None
    recency_score = 0.5 ** (age_days / half_life_days) if age_days is not None else 1.0

    reuse_count = int(record.get("reuse_count", 0) or 0)
    reuse_success_count = int(record.get("reuse_success_count", 0) or 0)
    # A beta(1, 1) prior keeps untried memory neutral instead of treating it as failed.
    utility_score = (reuse_success_count + 1.0) / (reuse_count + 2.0)
    confidence_value = record.get("confidence", 0.5)
    confidence_score = max(0.0, min(1.0, float(0.5 if confidence_value is None else confidence_value)))
    stage = lifecycle_stage(record)
    lifecycle_score = LIFECYCLE_RETENTION.get(stage, 0.5)
    score = (
        0.25 * recency_score
        + 0.30 * utility_score
        + 0.25 * confidence_score
        + 0.20 * lifecycle_score
    )
    if stage in {"validated", "generalized"}:
        score = max(score, 0.65)

    return {
        "retention_score": round(max(0.0, min(1.0, score)), 4),
        "memory_age_days": round(age_days, 4) if age_days is not None else None,
        "recency_score": round(recency_score, 4),
        "utility_score": round(utility_score, 4),
        "retention_confidence_score": round(confidence_score, 4),
        "retention_lifecycle_score": round(lifecycle_score, 4),
        "retention_lifecycle_stage": stage,
        "reuse_success_rate": round(reuse_success_rate(record), 4),
    }


def apply_decay(
    memory: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Attach retention metadata to a copied search view."""

    decayed: list[dict[str, Any]] = []
    for record in memory:
        next_record = deepcopy(record)
        next_record.update(retention_profile(next_record, now=now))
        decayed.append(next_record)
    return decayed


def prune_memory(
    memory: list[dict[str, Any]],
    *,
    now: datetime | None = None,
    min_retention_score: float = 0.2,
    max_records: int | None = None,
    aggressive: bool = False,
) -> list[dict[str, Any]]:
    """Prune a copied search view with conservative, auditable rules."""

    if not 0.0 <= min_retention_score <= 1.0:
        raise ValueError("min_retention_score must be between 0 and 1")
    if max_records is not None and max_records < 0:
        raise ValueError("max_records must be non-negative")

    retained: list[dict[str, Any]] = []
    for record in apply_decay(memory, now=now):
        memory_type = str(record.get("memory_type", "evidence") or "evidence")
        stage = str(record.get("retention_lifecycle_stage", ""))
        if record.get("pruned") is True:
            continue
        if memory_type == "experience" and stage == "deprecated":
            continue
        if aggressive and float(record.get("retention_score", 0.0)) < min_retention_score:
            continue
        retained.append(record)

    if max_records is not None and len(retained) > max_records:
        selected_ids = {
            str(record.get("memory_id"))
            for record in sorted(
                retained,
                key=lambda item: (-float(item.get("retention_score", 0.0)), str(item.get("memory_id", ""))),
            )[:max_records]
        }
        retained = [record for record in retained if str(record.get("memory_id")) in selected_ids]
    return retained


def prepare_searchable_memory(memory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create the default non-destructive search view: deduplicate, decay, safely prune."""

    return prune_memory(deduplicate_memory(memory), aggressive=False)
