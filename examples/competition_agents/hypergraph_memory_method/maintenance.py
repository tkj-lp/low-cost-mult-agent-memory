from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Callable

from .contracts import BatchPolicy, MemoryEncoder
from .evolution import merge_experience_records
from .indexed_retrieval import checked_vector
from .text import indexed_tokens


def _canonical(value: Any) -> str:
    return " ".join(
        str(value or "").casefold().replace("_", " ").replace("-", " ").split()
    )


def _conditions(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _canonical(record.get("problem_pattern")),
        tuple(
            sorted(_canonical(value) for value in record.get("trigger_conditions", []))
        ),
        _canonical(record.get("failure_pattern")),
        _canonical(record.get("task_type")),
    )


def _qualifiers(text: str) -> set[str]:
    words = {"no", "not", "never", "without", "before", "after", "min", "max"}
    return {
        token for token in indexed_tokens(text) if token.isdigit() or token in words
    }


def review_status(record: dict[str, Any], policy: BatchPolicy) -> str:
    successes = int(record.get("verified_success_count", 0))
    failures = int(record.get("verified_failure_count", 0))
    if failures >= policy.reject_after_failures:
        return "rejected"
    if failures or record.get("unresolved_failure"):
        return "pending"
    if successes >= policy.min_verified_successes:
        return "active"
    return "candidate"


def plan_maintenance(
    records: list[dict[str, Any]],
    *,
    batch_id: str,
    policy: BatchPolicy,
    encoder: MemoryEncoder | None = None,
    merge_verifier: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pure plan: no task scheduling, answer judging, database writes or clocks."""

    retained: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "batch_id": batch_id,
        "cleared_ids": [],
        "promoted_ids": [],
        "pending_ids": [],
        "rejected_ids": [],
        "superseded_ids": [],
        "capacity_pruned_ids": [],
        "merged": [],
        "merge_suggestions": [],
    }
    for raw in records:
        record = deepcopy(raw)
        memory_id = record["global_memory_id"]
        layer = record["memory_layer"]
        if layer != "experience":
            if record["batch_id"] == batch_id and (
                layer == "knowledge" or policy.clear_episodes
            ):
                report["cleared_ids"].append(memory_id)
                continue
            retained.append(record)
            continue
        if record.get("review_status") in {"superseded", "archived"}:
            if record.get("review_status") == "superseded":
                report["superseded_ids"].append(memory_id)
            record["batch_id"] = ""
            record["lifecycle_stage"] = "deprecated"
            record["archive_scope"] = "review-only"
            retained.append(record)
            continue
        status = review_status(record, policy)
        # Archived sources preserve provenance after batch knowledge is cleared.
        if status == "active" and not record.get("source_memory_ids"):
            status = "pending"
        if status == "active":
            if (
                record["batch_id"] == batch_id
                or record.get("review_status") != "active"
            ):
                report["promoted_ids"].append(memory_id)
            record["lifecycle_stage"] = "validated"
        elif status == "rejected":
            report["rejected_ids"].append(memory_id)
            record["lifecycle_stage"] = "deprecated"
        else:
            report["pending_ids"].append(memory_id)
            record["lifecycle_stage"] = "needs-validation"
        record["review_status"] = status
        record["batch_id"] = ""
        record["archive_scope"] = "long-term" if status == "active" else "review-only"
        retained.append(record)

    active = [
        item
        for item in retained
        if item["memory_layer"] == "experience" and item["review_status"] == "active"
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in active:
        groups.setdefault(_conditions(record), []).append(record)
    aliases: dict[str, str] = {}
    replacements: dict[str, dict[str, Any]] = {}
    for conditions, group in groups.items():
        group.sort(key=lambda item: (item["created_at"], item["global_memory_id"]))
        if not conditions[0]:
            continue
        vectors: list[list[float]] = []
        # Maintenance can suggest semantic duplicates, but similarity alone
        # cannot establish that two procedures have the same effect.
        if encoder is not None and 1 < len(group) <= 128:
            vectors = [
                checked_vector(value)
                for value in encoder.encode(
                    [str(item.get("solution_pattern", "")) for item in group]
                )
            ]
            if (
                len(vectors) != len(group)
                or len({len(value) for value in vectors}) != 1
            ):
                raise ValueError("merge encoder returned inconsistent vectors")
        for i, left in enumerate(group):
            left_id = left["global_memory_id"]
            if left_id in aliases:
                continue
            cluster = [left]
            for j in range(i + 1, len(group)):
                right = group[j]
                right_id = right["global_memory_id"]
                if right_id in aliases:
                    continue
                left_solution, right_solution = (
                    _canonical(item.get("solution_pattern")) for item in (left, right)
                )
                equivalent = bool(left_solution) and left_solution == right_solution
                if not equivalent and vectors:
                    a, b = vectors[i], vectors[j]
                    cosine = sum(x * y for x, y in zip(a, b)) / math.sqrt(
                        sum(x * x for x in a) * sum(y * y for y in b)
                    )
                    if cosine >= policy.merge_similarity and _qualifiers(
                        left_solution
                    ) == _qualifiers(right_solution):
                        accepted = (
                            merge_verifier is not None
                            and merge_verifier(deepcopy(left), deepcopy(right)) is True
                        )
                        if accepted:
                            equivalent = True
                        else:
                            report["merge_suggestions"].append(
                                {
                                    "left": left_id,
                                    "right": right_id,
                                    "similarity": round(cosine, 6),
                                    "requires_equivalence_review": True,
                                }
                            )
                if equivalent:
                    cluster.append(right)
                    aliases[right_id] = left_id
            if len(cluster) == 1:
                continue
            # The legacy merge helper requires literal identities. The verified
            # canonical procedure is retained; original variants stay in audit.
            compatible = [deepcopy(item) for item in cluster]
            for item in compatible:
                item["problem_pattern"] = left["problem_pattern"]
                item["solution_pattern"] = left["solution_pattern"]
            merged = merge_experience_records(compatible)
            for field in (
                "memory_id",
                "global_memory_id",
                "created_at",
                "problem_pattern",
                "solution_pattern",
            ):
                merged[field] = left[field]
            for field in ("source_memory_ids", "knowledge_refs"):
                merged[field] = sorted(
                    {str(value) for item in cluster for value in item.get(field, [])}
                )
            checks = {
                str(check["id"]): check
                for item in cluster
                for check in item.get("verification_history", [])
            }
            merged["verification_history"] = list(checks.values())
            merged["verified_success_count"] = sum(
                check.get("success") is True for check in checks.values()
            )
            merged["verified_failure_count"] = sum(
                check.get("failure") is True for check in checks.values()
            )
            merged["merged_variants"] = [
                {
                    "global_memory_id": item["global_memory_id"],
                    "solution_pattern": item["solution_pattern"],
                }
                for item in cluster
            ]
            merged["record_version"] = (
                max(int(item["record_version"]) for item in cluster) + 1
            )
            merged["review_status"] = review_status(merged, policy)
            if merged["review_status"] != "active":
                merged["lifecycle_stage"] = "needs-validation"
                merged["archive_scope"] = "review-only"
            replacements[left_id] = merged
            report["merged"].append(
                {
                    "canonical_id": left_id,
                    "merged_ids": [item["global_memory_id"] for item in cluster[1:]],
                }
            )

    output = [
        replacements.get(item["global_memory_id"], item)
        for item in retained
        if item["global_memory_id"] not in aliases
    ]
    for item in output:
        for field in ("source_memory_ids", "knowledge_refs"):
            item[field] = sorted(
                {aliases.get(value, value) for value in item.get(field, [])}
            )
    active = [
        item
        for item in output
        if item["memory_layer"] == "experience" and item["review_status"] == "active"
    ]
    active.sort(
        key=lambda item: (
            -int(item.get("verified_success_count", 0)),
            -int(item.get("reuse_success_count", 0)),
            item["global_memory_id"],
        )
    )
    for item in active[policy.max_active_experiences :]:
        item["review_status"] = "archived"
        item["archive_scope"] = "review-only"
        report["capacity_pruned_ids"].append(item["global_memory_id"])
    report["active_experience_count"] = sum(
        item.get("review_status") == "active" and item["memory_layer"] == "experience"
        for item in output
    )
    return output, report
