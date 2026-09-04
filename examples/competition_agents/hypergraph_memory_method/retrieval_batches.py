from __future__ import annotations

from copy import deepcopy
from typing import Any, Final, Literal

from .layers import EPISODIC_LAYER, EXPERIENCE_LAYER, KNOWLEDGE_LAYER


MEMORY_ROUTE_BATCH: Final = "memory_route"
KNOWLEDGE_FOLLOWUP_BATCH: Final = "knowledge_followup"
BatchDecision = Literal["run", "skip", "pending"]


def build_retrieval_batches(
    *,
    need_memory: bool,
    need_knowledge: bool,
) -> list[dict[str, Any]]:
    """Create a serializable two-batch retrieval contract.

    The first physical query searches the episodic and experience layers with
    separate quotas. The knowledge batch is a separate decision point so a
    later Planner implementation can run, rewrite, or skip it after batch one.
    """

    return [
        {
            "batch_id": MEMORY_ROUTE_BATCH,
            "sequence": 1,
            "layers": [EPISODIC_LAYER, EXPERIENCE_LAYER],
            "execution": "single_query_multi_layer",
            "decision_owner": "planner",
            "decision": "run" if need_memory else "skip",
            "status": "scheduled" if need_memory else "skipped",
            "query_count": 0,
            "hit_counts": {EPISODIC_LAYER: 0, EXPERIENCE_LAYER: 0},
        },
        {
            "batch_id": KNOWLEDGE_FOLLOWUP_BATCH,
            "sequence": 2,
            "layers": [KNOWLEDGE_LAYER],
            "execution": "conditional",
            "depends_on": MEMORY_ROUTE_BATCH,
            "decision_owner": "planner",
            "decision": "pending" if need_knowledge else "skip",
            "compatibility_default": "run" if need_knowledge else "skip",
            "status": "pending" if need_knowledge else "skipped",
            "query_count": 0,
            "hit_counts": {KNOWLEDGE_LAYER: 0},
        },
    ]


def resolve_followup_decision(
    retrieval_policy: dict[str, Any],
) -> tuple[bool, str, str]:
    """Resolve the knowledge-batch decision through the future Planner hook."""

    explicit = retrieval_policy.get("next_batch_decision")
    if isinstance(explicit, dict):
        value = str(explicit.get("decision") or "").strip().lower()
        if value in {"run", "skip"}:
            return (
                value == "run",
                "planner",
                str(explicit.get("reason") or "planner_followup_decision"),
            )
    elif isinstance(explicit, bool):
        return explicit, "planner", "planner_followup_decision"

    run = bool(retrieval_policy.get("need_knowledge", True))
    return (
        run,
        "compatibility_default",
        str(retrieval_policy.get("reason") or "existing_retrieval_policy"),
    )


def update_retrieval_batch(
    batches: list[dict[str, Any]],
    batch_id: str,
    *,
    status: str,
    query_count: int,
    hit_counts: dict[str, int],
    decision: BatchDecision | None = None,
    decision_source: str | None = None,
    decision_reason: str | None = None,
) -> list[dict[str, Any]]:
    """Return a new batch list with one execution result recorded."""

    updated = deepcopy(batches)
    for batch in updated:
        if batch.get("batch_id") != batch_id:
            continue
        batch["status"] = status
        batch["query_count"] = max(0, int(query_count))
        batch["hit_counts"] = {
            str(layer): max(0, int(count)) for layer, count in hit_counts.items()
        }
        if decision is not None:
            batch["decision"] = decision
        if decision_source:
            batch["decision_source"] = decision_source
        if decision_reason:
            batch["decision_reason"] = decision_reason
        break
    return updated
