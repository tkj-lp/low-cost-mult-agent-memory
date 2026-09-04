from __future__ import annotations

from typing import Any

from .text import normalize_text


def _clean_label(value: Any) -> str:
    return normalize_text(str(value)).replace(" ", "-")


def _message_payload(message: dict[str, Any]) -> dict[str, Any]:
    payload = message.get("payload", {})
    return payload if isinstance(payload, dict) else {}


def _payload_refs(message: dict[str, Any], short_key: str, long_key: str) -> list[str]:
    payload = _message_payload(message)
    refs = payload.get(short_key) or payload.get(long_key) or []
    if not isinstance(refs, list):
        refs = [refs]
    return [str(ref) for ref in refs if ref]


def archive_scope(trace: dict[str, Any]) -> str:
    """Classify where a memory should be reused."""

    task_meta = trace.get("task_meta", {}) if isinstance(trace.get("task_meta"), dict) else {}
    memory_hits = trace.get("memory_hits", [])
    if memory_hits and task_meta.get("group_id"):
        return "cross-turn-reuse"
    if task_meta.get("group_id"):
        return "group"
    if task_meta.get("dataset_type") == "continuous":
        return "continuous-task"
    return "task"


def reuse_success_rate(record: dict[str, Any]) -> float:
    """Return successful reuse ratio for a memory record."""

    reuse_count = int(record.get("reuse_count", 0) or 0)
    if reuse_count <= 0:
        return 0.0
    return max(0.0, min(1.0, float(record.get("reuse_success_count", 0) or 0) / reuse_count))


def lifecycle_stage(record: dict[str, Any], trace: dict[str, Any] | None = None) -> str:
    """Infer the current lifecycle stage of a reusable memory.

    The thresholds are intentionally conservative so one lucky reuse cannot turn
    an episodic memory into a long-term rule.
    """

    reuse_count = int(record.get("reuse_count", 0) or 0)
    reuse_success_count = int(record.get("reuse_success_count", 0) or 0)
    success_rate = reuse_success_rate(record)
    stored_stage = str(record.get("lifecycle_stage") or "").strip()

    if reuse_count >= 5 and success_rate < 0.2:
        return "deprecated"
    if reuse_success_count >= 4 and success_rate >= 0.8:
        return "generalized"
    if reuse_success_count >= 2 and success_rate >= 0.67:
        return "validated"
    if reuse_success_count >= 1 and success_rate >= 0.5:
        return "promoted-candidate"
    if reuse_count:
        return "needs-validation"
    if stored_stage in {"validated", "generalized", "deprecated", "needs-validation"}:
        return stored_stage

    trace = trace if isinstance(trace, dict) else record.get("trace")
    execution_result = trace.get("execution_result", {}) if isinstance(trace, dict) and isinstance(trace.get("execution_result"), dict) else {}
    memory_hits = trace.get("memory_hits", []) if isinstance(trace, dict) else []

    if execution_result.get("limitations"):
        return "needs-validation"
    if memory_hits:
        return "promoted-candidate"
    return stored_stage or "episodic"


def distill_experience(record: dict[str, Any]) -> dict[str, Any]:
    """Distill an episodic trace into reusable strategy and experience labels."""

    trace = record.get("trace")
    if not isinstance(trace, dict):
        return {
            "archive_scope": "task",
            "lifecycle_stage": "episodic",
            "strategies": [],
            "experiences": [],
            "reuse_benefits": [],
        }

    task_meta = trace.get("task_meta", {}) if isinstance(trace.get("task_meta"), dict) else {}
    messages = [message for message in trace.get("messages", []) if isinstance(message, dict)]
    memory_hits = [hit for hit in trace.get("memory_hits", []) if isinstance(hit, dict)]
    state_refs = trace.get("state_refs", {}) if isinstance(trace.get("state_refs"), dict) else {}
    evidence = trace.get("evidence", [])
    supporting_facts = record.get("supporting_facts", []) or record.get(
        "supporting_evidence", []
    )
    metrics = trace.get("metrics", {}) if isinstance(trace.get("metrics"), dict) else {}
    execution_result = trace.get("execution_result", {}) if isinstance(trace.get("execution_result"), dict) else {}

    strategies: set[str] = set()
    experiences: set[str] = set()
    reuse_benefits: set[str] = set()

    for message in messages:
        sender = _clean_label(message.get("from", "unknown"))
        receiver = _clean_label(message.get("to", "unknown"))
        action = _clean_label(message.get("action", "unknown"))
        strategies.add(f"handoff:{sender}->{receiver}:{action}")
        if _payload_refs(message, "s", "state_refs"):
            strategies.add("state-ref-passing")
        if _payload_refs(message, "m", "memory_refs"):
            strategies.add("memory-ref-passing")

    if state_refs:
        strategies.add("latent-state-exchange")
    if evidence or supporting_facts:
        strategies.add("evidence-grounding")
    if trace.get("cognitive_gaps"):
        strategies.add("gap-aware-coordination")
    if memory_hits:
        experiences.add("memory-grounded-reasoning")
        reuse_benefits.add(f"memory-hit-count:{len(memory_hits)}")
    else:
        experiences.add("cold-start-episode")
    if evidence or supporting_facts:
        experiences.add("supporting-fact-chain")
    if "answer" in state_refs or execution_result.get("answer_ref"):
        experiences.add("answer-state-captured")
    if "evidence" in state_refs or execution_result.get("evidence_ref"):
        experiences.add("evidence-state-captured")

    if metrics.get("latent_transfer_count"):
        reuse_benefits.add("latent-transfer-accounted")
    if metrics.get("text_chars"):
        reuse_benefits.add("communication-cost-accounted")
    retrieval_mode = execution_result.get("retrieval_mode")
    if retrieval_mode:
        reuse_benefits.add(f"retrieval-mode:{_clean_label(retrieval_mode)}")

    return {
        "archive_scope": archive_scope(trace),
        "lifecycle_stage": lifecycle_stage(record, trace),
        "experience_success_rate": round(reuse_success_rate(record), 4),
        "strategies": sorted(strategies),
        "experiences": sorted(experiences),
        "reuse_benefits": sorted(reuse_benefits),
    }
