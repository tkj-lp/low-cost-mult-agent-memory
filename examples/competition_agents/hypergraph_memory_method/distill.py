from __future__ import annotations

from typing import Any

from .text import normalize_text


def _clean_label(value: Any) -> str:
    return normalize_text(str(value)).replace(" ", "-")


def _artifact_from_expected_reuse(value: str) -> str:
    _, _, artifact = value.partition(".")
    return _clean_label(artifact or value)


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
    expected_reuse = task_meta.get("expected_reuse", [])
    if memory_hits and expected_reuse:
        return "cross-turn-reuse"
    if task_meta.get("group_id"):
        return "group"
    if task_meta.get("dataset_type") == "continuous":
        return "continuous-task"
    return "task"


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
    expected_reuse = [str(value) for value in task_meta.get("expected_reuse", []) if value]
    state_refs = trace.get("state_refs", {}) if isinstance(trace.get("state_refs"), dict) else {}
    evidence = trace.get("evidence", [])
    supporting_facts = task_meta.get("supporting_facts", [])
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
    if expected_reuse:
        strategies.add("expected-reuse-planning")

    for value in expected_reuse:
        artifact = _artifact_from_expected_reuse(value)
        experiences.add(f"reuse-{artifact}")
        reuse_benefits.add(f"expected-reuse:{artifact}")
    if memory_hits:
        experiences.add("memory-grounded-reasoning")
        reuse_benefits.add(f"memory-hit-count:{len(memory_hits)}")
    else:
        experiences.add("cold-start-episode")
    if supporting_facts:
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

    lifecycle_stage = "promoted-candidate" if memory_hits or expected_reuse else "episodic"
    if execution_result.get("limitations"):
        lifecycle_stage = "needs-validation"

    return {
        "archive_scope": archive_scope(trace),
        "lifecycle_stage": lifecycle_stage,
        "strategies": sorted(strategies),
        "experiences": sorted(experiences),
        "reuse_benefits": sorted(reuse_benefits),
    }
