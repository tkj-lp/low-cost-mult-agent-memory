from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Iterable

from .layers import memory_layer_for_record
from .text import stable_id

MEMORY_SCHEMA_VERSION = "1.0"
EVIDENCE_CHAIN_SCHEMA_VERSION = "1.0"
EVALUATION_ONLY_FIELDS = {
    "expected",
    "expected_answer",
    "expected_reuse",
    "gold",
    "reference_answer",
    "reference_code",
    "supporting_facts",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_list(value: Any) -> list[Any]:
    if value in (None, "", [], {}):
        return []
    return value if isinstance(value, list) else [value]


def _unique(values: Iterable[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for value in values:
        if value in (None, "", [], {}):
            continue
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if marker in seen:
            continue
        seen.add(marker)
        output.append(deepcopy(value))
    return output


def _confidence(value: Any, default: float = 0.5) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 4)
    except (TypeError, ValueError):
        return default


def _trace_knowledge_refs(trace: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for output in _as_list(trace.get("agent_outputs")):
        if not isinstance(output, dict) or output.get("agent") != "RetrieverAgent":
            continue
        refs.extend(
            str(value)
            for value in _as_list(output.get("context_hit_ids"))
            if str(value).strip()
        )
    return [str(value) for value in _unique(refs)]


def compact_trace(trace: dict[str, Any] | None) -> dict[str, Any]:
    """保留可重建来源的轻量轨迹，移除评测标签和递归记忆对象。"""

    if not isinstance(trace, dict):
        return {}

    compacted = deepcopy(trace)
    task_meta = compacted.get("task_meta")
    if isinstance(task_meta, dict):
        compacted["task_meta"] = {
            key: value
            for key, value in task_meta.items()
            if str(key) not in EVALUATION_ONLY_FIELDS
        }

    for key in ("memory_hits", "memory_context"):
        values = compacted.get(key)
        if not isinstance(values, list):
            continue
        compacted[key] = [
            {
                field: item.get(field)
                for field in (
                    "memory_id",
                    "memory_type",
                    "summary",
                    "source_agent",
                    "retrieval_score",
                    "evidence_chain_id",
                )
                if item.get(field) not in (None, "", [], {})
            }
            for item in values
            if isinstance(item, dict) and item.get("memory_id")
        ]

    latent_states = compacted.get("latent_states")
    if isinstance(latent_states, dict):
        compacted["latent_states"] = {
            str(state_id): {
                key: value
                for key, value in state.items()
                if key
                in {
                    "state_id",
                    "content_type",
                    "producer",
                    "consumers",
                    "topic",
                    "summary",
                    "embedding_provider",
                    "embedding_model",
                    "shape",
                    "scope",
                    "checksum",
                }
            }
            for state_id, state in latent_states.items()
            if isinstance(state, dict)
        }

    execution_result = compacted.get("execution_result")
    if isinstance(execution_result, dict):
        compacted["execution_result"] = {
            key: value
            for key, value in execution_result.items()
            if key not in EVALUATION_ONLY_FIELDS
            and key not in {"memory_hit_details", "memory_context", "experience_rules"}
        }
    return compacted


def _fact_parts(value: Any) -> tuple[str, str]:
    if isinstance(value, (list, tuple)) and value:
        return str(value[0]).strip(), str(value[1]).strip() if len(value) > 1 else ""
    if isinstance(value, dict):
        return (
            str(value.get("title") or value.get("source") or "").strip(),
            str(
                value.get("sentence") or value.get("text") or value.get("content") or ""
            ).strip(),
        )
    return "", str(value or "").strip()


def _source(
    *,
    source_type: str,
    source_id: str,
    source_agent: str,
    task_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_type": source_type,
        "source_id": source_id,
        "source_agent": source_agent,
        "task_id": str(task_meta.get("task_id") or ""),
        "group_id": str(task_meta.get("group_id") or ""),
        "turn_id": str(task_meta.get("turn_id") or ""),
        "dataset_type": str(task_meta.get("dataset_type") or ""),
    }


def build_evidence_chain(record: dict[str, Any]) -> dict[str, Any]:
    """从兼容字段构造有序、带来源且可校验的答案证据链。"""

    existing = record.get("evidence_chain")
    if isinstance(existing, dict) and existing.get("steps"):
        chain = deepcopy(existing)
        validation = validate_evidence_chain(chain)
        chain.update(validation)
        return chain

    trace = compact_trace(
        record.get("trace") if isinstance(record.get("trace"), dict) else {}
    )
    task_meta = (
        trace.get("task_meta") if isinstance(trace.get("task_meta"), dict) else {}
    )
    memory_id = str(
        record.get("memory_id") or f"mem-{stable_id(str(record), length=12)}"
    )
    source_agent = str(record.get("source_agent") or "unknown")
    question = str(task_meta.get("question") or record.get("task_topic") or "").strip()
    steps: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []

    facts = _unique(
        [
            *_as_list(record.get("supporting_facts")),
            *_as_list(task_meta.get("supporting_facts")),
        ]
    )
    for index, fact in enumerate(facts, start=1):
        title, sentence = _fact_parts(fact)
        text = sentence or title
        if not text:
            continue
        step_id = f"{memory_id}:evidence:{len(steps) + 1}"
        source_id = title or f"supporting-fact-{index}"
        steps.append(
            {
                "step_id": step_id,
                "position": len(steps) + 1,
                "kind": "evidence",
                "text": text,
                "title": title,
                "source": _source(
                    source_type="dataset_supporting_fact",
                    source_id=source_id,
                    source_agent=source_agent,
                    task_meta=task_meta,
                ),
                "confidence": 1.0,
                "depends_on": [],
                "memory_refs": [],
            }
        )

    evidence_values = _unique(
        [
            *_as_list(record.get("supporting_evidence")),
            *_as_list(record.get("evidence")),
            *_as_list(trace.get("evidence")),
        ]
    )
    known_evidence_text = {str(step.get("text") or "").strip() for step in steps}
    memory_refs = [
        str(item.get("memory_id"))
        for item in _as_list(trace.get("memory_hits"))
        if isinstance(item, dict) and item.get("memory_id")
    ]
    for index, value in enumerate(evidence_values, start=1):
        text = str(value or "").strip()
        if not text or text in known_evidence_text:
            continue
        step_id = f"{memory_id}:evidence:{len(steps) + 1}"
        steps.append(
            {
                "step_id": step_id,
                "position": len(steps) + 1,
                "kind": "evidence",
                "text": text,
                "title": "",
                "source": _source(
                    source_type="agent_retrieval",
                    source_id=f"retrieved-evidence-{index}",
                    source_agent="retriever",
                    task_meta=task_meta,
                ),
                "confidence": 0.75,
                "depends_on": [],
                "memory_refs": list(memory_refs),
            }
        )

    execution_result = (
        trace.get("execution_result")
        if isinstance(trace.get("execution_result"), dict)
        else {}
    )
    reasoning_summary = str(execution_result.get("reasoning_summary") or "").strip()
    evidence_ids = [str(step["step_id"]) for step in steps]
    dependency_ids = evidence_ids
    if reasoning_summary:
        reasoning_id = f"{memory_id}:claim:intermediate"
        steps.append(
            {
                "step_id": reasoning_id,
                "position": len(steps) + 1,
                "kind": "intermediate_claim",
                "text": reasoning_summary,
                "title": "",
                "source": _source(
                    source_type="agent_inference",
                    source_id="reasoning-summary",
                    source_agent="reasoner",
                    task_meta=task_meta,
                ),
                "confidence": _confidence(record.get("confidence"), 0.7),
                "depends_on": evidence_ids,
                "memory_refs": list(memory_refs),
            }
        )
        for evidence_id in evidence_ids:
            links.append(
                {"from": evidence_id, "to": reasoning_id, "relation": "supports"}
            )
        dependency_ids = [reasoning_id]

    final_text = str(record.get("answer") or record.get("summary") or "").strip()
    final_claim_id = ""
    if final_text:
        final_claim_id = f"{memory_id}:claim:final"
        steps.append(
            {
                "step_id": final_claim_id,
                "position": len(steps) + 1,
                "kind": "final_claim",
                "text": final_text,
                "title": "",
                "source": _source(
                    source_type="agent_conclusion",
                    source_id="final-answer",
                    source_agent=source_agent,
                    task_meta=task_meta,
                ),
                "confidence": _confidence(record.get("confidence"), 0.7),
                "depends_on": dependency_ids,
                "memory_refs": list(memory_refs),
            }
        )
        for dependency_id in dependency_ids:
            links.append(
                {"from": dependency_id, "to": final_claim_id, "relation": "supports"}
            )

    for raw_conflict in _as_list(record.get("contradictions")):
        conflict_text = str(raw_conflict or "").strip()
        if not conflict_text or not final_claim_id:
            continue
        conflict_id = f"{memory_id}:claim:conflict:{stable_id(conflict_text, length=8)}"
        steps.append(
            {
                "step_id": conflict_id,
                "position": len(steps) + 1,
                "kind": "contradicting_claim",
                "text": conflict_text,
                "title": "",
                "source": _source(
                    source_type="memory_conflict",
                    source_id=conflict_id,
                    source_agent=source_agent,
                    task_meta=task_meta,
                ),
                "confidence": 0.5,
                "depends_on": [],
                "memory_refs": [],
            }
        )
        links.append(
            {"from": conflict_id, "to": final_claim_id, "relation": "contradicts"}
        )

    chain = {
        "schema_version": EVIDENCE_CHAIN_SCHEMA_VERSION,
        "chain_id": f"chain-{stable_id(memory_id + '|' + question + '|' + final_text, length=16)}",
        "memory_id": memory_id,
        "question": question,
        "final_claim_id": final_claim_id,
        "steps": steps,
        "links": links,
        "source_memory_ids": sorted(set(memory_refs)),
    }
    chain.update(validate_evidence_chain(chain))
    return chain


def validate_evidence_chain(chain: dict[str, Any]) -> dict[str, Any]:
    """校验证据链依赖、来源、顺序和结论支撑完整性。"""

    steps = [step for step in _as_list(chain.get("steps")) if isinstance(step, dict)]
    step_ids = [str(step.get("step_id") or "") for step in steps]
    known = {step_id for step_id in step_ids if step_id}
    issues: list[dict[str, str]] = []

    if len(known) != len(step_ids):
        issues.append(
            {
                "code": "duplicate_or_empty_step_id",
                "message": "证据链步骤编号为空或重复。",
            }
        )
    for step in steps:
        step_id = str(step.get("step_id") or "")
        if not str(step.get("text") or "").strip():
            issues.append(
                {"code": "empty_step_text", "message": f"步骤 {step_id} 缺少内容。"}
            )
        source = step.get("source")
        if (
            not isinstance(source, dict)
            or not source.get("source_type")
            or not source.get("source_id")
        ):
            issues.append(
                {"code": "missing_provenance", "message": f"步骤 {step_id} 缺少来源。"}
            )
        for dependency in _as_list(step.get("depends_on")):
            if str(dependency) not in known:
                issues.append(
                    {
                        "code": "missing_dependency",
                        "message": f"步骤 {step_id} 依赖不存在的步骤 {dependency}。",
                    }
                )

    final_claim_id = str(chain.get("final_claim_id") or "")
    final_step = next(
        (step for step in steps if str(step.get("step_id")) == final_claim_id), None
    )
    if not final_step:
        issues.append(
            {"code": "missing_final_claim", "message": "证据链缺少最终结论。"}
        )
    elif not _as_list(final_step.get("depends_on")):
        issues.append(
            {"code": "unsupported_final_claim", "message": "最终结论没有证据依赖。"}
        )

    graph = {
        str(step.get("step_id")): [
            str(value) for value in _as_list(step.get("depends_on"))
        ]
        for step in steps
        if step.get("step_id")
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def has_cycle(step_id: str) -> bool:
        if step_id in visiting:
            return True
        if step_id in visited:
            return False
        visiting.add(step_id)
        if any(
            has_cycle(dependency)
            for dependency in graph.get(step_id, [])
            if dependency in graph
        ):
            return True
        visiting.remove(step_id)
        visited.add(step_id)
        return False

    if any(has_cycle(step_id) for step_id in graph):
        issues.append({"code": "cyclic_dependency", "message": "证据链存在循环依赖。"})

    evidence_count = sum(1 for step in steps if step.get("kind") == "evidence")
    supported_claim_count = sum(
        1
        for step in steps
        if step.get("kind") in {"intermediate_claim", "final_claim"}
        and _as_list(step.get("depends_on"))
    )
    completeness = 0.0
    if final_step:
        completeness += 0.35
    if evidence_count:
        completeness += 0.35
    if supported_claim_count:
        completeness += 0.2
    if steps and all(
        isinstance(step.get("source"), dict) and step["source"].get("source_id")
        for step in steps
    ):
        completeness += 0.1
    completeness = max(0.0, min(1.0, completeness - min(0.5, len(issues) * 0.1)))
    return {
        "valid": not issues,
        "issues": issues,
        "completeness": round(completeness, 4),
        "evidence_count": evidence_count,
        "claim_count": sum(1 for step in steps if "claim" in str(step.get("kind"))),
    }


def infer_memory_kinds(record: dict[str, Any]) -> list[str]:
    kinds: set[str] = set()
    memory_type = str(record.get("memory_type") or "evidence")
    if memory_type == "experience":
        kinds.add("experience")
    else:
        kinds.update({"fact", "evidence", "conclusion"})
    trace = record.get("trace") if isinstance(record.get("trace"), dict) else {}
    if trace.get("messages") or trace.get("agent_outputs"):
        kinds.add("trajectory")
    if trace.get("execution_result"):
        kinds.add("execution_result")
    if record.get("problem_pattern") or record.get("solution_pattern"):
        kinds.add("strategy")
    return sorted(kinds)


def canonical_search_document(record: dict[str, Any]) -> str:
    chain = (
        record.get("evidence_chain")
        if isinstance(record.get("evidence_chain"), dict)
        else {}
    )
    chain_text = " ".join(
        str(step.get("text") or "")
        for step in _as_list(chain.get("steps"))
        if isinstance(step, dict)
    )
    parts = [
        memory_layer_for_record(record),
        record.get("task_topic", ""),
        record.get("summary", ""),
        record.get("group_id", ""),
        record.get("turn_id", ""),
        " ".join(str(value) for value in _as_list(record.get("artifact_types"))),
        " ".join(str(tag) for tag in _as_list(record.get("tags"))),
        record.get("problem_pattern", ""),
        record.get("solution_pattern", ""),
        record.get("failure_pattern", ""),
        " ".join(str(value) for value in _as_list(record.get("trigger_conditions"))),
        chain_text,
    ]
    return "\n".join(str(part).strip() for part in parts if str(part).strip())


def normalize_memory_record(
    record: dict[str, Any], *, compact: bool = True
) -> dict[str, Any]:
    """补齐统一记忆契约，同时保持现有字段和主语义不变。"""

    normalized = deepcopy(record)
    created_at = str(normalized.get("created_at") or _now_iso())
    memory_id = str(
        normalized.get("memory_id")
        or f"mem-{stable_id(str(normalized.get('task_topic', '')) + '|' + str(normalized.get('summary', '')), length=16)}"
    )
    normalized.update(
        {
            "schema_version": str(
                normalized.get("schema_version") or MEMORY_SCHEMA_VERSION
            ),
            "memory_id": memory_id,
            "memory_type": str(normalized.get("memory_type") or "evidence"),
            "source_agent": str(normalized.get("source_agent") or "unknown"),
            "created_at": created_at,
            "updated_at": str(normalized.get("updated_at") or created_at),
            "task_topic": str(normalized.get("task_topic") or ""),
            "summary": str(normalized.get("summary") or ""),
            "tags": sorted(
                {
                    str(tag).strip().lower().replace("_", "-")
                    for tag in _as_list(normalized.get("tags"))
                    if str(tag).strip()
                }
            ),
            "confidence": _confidence(normalized.get("confidence"), 0.5),
            "archive_scope": str(normalized.get("archive_scope") or "task"),
            "lifecycle_stage": str(normalized.get("lifecycle_stage") or "episodic"),
            "record_version": int(normalized.get("record_version", 1) or 1),
            "retrieval_count": int(normalized.get("retrieval_count", 0) or 0),
            "reuse_count": int(normalized.get("reuse_count", 0) or 0),
            "reuse_success_count": int(normalized.get("reuse_success_count", 0) or 0),
            "reuse_failure_count": int(normalized.get("reuse_failure_count", 0) or 0),
            "entities": _as_list(normalized.get("entities")),
            "supporting_facts": _as_list(normalized.get("supporting_facts")),
            "supporting_evidence": _as_list(normalized.get("supporting_evidence")),
            "evidence": _as_list(normalized.get("evidence")),
            "state_refs": _as_list(normalized.get("state_refs")),
        }
    )
    normalized["memory_layer"] = memory_layer_for_record(normalized)
    trace = normalized.get("trace") if isinstance(normalized.get("trace"), dict) else {}
    task_meta = (
        trace.get("task_meta") if isinstance(trace.get("task_meta"), dict) else {}
    )
    artifact_types = _as_list(normalized.get("artifact_types"))
    if not artifact_types:
        artifact_types = ["experience", "strategy"] if normalized["memory_type"] == "experience" else ["answer", "evidence"]
        if normalized.get("supporting_facts") or normalized.get("supporting_evidence") or task_meta.get("supporting_facts"):
            artifact_types.append("supporting-facts")
    normalized["artifact_types"] = sorted({str(value).strip().lower().replace("_", "-") for value in artifact_types if str(value).strip()})
    normalized["task_id"] = str(normalized.get("task_id") or task_meta.get("task_id") or "")
    normalized["group_id"] = str(normalized.get("group_id") or task_meta.get("group_id") or task_meta.get("chain_id") or "")
    normalized["turn_id"] = str(normalized.get("turn_id") or task_meta.get("turn_id") or "")
    normalized["dataset_type"] = str(normalized.get("dataset_type") or task_meta.get("dataset_type") or "")
    normalized["task_type"] = str(normalized.get("task_type") or task_meta.get("task_type") or "")
    normalized["knowledge_refs"] = [
        str(value)
        for value in _unique(
            [
                *_as_list(normalized.get("knowledge_refs")),
                *_trace_knowledge_refs(trace),
            ]
        )
        if str(value).strip()
    ]
    normalized["source_memory_ids"] = [
        str(value)
        for value in _unique(_as_list(normalized.get("source_memory_ids")))
        if str(value).strip()
    ]
    normalized["global_memory_id"] = str(
        normalized.get("global_memory_id")
        or "gmem-"
        + stable_id(
            "|".join(
                [
                    memory_id,
                    str(task_meta.get("task_id") or ""),
                    str(task_meta.get("group_id") or ""),
                    str(task_meta.get("turn_id") or ""),
                    created_at,
                    normalized["source_agent"],
                ]
            ),
            length=20,
        )
    )
    if compact and isinstance(normalized.get("trace"), dict):
        normalized["trace"] = compact_trace(normalized["trace"])
    normalized["memory_kinds"] = infer_memory_kinds(normalized)
    normalized["evidence_chain"] = build_evidence_chain(normalized)
    normalized["evidence_chain_id"] = normalized["evidence_chain"].get("chain_id", "")
    normalized["evidence_chain_valid"] = bool(normalized["evidence_chain"].get("valid"))
    normalized["evidence_chain_completeness"] = float(
        normalized["evidence_chain"].get("completeness", 0.0) or 0.0
    )
    normalized["search_document"] = canonical_search_document(normalized)
    return normalized


def compact_reuse_projection(
    record: dict[str, Any], *, max_steps: int = 6
) -> dict[str, Any]:
    """返回供后续智能体直接复用的紧凑记忆与证据链投影。"""

    normalized = normalize_memory_record(record)
    chain = normalized["evidence_chain"]
    steps = [
        {
            "step_id": step.get("step_id"),
            "kind": step.get("kind"),
            "text": step.get("text"),
            "source": step.get("source"),
            "depends_on": step.get("depends_on", []),
            "memory_refs": step.get("memory_refs", []),
        }
        for step in chain.get("steps", [])[:max_steps]
        if isinstance(step, dict)
    ]
    return {
        "memory_id": normalized["memory_id"],
        "global_memory_id": normalized["global_memory_id"],
        "memory_type": normalized["memory_type"],
        "memory_layer": normalized["memory_layer"],
        "memory_kinds": normalized["memory_kinds"],
        "summary": normalized["summary"],
        "tags": normalized["tags"],
        "evidence_chain_id": chain.get("chain_id"),
        "evidence_chain_valid": chain.get("valid", False),
        "evidence_chain_completeness": chain.get("completeness", 0.0),
        "evidence_chain_steps": steps,
        "source_memory_ids": chain.get("source_memory_ids", []),
        "knowledge_refs": normalized.get("knowledge_refs", []),
        "abstracted_from_memory_ids": normalized.get("source_memory_ids", []),
    }
