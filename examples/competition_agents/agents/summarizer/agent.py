from __future__ import annotations

import json
import os
import time

from competition_agents.llm import get_llm_client
from competition_agents.memory import add_memory, increment_reuse_counts, refine_memory
from competition_agents.protocol import make_cognitive_gap
from competition_agents.state import CompetitionState


def _summarizer_task_text(state: CompetitionState) -> str:
    """在 hypergraph 模式下只把问题文本传给 Summarizer，不直接暴露原始上下文。"""

    task_meta = state.get("task_meta", {})
    if str(task_meta.get("context_source") or "") == "hypergraph":
        question = str(task_meta.get("question") or "").strip()
        if question:
            return f"Question: {question}"
    return state["task"]


def _direct_qa_summary(state: CompetitionState, execution_result: dict) -> str:
    mode = os.environ.get("COMPETITION_SUMMARIZER_MODE", "").strip().lower()
    if mode in {"llm", "model"}:
        return ""
    task_meta = state.get("task_meta", {})
    dataset_type = str(task_meta.get("dataset_type") or "")
    if dataset_type not in {"continuous", "multihop", "codeact"}:
        return ""
    provider = (
        os.environ.get("LLM_SUMMARIZER_PROVIDER")
        or os.environ.get("LLM_PROVIDER")
        or ""
    ).strip().lower()
    if mode not in {"direct", "deterministic"} and provider != "local_hf":
        return ""
    if dataset_type == "codeact":
        result = execution_result.get("result")
        if result is not None:
            return json.dumps(result, ensure_ascii=False, sort_keys=True)
        return str(execution_result.get("sandbox_status") or "").strip()
    answer = str(execution_result.get("answer_candidate") or "").strip()
    return answer


def _locate_latest_experience_write(
    memory: list[dict],
    *,
    refined: dict,
) -> dict:
    """在 merge 场景下定位本轮真正写入或更新的 experience record。"""

    problem_pattern = str(refined.get("problem_pattern") or "")
    solution_pattern = str(refined.get("solution_pattern") or "")
    failure_pattern = str(refined.get("failure_pattern") or "")
    source_case = refined.get("distillation_case", {}) or {}
    source_question = str(source_case.get("question") or "")

    for record in reversed(memory):
        if str(record.get("memory_type", "evidence")) not in {"experience", "failed_experience"}:
            continue
        if (
            str(record.get("problem_pattern") or "") == problem_pattern
            and str(record.get("solution_pattern") or "") == solution_pattern
            and str(record.get("failure_pattern") or "") == failure_pattern
        ):
            return record
        record_case = record.get("source_case", {}) or {}
        if source_question and str(record_case.get("question") or "") == source_question:
            return record
    return {}


def _normalize_latest_experience_write(latest: dict, refined: dict) -> dict:
    """补齐前端展示最关心的经验外层字段，避免只在内层 rule card 中可见。"""

    if not latest:
        return {}
    rule_card = latest.get("experience_rule_card", {}) or {}
    refined_rule_card = refined.get("experience_rule_card", {}) or {}
    return {
        **latest,
        "problem_pattern": str(
            latest.get("problem_pattern")
            or rule_card.get("problem_pattern")
            or refined.get("problem_pattern")
            or refined_rule_card.get("problem_pattern")
            or ""
        ),
        "solution_pattern": str(
            latest.get("solution_pattern")
            or rule_card.get("solution_pattern")
            or refined.get("solution_pattern")
            or refined_rule_card.get("solution_pattern")
            or ""
        ),
        "failure_pattern": str(
            latest.get("failure_pattern")
            or rule_card.get("failure_pattern")
            or refined.get("failure_pattern")
            or refined_rule_card.get("failure_pattern")
            or ""
        ),
        "trigger_conditions": list(
            latest.get("trigger_conditions")
            or rule_card.get("trigger_conditions")
            or refined.get("trigger_conditions")
            or refined_rule_card.get("trigger_conditions")
            or []
        ),
        "supporting_evidence": list(
            latest.get("supporting_evidence")
            or rule_card.get("supporting_evidence")
            or refined.get("supporting_evidence")
            or refined_rule_card.get("supporting_evidence")
            or []
        ),
        "confidence": float(
            latest.get("confidence")
            or rule_card.get("confidence")
            or refined.get("confidence")
            or refined_rule_card.get("confidence")
            or 0.0
        ),
        "abstraction_note": str(
            latest.get("abstraction_note")
            or rule_card.get("abstraction_note")
            or refined.get("abstraction_note")
            or refined_rule_card.get("abstraction_note")
            or ""
        ),
        "experience_source": str(
            latest.get("experience_source")
            or refined.get("experience_source")
            or ""
        ),
    }


def summarizer_agent(state: CompetitionState) -> dict:
    """生成最终答案，并沉淀为可复用共享记忆。"""

    hit_count = len(state.get("memory_hits", []))
    latent_count = len(state.get("latent_states", {}))
    execution_result = state.get("execution_result", {})
    task_text = _summarizer_task_text(state)
    print(
        "[summarizer] "
        f"task chars={len(task_text)} memory_hits={hit_count} latent_states={latent_count} "
        f"memory_context chars={len(json.dumps(state.get('memory_context', []), ensure_ascii=False))}",
        flush=True,
    )
    if state["mode"] in {"protocol", "hidden"}:
        answer_ref = state.get("state_refs", {}).get("answer", "")
        answer_state = state.get("latent_states", {}).get(answer_ref, {})
        if answer_state:
            execution_result = {
                **execution_result,
                "latent_answer_ref": answer_ref,
                "latent_answer_summary": answer_state.get("summary", ""),
                "latent_embedding_provider": answer_state.get("embedding_provider", ""),
                "latent_embedding_model": answer_state.get("embedding_model", ""),
                "latent_model": answer_state.get("model", ""),
                "latent_layer": answer_state.get("layer", ""),
                "latent_shape": answer_state.get("shape", []),
            }
    # Summarizer 可单独接真实 API，其他 Agent 保持 mock 以便快速验证。
    direct_summary = _direct_qa_summary(state, execution_result)
    if direct_summary:
        summary = direct_summary
        summarizer_llm_ms = 0.0
    else:
        summarize_started = time.perf_counter()
        summary = get_llm_client("summarizer").summarize(
            task=task_text,
            memory_hits=hit_count,
            latent_states=latent_count,
            plan=state.get("plan", []),
            evidence=state.get("evidence", []),
            execution_result=execution_result,
            cognitive_gaps=state.get("cognitive_gaps", []),
            mode=state["mode"],
        )
        summarizer_llm_ms = (time.perf_counter() - summarize_started) * 1000
    # 这里是共享记忆写入路径，也是后续接超图存储的钩子。
    trace = {
        "task_meta": state.get("task_meta", {}),
        "plan": state.get("plan", []),
        "messages": state.get("messages", []),
        "evidence": state.get("evidence", []),
        "memory_hits": state.get("memory_hits", []),
        "memory_context": state.get("memory_context", []),
        "latent_states": state.get("latent_states", {}),
        "state_refs": state.get("state_refs", {}),
        "execution_result": state.get("execution_result", {}),
        "cognitive_gaps": state.get("cognitive_gaps", []),
        "agent_outputs": state.get("agent_outputs", []),
        "metrics": state.get("metrics", {}),
    }
    refine_started = time.perf_counter()
    refined = refine_memory(
        task_topic=task_text,
        summary=summary,
        trace=trace,
    )
    memory_refine_ms = (time.perf_counter() - refine_started) * 1000
    refine_case = refined.get("distillation_case", {}) or {}
    exact_match = refine_case.get("exact_match")
    positive_memory_allowed = exact_match is not False
    positive_quality_status = "verified" if exact_match is True else "suspect"
    successful_experience_ids: list[str] = []
    if refined.get("experience_source") == "success-derived":
        successful_experience_ids = [
            str(memory_id)
            for memory_id in state.get("execution_result", {}).get("experience_used_memory_ids", [])
            if memory_id
        ]
    updated_memory = increment_reuse_counts(
        state["memory"],
        [hit.get("memory_id") for hit in state.get("memory_hits", [])],
        successful_memory_ids=successful_experience_ids,
    )
    write_started = time.perf_counter()
    if positive_memory_allowed:
        evidence_memory = add_memory(
            updated_memory,
            source_agent="summarizer",
            task_topic=task_text,
            summary=summary,
            tags=["langgraph", "multi-agent", "protocol", state["mode"], positive_quality_status],
            trace=trace,
            quality_status=positive_quality_status,
            source_task_correct=exact_match,
            use_as_positive=True,
        )
    else:
        evidence_memory = updated_memory
    latest_evidence_write = evidence_memory[-1] if positive_memory_allowed and evidence_memory else {}
    memory = evidence_memory
    latest_experience_write: dict = {}
    if refined.get("should_write", True):
        experience_is_failure = exact_match is False or refined.get("experience_source") == "failure-derived"
        experience_memory_type = "failed_experience" if experience_is_failure else "experience"
        experience_quality_status = "failed" if experience_is_failure else positive_quality_status
        memory = add_memory(
            evidence_memory,
            source_agent="summarizer",
            task_topic=task_text,
            summary=refined["summary"],
            tags=["langgraph", "multi-agent", experience_memory_type, state["mode"], experience_quality_status],
            trace=None,
            memory_type=experience_memory_type,
            problem_pattern=refined["problem_pattern"],
            solution_pattern=refined["solution_pattern"],
            failure_pattern=refined["failure_pattern"],
            trigger_conditions=refined["trigger_conditions"],
            supporting_evidence=refined["supporting_evidence"],
            confidence=refined["confidence"],
            archive_scope=refined["archive_scope"],
            lifecycle_stage=refined["lifecycle_stage"],
            strategies=refined["strategies"],
            experiences=refined["experiences"],
            reuse_benefits=refined["reuse_benefits"],
            experience_rule_card=refined["experience_rule_card"],
            experience_source=refined.get("experience_source", ""),
            quality_status=experience_quality_status,
            source_task_correct=exact_match,
            use_as_positive=not experience_is_failure,
            distillation_case=refined.get("distillation_case", {}),
            abstraction_note=refined.get("abstraction_note", ""),
            distillation_mode=refined.get("distillation_mode", ""),
            distillation_raw_output=refined.get("distillation_raw_output", ""),
        )
        latest_experience_write = _locate_latest_experience_write(memory, refined=refined)
        latest_experience_write = _normalize_latest_experience_write(latest_experience_write, refined)
    memory_write_ms = (time.perf_counter() - write_started) * 1000
    metrics = dict(state["metrics"])
    metrics["summarizer_llm_ms"] = metrics.get("summarizer_llm_ms", 0.0) + summarizer_llm_ms
    metrics["memory_refine_ms"] = metrics.get("memory_refine_ms", 0.0) + memory_refine_ms
    metrics["memory_write_ms"] = metrics.get("memory_write_ms", 0.0) + memory_write_ms
    cognitive_gap = make_cognitive_gap(
        agent="summarizer",
        known=["plan", "evidence", "execution_result", "latent_state_refs"],
        unknown=[],
        needed=[],
        gap_type="closed",
        confidence=0.9,
    )
    return {
        "final_answer": summary,
        "memory": memory,
        "memory_refine_case": refined.get("distillation_case", {}),
        "memory_refine_debug": {
            "should_write": refined.get("should_write", True),
            "memory_type": refined.get("memory_type", "experience"),
            "experience_source": refined.get("experience_source", ""),
            "abstraction_note": refined.get("abstraction_note", ""),
            "experience_rule_card": refined.get("experience_rule_card", {}),
            "distillation_mode": refined.get("distillation_mode", ""),
            "distillation_raw_output": refined.get("distillation_raw_output", ""),
            "exact_match": exact_match,
            "positive_memory_allowed": positive_memory_allowed,
        },
        "latest_evidence_write": latest_evidence_write,
        "latest_experience_write": latest_experience_write,
        "memory_write_count": int(positive_memory_allowed) + (1 if refined.get("should_write", True) else 0),
        "experience_write_count": 1 if refined.get("should_write", True) else 0,
        "metrics": metrics,
        "agent_outputs": [
            *state.get("agent_outputs", []),
            {"agent": "SummarizerAgent", "raw_output": summary},
        ],
        "cognitive_gaps": [*state["cognitive_gaps"], cognitive_gap],
    }
