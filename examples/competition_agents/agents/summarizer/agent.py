from __future__ import annotations

import json
import os
import time

from competition_agents.llm import get_llm_client
from competition_agents.memory import add_memory, record_memory_usage, refine_memory
from competition_agents.protocol import make_cognitive_gap
from competition_agents.state import CompetitionState
from competition_agents.state_store import resolve_state_payload
from hypergraph_memory_method import compact_trace


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
        if str(record.get("memory_type", "evidence")) != "experience":
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

    retrieval_ref = str(state.get("state_refs", {}).get("retrieval") or "")
    retrieval_bundle = (
        resolve_state_payload(retrieval_ref, expected_content_type="retrieval.bundle")
        if retrieval_ref
        else None
    ) or {}
    memory_hits = retrieval_bundle.get("memory_hits", state.get("memory_hits", []))
    memory_context = retrieval_bundle.get("memory_context", state.get("memory_context", []))
    experience_rules = retrieval_bundle.get("experience_rules", state.get("experience_rules", []))
    evidence = retrieval_bundle.get("evidence", state.get("evidence", []))
    hit_count = len(memory_hits)
    latent_count = len(state.get("latent_states", {}))
    execution_result = state.get("execution_result", {})
    task_text = _summarizer_task_text(state)
    print(
        "[summarizer] "
        f"task chars={len(task_text)} memory_hits={hit_count} latent_states={latent_count} "
        f"memory_context chars={len(json.dumps(memory_context, ensure_ascii=False))}",
        flush=True,
    )
    if state["mode"] in {"protocol", "hidden"}:
        answer_ref = state.get("state_refs", {}).get("answer", "")
        answer_state = state.get("latent_states", {}).get(answer_ref, {})
        answer_payload = resolve_state_payload(answer_ref) if answer_ref else None
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
                "latent_answer_content_type": (answer_payload or {}).get("content_type", ""),
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
            evidence=evidence,
            execution_result=execution_result,
            cognitive_gaps=state.get("cognitive_gaps", []),
            mode=state["mode"],
        )
        summarizer_llm_ms = (time.perf_counter() - summarize_started) * 1000
    # 这里是共享记忆写入路径，也是后续接超图存储的钩子。
    trace = compact_trace({
        "task_meta": state.get("task_meta", {}),
        "plan": state.get("plan", []),
        "messages": state.get("messages", []),
        "evidence": evidence,
        "memory_hits": memory_hits,
        "memory_context": memory_context,
        "latent_states": state.get("latent_states", {}),
        "state_refs": state.get("state_refs", {}),
        "execution_result": state.get("execution_result", {}),
        "cognitive_gaps": state.get("cognitive_gaps", []),
        "agent_outputs": state.get("agent_outputs", []),
        "metrics": state.get("metrics", {}),
    })
    refine_started = time.perf_counter()
    refined = refine_memory(
        task_topic=task_text,
        summary=summary,
        trace=trace,
    )
    memory_refine_ms = (time.perf_counter() - refine_started) * 1000
    execution_usage = state.get("execution_result", {})
    updated_memory = record_memory_usage(
        state["memory"],
        retrieved_memory_ids=execution_usage.get("memory_retrieved_ids", []),
        adopted_memory_ids=execution_usage.get(
            "memory_adopted_ids",
            execution_usage.get("memory_materialized_ids", []),
        ),
    )
    write_started = time.perf_counter()
    evidence_memory = add_memory(
        updated_memory,
        source_agent="summarizer",
        task_topic=task_text,
        summary=summary,
        tags=["langgraph", "multi-agent", "protocol", state["mode"]],
        trace=trace,
    )
    latest_evidence_write = evidence_memory[-1] if evidence_memory else {}
    memory = evidence_memory
    latest_experience_write: dict = {}
    if refined.get("should_write", True):
        memory = add_memory(
            evidence_memory,
            source_agent="summarizer",
            task_topic=task_text,
            summary=refined["summary"],
            tags=["langgraph", "multi-agent", "experience", state["mode"]],
            trace=None,
            memory_type="experience",
            problem_pattern=refined["problem_pattern"],
            solution_pattern=refined["solution_pattern"],
            failure_pattern=refined["failure_pattern"],
            trigger_conditions=refined["trigger_conditions"],
            supporting_evidence=refined["supporting_evidence"],
            confidence=refined["confidence"],
            archive_scope=refined["archive_scope"],
            lifecycle_stage="needs-validation",
            strategies=refined["strategies"],
            experiences=refined["experiences"],
            reuse_benefits=refined["reuse_benefits"],
            experience_rule_card=refined["experience_rule_card"],
            experience_source=refined.get("experience_source", ""),
            source_memory_ids=[
                str(latest_evidence_write.get("memory_id"))
            ]
            if latest_evidence_write.get("memory_id")
            else [],
            distillation_case=refined.get("distillation_case", {}),
            abstraction_note=refined.get("abstraction_note", ""),
            distillation_mode=refined.get("distillation_mode", ""),
            distillation_raw_output=refined.get("distillation_raw_output", ""),
        )
        latest_experience_write = _locate_latest_experience_write(memory, refined=refined)
        latest_experience_write = _normalize_latest_experience_write(latest_experience_write, refined)
    memory_write_ms = (time.perf_counter() - write_started) * 1000
    metrics = dict(state["metrics"])
    state_store_reads = int(bool(retrieval_bundle)) + int(
        bool(state["mode"] in {"protocol", "hidden"} and state.get("state_refs", {}).get("answer"))
    )
    metrics["state_store_read_count"] = metrics.get("state_store_read_count", 0) + state_store_reads
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
        },
        "latest_evidence_write": latest_evidence_write,
        "latest_experience_write": latest_experience_write,
        "memory_write_count": 2 if refined.get("should_write", True) else 1,
        "experience_write_count": 1 if refined.get("should_write", True) else 0,
        "metrics": metrics,
        "agent_outputs": [
            *state.get("agent_outputs", []),
            {"agent": "SummarizerAgent", "raw_output": summary},
        ],
        "cognitive_gaps": [*state["cognitive_gaps"], cognitive_gap],
    }
