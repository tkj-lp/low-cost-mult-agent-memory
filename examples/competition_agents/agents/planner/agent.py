from __future__ import annotations

import json
import os
import re
import time

from baselines.autoform import autoform_enabled, generate_autoform_message
from competition_agents.llm import get_llm_client
from competition_agents.memory import search_memory
from competition_agents.protocol import make_cognitive_gap, make_compact_packet, make_packet
from competition_agents.runtime_events import emit_memory_retrieval_events
from competition_agents.state import (
    CompetitionState,
    put_hidden_state,
    put_latent_state,
    put_shared_state,
    record_message,
)
from hypergraph_memory_method import build_retrieval_batches, update_retrieval_batch


def _contains_cross_turn_reference(text: str) -> bool:
    markers = (
        "previous",
        "earlier",
        "above",
        "again",
        "same",
        "former",
        "latter",
        "there",
        "that list",
        "that process",
        "that substance",
        "that experiment",
        "上一个",
        "前面",
        "之前",
        "刚才",
        "上述",
        "这些",
        "该",
    )
    lowered = str(text or "").lower()
    if any(marker in lowered for marker in markers):
        return True
    if re.search(r"\b(that|this|it|they|those|he|she|him|her|them|their|its)\b", lowered):
        return True
    return bool(
        re.search(
            r"\b(what|where|when|who|how) (is|was|are|were|does|did|do|has|have|had) "
            r"the (film|movie|place|country|city|sea|process|experiment|substance|"
            r"work|book|organization|company)\b",
            lowered,
        )
    )


def _plan_requests_memory(plan: list[str]) -> bool:
    text = " ".join(str(step or "").lower() for step in plan)
    return any(marker in text for marker in ("memory", "reuse", "historical", "prior", "previous"))


def _experience_query(state: CompetitionState) -> str:
    task_meta = state.get("task_meta", {}) or {}
    question = str(task_meta.get("question") or state.get("task") or "").strip()
    tags = task_meta.get("tags", []) or []
    if not isinstance(tags, list):
        tags = [tags]
    return "\n".join(
        part
        for part in (
            f"Question: {question}",
            f"Task type: {task_meta.get('task_type') or task_meta.get('dataset_type') or ''}",
            "Tags: " + json.dumps(tags, ensure_ascii=False),
        )
        if not part.endswith(": ")
    )


def _compact_experience_hits(hits: list[dict]) -> list[dict]:
    return [
        {
            "memory_id": hit.get("memory_id"),
            "problem_pattern": hit.get("problem_pattern", ""),
            "solution_pattern": hit.get("solution_pattern", ""),
            "failure_pattern": hit.get("failure_pattern", ""),
            "trigger_conditions": hit.get("trigger_conditions", []),
            "supporting_evidence": hit.get("supporting_evidence", []),
            "confidence": hit.get("confidence", 0.0),
            "reuse_count": hit.get("reuse_count", 0),
            "reuse_success_count": hit.get("reuse_success_count", 0),
            "summary": str(hit.get("summary") or "")[:240],
            "retrieval_score": hit.get("retrieval_score"),
        }
        for hit in hits
    ]


def _build_planner_task(state: CompetitionState, experience_hits: list[dict]) -> str:
    task = str(state.get("task") or "")
    task_meta = state.get("task_meta", {}) or {}
    question = str(task_meta.get("question") or "").strip()
    if not question:
        return task

    lines = [
        "Plan this QA task. Return task steps only, not a dialogue.",
        f"Question: {question}",
    ]
    topic = str(task_meta.get("topic") or "").strip()
    if topic:
        lines.append(f"Topic: {topic}")
    if experience_hits:
        lines.append(
            "Previously verified experience:\n"
            + json.dumps(_compact_experience_hits(experience_hits), ensure_ascii=False)
        )
    lines.append(
        "Decide whether pronouns or cross-turn references require question-answer memory, "
        "then specify the knowledge facts that Retriever should obtain."
    )
    return "\n".join(lines)


def _fallback_plan(state: CompetitionState) -> list[str]:
    return [
        "Identify the question target and requested answer type.",
        "Decide whether this task needs prior question-answer memory.",
        "Retrieve the most relevant supporting evidence for the target.",
        "Route evidence to reasoning and summarize a concise final answer.",
    ]


def _is_bad_plan_step(step: str) -> bool:
    text = str(step or "").strip()
    lowered = text.lower()
    if not text:
        return True
    dialogue_markers = (
        "ask:",
        "receive response",
        "user:",
        "assistant:",
        "protagonist",
        "love story",
    )
    if any(marker in lowered for marker in dialogue_markers):
        return True
    if len(text) > 180:
        return True
    return False


def _sanitize_plan(state: CompetitionState, plan: list[str]) -> tuple[list[str], bool]:
    cleaned = []
    for step in plan:
        text = str(step or "").strip("- 0123456789.\t ")
        if _is_bad_plan_step(text):
            continue
        cleaned.append(text)
    if len(cleaned) < 2:
        return _fallback_plan(state), True
    return cleaned[:4], len(cleaned) != len(plan)


def _experience_recommends_qa_recall(experience_hits: list[dict]) -> bool:
    text = " ".join(
        str(value or "")
        for hit in experience_hits
        for value in (
            hit.get("problem_pattern"),
            hit.get("solution_pattern"),
            " ".join(hit.get("trigger_conditions", []) or []),
        )
    ).lower()
    return any(marker in text for marker in ("coreference", "cross_turn", "pronoun", "prior_answer"))


def _knowledge_steps(state: CompetitionState, plan: list[str]) -> list[dict[str, object]]:
    task_meta = state.get("task_meta", {}) or {}
    question = str(task_meta.get("question") or state.get("task") or "").strip()
    candidates = [
        str(step).strip()
        for step in plan
        if str(step).strip()
        and any(
            marker in str(step).lower()
            for marker in ("identify", "retrieve", "locate", "find", "extract", "determine", "verify")
        )
        and "memory" not in str(step).lower()
    ]
    if not candidates:
        candidates = ["Retrieve evidence needed to answer the question."]
    steps: list[dict[str, object]] = []
    for index, goal in enumerate(candidates[:3], start=1):
        steps.append(
            {
                "step": index,
                "goal": goal.rstrip("."),
                "query": f"{question}\nRetrieval goal: {goal.rstrip('.')}",
                "depends_on": index - 1 if index > 1 else None,
            }
        )
    return steps


def _build_retrieval_policy(
    state: CompetitionState,
    plan: list[str],
    experience_hits: list[dict],
) -> tuple[dict[str, object], float]:
    started = time.perf_counter()
    task_meta = state.get("task_meta", {})
    task = state.get("task", "")
    qa_memory_available = any(
        str(item.get("memory_type", "evidence")) == "evidence"
        for item in state.get("memory", [])
    )
    dataset_type = str(task_meta.get("dataset_type") or "")
    turn_id = str(task_meta.get("turn_id") or "")
    cross_turn_reference = _contains_cross_turn_reference(
        str(task_meta.get("question") or task)
    )
    experience_recommends_recall = _experience_recommends_qa_recall(experience_hits)
    need_qa_memory = qa_memory_available and (
        cross_turn_reference or experience_recommends_recall
    )
    qa_reason = (
        "question contains a pronoun or cross-turn reference"
        if cross_turn_reference
        else "verified experience recommends question-answer recall"
        if experience_recommends_recall
        else "question is self-contained"
    )
    reason_signals = {
        "qa_memory_available": qa_memory_available,
        "dataset_type": dataset_type,
        "turn_id": turn_id,
        "cross_turn_reference": cross_turn_reference,
        "experience_hit_count": len(experience_hits),
        "experience_recommends_qa_recall": experience_recommends_recall,
    }
    reasons = [key for key, value in reason_signals.items() if value]
    if not qa_memory_available:
        reasons.append("qa_memory_empty")
    question = str(task_meta.get("question") or task or "").strip()
    policy = {
        "need_memory": need_qa_memory,
        "need_qa_memory": need_qa_memory,
        "need_knowledge": True,
        "memory_first": True,
        "qa_memory_query": question,
        "qa_memory_reason": qa_reason,
        "memory_query_hint": question,
        "knowledge_query_hint": question,
        "knowledge_steps": _knowledge_steps(state, plan),
        "experience_retrieval": {
            "performed": True,
            "query": _experience_query(state),
            "hit_ids": [hit.get("memory_id") for hit in experience_hits],
        },
        "reason": ", ".join(reasons) if reasons else "default_knowledge_only",
        "signals": reason_signals,
    }
    elapsed_ms = (time.perf_counter() - started) * 1000
    return policy, elapsed_ms


def planner_agent(state: CompetitionState) -> dict:
    """规划任务，并向 ReasonerAgent 发起推理请求。"""

    task = state["task"]
    memory_context = state.get("memory_context", [])
    experience_query = _experience_query(state)
    experience_started = time.perf_counter()
    experience_hits = search_memory(
        state.get("memory", []),
        experience_query,
        top_k=2,
        memory_types=["experience"],
        memory_layers=["experience"],
    )
    experience_elapsed_ms = (time.perf_counter() - experience_started) * 1000
    emit_memory_retrieval_events(
        state,
        query=experience_query,
        hits=experience_hits,
        requester="planner",
        memory_layer="experience",
    )
    print(
        "[planner] "
        f"task chars={len(task)} memory_hits={len(state.get('memory_hits', []))} "
        f"memory_context chars={len(json.dumps(memory_context, ensure_ascii=False))}",
        flush=True,
    )
    # Planner 可通过 LLM_PLANNER_PROVIDER 切换 mock 或真实 API。
    client = get_llm_client("planner")
    llm_started = time.perf_counter()
    previous_capture_hidden = os.environ.get("LOCAL_CAPTURE_HIDDEN")
    if state["mode"] == "hidden":
        os.environ["LOCAL_CAPTURE_HIDDEN"] = "1"
    try:
        planner_task = _build_planner_task(state, experience_hits)
        if hasattr(client, "plan_with_raw"):
            plan, raw_output = client.plan_with_raw(planner_task)
        else:
            plan = client.plan(planner_task)
            raw_output = "\n".join(plan)
    finally:
        if state["mode"] == "hidden":
            if previous_capture_hidden is None:
                os.environ.pop("LOCAL_CAPTURE_HIDDEN", None)
            else:
                os.environ["LOCAL_CAPTURE_HIDDEN"] = previous_capture_hidden
    llm_elapsed_ms = (time.perf_counter() - llm_started) * 1000
    plan, plan_sanitized = _sanitize_plan(state, plan)
    hidden_result = None
    hidden_tensor = None
    if state["mode"] == "hidden":
        consume_hidden = getattr(client, "consume_last_hidden_state", None)
        if callable(consume_hidden):
            hidden_result, hidden_tensor = consume_hidden()
    retrieval_policy, planner_decision_ms = _build_retrieval_policy(
        state,
        plan,
        experience_hits,
    )
    retrieval_batches = build_retrieval_batches(
        need_memory=bool(retrieval_policy.get("need_qa_memory")),
        need_knowledge=bool(retrieval_policy.get("need_knowledge", True)),
    )
    retrieval_batches = update_retrieval_batch(
        retrieval_batches,
        "memory_route",
        status="experience_complete",
        query_count=1,
        hit_counts={"episodic": 0, "experience": len(experience_hits)},
    )
    cognitive_gap = make_cognitive_gap(
        agent="planner",
        known=["task_goal", "available_agent_capabilities"],
        unknown=["external_supporting_facts", "historical_memory"],
        needed=["model_reasoning", "summary_generation"],
        gap_type="needs_reasoning",
        confidence=0.72,
    )
    plan_ref = ""
    latent_states = state.get("latent_states", {})
    metrics = state["metrics"]
    if state["mode"] in {"protocol", "hidden"}:
        if state["mode"] == "hidden":
            plan_ref, latent_states, metrics = put_hidden_state(
                state,
                producer="planner",
                topic=task,
                text="\n".join(plan),
                consumers=["retriever", "reasoner"],
                hidden_result=hidden_result,
                hidden_tensor=hidden_tensor,
            )
        else:
            plan_ref, latent_states, metrics = put_latent_state(
                state,
                producer="planner",
                topic=task,
                text="\n".join(plan),
                consumers=["reasoner", "summarizer"],
            )
        experience_ref, latent_states, metrics = put_shared_state(
            {**state, "latent_states": latent_states, "metrics": metrics},
            producer="planner",
            topic=task,
            content_type="retrieval.experience",
            payload={
                "summary": f"Planner retrieved {len(experience_hits)} experience memories.",
                "query": experience_query,
                "hits": _compact_experience_hits(experience_hits),
            },
            consumers=["planner", "retriever"],
            scope="task",
            prefix="experience",
        )
    else:
        experience_ref = ""

    if state["mode"] == "text":
        # text 模式故意重复上下文，用来模拟传统自然语言协作的高开销。
        payload = (
            f"PlannerAgent received task: {task}. "
            f"It creates a four-step plan: {', '.join(plan)}. "
            "ReasonerAgent should answer from the task and plan because retrieval is skipped in this run. "
            "For text-mode collaboration, the planner forwards the full task background, "
            "role description, expected output format, and repeated natural-language context "
            "so the next agent can reconstruct the internal planning state without reading "
            f"structured state references. The planner says its capabilities are task decomposition, "
            "capability routing, and cognitive gap detection. It also explains its cognitive gap: "
            "it already knows the task goal and available agent capabilities, but it does not yet "
            "know external supporting facts or historical memory, so it needs model reasoning and final "
            "summary generation. Full task background repeated for baseline cost: "
            f"{task}."
        )
        if autoform_enabled():
            payload, _autoform_audit = generate_autoform_message(
                sender="planner",
                receiver="retriever",
                action="retrieve",
                source_message=payload,
            )
    else:
        # protocol 模式只传紧凑的动作参数，不传长段自然语言。
        payload = make_compact_packet(
            sender="planner",
            receiver="retriever",
            action="retrieve",
            content_type="hidden_state_ref" if state["mode"] == "hidden" else "state_ref",
            params={"plan": plan_ref, "retrieval_policy": retrieval_policy},
            result={"next": "retriever", "retrieval_policy": retrieval_policy},
            state_refs=[plan_ref],
            capability_required=["retrieve"],
            cognitive_gap=cognitive_gap,
            confidence=0.72,
        )

    # record_message 会追加消息，并更新通信开销指标。
    messages, metrics = record_message(
        state,
        sender="planner",
        receiver="retriever",
        action="retrieve",
        payload=payload,
        metrics=metrics,
    )
    metrics = dict(metrics)
    metrics["planner_llm_ms"] = metrics.get("planner_llm_ms", 0.0) + llm_elapsed_ms
    metrics["retriever_memory_ms"] = metrics.get("retriever_memory_ms", 0.0) + experience_elapsed_ms
    metrics["memory_query_count"] = metrics.get("memory_query_count", 0) + 1
    metrics["retrieval_batch_count"] = metrics.get("retrieval_batch_count", 0) + 1
    metrics["memory_gate_count"] = metrics.get("memory_gate_count", 0) + 1
    if not retrieval_policy.get("need_memory"):
        metrics["memory_gate_skip_count"] = metrics.get("memory_gate_skip_count", 0) + 1
    metrics["planner_memory_decision_ms"] = metrics.get("planner_memory_decision_ms", 0.0) + planner_decision_ms
    return {
        "plan": plan,
        "retrieval_policy": retrieval_policy,
        "retrieval_batches": retrieval_batches,
        "experience_hits": experience_hits,
        "experience_rules": _compact_experience_hits(experience_hits),
        "memory_layer_hits": {
            **state.get("memory_layer_hits", {}),
            "experience": experience_hits,
        },
        "latent_states": latent_states,
        "state_refs": {
            **state.get("state_refs", {}),
            **({"plan": plan_ref} if plan_ref else {}),
            **({"experience_retrieval": experience_ref} if experience_ref else {}),
        },
        "messages": messages,
        "metrics": metrics,
        "agent_outputs": [
            *state.get("agent_outputs", []),
            {
                "agent": "PlannerAgent",
                "raw_output": raw_output,
                "plan_sanitized": plan_sanitized,
                "retrieval_policy": retrieval_policy,
            },
        ],
        "cognitive_gaps": [*state["cognitive_gaps"], cognitive_gap],
    }
