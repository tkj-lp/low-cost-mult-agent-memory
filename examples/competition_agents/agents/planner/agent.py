from __future__ import annotations

import json
import os
import re
import time

from baselines.autoform import autoform_enabled, generate_autoform_message
from competition_agents.llm import get_llm_client
from competition_agents.protocol import make_cognitive_gap, make_compact_packet, make_packet
from competition_agents.state import CompetitionState, put_hidden_state, put_latent_state, record_message
from hypergraph_memory_method.retrieval_batches import (
    KNOWLEDGE_FOLLOWUP_BATCH,
    build_retrieval_batches,
)


def _contains_cross_turn_reference(text: str) -> bool:
    markers = (
        "previous",
        "earlier",
        "above",
        "again",
        "same",
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
    return bool(re.search(r"\b(that|this|it|they|those)\b", lowered))


def _plan_requests_memory(plan: list[str]) -> bool:
    text = " ".join(str(step or "").lower() for step in plan)
    return any(marker in text for marker in ("memory", "reuse", "historical", "prior", "previous"))


def _build_planner_task(state: CompetitionState) -> str:
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
    supporting = str(task_meta.get("supporting_passage") or "").strip()
    if supporting:
        lines.append("Supporting passage hint: " + supporting[:360])
    return "\n".join(lines)


def _fallback_plan(state: CompetitionState) -> list[str]:
    task_meta = state.get("task_meta", {}) or {}
    task_text = " ".join(
        str(value or "")
        for value in (
            state.get("task"),
            task_meta.get("question"),
            task_meta.get("supporting_passage"),
        )
    )
    is_later_turn = str(task_meta.get("turn_id") or "") not in {"", "turn-1", "1"}
    memory_step = (
        "Resolve cross-turn references against available prior memory before retrieval."
        if is_later_turn or _contains_cross_turn_reference(task_text)
        else "Decide whether this task needs prior memory or only knowledge evidence."
    )
    return [
        "Identify the question target and requested answer type.",
        memory_step,
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


def _build_retrieval_policy(state: CompetitionState, plan: list[str]) -> tuple[dict[str, object], float]:
    started = time.perf_counter()
    task_meta = state.get("task_meta", {})
    task = state.get("task", "")
    memory_available = bool(state.get("memory"))
    dataset_type = str(task_meta.get("dataset_type") or "")
    turn_id = str(task_meta.get("turn_id") or "")
    cross_turn_reference = _contains_cross_turn_reference(
        " ".join(
            str(item or "")
            for item in (
                task,
                task_meta.get("question", ""),
                task_meta.get("supporting_passage", ""),
            )
        )
    )
    later_continuous_turn = (
        dataset_type == "continuous"
        and turn_id not in {"", "turn-1", "1"}
    )
    plan_memory_intent = _plan_requests_memory(plan)
    need_memory = memory_available and (
        later_continuous_turn
        or cross_turn_reference
        or (plan_memory_intent and dataset_type in {"continuous", "codeact"})
    )
    reason_signals = {
        "memory_available": memory_available,
        "dataset_type": dataset_type,
        "turn_id": turn_id,
        "later_continuous_turn": later_continuous_turn,
        "cross_turn_reference": cross_turn_reference,
        "plan_memory_intent": plan_memory_intent,
    }
    reasons = [key for key, value in reason_signals.items() if value]
    if not memory_available:
        reasons.append("memory_empty")
    policy = {
        "need_memory": need_memory,
        "need_knowledge": True,
        "memory_first": True,
        "memory_query_hint": task_meta.get("question") or task[:240],
        "knowledge_query_hint": task_meta.get("question") or task[:240],
        "reason": ", ".join(reasons) if reasons else "default_knowledge_only",
        "signals": reason_signals,
    }
    policy["retrieval_batches"] = build_retrieval_batches(
        need_memory=need_memory,
        need_knowledge=True,
    )
    policy["next_batch_interface"] = {
        "batch_id": KNOWLEDGE_FOLLOWUP_BATCH,
        "decision_owner": "planner",
        "allowed_decisions": ["run", "skip"],
        "decision_field": "next_batch_decision",
    }
    elapsed_ms = (time.perf_counter() - started) * 1000
    return policy, elapsed_ms


def planner_agent(state: CompetitionState) -> dict:
    """规划任务，并向 ReasonerAgent 发起推理请求。"""

    task = state["task"]
    memory_context = state.get("memory_context", [])
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
        planner_task = _build_planner_task(state)
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
    retrieval_policy, planner_decision_ms = _build_retrieval_policy(state, plan)
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
        wire_retrieval_policy = {
            key: value
            for key, value in retrieval_policy.items()
            if key not in {"retrieval_batches", "next_batch_interface"}
        }
        payload = make_compact_packet(
            sender="planner",
            receiver="retriever",
            action="retrieve",
            content_type="hidden_state_ref" if state["mode"] == "hidden" else "state_ref",
            params={"plan": plan_ref, "retrieval_policy": wire_retrieval_policy},
            result={"next": "retriever", "retrieval_policy": wire_retrieval_policy},
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
    metrics["memory_gate_count"] = metrics.get("memory_gate_count", 0) + 1
    if not retrieval_policy.get("need_memory"):
        metrics["memory_gate_skip_count"] = metrics.get("memory_gate_skip_count", 0) + 1
    metrics["planner_memory_decision_ms"] = metrics.get("planner_memory_decision_ms", 0.0) + planner_decision_ms
    return {
        "plan": plan,
        "retrieval_policy": retrieval_policy,
        "retrieval_batches": retrieval_policy["retrieval_batches"],
        "latent_states": latent_states,
        "state_refs": {**state.get("state_refs", {}), **({"plan": plan_ref} if plan_ref else {})},
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
