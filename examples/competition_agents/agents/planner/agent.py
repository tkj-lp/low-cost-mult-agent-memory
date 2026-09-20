from __future__ import annotations

import json
import os
import re
import time

from baselines.autoform import autoform_enabled, generate_autoform_message
from competition_agents.llm import get_llm_client
from competition_agents.handoff import parse_planner_output
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
        f"Question: {question}",
    ]
    topic = str(task_meta.get("topic") or "").strip()
    if topic:
        lines.append(f"Topic: {topic}")
    if experience_hits:
        lines.append(
            "Previously verified experience:\n"
            + json.dumps([{"solution": str(h.get("solution_pattern") or "")[:160],
                           "avoid": str(h.get("failure_pattern") or "")[:100]}
                          for h in experience_hits[:2]], ensure_ascii=False)
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
    topic = str(task_meta.get("topic") or "").strip()
    section = str(task_meta.get("topic_section") or task_meta.get("task_type") or "").strip()
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
        query_parts = [f"Question: {question}"]
        if topic:
            query_parts.append(f"Current topic: {topic}")
        if section:
            query_parts.append(f"Requested section or attribute: {section}")
        query_parts.append(f"Retrieval goal: {goal.rstrip('.')}")
        steps.append(
            {
                "step": index,
                "goal": goal.rstrip("."),
                "query": "\n".join(query_parts),
                "depends_on": index - 1 if index > 1 else None,
            }
        )
    return steps


def _build_retrieval_policy(
    state: CompetitionState,
    plan: list[str],
    experience_hits: list[dict],
    classification: dict | None = None,
) -> tuple[dict[str, object], float]:
    started = time.perf_counter()
    task_meta = state.get("task_meta", {})
    task = state.get("task", "")
    classification = classification or {"query_type": "unknown", "reference_target": ""}
    question = str(task_meta.get("question") or task or "").strip()
    policy = {
        "query_type": classification["query_type"],
        "reference_target": classification["reference_target"],
        "decision_source": "retriever_pending",
        "need_knowledge": True,
        "memory_first": True,
        "qa_memory_query": question,
        "memory_query_hint": question,
        "knowledge_query_hint": question,
        "knowledge_steps": _knowledge_steps(state, plan),
        "experience_retrieval": {
            "performed": True,
            "query": _experience_query(state),
            "hit_ids": [hit.get("memory_id") for hit in experience_hits],
        },
        "reason": "Planner classifies the query; Retriever decides QA retrieval.",
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
        if hasattr(client, "plan_query"):
            raw_output = client.plan_query(planner_task)
        elif hasattr(client, "plan_with_raw"):
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
    classification, fallback_reason = parse_planner_output(
        raw_output, getattr(client, "last_generation_metrics", {}))
    plan = classification["steps"]
    plan_sanitized = bool(fallback_reason)
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
        classification,
    )
    retrieval_batches = build_retrieval_batches(
        need_memory=True,
        need_knowledge=bool(retrieval_policy.get("need_knowledge", True)),
    )
    retrieval_batches = update_retrieval_batch(
        retrieval_batches,
        "memory_route",
        status="experience_complete",
        query_count=1,
        hit_counts={"episodic": 0, "experience": len(experience_hits)},
        decision="run",
        decision_reason="experience_complete_qa_decision_pending",
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
                consumers=["retriever", "reasoner", "summarizer"],
                hidden_result=hidden_result,
                hidden_tensor=hidden_tensor,
                semantic_payload={
                    "plan": plan,
                    "retrieval_policy": retrieval_policy,
                },
            )
        else:
            plan_ref, latent_states, metrics = put_latent_state(
                state,
                producer="planner",
                topic=task,
                text="\n".join(plan),
                consumers=["retriever", "reasoner", "summarizer"],
                semantic_payload={
                    "plan": plan,
                    "retrieval_policy": retrieval_policy,
                },
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
        question = str(state.get("task_meta", {}).get("question") or task).strip()
        knowledge_steps = retrieval_policy.get("knowledge_steps", [])
        payload = json.dumps({"question": question, "plan": plan, "retrieval_policy": retrieval_policy},
                             ensure_ascii=False, separators=(",", ":"))
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
            params={"operation": "execute_retrieval_plan"},
            state_refs=[plan_ref],
            capability_required=["retrieve"],
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
                "query_classification": classification,
                "planner_fallback_reason": fallback_reason,
                "llm_call": dict(getattr(client, "last_generation_metrics", {})),
                "raw_output": raw_output,
                "plan_sanitized": plan_sanitized,
                "retrieval_policy": retrieval_policy,
            },
        ],
        "cognitive_gaps": [*state["cognitive_gaps"], cognitive_gap],
    }
