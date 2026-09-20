from __future__ import annotations

import json
import os
import re
import time

from baselines.autoform import autoform_enabled, generate_autoform_message
from competition_agents.latent_hidden.adapter import cosine_similarity
from competition_agents.llm import get_llm_client
from competition_agents.handoff import decode_memory_object, encode_answer_handoff
from competition_agents.protocol import make_cognitive_gap, make_compact_packet, make_packet
from competition_agents.state import (
    CompetitionState,
    put_hidden_state,
    put_latent_state,
    record_message,
    resolve_state_ref,
    received_state_refs,
)


def _reasoner_task_text(state: CompetitionState) -> str:
    """在 hypergraph 模式下只把问题本身传给 Reasoner，避免意外回看原始上下文。"""

    task_meta = state.get("task_meta", {})
    if str(task_meta.get("context_source") or "") == "hypergraph":
        question = str(task_meta.get("question") or "").strip()
        if question:
            parts = [f"Question: {question}"]
            topic = str(task_meta.get("topic") or "").strip()
            if topic:
                parts.append(f"Resolved topic: {topic}")
            return "\n".join(parts)
    return state["task"]


def _build_codeact_execution_request(state: CompetitionState) -> dict | None:
    """为 CodeAct 数据集生成确定性的执行请求，暂不引入 LLM 动态代码。"""

    task_meta = state.get("task_meta", {})
    if task_meta.get("dataset_type") != "codeact":
        return None
    code = task_meta.get("reference_code")
    if not code:
        return None
    return {
        "needs_tool": True,
        "language": "python",
        "code": code,
        "allowed_imports": task_meta.get("allowed_imports", []),
        "timeout_ms": task_meta.get("timeout_ms", 3000),
        "code_source": "reference_code",
    }


def _unique_memory_ids(items: list[dict]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for item in items:
        memory_id = str(item.get("memory_id") or "").strip()
        if not memory_id or memory_id in seen:
            continue
        seen.add(memory_id)
        ids.append(memory_id)
    return ids


def _latest_message_payload(state: CompetitionState, *, sender: str, receiver: str) -> str:
    """Return the newest text payload sent between two agents."""

    for message in reversed(state.get("messages", [])):
        if message.get("from") != sender or message.get("to") != receiver:
            continue
        payload = message.get("payload", "")
        return payload if isinstance(payload, str) else str(payload)
    return ""


def _question_answer_rules(question: str) -> list[str]:
    """Build dataset-independent answer constraints from the question form."""

    question_lower = str(question or "").lower()
    rules = [
        "First resolve every pronoun, ordinal, and ellipsis from QA memory or the supplied topic.",
        "Identify the requested attribute before reading evidence.",
        "Use only facts supported by retrieved evidence; experience rules are not factual evidence.",
        "Return the shortest complete answer that satisfies the requested attribute.",
        "Do not repeat the topic definition when the question asks for another attribute.",
    ]
    if re.search(r"\b(when|what year|what date)\b", question_lower):
        rules.append("Answer type: time or date. Match all temporal, medium, and location constraints.")
    elif re.search(r"\b(where|which place|which country|which city|location)\b", question_lower):
        rules.append(
            "Answer type: location. If granularity is unspecified, include both the country and the more specific place stated in evidence."
        )
    elif re.search(r"\b(who|whose|which person)\b", question_lower):
        rules.append("Answer type: person or role. Return only the requested identity or role.")
    elif re.search(r"\b(how many|how much|what number|what percentage)\b", question_lower):
        rules.append("Answer type: number or quantity. Preserve the unit from evidence.")

    list_question = bool(
        re.search(
            r"\b(list|name|mention|which|what kinds|what types?|examples|members|species)\b",
            question_lower,
        )
    )
    if list_question:
        rules.append("Answer type: list. Return all directly requested items from the best evidence span, without unrelated examples.")

    if question_lower.startswith(
        ("is ", "are ", "was ", "were ", "do ", "does ", "did ", "has ", "have ", "had ")
    ):
        rules.append("Answer type: yes/no. Answer Yes or No and add only the minimum evidence-backed qualification.")

    if re.search(
        r"\b(describe|explain|tell me about|what .* like|how .* look|why)\b",
        question_lower,
    ):
        rules.append(
            "Answer type: descriptive. Include all directly relevant evidence clauses, while omitting unrelated background."
        )

    if re.match(r"^what (?:is|are)\b", question_lower):
        rules.append("Answer type: definition. Return the category or description, never only the entity named in the question.")

    if re.search(r"\b(first|second|third|fourth|last|former|latter)\b", question_lower):
        rules.append("Resolve the ordinal or relative reference against the ordered prior answer before extracting the new attribute.")
    return rules


def reasoner_agent(state: CompetitionState) -> dict:
    """基于任务、计划、检索证据和记忆命中生成候选答案。"""

    # 保留图状态字段用于前端和旧模式；非文本模式优先消费状态池中的数据面载荷。
    metrics = state["metrics"]
    memory_hits = state.get("memory_hits", [])
    memory_context = state.get("memory_context", [])
    experience_hits = state.get("experience_hits", [])
    experience_rules = state.get("experience_rules", [])
    evidence = state.get("evidence", [])
    plan = state.get("plan", [])
    plan_ref = ""
    evidence_ref = ""
    plan_state_payload: dict = {}
    evidence_state_payload: dict = {}
    if state["mode"] in {"protocol", "hidden"}:
        delivered_refs = received_state_refs(
            state,
            sender="retriever",
            receiver="reasoner",
        )
        evidence_ref = delivered_refs[0]
        plan_ref = delivered_refs[1] if len(delivered_refs) > 1 else ""
        evidence_state_payload, metrics = resolve_state_ref(
            {**state, "metrics": metrics},
            evidence_ref,
            consumer="reasoner",
            expected_content_types={"latent.embedding", "latent.llm_hidden_state"},
        )
        if plan_ref:
            plan_state_payload, metrics = resolve_state_ref(
                {**state, "metrics": metrics},
                plan_ref,
                consumer="reasoner",
                expected_content_types={"latent.embedding", "latent.llm_hidden_state"},
            )
        plan_semantic = plan_state_payload.get("semantic_payload", {})
        if isinstance(plan_semantic, dict):
            resolved_plan = plan_semantic.get("plan")
            if isinstance(resolved_plan, list) and resolved_plan:
                plan = [str(step) for step in resolved_plan]
        evidence_semantic = evidence_state_payload.get("semantic_payload", {})
        if not evidence_state_payload or not evidence_semantic.get("reasoning_input"):
            raise ValueError("Retriever handoff state is missing or invalid")
        if isinstance(evidence_semantic, dict):
            resolved_evidence = evidence_semantic.get("evidence")
            if isinstance(resolved_evidence, list):
                evidence = [str(item) for item in resolved_evidence]
            resolved_hits = evidence_semantic.get("memory_hits")
            if isinstance(resolved_hits, list):
                memory_hits = resolved_hits
            resolved_experience_hits = evidence_semantic.get("experience_hits")
            if isinstance(resolved_experience_hits, list):
                experience_hits = resolved_experience_hits
            resolved_rules = evidence_semantic.get("experience_rules")
            if isinstance(resolved_rules, list):
                experience_rules = resolved_rules
            resolved_context = evidence_semantic.get("memory_context")
            if isinstance(resolved_context, list):
                memory_context = resolved_context
    task_text = _reasoner_task_text(state)
    question = str(state.get("task_meta", {}).get("question") or "")
    answer_rules = _question_answer_rules(question)
    # Every mode exposes the same stage content to the model. Compression may
    # change the delivered text, but no mode appends full graph state to it.
    reasoning_input = (
        _latest_message_payload(state, sender="retriever", receiver="reasoner")
        if state["mode"] == "text" else evidence_semantic["reasoning_input"]
    )
    if not reasoning_input:
        raise ValueError("Missing Retriever -> Reasoner content")
    delivered_object = decode_memory_object(reasoning_input)
    model_evidence = [item["text"] for item in delivered_object["evidence"]] if delivered_object else []
    history_qa = delivered_object["history_qa"] if delivered_object else []
    delivered_rules = delivered_object["experience"] if delivered_object else []
    print(
        "[reasoner] "
        f"task chars={len(task_text)} evidence_count={len(evidence)} "
        f"memory_hits={len(memory_hits)} "
        f"memory_context chars={len(json.dumps(memory_context, ensure_ascii=False))}",
        flush=True,
    )
    latent_alignment_score = None
    def usable_vector(payload: dict) -> list[float]:
        content_type = str(payload.get("content_type") or "")
        if content_type == "latent.llm_hidden_state":
            return [float(value) for value in payload.get("pooled_vector", [])]
        provider = str(payload.get("embedding_provider") or "").lower()
        if content_type == "latent.embedding" and provider not in {"", "hash", "hash-v1"}:
            return [float(value) for value in payload.get("vector", [])]
        return []

    plan_vector = usable_vector(plan_state_payload)
    evidence_vector = usable_vector(evidence_state_payload)
    if plan_vector and evidence_vector and len(plan_vector) == len(evidence_vector):
        latent_alignment_score = cosine_similarity(plan_vector, evidence_vector)
        answer_rules.append(
            "Latent plan-evidence alignment is "
            f"{latent_alignment_score:.3f}; keep the answer grounded in the retrieved evidence."
        )

    client = get_llm_client("reasoner")
    llm_started = time.perf_counter()
    previous_capture_hidden = os.environ.get("LOCAL_CAPTURE_HIDDEN")
    if state["mode"] == "hidden":
        os.environ["LOCAL_CAPTURE_HIDDEN"] = "1"
    try:
        result = client.reason(
            task=task_text,
            plan=[],
            evidence=model_evidence,
            memory_hits=0,
            memory_hit_details=[],
            experience_rules=delivered_rules,
            history_qa=history_qa,
            delivered_context="" if delivered_object else reasoning_input,
            answer_rules=answer_rules,
            mode="stage_handoff",
        )
    finally:
        if state["mode"] == "hidden":
            if previous_capture_hidden is None:
                os.environ.pop("LOCAL_CAPTURE_HIDDEN", None)
            else:
                os.environ["LOCAL_CAPTURE_HIDDEN"] = previous_capture_hidden
    llm_elapsed_ms = (time.perf_counter() - llm_started) * 1000
    hidden_result = None
    hidden_tensor = None
    if state["mode"] == "hidden":
        consume_hidden = getattr(client, "consume_last_hidden_state", None)
        if callable(consume_hidden):
            hidden_result, hidden_tensor = consume_hidden()
    cognitive_gap = make_cognitive_gap(
        agent="reasoner",
        known=["task", "plan", "retrieved_evidence", "memory_hit_count"],
        unknown=[] if memory_hits else ["reusable_memory"],
        needed=["summary_generation"],
        gap_type="memory_grounded_reasoning" if memory_hits else "no_memory_hit",
        confidence=0.82 if memory_hits else 0.7,
    )
    answer_ref = ""
    answer_content = {key: result.get(key) for key in (
        "answer_candidate", "reasoning_summary", "limitations"
    )}
    latent_states = state.get("latent_states", {})
    if state["mode"] in {"protocol", "hidden"}:
        answer_text = (
            f"answer_candidate: {result.get('answer_candidate', '')}\n"
            f"reasoning_summary: {result.get('reasoning_summary', '')}\n"
            f"memory_refs: {[hit.get('memory_id') for hit in memory_hits]}\n"
            f"experience_rule_refs: {[rule.get('memory_id') for rule in experience_rules]}\n"
            f"memory_context_count: {len(memory_context)}\n"
            f"limitations: {result.get('limitations', [])}"
        )
        if state["mode"] == "hidden":
            answer_ref, latent_states, metrics = put_hidden_state(
                {**state, "metrics": metrics},
                producer="reasoner",
                topic=task_text,
                text=answer_text,
                consumers=["summarizer"],
                hidden_result=hidden_result,
                hidden_tensor=hidden_tensor,
                semantic_payload=answer_content,
            )
        else:
            answer_ref, latent_states, metrics = put_latent_state(
                {**state, "metrics": metrics},
                producer="reasoner",
                topic=task_text,
                text=answer_text,
                consumers=["summarizer"],
                semantic_payload=answer_content,
            )

    execution_request = _build_codeact_execution_request(state)

    if state["mode"] == "text":
        if execution_request:
            payload = (
                "ReasonerAgent prepared a restricted Python CodeAct execution request. "
                "The code comes from deterministic task metadata reference_code, and ExecutorAgent "
                "must run AST safety checks before executing it. "
                f"Allowed imports: {execution_request.get('allowed_imports', [])}."
            )
            if autoform_enabled():
                payload, _autoform_audit = generate_autoform_message(
                    sender="reasoner",
                    receiver="executor",
                    action="execute",
                    source_message=payload,
                )
        else:
            payload = (
                f"Answer candidate: {result.get('answer_candidate', '')}\n"
                f"Reasoning summary: {result.get('reasoning_summary', '')}\n"
                f"Limitations: {json.dumps(result.get('limitations', []), ensure_ascii=False)}"
            )
            if autoform_enabled():
                payload, _autoform_audit = generate_autoform_message(
                    sender="reasoner",
                    receiver="summarizer",
                    action="summarize",
                    source_message=payload,
                )
                payload = encode_answer_handoff(
                    result.get("answer_candidate", ""),
                    payload,
                    "autoform",
                )
    else:
        if execution_request:
            payload = make_compact_packet(
                sender="reasoner",
                receiver="executor",
                action="execute",
                content_type="protocol.execution_request",
                params={
                    "language": execution_request["language"],
                    "allowed_imports": execution_request.get("allowed_imports", []),
                    "timeout_ms": execution_request.get("timeout_ms", 3000),
                    "code_source": execution_request.get("code_source", "reference_code"),
                },
                result={"needs_tool": True},
                state_refs=[ref for ref in [plan_ref, evidence_ref, answer_ref] if ref],
                memory_refs=[hit.get("memory_id") for hit in memory_hits],
                capability_required=["tool_execution"],
                cognitive_gap=cognitive_gap,
                confidence=0.86,
            )
        else:
            payload = make_compact_packet(
                sender="reasoner",
                receiver="summarizer",
                action="summarize",
                content_type="hidden_answer_ref" if state["mode"] == "hidden" else "answer_ref",
                state_refs=[answer_ref],
                capability_required=["summarize", "memory_write"],
            )

    receiver = "executor" if execution_request else "summarizer"
    action = "execute" if execution_request else "summarize"
    messages, metrics = record_message(
        state,
        sender="reasoner",
        receiver=receiver,
        action=action,
        payload=payload,
        metrics=metrics,
    )
    metrics = dict(metrics)
    metrics["reasoner_llm_ms"] = metrics.get("reasoner_llm_ms", 0.0) + llm_elapsed_ms
    evidence_used_memory_ids = _unique_memory_ids(
        [hit for hit in memory_hits if str(hit.get("memory_type", "evidence") or "evidence") != "experience"]
    )
    experience_used_memory_ids = _unique_memory_ids(experience_rules)
    memory_used_ids = _unique_memory_ids([*memory_hits, *experience_rules])
    return {
        "evidence": evidence,
        "memory_hits": memory_hits,
        "experience_hits": experience_hits,
        "experience_rules": experience_rules,
        "memory_context": memory_context,
        "latent_states": latent_states,
        "state_refs": {**state.get("state_refs", {}), **({"answer": answer_ref} if answer_ref else {})},
        "execution_result": {
            **result,
            "execution_mode": "model_reasoning",
            "retrieval_mode": "hypergraph_memory" if memory_hits else "retrieved_no_hit",
            "memory_hit_details": memory_context,
            "memory_context": memory_context,
            "experience_rules": experience_rules,
            "answer_rules": answer_rules,
            "experience_used_count": len(experience_rules),
            "memory_used_ids": memory_used_ids,
            "evidence_used_memory_ids": evidence_used_memory_ids,
            "experience_used_memory_ids": experience_used_memory_ids,
            "plan_ref": plan_ref,
            "evidence_ref": evidence_ref,
            "answer_ref": answer_ref,
            "latent_alignment_score": latent_alignment_score,
            **({"execution_request": execution_request} if execution_request else {}),
        },
        "messages": messages,
        "metrics": metrics,
        "agent_outputs": [
            *state.get("agent_outputs", []),
            {
                "agent": "ReasonerAgent",
                "handoff_parse_status": "structured" if delivered_object else "compressed_or_legacy_text",
                "llm_call": dict(getattr(client, "last_generation_metrics", {})),
                "raw_output": result.get("raw_model_output") or result.get("answer_candidate", ""),
                "plan_state_resolved": bool(plan_state_payload),
                "evidence_state_resolved": bool(evidence_state_payload),
                "latent_alignment_score": latent_alignment_score,
            },
        ],
        "cognitive_gaps": [*state["cognitive_gaps"], cognitive_gap],
    }
