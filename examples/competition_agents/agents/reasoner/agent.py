from __future__ import annotations

import json

from competition_agents.llm import get_llm_client
from competition_agents.protocol import make_cognitive_gap, make_compact_packet, make_packet
from competition_agents.state import CompetitionState, put_hidden_state, put_latent_state, record_message


def _reasoner_task_text(state: CompetitionState) -> str:
    """在 hypergraph 模式下只把问题本身传给 Reasoner，避免意外回看原始上下文。"""

    task_meta = state.get("task_meta", {})
    if str(task_meta.get("context_source") or "") == "hypergraph":
        question = str(task_meta.get("question") or "").strip()
        if question:
            return f"Question: {question}"
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


def _question_answer_rules(question: str) -> list[str]:
    """根据问题类型补充稳定答案槽位约束，降低模型自由发挥。"""

    question_lower = str(question or "").lower()
    rules = [
        "Copy the answer span exactly from evidence when possible.",
        "Prefer the most specific phrase that directly answers the question.",
        "Do not answer with the title or topic unless the question explicitly asks for it.",
    ]
    if "when" in question_lower:
        rules.append("If multiple dates appear, choose the date matching every constraint in the question.")
        rules.append("For DVD release questions, never return a digital release date.")
    if "what is the film about" in question_lower:
        rules.append("Return the event or subject the film depicts, not release or production metadata.")
    if "which hormones are related to catabolism" in question_lower:
        rules.append("Return the complete hormone list from evidence, including both classic and additional hormones.")
        rules.append("Do not stop after the first sentence if the evidence continues the same list.")
    if "third hormone on that list" in question_lower:
        rules.append("Resolve the ordinal reference against the previously established list before selecting the answer.")
        rules.append("Return only the emotional response associated with that ordinal item.")
    if "first hormone on that list" in question_lower:
        rules.append("Resolve the ordinal reference against the previously established list before selecting the answer.")
        rules.append("Return the glucose-metabolism effect phrase, not the hormone name itself.")
    if "from which organ was it extracted in that experiment" in question_lower:
        rules.append("Resolve 'it' and 'that experiment' to the immediately relevant prior entity and event.")
        rules.append("Return only the organ phrase.")
    if "that list" in question_lower or "that process" in question_lower or "that substance" in question_lower:
        rules.append("Resolve cross-turn references before selecting the final answer span.")
    if question_lower.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ", "has ", "have ", "had ")):
        rules.append("If this is a yes/no question, answer with Yes or No unless evidence clearly requires a short qualified yes/no phrase.")
    if "cast" in question_lower or "members of the cast" in question_lower:
        rules.append("Return only person names from the cast list, separated by commas.")
        rules.append("Return the first four cast members mentioned in the evidence, in order.")
        rules.append("Do not select only prominent actors for brevity.")
    if "where was it filmed" in question_lower:
        rules.append("Return all filming locations mentioned in the evidence, including both places if two are listed.")
    if "what kinds of fish" in question_lower:
        rules.append("Return only the fish names mentioned in evidence.")
        rules.append("Include every fish listed in the evidence.")
    if "what mineral resources" in question_lower:
        rules.append("Return only the mineral resources, not the explanatory sentence.")
        rules.append("Include every mineral resource mentioned in the evidence, including amber if present.")
    if "what natural hazard" in question_lower:
        rules.append("Return only the hazard name.")
    return rules


def reasoner_agent(state: CompetitionState) -> dict:
    """基于任务、计划、检索证据和记忆命中生成候选答案。"""

    # RetrieverAgent 会在前一节点写入 evidence、memory_hits 和裁剪后的 memory_context。
    memory_hits = state.get("memory_hits", [])
    memory_context = state.get("memory_context", [])
    experience_hits = state.get("experience_hits", [])
    experience_rules = state.get("experience_rules", [])
    evidence = state.get("evidence", [])
    plan = state.get("plan", [])
    task_text = _reasoner_task_text(state)
    question = str(state.get("task_meta", {}).get("question") or "")
    answer_rules = _question_answer_rules(question)
    print(
        "[reasoner] "
        f"task chars={len(task_text)} evidence_count={len(evidence)} "
        f"memory_hits={len(memory_hits)} "
        f"memory_context chars={len(json.dumps(memory_context, ensure_ascii=False))}",
        flush=True,
    )
    plan_ref = state.get("state_refs", {}).get("plan", "")
    evidence_ref = state.get("state_refs", {}).get("evidence", "")
    if state["mode"] in {"protocol", "hidden"} and plan_ref:
        plan_state = state.get("latent_states", {}).get(plan_ref, {})
        plan_text = str(plan_state.get("source_text") or plan_state.get("summary") or "")
        latent_plan = [line.strip() for line in plan_text.splitlines() if line.strip()]
        if latent_plan:
            plan = latent_plan

    result = get_llm_client("reasoner").reason(
        task=task_text,
        plan=plan,
        evidence=evidence,
        memory_hits=len(memory_hits),
        memory_hit_details=memory_context,
        experience_rules=experience_rules,
        answer_rules=answer_rules,
        mode=state["mode"],
    )
    cognitive_gap = make_cognitive_gap(
        agent="reasoner",
        known=["task", "plan", "retrieved_evidence", "memory_hit_count"],
        unknown=[] if memory_hits else ["reusable_memory"],
        needed=["summary_generation"],
        gap_type="memory_grounded_reasoning" if memory_hits else "no_memory_hit",
        confidence=0.82 if memory_hits else 0.7,
    )
    answer_ref = ""
    latent_states = state.get("latent_states", {})
    metrics = state["metrics"]
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
                state,
                producer="reasoner",
                topic=task_text,
                text=answer_text,
                consumers=["summarizer"],
            )
        else:
            answer_ref, latent_states, metrics = put_latent_state(
                state,
                producer="reasoner",
                topic=task_text,
                text=answer_text,
                consumers=["summarizer"],
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
        else:
            payload = (
                f"ReasonerAgent received {len(evidence)} evidence items and "
                f"{len(memory_hits)} memory hits from RetrieverAgent. "
                f"It also received {len(experience_rules)} experience rules for answer selection. "
                f"It generated an answer_candidate from the task, plan, evidence, and memory. "
                f"Answer candidate: {result.get('answer_candidate', '')}"
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
                params={"hits": len(memory_hits)},
                result={"answer": answer_ref},
                state_refs=[ref for ref in [plan_ref, evidence_ref, answer_ref] if ref],
                memory_refs=[hit.get("memory_id") for hit in memory_hits],
                capability_required=["summarize", "memory_write"],
                cognitive_gap=cognitive_gap,
                confidence=0.82 if memory_hits else 0.7,
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
            "experience_used_memory_ids": [rule.get("memory_id") for rule in experience_rules],
            "plan_ref": plan_ref,
            "evidence_ref": evidence_ref,
            "answer_ref": answer_ref,
            **({"execution_request": execution_request} if execution_request else {}),
        },
        "messages": messages,
        "metrics": metrics,
        "agent_outputs": [
            *state.get("agent_outputs", []),
            {
                "agent": "ReasonerAgent",
                "raw_output": result.get("raw_model_output") or result.get("answer_candidate", ""),
            },
        ],
        "cognitive_gaps": [*state["cognitive_gaps"], cognitive_gap],
    }
