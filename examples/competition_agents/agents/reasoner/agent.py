from __future__ import annotations

from competition_agents.llm import get_llm_client
from competition_agents.protocol import make_cognitive_gap, make_compact_packet, make_packet
from competition_agents.state import CompetitionState, put_latent_state, record_message


def reasoner_agent(state: CompetitionState) -> dict:
    """基于任务、计划、检索证据和记忆命中生成候选答案。"""

    # RetrieverAgent 会在前一节点写入 evidence、memory_hits 和裁剪后的 memory_context。
    memory_hits = state.get("memory_hits", [])
    memory_context = state.get("memory_context", [])
    evidence = state.get("evidence", [])
    plan = state.get("plan", [])
    plan_ref = state.get("state_refs", {}).get("plan", "")
    evidence_ref = state.get("state_refs", {}).get("evidence", "")
    if state["mode"] == "protocol" and plan_ref:
        plan_state = state.get("latent_states", {}).get(plan_ref, {})
        plan_text = str(plan_state.get("source_text") or plan_state.get("summary") or "")
        latent_plan = [line.strip() for line in plan_text.splitlines() if line.strip()]
        if latent_plan:
            plan = latent_plan

    result = get_llm_client("reasoner").reason(
        task=state["task"],
        plan=plan,
        evidence=evidence,
        memory_hits=len(memory_hits),
        memory_hit_details=memory_context,
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
    if state["mode"] == "protocol":
        answer_text = (
            f"answer_candidate: {result.get('answer_candidate', '')}\n"
            f"reasoning_summary: {result.get('reasoning_summary', '')}\n"
            f"memory_refs: {[hit.get('memory_id') for hit in memory_hits]}\n"
            f"memory_context_count: {len(memory_context)}\n"
            f"limitations: {result.get('limitations', [])}"
        )
        answer_ref, latent_states, metrics = put_latent_state(
            state,
            producer="reasoner",
            topic=state["task"],
            text=answer_text,
            consumers=["summarizer"],
        )

    if state["mode"] == "text":
        payload = (
            f"ReasonerAgent received {len(evidence)} evidence items and "
            f"{len(memory_hits)} memory hits from RetrieverAgent. "
            f"It generated an answer_candidate from the task, plan, evidence, and memory. "
            f"Answer candidate: {result.get('answer_candidate', '')}"
        )
    else:
        payload = make_compact_packet(
            sender="reasoner",
            receiver="summarizer",
            action="summarize",
            content_type="answer_ref",
            params={"hits": len(memory_hits)},
            result={"answer": answer_ref},
            state_refs=[ref for ref in [plan_ref, evidence_ref, answer_ref] if ref],
            memory_refs=[hit.get("memory_id") for hit in memory_hits],
            capability_required=["summarize", "memory_write"],
            cognitive_gap=cognitive_gap,
            confidence=0.82 if memory_hits else 0.7,
        )

    messages, metrics = record_message(
        state,
        sender="reasoner",
        receiver="summarizer",
        action="summarize",
        payload=payload,
        metrics=metrics,
    )
    return {
        "evidence": evidence,
        "memory_hits": memory_hits,
        "memory_context": memory_context,
        "latent_states": latent_states,
        "state_refs": {**state.get("state_refs", {}), **({"answer": answer_ref} if answer_ref else {})},
        "execution_result": {
            **result,
            "execution_mode": "model_reasoning",
            "retrieval_mode": "hypergraph_memory" if memory_hits else "retrieved_no_hit",
            "memory_hit_details": memory_context,
            "memory_context": memory_context,
            "plan_ref": plan_ref,
            "evidence_ref": evidence_ref,
            "answer_ref": answer_ref,
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
