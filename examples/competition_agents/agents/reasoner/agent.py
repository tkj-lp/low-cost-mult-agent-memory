from __future__ import annotations

from competition_agents.llm import get_llm_client
from competition_agents.protocol import make_cognitive_gap, make_packet
from competition_agents.state import CompetitionState, put_latent_state, record_message


def reasoner_agent(state: CompetitionState) -> dict:
    """基于任务和 Planner 输出生成候选答案；当前不依赖检索证据。"""

    memory_hits: list[dict] = []
    evidence: list[str] = []
    plan = state.get("plan", [])
    plan_ref = state.get("state_refs", {}).get("plan", "")
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
        mode=state["mode"],
    )
    cognitive_gap = make_cognitive_gap(
        agent="reasoner",
        known=["task", "plan", "empty_memory", "no_retrieved_evidence"],
        unknown=["external_supporting_facts", "hypergraph_memory"],
        needed=["summary_generation"],
        gap_type="no_retrieval_context",
        confidence=0.7,
    )
    answer_ref = ""
    latent_states = state.get("latent_states", {})
    metrics = state["metrics"]
    if state["mode"] == "protocol":
        answer_text = (
            f"answer_candidate: {result.get('answer_candidate', '')}\n"
            f"reasoning_summary: {result.get('reasoning_summary', '')}\n"
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
            "ReasonerAgent received the task and planner steps, with memory_hits=0 "
            "and no retrieved evidence. It generated an answer_candidate directly from "
            f"the task and plan. Answer candidate: {result.get('answer_candidate', '')}"
        )
    else:
        payload = make_packet(
            sender="reasoner",
            receiver="summarizer",
            action="summarize",
            content_type="protocol.answer_candidate",
            schema_ref="competition.protocol.answer_candidate.v0.2",
            params={
                "memory_hits": 0,
                "evidence_refs": [],
                "plan_ref": plan_ref,
                "answer_ref": answer_ref,
            },
            result={
                "answer_ref": answer_ref,
            },
            state_refs=[ref for ref in [plan_ref, answer_ref] if ref],
            capability_required=["summary_generation", "memory_write"],
            receiver_capability="summary_generation",
            cognitive_gap=cognitive_gap,
            confidence=0.7,
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
        "latent_states": latent_states,
        "state_refs": {**state.get("state_refs", {}), **({"answer": answer_ref} if answer_ref else {})},
        "execution_result": {
            **result,
            "execution_mode": "model_reasoning",
            "retrieval_mode": "skipped",
            "plan_ref": plan_ref,
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
