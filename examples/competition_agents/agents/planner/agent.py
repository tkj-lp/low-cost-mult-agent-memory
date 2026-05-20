from __future__ import annotations

from competition_agents.llm import get_llm_client
from competition_agents.protocol import make_cognitive_gap, make_packet
from competition_agents.state import CompetitionState, record_message


def planner_agent(state: CompetitionState) -> dict:
    """规划任务，并向 RetrieverAgent 发起检索请求。"""

    task = state["task"]
    # Planner 可通过 LLM_PLANNER_PROVIDER 切换 mock 或真实 API。
    plan = get_llm_client("planner").plan(task)
    cognitive_gap = make_cognitive_gap(
        agent="planner",
        known=["task_goal", "available_agent_capabilities"],
        unknown=["historical_evidence", "reusable_latent_state"],
        needed=["memory_recall", "evidence_search", "latent_state_write"],
        gap_type="missing_context",
        confidence=0.72,
    )

    if state["mode"] == "text":
        # text 模式故意重复上下文，用来模拟传统自然语言协作的高开销。
        payload = (
            f"PlannerAgent received task: {task}. "
            f"It creates a four-step plan: {', '.join(plan)}. "
            "RetrieverAgent should search historical memory and evidence before execution. "
            "For text-mode collaboration, the planner forwards the full task background, "
            "role description, expected output format, and repeated natural-language context "
            "so the next agent can reconstruct the internal planning state without reading "
            f"structured state references. The planner says its capabilities are task decomposition, "
            "capability routing, and cognitive gap detection. It also explains its cognitive gap: "
            "it already knows the task goal and available agent capabilities, but it does not yet "
            "know historical evidence or reusable latent states, so it needs memory recall, evidence "
            "search, and latent state writing. Full task background repeated for baseline cost: "
            f"{task}."
        )
    else:
        # protocol 模式只传紧凑的动作参数，不传长段自然语言。
        payload = make_packet(
            sender="planner",
            receiver="retriever",
            action="retrieve",
            content_type="protocol.intent",
            schema_ref="competition.protocol.intent.v0.2",
            params={"query": task, "top_k": 3},
            capability_required=["memory_recall", "evidence_search"],
            receiver_capability="memory_recall",
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
    )
    return {
        "plan": plan,
        "messages": messages,
        "metrics": metrics,
        "cognitive_gaps": [*state["cognitive_gaps"], cognitive_gap],
    }
