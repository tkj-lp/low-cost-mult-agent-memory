from __future__ import annotations

import json

from competition_agents.llm import get_llm_client
from competition_agents.protocol import make_cognitive_gap, make_compact_packet, make_packet
from competition_agents.state import CompetitionState, put_hidden_state, put_latent_state, record_message


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
    if hasattr(client, "plan_with_raw"):
        plan, raw_output = client.plan_with_raw(task)
    else:
        plan = client.plan(task)
        raw_output = "\n".join(plan)
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
    else:
        # protocol 模式只传紧凑的动作参数，不传长段自然语言。
        payload = make_compact_packet(
            sender="planner",
            receiver="retriever",
            action="retrieve",
            content_type="hidden_state_ref" if state["mode"] == "hidden" else "state_ref",
            params={"plan": plan_ref},
            result={"next": "retriever"},
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
    return {
        "plan": plan,
        "latent_states": latent_states,
        "state_refs": {**state.get("state_refs", {}), **({"plan": plan_ref} if plan_ref else {})},
        "messages": messages,
        "metrics": metrics,
        "agent_outputs": [
            *state.get("agent_outputs", []),
            {"agent": "PlannerAgent", "raw_output": raw_output},
        ],
        "cognitive_gaps": [*state["cognitive_gaps"], cognitive_gap],
    }
