from __future__ import annotations

from competition_agents.protocol import make_cognitive_gap, make_packet
from competition_agents.state import CompetitionState, record_message


def executor_agent(state: CompetitionState) -> dict:
    """执行一个确定性的 CodeAct 风格工具步骤。"""

    # 当前用确定性统计结果占位，后续可替换成真实 CodeAct 沙箱。
    result = {
        "message_count_so_far": state["metrics"]["message_count"] + 1,
        "memory_hit_count": len(state.get("memory_hits", [])),
        "latent_state_count": len(state.get("latent_states", {})),
        "execution_mode": "simulated_codeact_stats",
        "sandbox_status": "not_implemented",
    }
    cognitive_gap = make_cognitive_gap(
        agent="executor",
        known=["execution_metrics", "latent_state_count", "memory_hit_count"],
        unknown=["real_sandbox_result"],
        needed=["summary_generation", "memory_write"],
        gap_type="missing_tool",
        confidence=0.81,
    )

    if state["mode"] == "text":
        # baseline 模式用较长自然语言解释工具执行结果。
        payload = (
            "ExecutorAgent ran a lightweight CodeAct statistics step. "
            f"The current collaboration has {result['message_count_so_far']} messages, "
            f"{result['memory_hit_count']} memory hits, and "
            f"{result['latent_state_count']} non-text state transfers. "
            "For text-mode collaboration, the executor forwards a verbose explanation of "
            "the tool result, how it interpreted previous messages, and why the summarizer "
            "should write a reusable memory. This simulates repeated natural-language "
            "state reconstruction between agents. The executor also states that it supports "
            "tool execution, CodeAct simulation, and latent state reading. Its remaining "
            "cognitive gap is the absence of a real sandbox result, so the summarizer should "
            "describe the current result as a deterministic CodeAct-style statistics step."
        )
    else:
        # protocol 模式只返回 SummarizerAgent 需要的结构化结果字段。
        payload = make_packet(
            sender="executor",
            receiver="summarizer",
            action="summarize",
            content_type="protocol.execution_result",
            schema_ref="competition.protocol.execution_result.v0.2",
            params={"format": "brief_report"},
            result=result,
            capability_required=["summary_generation", "memory_write"],
            receiver_capability="summary_generation",
            cognitive_gap=cognitive_gap,
            confidence=0.81,
        )

    # 保存 executor 到 summarizer 的消息，并统计字符开销。
    messages, metrics = record_message(
        state,
        sender="executor",
        receiver="summarizer",
        action="summarize",
        payload=payload,
    )
    return {
        "execution_result": result,
        "messages": messages,
        "metrics": metrics,
        "cognitive_gaps": [*state["cognitive_gaps"], cognitive_gap],
    }
