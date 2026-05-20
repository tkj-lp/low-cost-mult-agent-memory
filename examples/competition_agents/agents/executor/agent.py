from __future__ import annotations

from competition_agents.protocol import make_packet
from competition_agents.state import CompetitionState, record_message


def executor_agent(state: CompetitionState) -> dict:
    """执行一个确定性的 CodeAct 风格工具步骤。"""

    # 当前用确定性统计结果占位，后续可替换成真实 CodeAct 沙箱。
    result = {
        "message_count_so_far": state["metrics"]["message_count"],
        "memory_hit_count": len(state.get("memory_hits", [])),
        "latent_state_count": len(state.get("latent_states", {})),
    }

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
            "state reconstruction between agents."
        )
    else:
        # protocol 模式只返回 SummarizerAgent 需要的结构化结果字段。
        payload = make_packet(
            sender="executor",
            receiver="summarizer",
            action="summarize",
            params={"format": "brief_report"},
            result=result,
            capability_required=["summary_generation", "memory_write"],
        )

    # 保存 executor 到 summarizer 的消息，并统计字符开销。
    messages, metrics = record_message(
        state,
        sender="executor",
        receiver="summarizer",
        action="summarize",
        payload=payload,
    )
    return {"execution_result": result, "messages": messages, "metrics": metrics}
