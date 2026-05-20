from __future__ import annotations

from competition_agents.llm import get_llm_client
from competition_agents.memory import add_memory
from competition_agents.state import CompetitionState


def summarizer_agent(state: CompetitionState) -> dict:
    """生成最终答案，并沉淀为可复用共享记忆。"""

    hit_count = len(state.get("memory_hits", []))
    latent_count = len(state.get("latent_states", {}))
    # Summarizer 可单独接真实 API，其他 Agent 保持 mock 以便快速验证。
    summary = get_llm_client("summarizer").summarize(
        task=state["task"],
        memory_hits=hit_count,
        latent_states=latent_count,
    )
    # 这里是共享记忆写入路径，也是后续接超图存储的钩子。
    memory = add_memory(
        state["memory"],
        source_agent="summarizer",
        task_topic=state["task"],
        summary=summary,
        tags=["langgraph", "multi-agent", "protocol", state["mode"]],
    )
    return {"final_answer": summary, "memory": memory}
