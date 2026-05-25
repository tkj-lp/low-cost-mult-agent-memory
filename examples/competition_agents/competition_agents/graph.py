from __future__ import annotations

from agents.planner import planner_agent
from agents.reasoner import reasoner_agent
from agents.retriever import retriever_agent
from agents.summarizer import summarizer_agent
from competition_agents.state import CompetitionState


class SequentialCompetitionGraph:
    """在 LangGraph 依赖未就绪时使用的简易顺序运行器。"""

    def invoke(self, state: CompetitionState) -> CompetitionState:
        # 每个 Agent 读取共享状态，并返回自己负责的局部更新。
        for agent in (planner_agent, retriever_agent, reasoner_agent, summarizer_agent):
            state = {**state, **agent(state)}
        return state


def build_graph():
    """构建 LangGraph 工作流；依赖失败时回退到本地顺序运行器。"""

    try:
        from langgraph.graph import END, START, StateGraph
    except (ImportError, TypeError):
        # 本机 Python 依赖还没修好时，仍然让比赛 demo 可以运行。
        # Agent 代码和 state 接口保持不变。
        return SequentialCompetitionGraph()

    graph = StateGraph(CompetitionState)
    # 每个 Agent 注册成一个图节点。
    graph.add_node("planner", planner_agent)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("reasoner", reasoner_agent)
    graph.add_node("summarizer", summarizer_agent)

    # 固定流水线：规划 -> 检索/状态写入 -> 推理 -> 总结/记忆沉淀。
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "reasoner")
    graph.add_edge("reasoner", "summarizer")
    graph.add_edge("summarizer", END)
    return graph.compile()
