from __future__ import annotations

from typing import Any


AGENT_CAPABILITIES: dict[str, dict[str, Any]] = {
    "planner": {
        "agent": "planner",
        "display_name": "PlannerAgent",
        "capabilities": [
            "task_decomposition",
            "capability_routing",
            "cognitive_gap_detection",
        ],
        "input_schema": "task.text",
        "output_schema": "plan.steps",
    },
    "retriever": {
        "agent": "retriever",
        "display_name": "RetrieverAgent",
        "capabilities": [
            "memory_recall",
            "evidence_search",
            "latent_state_write",
        ],
        "input_schema": "protocol.intent",
        "output_schema": "evidence.refs",
    },
    "reasoner": {
        "agent": "reasoner",
        "display_name": "ReasonerAgent",
        "capabilities": [
            "model_reasoning",
            "answer_candidate_generation",
            "limitation_reporting",
        ],
        "input_schema": "task.plan",
        "output_schema": "answer.candidate",
    },
    "executor": {
        "agent": "executor",
        "display_name": "ExecutorAgent",
        "capabilities": [
            "tool_execution",
            "codeact_simulation",
            "latent_state_read",
        ],
        "input_schema": "evidence.refs",
        "output_schema": "execution.stats",
    },
    "summarizer": {
        "agent": "summarizer",
        "display_name": "SummarizerAgent",
        "capabilities": [
            "summary_generation",
            "memory_write",
            "cognitive_gap_closure",
        ],
        "input_schema": "execution.stats",
        "output_schema": "summary.text",
    },
}


def get_agent_capabilities(agent: str) -> list[str]:
    """返回指定 Agent 的能力列表，用于协议握手和能力发现。"""

    return list(AGENT_CAPABILITIES[agent]["capabilities"])


def build_capability_registry() -> dict[str, dict[str, Any]]:
    """创建运行期能力注册表，后续可替换为动态握手结果。"""

    return {agent: dict(profile) for agent, profile in AGENT_CAPABILITIES.items()}


def find_agents_by_capability(capability: str) -> list[str]:
    """根据能力反查可承接该动作的 Agent。"""

    return [
        agent
        for agent, profile in AGENT_CAPABILITIES.items()
        if capability in profile["capabilities"]
    ]
