from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """比赛 Agent 使用的最小模型接口。"""

    def generate(self, prompt: str) -> str:
        """根据 prompt 生成文本。"""

    def plan(self, task: str) -> list[str]:
        """为 PlannerAgent 生成任务步骤。"""

    def summarize(self, *, task: str, memory_hits: int, latent_states: int) -> str:
        """为 SummarizerAgent 生成最终答案。"""
