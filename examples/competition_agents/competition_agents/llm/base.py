from __future__ import annotations

from typing import Any, Protocol


class LLMClient(Protocol):
    """比赛 Agent 使用的最小模型接口。"""

    def generate(self, prompt: str) -> str:
        """根据 prompt 生成文本。"""

    def plan(self, task: str) -> list[str]:
        """为 PlannerAgent 生成任务步骤。"""

    def plan_with_raw(self, task: str) -> tuple[list[str], str]:
        """为 PlannerAgent 生成任务步骤，并返回模型原始输出。"""

    def reason(
        self,
        *,
        task: str,
        plan: list[str],
        evidence: list[str],
        memory_hits: int,
        mode: str,
    ) -> dict[str, Any]:
        """为 ReasonerAgent 生成候选答案和推理摘要。"""

    def summarize(
        self,
        *,
        task: str,
        memory_hits: int,
        latent_states: int,
        plan: list[str] | None = None,
        evidence: list[str] | None = None,
        execution_result: dict[str, Any] | None = None,
        cognitive_gaps: list[dict[str, Any]] | None = None,
        mode: str | None = None,
    ) -> str:
        """为 SummarizerAgent 生成最终答案。"""
