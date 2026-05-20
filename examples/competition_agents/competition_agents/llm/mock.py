from __future__ import annotations


class MockLLMClient:
    """用于演示和队友开发的确定性本地模型替身。"""

    def generate(self, prompt: str) -> str:
        # 返回可预测结果，保证实验输出稳定。
        return f"Mock model response for: {prompt[:120]}"

    def plan(self, task: str) -> list[str]:
        # mock 规划保持简短稳定，方便复现实验指标。
        return [
            "identify reusable memory",
            "retrieve supporting evidence",
            "execute a tool step",
            "summarize and store memory",
        ]

    def summarize(self, *, task: str, memory_hits: int, latent_states: int) -> str:
        # mock 总结保持和真实 API 总结相近的字段含义。
        return (
            f"Task '{task}' was completed by Planner, Retriever, Executor, "
            f"and Summarizer. The run reused {memory_hits} memories and exchanged "
            f"{latent_states} non-text latent states."
        )
