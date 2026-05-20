from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class OpenAICompatibleClient:
    """只使用 Python 标准库实现的 OpenAI-compatible 聊天客户端。"""

    def __init__(self) -> None:
        # 密钥和模型配置必须放在环境变量里，不要写进源码。
        self.api_key = os.environ["LLM_API_KEY"]
        self.base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    def generate(self, prompt: str) -> str:
        # 使用 OpenAI-compatible 服务通用的 chat/completions 请求格式。
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            # 用 urllib 避免为原型额外引入第三方 HTTP 依赖。
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM API request failed: {exc}") from exc

        return body["choices"][0]["message"]["content"].strip()

    def plan(self, task: str) -> list[str]:
        # 要求每行一个步骤，便于简单稳定地解析。
        prompt = (
            "You are PlannerAgent in a multi-agent competition prototype. "
            "Return exactly four short English task steps, one per line. "
            f"Task: {task}"
        )
        lines = [line.strip("- 0123456789.\t") for line in self.generate(prompt).splitlines()]
        steps = [line for line in lines if line]
        return steps[:4] or [
            "identify reusable memory",
            "retrieve supporting evidence",
            "execute a tool step",
            "summarize and store memory",
        ]

    def summarize(self, *, task: str, memory_hits: int, latent_states: int) -> str:
        # prompt 中带上指标，让生成内容体现记忆和状态复用。
        prompt = (
            "You are SummarizerAgent. Write one concise English sentence for the final result. "
            f"Task: {task}\n"
            f"Memory hits: {memory_hits}\n"
            f"Non-text latent states exchanged: {latent_states}"
        )
        return self.generate(prompt)
