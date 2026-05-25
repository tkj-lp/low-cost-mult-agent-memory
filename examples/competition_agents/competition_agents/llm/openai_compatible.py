from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


def _loads_json_object(text: str) -> dict[str, Any] | None:
    """解析模型返回的 JSON 对象，兼容 ```json 包裹和前后说明文本。"""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
    return None


class OpenAICompatibleClient:
    """只使用 Python 标准库实现的 OpenAI-compatible 聊天客户端。"""

    def __init__(self) -> None:
        # 密钥和模型配置必须放在环境变量里，不要写进源码。
        self.api_key = os.environ["LLM_API_KEY"]
        self.base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        if self.api_key in {"", "replace_with_your_api_key", "your_api_key", "你的 key"}:
            raise RuntimeError(
                "LLM_API_KEY is not configured. Edit examples/competition_agents/.env.local "
                "and replace the placeholder with your real API key."
            )

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
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LLM API request failed with HTTP {exc.code}: {exc.reason}. "
                f"Response body: {error_body}"
            ) from exc
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
        raw = self.generate(prompt)
        lines = [line.strip("- 0123456789.\t") for line in raw.splitlines()]
        steps = [line for line in lines if line]
        return steps[:4] or [
            "identify reusable memory",
            "retrieve supporting evidence",
            "execute a tool step",
            "summarize and store memory",
        ]

    def plan_with_raw(self, task: str) -> tuple[list[str], str]:
        prompt = (
            "You are PlannerAgent in a multi-agent competition prototype. "
            "Return exactly four short English task steps, one per line. "
            f"Task: {task}"
        )
        raw = self.generate(prompt)
        lines = [line.strip("- 0123456789.\t") for line in raw.splitlines()]
        steps = [line for line in lines if line]
        return (
            steps[:4]
            or [
                "identify reusable memory",
                "retrieve supporting evidence",
                "execute a tool step",
                "summarize and store memory",
            ],
            raw,
        )

    def reason(
        self,
        *,
        task: str,
        plan: list[str],
        evidence: list[str],
        memory_hits: int,
        mode: str,
    ) -> dict[str, Any]:
        # Reasoner 是主答案生成者；没有检索证据时也必须直接回答任务。
        prompt = (
            "You are ReasonerAgent in a multi-agent system. "
            "Answer the user's task directly in English. "
            "Do not invent retrieved documents, memory contents, sandbox execution, hidden states, or tool results. "
            "If the task contains a Context field, use only that context to answer the question. "
            "For QA tasks, make answer_candidate the shortest correct answer string, not a paragraph. "
            "If evidence is empty, solve from the task and plan only. "
            "Return JSON with keys answer_candidate, reasoning_summary, limitations.\n"
            f"Task: {task}\n"
            f"Mode: {mode}\n"
            f"Plan: {json.dumps(plan, ensure_ascii=False)}\n"
            f"Evidence: {json.dumps(evidence, ensure_ascii=False)}\n"
            f"Memory hits: {memory_hits}\n"
        )
        raw = self.generate(prompt)
        parsed = _loads_json_object(raw)
        if parsed is not None:
            limitations = parsed.get("limitations", [])
            if isinstance(limitations, str):
                limitations = [limitations]
            elif not isinstance(limitations, list):
                limitations = [str(limitations)]
            return {
                "answer_candidate": str(parsed.get("answer_candidate", "")).strip() or raw,
                "reasoning_summary": str(parsed.get("reasoning_summary", "")).strip(),
                "limitations": limitations,
                "raw_model_output": raw,
            }
        return {
            "answer_candidate": raw,
            "reasoning_summary": "Model returned free-form text instead of JSON.",
            "limitations": [],
            "raw_model_output": raw,
        }

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
        # prompt 中带上上下文，让模型回答任务本身，而不是只复述运行状态。
        prompt = (
            "You are SummarizerAgent in a LangGraph multi-agent competition prototype. "
            "Your first priority is to answer the user's task directly in English. "
            "If execution_result contains answer_candidate, use it as the primary answer and do not replace it with runtime commentary. "
            "Do not make the answer mainly about runtime metrics, memory hits, or latent state counts. "
            "Use those values only in a short final diagnostics section. "
            "Return exactly these sections:\n"
            "Answer:\n"
            "<a direct, useful answer to the task>\n\n"
            "Implementation notes:\n"
            "<briefly mention what context from plan/evidence/execution was used>\n\n"
            "Run diagnostics:\n"
            "<one or two lines with mode, memory_hits, latent_states, and execution caveats>\n"
            "Be faithful to the provided implementation state: if execution_result says sandbox_status is "
            "not_implemented, say the current run simulates CodeAct statistics rather than running a real sandbox. "
            "If the evidence says vector similarity is planned but not implemented, do not claim vector search is active. "
            "Separate implemented behavior from planned extensions.\n"
            f"Task: {task}\n"
            f"Mode: {mode}\n"
            f"Plan: {json.dumps(plan or [], ensure_ascii=False)}\n"
            f"Evidence: {json.dumps(evidence or [], ensure_ascii=False)}\n"
            f"Execution result: {json.dumps(execution_result or {}, ensure_ascii=False)}\n"
            f"Cognitive gaps: {json.dumps(cognitive_gaps or [], ensure_ascii=False)}\n"
            f"Memory hits: {memory_hits}\n"
            f"Non-text latent states exchanged: {latent_states}"
        )
        return self.generate(prompt)
