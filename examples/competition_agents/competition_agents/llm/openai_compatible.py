from __future__ import annotations

import json
import os
import re
import time
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


def _normalize_answer_span(text: str) -> str:
    """Normalize whitespace without injecting dataset-specific answer content."""

    return " ".join(str(text or "").strip().split())


def _extract_question_from_task(task: str) -> str:
    """尽量从任务文本中恢复当前问题，供答案后处理使用。"""

    text = str(task or "").strip()
    if not text:
        return ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("question:"):
            return stripped.split(":", 1)[1].strip()
    return text


def _postprocess_answer_candidate(question: str, evidence: list[str], answer_candidate: str) -> str:
    """Apply format-only cleanup without encoding benchmark answers."""

    del question, evidence
    cleaned = _normalize_answer_span(answer_candidate)
    cleaned = re.sub(
        r"^(?:answer|final answer|answer_candidate)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _extract_answer_line(text: str) -> str:
    """Extract one answer line from small-model output without adding facts."""

    cleaned = str(text or "").strip().strip("`").strip()
    if not cleaned:
        return ""
    parsed = _loads_json_object(cleaned)
    if parsed is not None and parsed.get("answer_candidate"):
        return str(parsed["answer_candidate"]).strip()
    for line in cleaned.splitlines():
        candidate = line.strip().strip("- ")
        if not candidate:
            continue
        match = re.match(
            r"^(?:answer|final answer|answer_candidate)\s*:\s*(.+)$",
            candidate,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
    return next((line.strip() for line in cleaned.splitlines() if line.strip()), cleaned)


def _content_tokens(text: str) -> set[str]:
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
        "was", "were", "with",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if token not in stopwords
    }


def _resolved_topic(task: str) -> str:
    for line in str(task or "").splitlines():
        if line.lower().startswith("resolved topic:"):
            return line.split(":", 1)[1].strip()
    return ""


def _location_relation_values(evidence: list[str]) -> list[str]:
    values: list[str] = []
    for item in evidence:
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", str(item or "")):
            for pattern in (
                r"\b(?:from|born(?:\s+and\s+raised)?\s+in|raised\s+in|originated\s+in|headquartered\s+in)\s+([^,.;]+)",
                r"\b(?:located|situated|found|based|occurs?)\s+(?:in|at|on|near|within)\s+([^,.;]+)",
                r"\b(?:is|are|was|were)\s+(?:in|at|on|near|within)\s+([^,.;]+)",
                r"\b(?:is|are|was|were)\s+(?:an?\s+)?[^,.;]{0,90}?\s+in\s+([^,.;]+)",
            ):
                match = re.search(pattern, sentence, flags=re.IGNORECASE)
                if match:
                    values.append(match.group(1).strip(" ,;."))
    return values


def _answer_repair_reason(
    *,
    task: str,
    evidence: list[str],
    answer: str,
) -> str:
    """Return a portable validation failure, or an empty string when valid."""

    cleaned = _normalize_answer_span(answer)
    if not cleaned:
        return "the answer is empty"
    topic = _normalize_answer_span(_resolved_topic(task)).lower().strip('"')
    candidate = cleaned.lower().strip('"')
    if topic and _content_tokens(candidate) == _content_tokens(topic):
        return "the answer merely repeats the resolved topic"
    if len(cleaned.split()) > 48:
        return "the answer is an explanation instead of a concise answer"

    evidence_text = " ".join(str(item or "") for item in evidence).lower()
    candidate_tokens = _content_tokens(cleaned)
    evidence_tokens = _content_tokens(evidence_text)
    if candidate_tokens and candidate not in {"yes", "yes.", "no", "no."}:
        grounding = len(candidate_tokens & evidence_tokens) / len(candidate_tokens)
        if grounding < 0.6:
            return "most answer terms are not grounded in evidence"

    question = _extract_question_from_task(task).lower()
    question_tokens = _content_tokens(question)
    topic_tokens = _content_tokens(_resolved_topic(task))
    requested_tokens = question_tokens - topic_tokens
    if candidate_tokens and len(candidate_tokens) <= 2 and candidate_tokens <= requested_tokens:
        return "the answer only repeats the requested attribute"

    list_question = bool(
        re.search(
            r"\b(list|name|mention|members|what kinds|what types?|species|examples)\b",
            question,
        )
    )
    evidence_has_explicit_list = any(
        re.search(r"\b(?:include|includes|including|such as)\b", str(item), re.IGNORECASE)
        for item in evidence
    )
    answer_has_list = cleaned.count(",") >= 1 or bool(re.search(r"\s+and\s+", cleaned.lower()))
    if list_question and evidence_has_explicit_list and not answer_has_list:
        return "the question requests explicit items but the answer is only a count or category"

    descriptive_question = bool(
        re.search(r"\b(describe|explain|tell me about|what .* like|how .* look|why)\b", question)
    )
    if descriptive_question and len(candidate_tokens) < 5:
        return "the descriptive answer omits the requested evidence details"

    yes_no_question = question.startswith(
        ("is ", "are ", "was ", "were ", "do ", "does ", "did ", "has ", "have ", "had ")
    )
    if yes_no_question and candidate.rstrip(".") in {"yes", "no"}:
        return "the yes/no answer lacks an evidence-backed qualification"
    if candidate.startswith("no") and not re.search(
        r"\b(no|not|never|none|without|lack|lacks|absent)\b",
        evidence_text,
    ):
        return "the negative answer has no negative evidence"
    if re.search(r"\b(where|which place|which country|which city)\b", question):
        relation_values = _location_relation_values(evidence)
        relation_tokens = set().union(*(_content_tokens(value) for value in relation_values)) if relation_values else set()
        if relation_tokens and not (candidate_tokens & relation_tokens):
            return "the answer is mentioned in evidence but is not a location-relation value"
    return ""


def _evidence_sentences(evidence: list[str]) -> list[str]:
    sentences: list[str] = []
    for item in evidence:
        sentences.extend(
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", str(item or ""))
            if sentence.strip()
        )
    return sentences


def _extractive_fallback(*, task: str, evidence: list[str]) -> str:
    """Extract common relation values when a small model fails validation.

    This fallback is benchmark-independent: it uses only the current question,
    resolved topic, and retrieved evidence. It never reads labels or gold answers.
    """

    question = _extract_question_from_task(task).lower()
    sentences = _evidence_sentences(evidence)
    if not sentences:
        return ""

    if re.search(r"\b(list|name|mention|members|what kinds|what types?|species|examples)\b", question):
        for sentence in sentences:
            match = re.search(
                r"\b(?:include|includes|including|such as)\s+(.+?)"
                r"(?=\s+is\s+(?:found|located|recorded|listed)\b|[.;]|$)",
                sentence,
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(1).strip(" ,;.")

    if re.search(r"\b(where|location|located|situated)\b", question):
        location_candidates = _location_relation_values(evidence)
        if location_candidates:
            return min(location_candidates, key=lambda value: (len(value.split()), len(value)))

    if re.search(r"^\s*what\s+(?:is|are|was|were)\b", question):
        for sentence in sentences:
            match = re.search(
                r"\b(?:is|are|was|were)\s+(.+?)"
                r"(?=\s+(?:found|located|which|that|who|where)\b|[.;]|$)",
                sentence,
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(1).strip(" ,;.")
    return ""


def _coerce_answer_candidate(answer_candidate: str, evidence: list[str], question: str = "") -> str:
    """Normalize model output without inserting benchmark-specific facts."""

    cleaned = " ".join(str(answer_candidate or "").strip().split())
    if not cleaned:
        return cleaned

    cleaned = _postprocess_answer_candidate(question, evidence, cleaned)
    if not cleaned:
        return cleaned

    return _normalize_answer_span(cleaned)


class OpenAICompatibleClient:
    """只使用 Python 标准库实现的 OpenAI-compatible 聊天客户端。"""

    def __init__(self) -> None:
        # 密钥和模型配置必须放在环境变量里，不要写进源码。
        self.api_key = os.environ.get("LLM_API_KEY", "")
        self.base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.request_timeout = float(os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS", "180"))
        self.max_retries = max(0, int(os.environ.get("LLM_MAX_RETRIES", "2")))
        if self.api_key in {
            "",
            "replace_with_your_api_key",
            "your_api_key",
            "你的 key",
            "sk-competition-console",
        }:
            raise RuntimeError(
                "LLM_API_KEY is not configured. Edit examples/competition_agents/.env.local "
                "and replace the placeholder with your real API key."
            )

    def _post_json(self, request: urllib.request.Request) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"LLM API request failed with HTTP {exc.code}: {exc.reason}. "
                    f"Response body: {error_body}"
                ) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2**attempt, 4))

        raise RuntimeError(f"LLM API request failed after retries: {last_error}") from last_error

    def generate(self, prompt: str) -> str:
        # 使用 OpenAI-compatible 服务通用的 chat/completions 请求格式。
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        print(f"[openai_compatible] model={self.model} prompt chars: {len(prompt)}", flush=True)
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        body = self._post_json(request)
        return body["choices"][0]["message"]["content"].strip()

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

    def plan_query(self, task: str) -> str:
        return self.generate(
            "Classify the question and plan knowledge lookup. Return JSON only: "
            '{"query_type":"standalone|conversational_reference|follow_up_fact",'
            '"reference_target":"short unresolved reference or empty",'
            '"steps":["Retrieve the required fact"]}. '
            "Choose one query_type. Use one or two brief complete retrieval goals, "
            "at most eight words each. Do not decide whether QA memory is retrieved. "
            "Do not answer the question or invent a resolved entity.\n" + task
        )

    def reason(
        self,
        *,
        task: str,
        plan: list[str],
        evidence: list[str],
        memory_hits: int,
        mode: str,
        memory_hit_details: list[dict[str, Any]] | None = None,
        experience_rules: list[dict[str, Any]] | None = None,
        answer_rules: list[str] | None = None,
        history_qa: list[dict[str, Any]] | None = None,
        delivered_context: str = "",
    ) -> dict[str, Any]:
        # Keep the generation contract deliberately small so sub-billion local
        # models spend capacity on evidence selection instead of JSON syntax.
        compact_experience = [
            {
                "solution": item.get("solution", item.get("solution_pattern", "")),
                "avoid": item.get("avoid", item.get("failure_pattern", "")),
            }
            for item in (experience_rules or [])[:2]
        ]
        history_text = "\n\n".join(
            f"[Turn {qa.get('turn', '')}]\nQuestion: {qa.get('question', '')}\n"
            f"Answer: {qa.get('answer', '')}\nVerification: {qa.get('verification_status', 'unverified')}"
            for qa in (history_qa or [])
        )
        context_text = (
            "Historical conversation (previous answers may be wrong; use to resolve references, not as external evidence):\n"
            + history_text + "\n"
        )
        if delivered_context:
            context_text += (
                "Received compressed stage content: distinguish evidence, historical QA and strategy by their labels. "
                "Missing fields are unavailable; do not invent them.\n" + delivered_context + "\n"
            )
        prompt = (
            "You are an evidence-grounded extractive QA component.\n"
            "Answer the CURRENT question, not an earlier question.\n"
            "The resolved topic and requested attribute are part of the question.\n"
            "Select the evidence sentence whose predicate matches that attribute.\n"
            "Metadata labels and section names are retrieval hints, never answers.\n"
            "Never replace current evidence with a remembered answer or general topic definition.\n"
            "Return a concise but complete answer span from evidence.\n"
            "For descriptive questions, include every directly relevant clause; for yes/no questions, add one supporting qualification.\n"
            "Do not explain your process or invent examples.\n"
            "Output exactly one line in this format: ANSWER: <answer>\n\n"
            f"Current task:\n{task}\n\n"
            + context_text + "\n"
            + f"Evidence:\n{json.dumps(evidence or [], ensure_ascii=False)}\n\n"
            f"Reusable strategy only (not facts):\n{json.dumps(compact_experience, ensure_ascii=False)}\n"
            f"Answer constraints:\n{json.dumps((answer_rules or [])[:8], ensure_ascii=False)}\n"
        )
        raw = self.generate(prompt)
        question = _extract_question_from_task(task)
        answer_candidate = _coerce_answer_candidate(
            _extract_answer_line(raw),
            evidence,
            question=question,
        )
        repair_reason = _answer_repair_reason(
            task=task,
            evidence=evidence,
            answer=answer_candidate,
        )
        repair_applied = False
        repair_failure = ""
        answer_resolution_method = "model"
        if repair_reason and evidence:
            repair_prompt = (
                "You repair an evidence-grounded QA answer.\n"
                f"The previous candidate failed because {repair_reason}.\n"
                "Extract the value requested by the CURRENT question from evidence.\n"
                "Read the sentence relation: return the value after the relation, not the subject before it.\n"
                "For a list question, copy the relevant evidence list.\n"
                "For a descriptive question, include all directly relevant evidence clauses.\n"
                "For a yes/no question, add the shortest supporting evidence clause.\n"
                "Do not output labels such as definition, location, list, flora, or fauna.\n"
                "Do not explain and do not repeat the question subject.\n"
                "Output answer words only, with no prefix.\n\n"
                f"Current task:\n{task}\n\n"
                f"Previous candidate: {answer_candidate}\n"
                + context_text
                + f"Evidence:\n{json.dumps(evidence or [], ensure_ascii=False)}"
            )
            repaired_raw = self.generate(repair_prompt)
            repaired_answer = _coerce_answer_candidate(
                _extract_answer_line(repaired_raw),
                evidence,
                question=question,
            )
            repaired_failure = _answer_repair_reason(
                task=task,
                evidence=evidence,
                answer=repaired_answer,
            )
            if repaired_answer and not repaired_failure:
                answer_candidate = repaired_answer
                raw = f"{raw}\n\n[repair]\n{repaired_raw}"
                repair_applied = True
                answer_resolution_method = "llm_repair"
            else:
                repair_failure = repaired_failure or "the repaired answer is empty"
                fallback_enabled = os.environ.get(
                    "COMPETITION_EXTRACTIVE_FALLBACK", "1"
                ).strip().lower() not in {"0", "false", "no", "off"}
                fallback_answer = (
                    _coerce_answer_candidate(
                        _extractive_fallback(task=task, evidence=evidence),
                        evidence,
                        question=question,
                    )
                    if fallback_enabled
                    else ""
                )
                fallback_failure = _answer_repair_reason(
                    task=task,
                    evidence=evidence,
                    answer=fallback_answer,
                )
                if fallback_answer and not fallback_failure:
                    answer_candidate = fallback_answer
                    raw = f"{raw}\n\n[extractive-fallback]\n{fallback_answer}"
                    repair_applied = True
                    repair_failure = ""
                    answer_resolution_method = "extractive_fallback"
        return {
            "answer_candidate": answer_candidate,
            "answer_resolution_method": answer_resolution_method,
            "reasoning_summary": (
                "Repaired a candidate that failed evidence validation."
                if repair_applied
                else "Selected an answer span from current retrieved evidence."
            ),
            "limitations": (
                [f"Answer repair was rejected: {repair_failure}."]
                if repair_reason and not repair_applied and evidence
                else ([] if evidence else ["No retrieved evidence was available."])
            ),
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
            "<one short sentence about which evidence or execution result was used>\n\n"
            "Run diagnostics:\n"
            "<one short line with mode, memory_hits, latent_states, and caveats>\n"
            "Be faithful to the provided implementation state: if execution_result says sandbox_status is "
            "not_implemented, say the current run simulates CodeAct statistics rather than running a real sandbox. "
            "Treat memory retrieval as hypergraph structural matching plus embedding similarity fusion when the evidence says so. "
            "Separate implemented behavior from planned extensions.\n"
            f"Task: {task}\n"
            f"Mode: {mode}\n"
            f"Plan: {json.dumps((plan or [])[:2], ensure_ascii=False)}\n"
            f"Evidence: {json.dumps((evidence or [])[:3], ensure_ascii=False)}\n"
            f"Execution result: {json.dumps({key: (execution_result or {}).get(key) for key in ['answer_candidate', 'reasoning_summary', 'execution_mode', 'retrieval_mode', 'sandbox_status'] if key in (execution_result or {})}, ensure_ascii=False)}\n"
            f"Cognitive gaps: {json.dumps((cognitive_gaps or [])[:1], ensure_ascii=False)}\n"
            f"Memory hits: {memory_hits}\n"
            f"Non-text latent states exchanged: {latent_states}"
        )
        return self.generate(prompt)
