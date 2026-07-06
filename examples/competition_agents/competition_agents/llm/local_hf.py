from __future__ import annotations

import os
import re
from typing import Any

from competition_agents.latent_hidden import generate_text_with_last_hidden_state
from competition_agents.latent_hidden.hidden_state import HiddenStateResult
from competition_agents.latent_hidden.local_model import load_local_hidden_model

from .openai_compatible import OpenAICompatibleClient


def _compact_local_task(task: str, *, limit: int = 1800) -> str:
    """Optionally shorten very long local-model prompts when explicitly enabled."""

    text = str(task or "").strip()
    context_match = re.search(r"\nContext\s*:", text, flags=re.IGNORECASE)
    if context_match:
        text = text[: context_match.start()].rstrip() + "\nContext: [provided separately as evidence]"
    if len(text) <= limit:
        return text
    return f"{text[:1200]}\n...[task compacted for local inference]...\n{text[-500:]}"


class LocalHFClient(OpenAICompatibleClient):
    """使用本机 HuggingFace causal LM 的 Agent 文本生成客户端。"""

    def __init__(self, role: str | None = None) -> None:
        self.bundle = load_local_hidden_model()
        self.model = self.bundle.model_name
        self.role = (role or "agent").lower()
        role_limit_name = f"LOCAL_LLM_{self.role.upper()}_MAX_NEW_TOKENS"
        self.max_new_tokens = int(
            os.environ.get(
                role_limit_name,
                os.environ.get("LOCAL_LLM_MAX_NEW_TOKENS", "160"),
            )
        )
        self.max_input_tokens = int(os.environ.get("LOCAL_LLM_MAX_INPUT_TOKENS", "0") or 0)
        self.compact_task = os.environ.get("LOCAL_LLM_COMPACT_TASK", "").lower() in {
            "1",
            "true",
            "yes",
        }
        self._last_hidden_result: HiddenStateResult | None = None
        self._last_hidden_tensor: object | None = None

    def consume_last_hidden_state(self) -> tuple[HiddenStateResult | None, object | None]:
        """Return and clear the hidden state produced by the last local generation."""

        result = self._last_hidden_result
        tensor = self._last_hidden_tensor
        self._last_hidden_result = None
        self._last_hidden_tensor = None
        return result, tensor

    def _prepare_inputs(self, prompt: str) -> dict[str, Any]:
        tokenizer = self.bundle.tokenizer
        messages = [{"role": "user", "content": prompt}]
        if "deepseek-coder" in self.model.lower():
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": (
                        "You are a general question-answering assistant in a multi-agent experiment. "
                        "Answer the user's task directly and concisely. Do not refuse non-programming "
                        "questions and do not describe implementation details unless explicitly asked."
                    ),
                },
            )
        apply_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            try:
                inputs = apply_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                if inputs is not None:
                    if hasattr(inputs, "items"):
                        prepared = {
                            key: value.to(self.bundle.device) if hasattr(value, "to") else value
                            for key, value in inputs.items()
                        }
                        return self._truncate_inputs(prepared)
                    if hasattr(inputs, "to"):
                        inputs = inputs.to(self.bundle.device)
                    return self._truncate_inputs({"input_ids": inputs})
            except Exception:
                pass
        inputs = tokenizer(prompt, return_tensors="pt")
        return self._truncate_inputs(
            {key: value.to(self.bundle.device) for key, value in inputs.items()}
        )

    def _truncate_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        input_ids = inputs.get("input_ids")
        if (
            input_ids is None
            or self.max_input_tokens <= 0
            or int(input_ids.shape[-1]) <= self.max_input_tokens
        ):
            return inputs
        head_tokens = min(256, self.max_input_tokens // 3)
        tail_tokens = self.max_input_tokens - head_tokens
        for key, value in list(inputs.items()):
            if hasattr(value, "shape") and len(value.shape) >= 2 and value.shape[-1] == input_ids.shape[-1]:
                inputs[key] = self.bundle.torch.cat(
                    [value[..., :head_tokens], value[..., -tail_tokens:]], dim=-1
                )
        return inputs

    def plan(self, task: str) -> list[str]:
        return super().plan(_compact_local_task(task, limit=1000) if self.compact_task else task)

    def plan_with_raw(self, task: str) -> tuple[list[str], str]:
        return super().plan_with_raw(_compact_local_task(task, limit=1000) if self.compact_task else task)

    def reason(self, **kwargs: Any) -> dict[str, Any]:
        if self.compact_task:
            kwargs["task"] = _compact_local_task(str(kwargs.get("task", "")))
        return super().reason(**kwargs)

    def summarize(self, **kwargs: Any) -> str:
        if self.compact_task:
            kwargs["task"] = _compact_local_task(str(kwargs.get("task", "")))
        return super().summarize(**kwargs)

    def _generate_text_only(self, prompt: str) -> str:
        inputs = self._prepare_inputs(prompt)
        tokenizer = self.bundle.tokenizer
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
        with self.bundle.torch.no_grad():
            output_ids = self.bundle.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        generated_ids = output_ids[0][prompt_tokens:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def generate(self, prompt: str) -> str:
        print(
            f"[local_hf] role={self.role} model={self.model} "
            f"device={self.bundle.device} max_new_tokens={self.max_new_tokens} "
            f"prompt_chars={len(prompt)}",
            flush=True,
        )
        capture_hidden = os.environ.get("LOCAL_CAPTURE_HIDDEN", "").lower() in {"1", "true", "yes"}
        if not capture_hidden:
            self._last_hidden_result = None
            self._last_hidden_tensor = None
            return self._generate_text_only(prompt)
        generated, tensor = generate_text_with_last_hidden_state(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        self._last_hidden_result = generated.hidden
        self._last_hidden_tensor = tensor
        return generated.generated_text.strip()
