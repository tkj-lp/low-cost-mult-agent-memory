from __future__ import annotations

import os
from typing import Any

from competition_agents.latent_hidden.local_model import load_local_hidden_model

from .openai_compatible import OpenAICompatibleClient


class LocalHFClient(OpenAICompatibleClient):
    """使用本机 HuggingFace causal LM 的 Agent 文本生成客户端。"""

    def __init__(self) -> None:
        self.bundle = load_local_hidden_model()
        self.model = self.bundle.model_name
        self.max_new_tokens = int(os.environ.get("LOCAL_LLM_MAX_NEW_TOKENS", "160"))

    def _prepare_inputs(self, prompt: str) -> dict[str, Any]:
        tokenizer = self.bundle.tokenizer
        messages = [{"role": "user", "content": prompt}]
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
                        return {
                            key: value.to(self.bundle.device) if hasattr(value, "to") else value
                            for key, value in inputs.items()
                        }
                    if hasattr(inputs, "to"):
                        inputs = inputs.to(self.bundle.device)
                    return {"input_ids": inputs}
            except Exception:
                pass
        inputs = tokenizer(prompt, return_tensors="pt")
        return {key: value.to(self.bundle.device) for key, value in inputs.items()}

    def generate(self, prompt: str) -> str:
        tokenizer = self.bundle.tokenizer
        model = self.bundle.model
        torch = self.bundle.torch
        inputs = self._prepare_inputs(prompt)
        input_len = int(inputs["input_ids"].shape[-1])
        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "use_cache": True,
        }
        if tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id or tokenizer.eos_token_id

        print(
            f"[local_hf] model={self.model} device={self.bundle.device} prompt chars: {len(prompt)}",
            flush=True,
        )
        with torch.no_grad():
            output_ids = model.generate(**generation_kwargs)
        generated_ids = output_ids[0][input_len:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
