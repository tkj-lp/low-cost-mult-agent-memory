from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass
class LocalHiddenModel:
    """保存本地因果语言模型和 tokenizer，避免每次重复加载。"""

    model_name: str
    tokenizer: Any
    model: Any
    torch: Any
    device: str


def get_hidden_model_name() -> str:
    """读取用于 hidden-state 实验的本地模型名称。"""

    return os.environ.get("LOCAL_HIDDEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


@lru_cache(maxsize=1)
def load_local_hidden_model() -> LocalHiddenModel:
    """加载本地 HuggingFace causal LM，用来提取最后一层 hidden states。"""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "hidden-state latent demo requires torch and transformers. "
            "Install them with: pip install torch transformers"
        ) from exc

    model_name = get_hidden_model_name()
    cache_dir = os.environ.get("LOCAL_HIDDEN_CACHE_DIR") or None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return LocalHiddenModel(
        model_name=model_name,
        tokenizer=tokenizer,
        model=model,
        torch=torch,
        device=device,
    )
