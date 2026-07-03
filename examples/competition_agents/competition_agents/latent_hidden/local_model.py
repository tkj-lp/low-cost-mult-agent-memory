from __future__ import annotations

import os
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
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

    return os.environ.get("LOCAL_HIDDEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


def _load_deepseek_coder_tokenizer(model_name: str, cache_dir: str | None):
    """DeepSeek-Coder 的 tokenizer.json 需要走 fast tokenizer，避免 byte-level 符号泄漏。"""

    from huggingface_hub import snapshot_download
    from transformers import PreTrainedTokenizerFast

    snapshot_dir = Path(snapshot_download(model_name, cache_dir=cache_dir, local_files_only=True))
    config_path = snapshot_dir / "tokenizer_config.json"
    tokenizer_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(snapshot_dir / "tokenizer.json"),
        bos_token=tokenizer_config.get("bos_token", {}).get("content", "<｜begin▁of▁sentence｜>"),
        eos_token=tokenizer_config.get("eos_token", {}).get("content", "<|EOT|>"),
        pad_token=tokenizer_config.get("pad_token", {}).get("content", "<｜end▁of▁sentence｜>"),
    )
    chat_template = tokenizer_config.get("chat_template")
    if chat_template:
        tokenizer.chat_template = chat_template
    return tokenizer


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

    if "deepseek-coder" in model_name.lower():
        tokenizer = _load_deepseek_coder_tokenizer(model_name, cache_dir)
    else:
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
