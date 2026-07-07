from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache

from competition_agents.latent_hidden.local_model import get_hidden_model_name, load_local_hidden_model


@dataclass
class HiddenStateResult:
    """本地 LLM 最后一层 hidden state 的元信息和池化向量。"""

    model_name: str
    layer: str
    shape: list[int]
    pooling: str
    pooled_vector: list[float]
    checksum: str
    token_count: int
    dtype: str
    device: str


@dataclass
class GeneratedHiddenResult:
    """同一个本地模型在生成文本时直接产出的 hidden state 结果。"""

    prompt: str
    generated_text: str
    full_text: str
    hidden: HiddenStateResult


def _prepare_generation_inputs(bundle, prompt: str) -> dict[str, object]:
    """优先走 chat template，避免 instruct 模型把输入当成裸续写文本。"""

    def truncate(inputs: dict[str, object]) -> dict[str, object]:
        max_input_tokens = int(os.environ.get("LOCAL_LLM_MAX_INPUT_TOKENS", "0") or 0)
        input_ids = inputs.get("input_ids")
        if (
            input_ids is None
            or max_input_tokens <= 0
            or not hasattr(input_ids, "shape")
            or int(input_ids.shape[-1]) <= max_input_tokens
        ):
            return inputs
        head_tokens = min(256, max_input_tokens // 3)
        tail_tokens = max_input_tokens - head_tokens
        for key, value in list(inputs.items()):
            if hasattr(value, "shape") and len(value.shape) >= 2 and value.shape[-1] == input_ids.shape[-1]:
                inputs[key] = bundle.torch.cat(
                    [value[..., :head_tokens], value[..., -tail_tokens:]], dim=-1
                )
        return inputs

    tokenizer = bundle.tokenizer
    messages = [{"role": "user", "content": prompt}]
    if "deepseek-coder" in bundle.model_name.lower():
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
                    return truncate({
                        key: value.to(bundle.device) if hasattr(value, "to") else value
                        for key, value in inputs.items()
                    })
                if hasattr(inputs, "to"):
                    inputs = inputs.to(bundle.device)
                return truncate({"input_ids": inputs})
        except Exception:
            pass

    inputs = tokenizer(prompt, return_tensors="pt")
    return truncate({key: value.to(bundle.device) for key, value in inputs.items()})


def hidden_storage_mode() -> str:
    """读取 hidden state 存储模式。默认只保留 pooled 向量。"""

    raw = os.environ.get("LOCAL_HIDDEN_STORAGE_MODE", "pooled_only").strip().lower()
    return raw if raw in {"pooled_only", "full"} else "pooled_only"


def should_store_full_hidden_tensor() -> bool:
    """是否在进程内保留完整 hidden tensor。"""

    return hidden_storage_mode() == "full"


def stored_hidden_shape(result: HiddenStateResult) -> list[int]:
    """返回当前模式下实际保留的 hidden 形状。"""

    if should_store_full_hidden_tensor():
        return result.shape
    return [len(result.pooled_vector)]


def stored_hidden_bytes(result: HiddenStateResult) -> int:
    """按 float32 估算当前模式下的实际 hidden 存储开销。"""

    total = 1
    for dim in stored_hidden_shape(result):
        total *= int(dim)
    return total * 4


def hidden_pooled_dim() -> int:
    raw = os.environ.get("LOCAL_HIDDEN_POOLED_DIM", "128").strip()
    try:
        value = int(raw)
    except ValueError:
        return 128
    return max(0, value)


def _project_pooled_vector(values: list[float]) -> list[float]:
    target_dim = hidden_pooled_dim()
    if target_dim <= 0 or target_dim >= len(values):
        return [float(value) for value in values]

    projected = [0.0 for _ in range(target_dim)]
    counts = [0 for _ in range(target_dim)]
    for index, value in enumerate(values):
        digest = hashlib.blake2b(str(index).encode("ascii"), digest_size=4).digest()
        bucket = int.from_bytes(digest[:2], "little") % target_dim
        sign = 1.0 if digest[2] & 1 else -1.0
        projected[bucket] += sign * float(value)
        counts[bucket] += 1

    compact = []
    for value, count in zip(projected, counts, strict=False):
        compact.append(round(value / (max(count, 1) ** 0.5), 6))
    return compact


def _checksum_tensor(tensor) -> str:
    """对 tensor 内容生成短 checksum，证明状态确实发生了传递。"""

    array = tensor.detach().to("cpu").contiguous().float().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def _build_hidden_result(bundle, hidden_tensor, token_count: int) -> HiddenStateResult:
    pooled = hidden_tensor[:, -1, :].squeeze(0).to("cpu").float()
    pooled_values = [float(value) for value in pooled.tolist()]
    compact_values = _project_pooled_vector(pooled_values)
    return HiddenStateResult(
        model_name=bundle.model_name,
        layer="last",
        shape=list(hidden_tensor.shape),
        pooling=f"last_token_projected_{len(compact_values)}",
        pooled_vector=compact_values,
        checksum=_checksum_tensor(hidden_tensor),
        token_count=token_count,
        dtype=str(hidden_tensor.dtype),
        device=str(hidden_tensor.device),
    )


def _build_pooled_hidden_result(bundle, hidden_tensor, token_count: int, shape: list[int]) -> HiddenStateResult:
    """只基于最后一个 token 的 hidden 构造轻量结果，避免完整 tensor 搬回 CPU。"""

    pooled = hidden_tensor[:, -1, :].squeeze(0).to("cpu").float()
    checksum = hashlib.sha256(pooled.contiguous().numpy().tobytes()).hexdigest()[:16]
    pooled_values = [float(value) for value in pooled.tolist()]
    compact_values = _project_pooled_vector(pooled_values)
    return HiddenStateResult(
        model_name=bundle.model_name,
        layer="last",
        shape=shape,
        pooling=f"last_token_projected_{len(compact_values)}",
        pooled_vector=compact_values,
        checksum=checksum,
        token_count=token_count,
        dtype=str(hidden_tensor.dtype),
        device=str(hidden_tensor.device),
    )


def _should_stop_generation(bundle, generated_token_ids: list[int]) -> bool:
    """检测是否已经生成出明显的串台标记，尽早停止解码。"""

    if not generated_token_ids:
        return False
    recent_text = bundle.tokenizer.decode(generated_token_ids[-32:], skip_special_tokens=True)
    lowered = recent_text.lower()
    stop_markers = ["human:", "assistant:", "question:", "answer:", "evidence:"]
    return any(marker in lowered for marker in stop_markers)


def _extract_last_hidden_state_uncached(text: str) -> tuple[HiddenStateResult, object]:
    bundle = load_local_hidden_model()
    inputs = _prepare_generation_inputs(bundle, text)

    with bundle.torch.no_grad():
        outputs = bundle.model(
            **inputs,
            output_hidden_states=True,
            use_cache=True,
        )

    last_hidden = outputs.hidden_states[-1].detach()
    token_count = int(inputs["input_ids"].shape[-1])
    if should_store_full_hidden_tensor():
        result = _build_hidden_result(bundle, last_hidden, token_count)
        return result, last_hidden.to("cpu")

    result = _build_pooled_hidden_result(bundle, last_hidden, token_count, list(last_hidden.shape))
    return result, last_hidden[:, -1, :].detach().to("cpu")


@lru_cache(maxsize=512)
def _extract_last_hidden_state_cached(
    model_name: str,
    storage_mode_value: str,
    max_input_tokens: str,
    text: str,
) -> tuple[HiddenStateResult, object]:
    """缓存轻量 hidden 抽取结果，避免连续任务里重复前向同一段状态文本。"""

    del model_name, storage_mode_value, max_input_tokens
    return _extract_last_hidden_state_uncached(text)


def extract_last_hidden_state(text: str) -> tuple[HiddenStateResult, object]:
    """运行本地 LLM 前向传播，直接提取最后一层 hidden state。"""

    if should_store_full_hidden_tensor():
        return _extract_last_hidden_state_uncached(text)

    return _extract_last_hidden_state_cached(
        get_hidden_model_name(),
        hidden_storage_mode(),
        os.environ.get("LOCAL_LLM_MAX_INPUT_TOKENS", "0"),
        text,
    )


def generate_text_with_last_hidden_state(
    prompt: str,
    *,
    max_new_tokens: int = 48,
    do_sample: bool = False,
    temperature: float = 0.7,
) -> tuple[GeneratedHiddenResult, object]:
    """在同一条生成流程里完成文本生成，并直接收集最后一层 hidden state。

    这里不再走“先 generate，再把完整文本重新 full forward 一遍”的双计算流程。
    而是用同一个本地模型：
    1. 对 prompt 做第一次前向
    2. 在增量解码过程中复用 `past_key_values`
    3. 直接拼接 prompt 与生成 token 的最后一层 hidden state
    """

    bundle = load_local_hidden_model()
    inputs = _prepare_generation_inputs(bundle, prompt)
    tokenizer = bundle.tokenizer
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "return_dict_in_generate": True,
        "output_hidden_states": True,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature

    with bundle.torch.no_grad():
        generated = bundle.model.generate(**inputs, **generate_kwargs)

    sequences = generated.sequences
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    generated_ids = sequences[0][prompt_tokens:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    full_text = tokenizer.decode(sequences[0], skip_special_tokens=True)

    hidden_steps = getattr(generated, "hidden_states", None) or ()
    latest_hidden = None
    for step in reversed(hidden_steps):
        if step:
            latest_hidden = step[-1].detach()
            break
    if latest_hidden is None:
        with bundle.torch.no_grad():
            outputs = bundle.model(
                input_ids=sequences,
                output_hidden_states=True,
                use_cache=True,
            )
        latest_hidden = outputs.hidden_states[-1].detach()

    full_shape = [int(sequences.shape[0]), int(sequences.shape[-1]), int(latest_hidden.shape[-1])]
    if should_store_full_hidden_tensor() and int(latest_hidden.shape[-2]) == int(sequences.shape[-1]):
        hidden = _build_hidden_result(bundle, latest_hidden, int(sequences.shape[-1]))
        returned_tensor = latest_hidden.to("cpu")
    else:
        hidden = _build_pooled_hidden_result(bundle, latest_hidden, int(sequences.shape[-1]), full_shape)
        returned_tensor = latest_hidden[:, -1, :].detach().to("cpu")
    return (
        GeneratedHiddenResult(
            prompt=prompt,
            generated_text=generated_text,
            full_text=full_text,
            hidden=hidden,
        ),
        returned_tensor,
    )
