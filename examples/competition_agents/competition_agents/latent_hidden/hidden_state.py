from __future__ import annotations

import hashlib
from dataclasses import dataclass

from competition_agents.latent_hidden.local_model import load_local_hidden_model


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


def _checksum_tensor(tensor) -> str:
    """对 tensor 内容生成短 checksum，便于证明状态确实发生了传递。"""

    array = tensor.detach().to("cpu").contiguous().float().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def extract_last_hidden_state(text: str) -> tuple[HiddenStateResult, object]:
    """运行本地 LLM 前向传播，直接提取最后一层 hidden state。

    注意：这里不是把文本再编码成 embedding，而是读取 causal LM 前向传播产生的
    `outputs.hidden_states[-1]`。这是更接近 LatentMAS 论文的 latent state。
    """

    bundle = load_local_hidden_model()
    inputs = bundle.tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(bundle.device) for key, value in inputs.items()}

    with bundle.torch.no_grad():
        outputs = bundle.model(
            **inputs,
            output_hidden_states=True,
            use_cache=True,
        )

    last_hidden = outputs.hidden_states[-1].detach()
    # 第一版用最后一个 token 的向量作为低开销可传递 latent 表示。
    pooled = last_hidden[:, -1, :].squeeze(0).to("cpu").float()
    result = HiddenStateResult(
        model_name=bundle.model_name,
        layer="last",
        shape=list(last_hidden.shape),
        pooling="last_token",
        pooled_vector=[float(value) for value in pooled.tolist()],
        checksum=_checksum_tensor(last_hidden),
        token_count=int(inputs["input_ids"].shape[-1]),
        dtype=str(last_hidden.dtype),
        device=bundle.device,
    )
    return result, last_hidden.to("cpu")
