from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompressionResult:
    payload: str
    raw_chars: int
    compressed_chars: int
    latency_ms: float
    provider: str
    ratio: float


_COMPRESSOR: Any | None = None


def compress_message(
    payload: str,
    *,
    sender: str,
    receiver: str,
    action: str,
) -> CompressionResult:
    """压缩 Agent 间自然语言消息，作为 LLMLingua-2 对比基线。"""

    raw_chars = len(payload)
    started = time.perf_counter()
    compressed, provider = _compress_with_llmlingua(payload)
    latency_ms = (time.perf_counter() - started) * 1000
    compressed = compressed if compressed.strip() else payload
    compressed_chars = len(compressed)
    ratio = compressed_chars / raw_chars if raw_chars else 1.0
    header = f"[LLMLingua-2 compressed message | {sender}->{receiver} | {action}]\n"
    return CompressionResult(
        payload=header + compressed,
        raw_chars=raw_chars,
        compressed_chars=len(header) + compressed_chars,
        latency_ms=latency_ms,
        provider=provider,
        ratio=ratio,
    )


def _compress_with_llmlingua(payload: str) -> tuple[str, str]:
    compressor = _get_official_compressor()
    if compressor is None:
        return _rule_based_compress(payload), "rule_fallback"

    rate = float(os.environ.get("LLMLINGUA2_RATE", "0.5") or 0.5)
    force_tokens = [
        "Question:",
        "Evidence:",
        "Answer:",
        "Plan:",
        "memory",
        "state",
        "final",
    ]
    try:
        result = compressor.compress_prompt(
            payload,
            rate=rate,
            force_tokens=force_tokens,
        )
        if isinstance(result, dict):
            return str(result.get("compressed_prompt") or payload), "llmlingua2"
        return str(result), "llmlingua2"
    except Exception:
        if os.environ.get("LLMLINGUA2_STRICT", "").lower() in {"1", "true", "yes"}:
            raise
        return _rule_based_compress(payload), "rule_fallback_after_error"


def _get_official_compressor() -> Any | None:
    global _COMPRESSOR
    if _COMPRESSOR is not None:
        return _COMPRESSOR
    if os.environ.get("LLMLINGUA2_DISABLE_OFFICIAL", "").lower() in {"1", "true", "yes"}:
        return None
    enable_official = os.environ.get("LLMLINGUA2_ENABLE_OFFICIAL", "").lower() in {"1", "true", "yes"}
    model_name = os.environ.get(
        "LLMLINGUA2_MODEL",
        "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    )
    model_is_local = os.path.isdir(model_name)
    if not enable_official and not model_is_local:
        return None
    try:
        from llmlingua import PromptCompressor  # type: ignore
    except Exception:
        return None

    device_map = os.environ.get("LLMLINGUA2_DEVICE_MAP") or "cpu"
    try:
        _COMPRESSOR = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,
            device_map=device_map,
        )
    except Exception:
        if os.environ.get("LLMLINGUA2_STRICT", "").lower() in {"1", "true", "yes"}:
            raise
        return None
    return _COMPRESSOR


def _rule_based_compress(payload: str) -> str:
    """无官方包时的可运行兜底：保留关键信息，删除明显冗余句。"""

    text = re.sub(r"\s+", " ", payload).strip()
    if len(text) <= 320:
        return text

    keep_markers = (
        "Question",
        "Answer",
        "candidate",
        "evidence",
        "memory",
        "experience",
        "plan",
        "task",
        "result",
        "stdout",
        "stderr",
        "final",
    )
    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    kept: list[str] = []
    budget = max(240, int(len(text) * float(os.environ.get("LLMLINGUA2_RATE", "0.5") or 0.5)))
    for sentence in sentences:
        if any(marker.lower() in sentence.lower() for marker in keep_markers):
            kept.append(sentence)
        if sum(len(item) for item in kept) >= budget:
            break
    if not kept:
        kept = sentences[: max(1, min(3, len(sentences)))]
    compressed = " ".join(kept)
    return compressed[:budget]
