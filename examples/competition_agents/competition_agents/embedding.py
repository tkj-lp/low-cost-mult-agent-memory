from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class EmbeddingResult:
    vector: list[float]
    provider: str
    model: str


def get_embedding_provider() -> str:
    return os.environ.get("EMBEDDING_PROVIDER", "hash").lower()


def hash_embedding(text: str, dimensions: int = 16) -> list[float]:
    """生成确定性的 hash 向量，作为真实 embedding API 不可用时的回退实现。"""

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    for index in range(dimensions):
        raw = digest[index % len(digest)]
        values.append(round((raw / 255.0) * 2.0 - 1.0, 4))
    return values


def _embed_openai_compatible(text: str) -> EmbeddingResult:
    api_key = os.environ.get("EMBEDDING_API_KEY") or ""
    if api_key in {"replace_with_your_api_key", "your_api_key"}:
        api_key = ""
    api_key = api_key or os.environ.get("LLM_API_KEY") or ""
    if not api_key or api_key in {"replace_with_your_api_key", "your_api_key"}:
        raise RuntimeError(
            "EMBEDDING_API_KEY/LLM_API_KEY is not configured. "
            "Set EMBEDDING_PROVIDER=hash to use the local fallback."
        )

    base_url = os.environ.get("EMBEDDING_BASE_URL") or os.environ.get(
        "LLM_BASE_URL", "https://api.openai.com/v1"
    )
    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-v4")
    request_timeout = float(os.environ.get("EMBEDDING_REQUEST_TIMEOUT_SECONDS", "180"))
    max_retries = max(0, int(os.environ.get("EMBEDDING_MAX_RETRIES", "2")))
    payload = {"model": model, "input": text}
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            vector = body["data"][0]["embedding"]
            return EmbeddingResult(
                vector=[float(value) for value in vector],
                provider="openai_compatible",
                model=model,
            )
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Embedding API request failed with HTTP {exc.code}: {exc.reason}. "
                f"Response body: {error_body}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(2**attempt, 4))

    raise RuntimeError(f"Embedding API request failed after retries: {last_error}") from last_error

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Embedding API request failed with HTTP {exc.code}: {exc.reason}. "
            f"Response body: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Embedding API request failed: {exc}") from exc

    vector = body["data"][0]["embedding"]
    return EmbeddingResult(
        vector=[float(value) for value in vector],
        provider="openai_compatible",
        model=model,
    )


@lru_cache(maxsize=1)
def _get_local_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Local embedding requires sentence-transformers. "
            f"Current Python is {sys.executable}. "
            "Activate the conda environment that contains sentence-transformers, "
            "for example: conda activate langgraph"
        ) from exc

    model_name = os.environ.get("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    cache_folder = os.environ.get("LOCAL_EMBEDDING_CACHE_DIR") or None
    return SentenceTransformer(model_name, cache_folder=cache_folder)


def _embed_local(text: str) -> EmbeddingResult:
    model_name = os.environ.get("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    model = _get_local_embedding_model()
    vector = model.encode(text, normalize_embeddings=True).tolist()
    return EmbeddingResult(
        vector=[float(value) for value in vector],
        provider="local",
        model=model_name,
    )




@lru_cache(maxsize=2048)
def _embed_text_cached(
    provider: str,
    model_key: str,
    base_url: str,
    text: str,
) -> tuple[tuple[float, ...], str, str]:
    """Cache embedding calls by provider/model/text to avoid repeated state encoding."""

    del model_key, base_url
    if provider in {"openai", "openai_compatible", "api"}:
        result = _embed_openai_compatible(text)
    elif provider == "local":
        result = _embed_local(text)
    elif provider == "hash":
        result = EmbeddingResult(
            vector=hash_embedding(text),
            provider="hash",
            model="hash-v1",
        )
    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")
    return tuple(result.vector), result.provider, result.model


def embed_text(text: str, *, provider: str | None = None) -> EmbeddingResult:
    provider = provider or get_embedding_provider()
    model_key = (
        os.environ.get("EMBEDDING_MODEL")
        or os.environ.get("LOCAL_EMBEDDING_MODEL")
        or os.environ.get("LOCAL_HIDDEN_MODEL")
        or ""
    )
    base_url = os.environ.get("EMBEDDING_BASE_URL") or os.environ.get("LLM_BASE_URL", "")
    vector, actual_provider, model = _embed_text_cached(provider, model_key, base_url, text)
    return EmbeddingResult(
        vector=[float(value) for value in vector],
        provider=actual_provider,
        model=model,
    )
