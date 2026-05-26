from __future__ import annotations

import hashlib
import re


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def normalize_text(text: str) -> str:
    return text.lower().replace("_", "-").strip()


def slug(text: str) -> str:
    text_slug = re.sub(r"[^a-zA-Z0-9-]+", "-", normalize_text(text)).strip("-")
    return text_slug[:48] or "empty"


def stable_id(text: str, *, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def tokenize(text: str) -> list[str]:
    """把任务和总结文本转成可复用的关键词节点。"""

    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9-]*", normalize_text(text))
    return sorted({token for token in tokens if len(token) > 1 and token not in STOPWORDS})


def node_id(kind: str, value: str) -> str:
    """统一节点 id 格式，避免 tag 和 keyword 的同名值混在一起。"""

    return f"{kind}:{normalize_text(value)}"
