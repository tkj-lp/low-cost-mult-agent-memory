from __future__ import annotations

import hashlib
import re
import unicodedata


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

NOISY_TOKENS = {
    "context",
    "question",
    "sentences",
    "title",
}


def normalize_text(text: str) -> str:
    return str(text).lower().replace("_", "-").strip()


def slug(text: str) -> str:
    text_slug = re.sub(r"[^a-zA-Z0-9-]+", "-", normalize_text(text)).strip("-")
    return text_slug[:48] or "empty"


def stable_id(text: str, *, length: int = 12) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:length]


def tokenize(text: str) -> list[str]:
    """Convert text into stable keyword tokens for graph nodes."""

    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9-]*", normalize_text(text))
    return sorted(
        {
            token
            for token in tokens
            if len(token) > 1
            and token not in STOPWORDS
            and token not in NOISY_TOKENS
            and not any(char.isdigit() for char in token)
        }
    )


def limited_tokens(text: str, *, limit: int = 32) -> list[str]:
    """Return a bounded token list so long contexts cannot dominate retrieval."""

    return tokenize(text)[:limit]


def node_id(kind: str, value: str) -> str:
    """Build a typed node id so equal values in different namespaces stay apart."""

    return f"{kind}:{normalize_text(value)}"


def indexed_tokens(text: str) -> list[str]:
    """FTS terms retain numbers and use CJK bigrams without a model call."""

    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    tokens: list[str] = []
    for match in re.finditer(r"[a-z0-9]+|[\u3400-\u9fff]+", normalized):
        value = match.group()
        if "\u3400" <= value[0] <= "\u9fff":
            tokens.extend(value[i : i + 2] for i in range(max(1, len(value) - 1)))
        elif value not in STOPWORDS:
            tokens.append(value)
    return tokens
