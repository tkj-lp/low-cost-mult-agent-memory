from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


_ENTITY_STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "as",
    "context",
    "evidence",
    "how",
    "it",
    "only",
    "original",
    "question",
    "return",
    "same",
    "task",
    "the",
    "this",
    "using",
    "what",
    "when",
    "where",
    "which",
    "who",
}

_ENTITY_PATTERNS = (
    r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r'"([^"\r\n]{2,80})"',
    r"\b(?:[A-Z][A-Za-z0-9'-]*|[A-Z]{2,})(?:\s+(?:(?:of|the|and|de|van|for|in)\s+)?(?:[A-Z][A-Za-z0-9'-]*|[A-Z]{2,})){0,5}\b",
)


def canonical_entity(value: object) -> str:
    """Normalize an entity label so separate tasks share one graph node."""

    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.strip(" \t\r\n\"'`[]{}.,;:!?")
    text = re.sub(r"\s+", " ", text).strip().lower()
    if text.endswith("'s"):
        text = text[:-2].rstrip()
    if len(text) < 2 or text in _ENTITY_STOPWORDS:
        return ""
    return text


def extract_canonical_entities(
    *texts: object,
    explicit: Iterable[object] | None = None,
    limit: int = 24,
) -> list[str]:
    """Extract bounded entity labels only from runtime-visible text."""

    if limit <= 0:
        return []
    explicit_values = list(explicit or [])
    candidates: list[object] = [*explicit_values]
    for raw_text in [*explicit_values, *texts]:
        text = str(raw_text or "")
        for pattern in _ENTITY_PATTERNS:
            candidates.extend(re.findall(pattern, text))

    entities: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        entity = canonical_entity(candidate)
        if not entity or entity in seen:
            continue
        seen.add(entity)
        entities.append(entity)
        if len(entities) >= limit:
            break
    return entities
