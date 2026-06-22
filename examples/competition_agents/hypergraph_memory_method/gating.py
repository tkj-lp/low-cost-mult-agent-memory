from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .text import normalize_text, tokenize


GENERIC_TOPIC_TOKENS = {
    "answer",
    "chain",
    "continuous",
    "evidence",
    "experience",
    "fact",
    "memory",
    "previous",
    "prior",
    "result",
    "reuse",
    "supporting",
}

TRIGGER_FAMILY_HINTS = {
    "temporal": {
        "date",
        "dvd",
        "released",
        "release",
        "temporal",
        "time",
        "when",
        "year",
    },
    "location": {
        "country",
        "filmed",
        "location",
        "locations",
        "place",
        "places",
        "region",
        "shot",
        "where",
    },
    "attribute": {
        "attribute",
        "economy",
        "fishery",
        "hazard",
        "history",
        "population",
        "property",
        "resource",
        "resources",
    },
    "entity-list": {
        "actor",
        "actors",
        "actress",
        "cast",
        "members",
        "people",
        "stars",
        "who",
    },
    "reference": {
        "former",
        "it",
        "its",
        "latter",
        "nearby",
        "reference",
        "that",
        "their",
        "this",
        "those",
    },
    "comparison": {"compare", "comparison", "difference", "different", "versus"},
    "causal": {"cause", "caused", "reason", "why"},
}

TRIGGER_LABEL_FAMILIES = {
    "attribute-lookup-query": "attribute",
    "entity-list-query": "entity-list",
    "format-constraint": "temporal",
    "multi-location-query": "location",
    "question-targets-single-attribute": "attribute",
    "reference-resolution-query": "reference",
    "region-constraint": "location",
    "temporal-query": "temporal",
}


@dataclass(frozen=True)
class CandidateGate:
    accepted: bool
    score: float
    trigger_score: float
    topic_score: float
    query_trigger_families: tuple[str, ...]
    record_trigger_families: tuple[str, ...]
    query_primary_families: tuple[str, ...]
    record_primary_families: tuple[str, ...]
    topic_overlap: tuple[str, ...]
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_gate_passed": self.accepted,
            "candidate_gate_score": round(self.score, 4),
            "trigger_compatibility_score": round(self.trigger_score, 4),
            "topic_compatibility_score": round(self.topic_score, 4),
            "query_trigger_families": list(self.query_trigger_families),
            "record_trigger_families": list(self.record_trigger_families),
            "query_primary_trigger_families": list(self.query_primary_families),
            "record_primary_trigger_families": list(self.record_primary_families),
            "topic_overlap": list(self.topic_overlap),
            "candidate_gate_reasons": list(self.reasons),
        }


def infer_trigger_families(text: str) -> set[str]:
    """Map free text or trigger labels into a small, stable trigger taxonomy."""

    normalized = normalize_text(text)
    tokens = set(tokenize(normalized))
    families = {
        family
        for family, hints in TRIGGER_FAMILY_HINTS.items()
        if tokens & hints
    }
    for label, family in TRIGGER_LABEL_FAMILIES.items():
        if label in normalized:
            families.add(family)
    return families


def extract_retrieval_question(query: str) -> str:
    """Extract the real task question from a planner-expanded retrieval query."""

    match = re.search(r"(?:^|\n)question:\s*(.+?)(?:\n\s*\n|\Z)", str(query), flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else str(query)


def record_trigger_families(record: dict[str, Any]) -> set[str]:
    values = record.get("trigger_conditions") or []
    if not isinstance(values, list):
        values = [values]
    families: set[str] = set()
    for value in values:
        families.update(infer_trigger_families(str(value)))
    return families


def query_primary_trigger_families(text: str) -> set[str]:
    """Infer the question's requested operation, excluding reference modifiers."""

    normalized = normalize_text(text)
    tokens = set(tokenize(normalized))
    primary: set[str] = set()
    for family in ("temporal", "attribute", "entity-list", "comparison", "causal"):
        hints = TRIGGER_FAMILY_HINTS[family]
        if tokens & hints:
            primary.add(family)
    location_hints = {"where", "filmed", "location", "locations", "shot"}
    if tokens & location_hints:
        primary.add("location")
    first_word = normalized.split(maxsplit=1)[0] if normalized else ""
    if not primary and ("?" in str(text) or first_word in {"how", "what", "which", "who", "why"}):
        primary.add("other")
    return primary


def record_primary_trigger_families(record: dict[str, Any]) -> set[str]:
    """Infer the operation encoded by experience trigger labels."""

    values = record.get("trigger_conditions") or []
    if not isinstance(values, list):
        values = [values]
    primary: set[str] = set()
    for value in values:
        label = normalize_text(str(value))
        if "temporal-query" in label:
            primary.add("temporal")
        elif "attribute-lookup-query" in label or "question-targets-single-attribute" in label:
            primary.add("attribute")
        elif "multi-location-query" in label:
            primary.add("location")
        elif "entity-list-query" in label:
            primary.add("entity-list")
        elif "comparison" in label:
            primary.add("comparison")
        elif "causal" in label:
            primary.add("causal")
    return primary


def _record_topic_tokens(record: dict[str, Any]) -> set[str]:
    trace = record.get("trace") if isinstance(record.get("trace"), dict) else {}
    task_meta = trace.get("task_meta") if isinstance(trace.get("task_meta"), dict) else {}
    values = [
        task_meta.get("topic", ""),
        task_meta.get("group_topic", ""),
        task_meta.get("question", ""),
        record.get("task_topic", ""),
        record.get("problem_pattern", ""),
    ]
    tokens: set[str] = set()
    for value in values:
        tokens.update(tokenize(str(value)))
    return tokens - GENERIC_TOPIC_TOKENS


def evaluate_candidate_gate(
    query: str,
    query_profile: dict[str, Any],
    record: dict[str, Any],
) -> CandidateGate:
    """Reject incompatible experience rules and softly score topic agreement.

    Trigger mismatch is a hard rejection only for experience memory. Evidence is
    allowed through because follow-up questions often replace the entity name
    with a pronoun and therefore lack enough lexical topic information.
    """

    effective_query = extract_retrieval_question(query)
    query_families = infer_trigger_families(effective_query)
    record_families = record_trigger_families(record)
    trigger_overlap = query_families & record_families
    query_primary = query_primary_trigger_families(effective_query)
    record_primary = record_primary_trigger_families(record)
    primary_overlap = query_primary & record_primary
    memory_type = str(record.get("memory_type", "evidence") or "evidence")

    reasons: list[str] = []
    accepted = True
    if memory_type == "experience" and query_primary and record_primary and not primary_overlap:
        accepted = False
        reasons.append("experience-primary-trigger-mismatch")
    elif memory_type == "experience" and query_families and record_families and not trigger_overlap:
        accepted = False
        reasons.append("experience-trigger-mismatch")

    effective_query = query_primary if query_primary and record_primary else query_families
    effective_record = record_primary if query_primary and record_primary else record_families
    effective_overlap = effective_query & effective_record
    if not record_families:
        trigger_score = 0.0
        reasons.append("record-has-no-explicit-trigger")
    elif not query_families:
        trigger_score = 0.0
        reasons.append("query-trigger-unknown")
    else:
        trigger_score = len(effective_overlap) / max(1, len(effective_record))
        reasons.append("trigger-compatible" if effective_overlap else "trigger-incompatible")

    query_topics = set(tokenize(effective_query)) - GENERIC_TOPIC_TOKENS
    record_topics = _record_topic_tokens(record)
    topic_overlap = query_topics & record_topics
    topic_score = len(topic_overlap) / max(1, min(len(query_topics), len(record_topics)))
    if topic_overlap:
        reasons.append("topic-overlap")
    else:
        reasons.append("no-explicit-topic-overlap")

    score = trigger_score * 4.0 + min(topic_score, 1.0) * 2.0
    return CandidateGate(
        accepted=accepted,
        score=score,
        trigger_score=trigger_score,
        topic_score=topic_score,
        query_trigger_families=tuple(sorted(query_families)),
        record_trigger_families=tuple(sorted(record_families)),
        query_primary_families=tuple(sorted(query_primary)),
        record_primary_families=tuple(sorted(record_primary)),
        topic_overlap=tuple(sorted(topic_overlap)),
        reasons=tuple(reasons),
    )
