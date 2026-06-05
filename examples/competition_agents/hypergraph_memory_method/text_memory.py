from __future__ import annotations

import time
from typing import Any

from .context import _context_units, _dataset_name, _task_id, _text
from .text import tokenize


def _token_set(*values: Any) -> set[str]:
    return set(tokenize(" ".join(str(value or "") for value in values)))


def _stringify_value(value: Any, *, max_items: int = 12) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        parts: list[str] = []
        for key, child in value.items():
            text = _stringify_value(child, max_items=max_items)
            if text:
                parts.append(f"{key}: {text}")
            if len(parts) >= max_items:
                break
        return " ".join(parts)
    if isinstance(value, list):
        parts = [_stringify_value(item, max_items=max_items) for item in value[:max_items]]
        return " ".join(part for part in parts if part)
    return str(value)


def _score_text(query_text: str, candidate_text: str) -> tuple[float, list[str]]:
    query_tokens = _token_set(query_text)
    if not query_tokens:
        return 0.0, []
    candidate_tokens = _token_set(candidate_text)
    if not candidate_tokens:
        return 0.0, []

    matched = sorted(query_tokens & candidate_tokens)
    if not matched:
        return 0.0, []

    coverage = len(matched) / len(query_tokens)
    precision = len(matched) / len(candidate_tokens)
    jaccard = len(matched) / len(query_tokens | candidate_tokens)
    score = len(matched) + coverage * 3.0 + precision + jaccard
    return round(score, 4), matched


def _text_reasoning_path(matched_terms: list[str], target_id: str) -> list[dict[str, str]]:
    return [{"matched_term": term, "target_id": target_id} for term in matched_terms[:5]]


def search_text_context(
    task_meta: dict[str, Any],
    query: str | None = None,
    *,
    dataset_type: str | None = None,
    planner_focus: list[str] | None = None,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Search dataset context with a plain-text lexical baseline.

    This deliberately avoids graph nodes and hyperedges. It flattens each
    candidate context unit into text and ranks by token overlap, so it can be
    used as a baseline against `search_context`.
    """

    started = time.perf_counter()
    resolved_dataset = _dataset_name(task_meta, dataset_type)
    task_id = _task_id(task_meta)
    units = _context_units(task_meta, resolved_dataset, task_id)
    query_text = " ".join(
        [
            _text(task_meta.get("question")),
            _text(task_meta.get("original_question")),
            _text(task_meta.get("task")),
            _text(query),
            " ".join(planner_focus or []),
        ]
    )

    hits: list[dict[str, Any]] = []
    for unit in units:
        candidate_text = " ".join([unit.get("title", ""), unit.get("section", ""), unit.get("text", "")])
        score, matched_terms = _score_text(query_text, candidate_text)
        if score <= 0:
            continue
        hits.append(
            {
                "retrieval_backend": "text_context",
                "context_id": unit["context_id"],
                "dataset_type": resolved_dataset,
                "kind": unit["kind"],
                "title": unit.get("title", ""),
                "section": unit.get("section", ""),
                "text": unit["text"],
                "context_score": score,
                "text_score": score,
                "retrieval_score": score,
                "matched_terms": matched_terms[:16],
                "matched_nodes": [f"text_term:{term}" for term in matched_terms[:10]],
                "matched_hyperedges": [],
                "reasoning_path": _text_reasoning_path(matched_terms, unit["context_id"]),
                "returned_chars": len(unit["text"]),
            }
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    ranked = sorted(hits, key=lambda hit: (-hit["retrieval_score"], hit["context_id"]))[:top_k]
    for hit in ranked:
        hit["retrieval_latency_ms"] = round(elapsed_ms, 4)
    return ranked


def _memory_text(record: dict[str, Any]) -> str:
    trace = record.get("trace", {})
    task_meta = trace.get("task_meta", {}) if isinstance(trace, dict) else {}
    pieces = [
        record.get("memory_id", ""),
        record.get("source_agent", ""),
        record.get("task_topic", ""),
        record.get("summary", ""),
        record.get("answer", ""),
        _stringify_value(record.get("tags", [])),
        _stringify_value(record.get("evidence", [])),
        _stringify_value(record.get("supporting_facts", [])),
        _stringify_value(record.get("state_refs", [])),
    ]
    if isinstance(task_meta, dict):
        pieces.extend(
            [
                task_meta.get("task_id", ""),
                task_meta.get("group_id", ""),
                task_meta.get("turn_id", ""),
                task_meta.get("dataset_type", ""),
                task_meta.get("question", ""),
                _stringify_value(task_meta.get("expected_reuse", [])),
            ]
        )
    return " ".join(str(piece or "") for piece in pieces)


def search_text_memory(memory: list[dict[str, Any]], query: str, *, top_k: int = 2) -> list[dict[str, Any]]:
    """Search historical memory records with a plain-text lexical baseline."""

    if not memory:
        return []

    started = time.perf_counter()
    hits: list[dict[str, Any]] = []
    for record in memory:
        candidate_text = _memory_text(record)
        score, matched_terms = _score_text(query, candidate_text)
        if score <= 0:
            continue
        hit = dict(record)
        hit.update(
            {
                "retrieval_backend": "text_memory",
                "retrieval_score": score,
                "text_score": score,
                "matched_terms": matched_terms[:20],
                "matched_nodes": [f"text_term:{term}" for term in matched_terms[:12]],
                "matched_hyperedges": [],
                "reasoning_path": _text_reasoning_path(matched_terms, str(record.get("memory_id", ""))),
                "returned_chars": len(str(record.get("summary", ""))),
            }
        )
        hits.append(hit)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    ranked = sorted(hits, key=lambda hit: (-hit["retrieval_score"], hit.get("memory_id", "")))[:top_k]
    for hit in ranked:
        hit["retrieval_latency_ms"] = round(elapsed_ms, 4)
    return ranked


def _hit_id(hit: dict[str, Any]) -> str:
    return str(hit.get("memory_id") or hit.get("context_id") or "")


def retrieval_metrics(
    hits: list[dict[str, Any]],
    *,
    expected_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    total_relevant: int | None = None,
    top_k: int | None = None,
) -> dict[str, float]:
    """Summarize retrieval quality for either text or hypergraph hits."""

    k = top_k or len(hits)
    evaluated = hits[:k] if k else hits
    expected = {str(item) for item in (expected_ids or []) if str(item)}
    hit_ids = [_hit_id(hit) for hit in evaluated]
    relevant_hits = [hit_id for hit_id in hit_ids if hit_id in expected] if expected else []
    first_relevant_rank = 0
    if expected:
        for index, hit_id in enumerate(hit_ids, start=1):
            if hit_id in expected:
                first_relevant_rank = index
                break

    denominator = total_relevant if total_relevant is not None else len(expected)
    returned_chars = sum(int(hit.get("returned_chars", len(str(hit.get("summary", hit.get("text", ""))))) or 0) for hit in evaluated)
    latency_values = [float(hit.get("retrieval_latency_ms", 0.0) or 0.0) for hit in evaluated]
    explanation_items = sum(len(hit.get("reasoning_path", []) or []) for hit in evaluated)

    return {
        "hit_rate": 1.0 if evaluated else 0.0,
        "hit_count": float(len(evaluated)),
        "precision_at_k": len(relevant_hits) / len(evaluated) if expected and evaluated else 0.0,
        "recall_at_k": len(set(relevant_hits)) / denominator if expected and denominator else 0.0,
        "mrr": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
        "avg_retrieval_score": sum(float(hit.get("retrieval_score", 0.0) or 0.0) for hit in evaluated) / len(evaluated)
        if evaluated
        else 0.0,
        "returned_chars": float(returned_chars),
        "avg_returned_chars": returned_chars / len(evaluated) if evaluated else 0.0,
        "retrieval_latency_ms": max(latency_values) if latency_values else 0.0,
        "explanation_items": float(explanation_items),
    }


def compare_retrieval_sets(
    *,
    text_hits: list[dict[str, Any]],
    hypergraph_hits: list[dict[str, Any]],
    expected_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Compare plain-text retrieval against hypergraph retrieval."""

    text_metrics = retrieval_metrics(text_hits, expected_ids=expected_ids, top_k=top_k)
    hypergraph_metrics = retrieval_metrics(hypergraph_hits, expected_ids=expected_ids, top_k=top_k)
    text_ids = {_hit_id(hit) for hit in text_hits if _hit_id(hit)}
    hypergraph_ids = {_hit_id(hit) for hit in hypergraph_hits if _hit_id(hit)}
    overlap = text_ids & hypergraph_ids

    return {
        "text": text_metrics,
        "hypergraph": hypergraph_metrics,
        "overlap_count": len(overlap),
        "overlap_ids": sorted(overlap),
        "hit_count_delta": hypergraph_metrics["hit_count"] - text_metrics["hit_count"],
        "precision_delta": hypergraph_metrics["precision_at_k"] - text_metrics["precision_at_k"],
        "recall_delta": hypergraph_metrics["recall_at_k"] - text_metrics["recall_at_k"],
        "returned_chars_delta": hypergraph_metrics["returned_chars"] - text_metrics["returned_chars"],
        "latency_delta_ms": hypergraph_metrics["retrieval_latency_ms"] - text_metrics["retrieval_latency_ms"],
    }
