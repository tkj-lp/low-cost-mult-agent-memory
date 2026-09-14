from __future__ import annotations

import json
import math
import time
from typing import Any, Iterable

from .retrieval import RetrievalMode, search_memory_by_mode

RETRIEVAL_MODES: tuple[RetrievalMode, ...] = (
    "keyword",
    "tag",
    "semantic",
    "hypergraph",
    "fusion",
)


def _hit_identifier(hit: dict[str, Any]) -> set[str]:
    return {
        str(value)
        for value in (hit.get("memory_id"), hit.get("global_memory_id"))
        if value
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[position]


def _safe_f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_retrieval_modes(
    memory: list[dict[str, Any]],
    cases: Iterable[dict[str, Any]],
    *,
    top_k: int = 3,
    modes: Iterable[RetrievalMode] = RETRIEVAL_MODES,
) -> dict[str, Any]:
    """在同一记忆、查询和标注上公平比较全部检索模式。"""

    prepared_cases = [case for case in cases if isinstance(case, dict)]
    reports: dict[str, Any] = {}
    for mode in modes:
        rows: list[dict[str, Any]] = []
        latencies: list[float] = []
        relevant_returned = 0
        returned = 0
        relevant_total = 0
        reciprocal_ranks: list[float] = []
        chain_relevant = 0
        returned_chars = 0

        for case in prepared_cases:
            query = str(case.get("query") or "")
            expected_ids = {
                str(value) for value in case.get("expected_memory_ids", []) if value
            }
            available_ids = {
                str(value) for value in case.get("available_memory_ids", []) if value
            }
            case_memory = (
                [
                    record
                    for record in memory
                    if {
                        str(record.get("memory_id") or ""),
                        str(record.get("global_memory_id") or ""),
                    }
                    & available_ids
                ]
                if "available_memory_ids" in case
                else memory
            )
            started = time.perf_counter()
            hits = search_memory_by_mode(case_memory, query, top_k=top_k, mode=mode)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(elapsed_ms)
            returned += len(hits)
            relevant_total += len(expected_ids)
            compact_payload = [
                hit.get("reusable_memory")
                or {
                    "memory_id": hit.get("memory_id"),
                    "summary": hit.get("summary"),
                    "memory_type": hit.get("memory_type"),
                }
                for hit in hits
            ]
            returned_chars += len(
                json.dumps(compact_payload, ensure_ascii=False, default=str)
            )

            matched_ids: set[str] = set()
            first_relevant_rank = 0
            chain_matched_ids: set[str] = set()
            for rank, hit in enumerate(hits, start=1):
                identifiers = _hit_identifier(hit)
                overlap = identifiers & expected_ids
                if not overlap:
                    continue
                matched_ids.update(overlap)
                if first_relevant_rank == 0:
                    first_relevant_rank = rank
                chain = (
                    hit.get("evidence_chain")
                    if isinstance(hit.get("evidence_chain"), dict)
                    else {}
                )
                if (
                    bool(chain.get("valid"))
                    and float(chain.get("completeness", 0.0) or 0.0) >= 0.8
                ):
                    chain_matched_ids.update(overlap)

            relevant_returned += len(matched_ids)
            chain_relevant += len(chain_matched_ids)
            reciprocal_ranks.append(
                1.0 / first_relevant_rank if first_relevant_rank else 0.0
            )
            row_precision = len(matched_ids) / len(hits) if hits else 0.0
            row_recall = len(matched_ids) / len(expected_ids) if expected_ids else 0.0
            rows.append(
                {
                    "case_id": case.get("case_id"),
                    "query": query,
                    "expected_memory_ids": sorted(expected_ids),
                    "hit_ids": [
                        str(hit.get("memory_id") or hit.get("global_memory_id"))
                        for hit in hits
                    ],
                    "precision_at_k": row_precision,
                    "recall_at_k": row_recall,
                    "f1_at_k": _safe_f1(row_precision, row_recall),
                    "reciprocal_rank": (
                        1.0 / first_relevant_rank if first_relevant_rank else 0.0
                    ),
                    "evidence_chain_relevant_count": len(chain_matched_ids),
                    "latency_ms": elapsed_ms,
                }
            )

        precision = relevant_returned / returned if returned else 0.0
        recall = relevant_returned / relevant_total if relevant_total else 0.0
        chain_recall = chain_relevant / relevant_total if relevant_total else 0.0
        reports[mode] = {
            "summary": {
                "case_count": len(prepared_cases),
                "precision_at_k": precision,
                "recall_at_k": recall,
                "f1_at_k": _safe_f1(precision, recall),
                "mrr": (
                    sum(reciprocal_ranks) / len(reciprocal_ranks)
                    if reciprocal_ranks
                    else 0.0
                ),
                "evidence_chain_recall_at_k": chain_recall,
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
                "p95_latency_ms": _percentile(latencies, 0.95),
                "returned_chars": returned_chars,
            },
            "rows": rows,
        }

    single_modes = [
        mode for mode in ("keyword", "tag", "semantic", "hypergraph") if mode in reports
    ]
    best_single_score = max(
        (float(reports[mode]["summary"]["f1_at_k"]) for mode in single_modes),
        default=0.0,
    )
    hypergraph_score = float(
        reports.get("hypergraph", {}).get("summary", {}).get("f1_at_k", 0.0)
    )
    fusion_score = float(
        reports.get("fusion", {}).get("summary", {}).get("f1_at_k", 0.0)
    )
    best_single_modes = [
        mode
        for mode in single_modes
        if float(reports[mode]["summary"]["f1_at_k"]) == best_single_score
    ]
    all_ranked_modes = sorted(
        reports,
        key=lambda mode: (-float(reports[mode]["summary"]["f1_at_k"]), mode),
    )
    hypergraph_rank = (
        all_ranked_modes.index("hypergraph") + 1
        if "hypergraph" in all_ranked_modes
        else 0
    )
    return {
        "meta": {"top_k": top_k, "case_count": len(prepared_cases)},
        "modes": reports,
        "comparison": {
            "best_single_modes": best_single_modes,
            "best_single_f1": best_single_score,
            "hypergraph_f1": hypergraph_score,
            "fusion_f1": fusion_score,
            "hypergraph_is_best_single": hypergraph_score >= best_single_score,
            "overall_ranking": all_ranked_modes,
            "hypergraph_rank": hypergraph_rank,
            "hypergraph_is_at_most_second_to_fusion": hypergraph_score
            >= best_single_score
            and hypergraph_rank in {1, 2}
            and (hypergraph_rank == 1 or all_ranked_modes[0] == "fusion"),
        },
    }
