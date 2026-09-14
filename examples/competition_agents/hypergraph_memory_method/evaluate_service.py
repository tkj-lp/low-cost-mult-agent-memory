from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .contracts import RetrievalRequest
from .encoders import SentenceTransformerEncoder
from .service import MemoryService


def evaluate_service(
    service: MemoryService,
    cases: Iterable[dict[str, Any]],
    *,
    modes: tuple[str, ...] = ("keyword", "hypergraph"),
    repeats: int = 3,
) -> dict[str, Any]:
    """Score annotations after retrieval; the request never receives gold IDs."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    prepared = list(cases)
    output: dict[str, Any] = {
        "scope": "memory_retrieval_only",
        "case_count": len(prepared),
        "answer_accuracy": None,
        "modes": {},
    }
    for mode in modes:
        rows = []
        timings = []
        for case in prepared:
            request = replace(RetrievalRequest(**case["request"]), mode=mode)
            relevant = {
                service.get_memory(request.batch_id, str(value))["global_memory_id"]
                for value in case.get("relevant_memory_ids", [])
            }
            results = [service.retrieve(request) for _ in range(repeats)]
            rankings = [
                [hit["memory_id"] for hit in result["hits"]] for result in results
            ]
            if any(ranking != rankings[0] for ranking in rankings[1:]):
                raise ValueError(
                    "retrieval ranking changed across identical repeated requests"
                )
            ids = rankings[0]
            correct = len(set(ids) & relevant)
            first = next((i for i, key in enumerate(ids, 1) if key in relevant), 0)
            latency = [result["diagnostics"]["total_ms"] for result in results]
            timings.extend(latency)
            rows.append(
                {
                    "case_id": case.get("case_id"),
                    "returned_ids": ids,
                    "precision_at_k": correct / request.top_k,
                    "returned_precision": correct / len(ids) if ids else 0.0,
                    "recall_at_k": correct / len(relevant) if relevant else None,
                    "reciprocal_rank": 1.0 / first if first else 0.0,
                    "no_match_correct": not ids if not relevant else None,
                    "first_query_ms": latency[0],
                    "median_query_ms": statistics.median(latency),
                    "payload_bytes": results[0]["diagnostics"]["payload_bytes"],
                    "records_loaded": results[0]["diagnostics"]["records_loaded"],
                }
            )
        recalls = [row["recall_at_k"] for row in rows if row["recall_at_k"] is not None]
        output["modes"][mode] = {
            "cases": rows,
            "summary": {
                "precision_at_k": (
                    statistics.mean(row["precision_at_k"] for row in rows)
                    if rows
                    else None
                ),
                "recall_at_k": statistics.mean(recalls) if recalls else None,
                "mrr": (
                    statistics.mean(row["reciprocal_rank"] for row in rows)
                    if rows
                    else None
                ),
                "mean_query_ms": statistics.mean(timings) if timings else None,
                "p95_query_ms": (
                    sorted(timings)[max(0, (95 * len(timings) + 99) // 100 - 1)]
                    if timings
                    else None
                ),
                "mean_payload_bytes": (
                    statistics.mean(row["payload_bytes"] for row in rows)
                    if rows
                    else None
                ),
            },
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate memory interfaces using held-out retrieval annotations."
    )
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--modes", nargs="+", default=["keyword", "hypergraph"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--embedding-model", type=Path)
    parser.add_argument("--encoder-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    encoder = None
    if args.embedding_model:
        if not args.encoder_id:
            parser.error("--embedding-model requires --encoder-id")
        encoder = SentenceTransformerEncoder(
            args.embedding_model, model_id=args.encoder_id
        )
    with MemoryService(
        ":memory:", namespace=snapshot["namespace"], encoder=encoder
    ) as service:
        service.restore_snapshot(snapshot)
        report = evaluate_service(
            service, cases, modes=tuple(args.modes), repeats=args.repeats
        )
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
