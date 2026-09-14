from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from .contracts import RetrievalRequest
from .retrieval import search_keyword_memory
from .service import MemoryService


def _label(number: int) -> str:
    letters = []
    for _ in range(5):
        letters.append(chr(ord("a") + number % 26))
        number //= 26
    return "item" + "".join(reversed(letters))


def run(*, record_count: int = 1000, queries: int = 20) -> dict[str, Any]:
    if record_count < 1 or queries < 1:
        raise ValueError("record_count and queries must be positive")
    with MemoryService(":memory:", namespace="synthetic-index-check") as service:
        service.begin_batch(
            "fixture",
            task_family="synthetic_lookup",
            dataset_id="synthetic-index-check",
        )
        records = [
            {
                "memory_id": _label(i),
                "source_agent": "fixture",
                "task_topic": _label(i),
                "summary": f"{_label(i)} has a verified registry entry.",
                "evidence": [f"Registry entry for {_label(i)}."],
                "entities": [],
            }
            for i in range(record_count)
        ]
        started = time.perf_counter()
        service.write_knowledge("fixture", records, operation_id="bulk")
        build_ms = (time.perf_counter() - started) * 1000
        normalized = service.store.scoped_records(service.namespace, "fixture")
        samples: dict[str, list[float]] = {"legacy_keyword": [], "indexed_keyword": []}
        correct = {mode: 0 for mode in samples}
        loaded = []
        for i in range(queries):
            query = _label(i * 47 % record_count)
            request = RetrievalRequest(
                "fixture", query, layers=("knowledge",), mode="keyword", top_k=1
            )
            order = tuple(samples) if i % 2 == 0 else tuple(reversed(samples))
            for mode in order:
                started = time.perf_counter()
                if mode == "legacy_keyword":
                    hits = search_keyword_memory(normalized, query, top_k=1)
                    answer = hits[0]["memory_id"] if hits else None
                else:
                    result = service.retrieve(request)
                    hits = result["hits"]
                    answer = hits[0]["memory"]["memory_id"] if hits else None
                    loaded.append(result["diagnostics"]["records_loaded"])
                samples[mode].append((time.perf_counter() - started) * 1000)
                correct[mode] += int(answer == query)
    return {
        "fixture_type": "synthetic_exact_lookup_only",
        "answer_accuracy": None,
        "record_count": record_count,
        "query_count": queries,
        "write_and_index_ms": round(build_ms, 3),
        "results": {
            mode: {
                "correct_lookups": correct[mode],
                "mean_ms": round(statistics.mean(values), 3),
                "p95_ms": round(
                    sorted(values)[max(0, (95 * len(values) + 99) // 100 - 1)], 3
                ),
            }
            for mode, values in samples.items()
        },
        "indexed_max_records_loaded": max(loaded),
        "limitation": "This verifies indexed exact lookup and measures its API latency. It does not measure semantic relevance, answer accuracy or end-to-end agent performance.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a synthetic memory-index scaling check."
    )
    parser.add_argument("--records", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(
        run(record_count=args.records, queries=args.queries),
        ensure_ascii=False,
        indent=2,
    )
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
