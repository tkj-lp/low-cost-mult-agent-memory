from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from competition_agents.embedding import embed_text
from competition_agents.graph import build_graph
from competition_agents.state import initial_state
from hypergraph_memory_method.schema import STRUCTURED_NODE_KINDS
from hypergraph_memory_method.scoring import node_kind
from hypergraph_memory_method.text import tokenize
from run_demo import load_tasks, parse_modes, task_meta_from_item


def _cosine(left: list[float], right: list[float]) -> float:
    dimensions = min(len(left), len(right))
    if dimensions == 0:
        return 0.0
    left_values = left[:dimensions]
    right_values = right[:dimensions]
    dot = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(a * a for a in left_values))
    right_norm = math.sqrt(sum(b * b for b in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return (dot / (left_norm * right_norm) + 1.0) / 2.0


def _query_text(task_item: dict[str, Any]) -> str:
    return str(task_item.get("question") or task_item.get("task") or "")


def _expected_turns(task_item: dict[str, Any]) -> set[str]:
    turns: set[str] = set()
    for value in task_item.get("expected_reuse", []) or []:
        turn, _, _ = str(value).partition(".")
        if turn:
            turns.add(turn)
    return turns


def _hit_turn(hit: dict[str, Any]) -> str:
    trace = hit.get("trace", {})
    if isinstance(trace, dict):
        task_meta = trace.get("task_meta", {})
        if isinstance(task_meta, dict) and task_meta.get("turn_id"):
            return str(task_meta["turn_id"])
    return ""


def _expected_reuse_hit(task_item: dict[str, Any], hits: list[dict[str, Any]]) -> bool | None:
    expected_turns = _expected_turns(task_item)
    if not expected_turns:
        return None
    return any(_hit_turn(hit) in expected_turns for hit in hits)


def _keyword_baseline(memory: list[dict[str, Any]], query: str, *, top_k: int) -> list[dict[str, Any]]:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return []
    scored: list[tuple[float, dict[str, Any]]] = []
    for record in memory:
        tags = {str(tag).lower().replace("_", "-") for tag in record.get("tags", [])}
        text_tokens = set(
            tokenize(
                " ".join(
                    [
                        str(record.get("task_topic", "")),
                        str(record.get("summary", "")),
                        " ".join(tags),
                    ]
                )
            )
        )
        keyword_overlap = len(query_tokens & text_tokens)
        tag_overlap = len(query_tokens & tags)
        score = keyword_overlap + tag_overlap * 3.0
        if score > 0:
            scored.append((score, record))
    return [record for _, record in sorted(scored, key=lambda item: (-item[0], item[1].get("memory_id", "")))[:top_k]]


def _embedding_baseline(memory: list[dict[str, Any]], query: str, *, top_k: int) -> list[dict[str, Any]]:
    query_embedding = embed_text(query).vector
    scored: list[tuple[float, dict[str, Any]]] = []
    for record in memory:
        embedding = record.get("embedding")
        if not isinstance(embedding, list):
            continue
        try:
            score = _cosine(query_embedding, [float(value) for value in embedding])
        except (TypeError, ValueError):
            continue
        scored.append((score, record))
    return [record for score, record in sorted(scored, key=lambda item: (-item[0], item[1].get("memory_id", "")))[:top_k] if score > 0]


def _hit_node_stats(hits: list[dict[str, Any]]) -> dict[str, float]:
    total = 0
    structured = 0
    keyword = 0
    noisy_keyword = 0
    for hit in hits:
        for node in hit.get("matched_nodes", []):
            total += 1
            kind = node_kind(str(node))
            if kind in STRUCTURED_NODE_KINDS:
                structured += 1
            if kind == "keyword":
                keyword += 1
                _, _, value = str(node).partition(":")
                if any(char.isdigit() for char in value):
                    noisy_keyword += 1
    return {
        "matched_node_count": float(total),
        "structured_hit_ratio": structured / total if total else 0.0,
        "keyword_hit_ratio": keyword / total if total else 0.0,
        "keyword_noise_ratio": noisy_keyword / keyword if keyword else 0.0,
    }


def _score_stats(hits: list[dict[str, Any]]) -> dict[str, float]:
    fields = [
        "retrieval_score",
        "hypergraph_score",
        "embedding_score",
        "gap_coverage_score",
        "reuse_score",
        "evidence_score",
        "state_ref_score",
        "strategy_score",
        "experience_score",
    ]
    stats = {}
    for field in fields:
        values = [float(hit.get(field, 0.0) or 0.0) for hit in hits]
        stats[f"avg_{field}"] = sum(values) / len(values) if values else 0.0
    return stats


def _summarize_rates(values: list[bool | None]) -> tuple[int, float]:
    filtered = [value for value in values if value is not None]
    return len(filtered), sum(1 for value in filtered if value) / len(filtered) if filtered else 0.0


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    tasks = load_tasks(args)
    mode = parse_modes(args.modes)[0]
    graph = build_graph()
    global_memory: list[dict[str, Any]] = []
    group_memory: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    totals = {
        "message_count": 0,
        "text_chars": 0,
        "latent_transfer_count": 0,
        "latent_bytes": 0,
    }
    started = time.perf_counter()

    for index, task_item in enumerate(tasks, start=1):
        group_id = task_item.get("group_id", "default")
        prior_memory = group_memory.get(group_id, []) if args.dataset_type == "continuous" else global_memory
        query = _query_text(task_item)
        keyword_hits = _keyword_baseline(prior_memory, query, top_k=args.top_k)
        embedding_hits = _embedding_baseline(prior_memory, query, top_k=args.top_k)

        result = graph.invoke(initial_state(task_item["task"], mode, prior_memory, task_meta=task_meta_from_item(task_item)))
        if args.dataset_type == "continuous":
            group_memory[group_id] = result["memory"]
        else:
            global_memory = result["memory"]

        metrics = result.get("metrics", {})
        for key in totals:
            totals[key] += int(metrics.get(key, 0))

        hypergraph_hits = result.get("memory_hits", [])
        node_stats = _hit_node_stats(hypergraph_hits)
        score_stats = _score_stats(hypergraph_hits)
        row = {
            "task_index": index,
            "task_id": task_item.get("task_id"),
            "group_id": group_id,
            "turn_id": task_item.get("turn_id"),
            "expected_reuse": task_item.get("expected_reuse", []),
            "prior_memory_count": len(prior_memory),
            "hypergraph_hit_count": len(hypergraph_hits),
            "keyword_hit_count": len(keyword_hits),
            "embedding_hit_count": len(embedding_hits),
            "hypergraph_expected_reuse_hit": _expected_reuse_hit(task_item, hypergraph_hits),
            "keyword_expected_reuse_hit": _expected_reuse_hit(task_item, keyword_hits),
            "embedding_expected_reuse_hit": _expected_reuse_hit(task_item, embedding_hits),
            "hypergraph_hit_ids": [hit.get("memory_id") for hit in hypergraph_hits],
            "keyword_hit_ids": [hit.get("memory_id") for hit in keyword_hits],
            "embedding_hit_ids": [hit.get("memory_id") for hit in embedding_hits],
            **node_stats,
            **score_stats,
            "message_count": metrics.get("message_count", 0),
            "text_chars": metrics.get("text_chars", 0),
            "latent_transfer_count": metrics.get("latent_transfer_count", 0),
            "latent_bytes": metrics.get("latent_bytes", 0),
        }
        rows.append(row)

    expected_count, hypergraph_expected_rate = _summarize_rates(
        [row["hypergraph_expected_reuse_hit"] for row in rows]
    )
    _, keyword_expected_rate = _summarize_rates([row["keyword_expected_reuse_hit"] for row in rows])
    _, embedding_expected_rate = _summarize_rates([row["embedding_expected_reuse_hit"] for row in rows])
    hit_tasks = [row for row in rows if row["hypergraph_hit_count"] > 0]
    avg = lambda key: sum(float(row.get(key, 0.0) or 0.0) for row in hit_tasks) / len(hit_tasks) if hit_tasks else 0.0

    return {
        "meta": {
            "dataset_type": args.dataset_type,
            "dataset_file": args.dataset_file or "<builtin>",
            "mode": mode,
            "tasks": len(tasks),
            "top_k": args.top_k,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        },
        "summary": {
            "memory_hit_rate": sum(1 for row in rows if row["hypergraph_hit_count"] > 0) / len(rows) if rows else 0.0,
            "expected_reuse_task_count": expected_count,
            "hypergraph_expected_reuse_hit_rate": hypergraph_expected_rate,
            "keyword_expected_reuse_hit_rate": keyword_expected_rate,
            "embedding_expected_reuse_hit_rate": embedding_expected_rate,
            "avg_structured_hit_ratio": avg("structured_hit_ratio"),
            "avg_keyword_hit_ratio": avg("keyword_hit_ratio"),
            "avg_keyword_noise_ratio": avg("keyword_noise_ratio"),
            "avg_retrieval_score": avg("avg_retrieval_score"),
            "avg_hypergraph_score": avg("avg_hypergraph_score"),
            "avg_embedding_score": avg("avg_embedding_score"),
            **totals,
        },
        "rows": rows,
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "retrieval_eval.json"
    md_path = out_dir / "retrieval_eval.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Hypergraph Memory Retrieval Evaluation",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"- `{key}`: {value:.4f}")
        else:
            lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Per Task", ""])
    for row in report["rows"]:
        lines.extend(
            [
                f"### {row['task_index']}. {row['task_id']}",
                "",
                f"- expected_reuse: `{row['expected_reuse']}`",
                f"- hypergraph_hit_ids: `{row['hypergraph_hit_ids']}`",
                f"- keyword_hit_ids: `{row['keyword_hit_ids']}`",
                f"- embedding_hit_ids: `{row['embedding_hit_ids']}`",
                f"- hypergraph_expected_reuse_hit: `{row['hypergraph_expected_reuse_hit']}`",
                f"- structured_hit_ratio: `{row['structured_hit_ratio']:.4f}`",
                f"- keyword_hit_ratio: `{row['keyword_hit_ratio']:.4f}`",
                f"- keyword_noise_ratio: `{row['keyword_noise_ratio']:.4f}`",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate real hypergraph-memory retrieval quality.")
    parser.add_argument("--dataset-type", choices=["builtin", "multihop", "continuous"], default="continuous")
    parser.add_argument(
        "--dataset-file",
        default="examples/competition_agents/data/hotpot_continuous_tasks.json",
        help="Path to multihop or continuous JSON data.",
    )
    parser.add_argument("--limit", type=int, default=1, help="Limit samples or continuous groups.")
    parser.add_argument("--modes", default="protocol", help="Use the first mode only.")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--out-dir",
        default="examples/competition_agents/hypergraph_memory_method/artifacts",
        help="Directory for retrieval_eval.json and retrieval_eval.md.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_evaluation(args)
    paths = write_outputs(report, Path(args.out_dir))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"retrieval_eval_json: {paths['json']}")
    print(f"retrieval_eval_markdown: {paths['markdown']}")


if __name__ == "__main__":
    main()
