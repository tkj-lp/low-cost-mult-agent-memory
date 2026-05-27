from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from competition_agents.graph import build_graph
from competition_agents.state import initial_state
from hypergraph_memory_method.visualization import export_visualization
from run_demo import load_tasks, parse_modes, task_meta_from_item


def _flatten_group_memory(group_memory: dict[str, list[dict[str, Any]]], global_memory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if group_memory:
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for memory in group_memory.values():
            for record in memory:
                memory_id = str(record.get("memory_id", ""))
                key = f"{memory_id}:{record.get('created_at', '')}"
                if key in seen:
                    continue
                seen.add(key)
                records.append(record)
        return records
    return global_memory


def run_real_tasks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tasks = load_tasks(args)
    mode = parse_modes(args.modes)[0]
    graph = build_graph()
    global_memory: list[dict[str, Any]] = []
    group_memory: dict[str, list[dict[str, Any]]] = {}
    retrievals: list[dict[str, Any]] = []
    started = time.perf_counter()

    for task_item in tasks:
        group_id = task_item.get("group_id", "default")
        memory = group_memory.get(group_id, []) if args.dataset_type == "continuous" else global_memory
        result = graph.invoke(initial_state(task_item["task"], mode, memory, task_meta=task_meta_from_item(task_item)))
        if args.dataset_type == "continuous":
            group_memory[group_id] = result["memory"]
        else:
            global_memory = result["memory"]
        retrievals.append(
            {
                "task_id": task_item.get("task_id"),
                "query": task_item.get("question") or task_item["task"],
                "hits": result.get("memory_hits", []),
                "final_answer": result.get("final_answer", ""),
            }
        )

    records = _flatten_group_memory(group_memory, global_memory)
    meta = {
        "dataset_type": args.dataset_type,
        "dataset_file": args.dataset_file or "<builtin>",
        "mode": mode,
        "tasks": len(tasks),
        "memory_records": len(records),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    return records, retrievals, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real competition-agent tasks and visualize hypergraph memory.")
    parser.add_argument("--dataset-type", choices=["builtin", "multihop", "continuous"], default="continuous")
    parser.add_argument(
        "--dataset-file",
        default="examples/competition_agents/data/hotpot_continuous_tasks.json",
        help="Path to multihop or continuous JSON data.",
    )
    parser.add_argument("--limit", type=int, default=1, help="Limit samples or continuous groups.")
    parser.add_argument("--modes", default="protocol", help="Use the first mode only for visualization.")
    parser.add_argument(
        "--out-dir",
        default="examples/competition_agents/hypergraph_memory_method/artifacts",
        help="Directory for memory_graph.html and memory_graph.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, retrievals, meta = run_real_tasks(args)
    paths = export_visualization(
        records=records,
        retrievals=retrievals,
        out_dir=Path(args.out_dir),
        title=(
            "真实运行超图记忆可视化 "
            f"({meta['dataset_type']}, {meta['mode']}, tasks={meta['tasks']}, memories={meta['memory_records']})"
        ),
    )
    print(f"visualization_html: {paths['html']}")
    print(f"visualization_json: {paths['json']}")
    print(f"run_meta: {meta}")


if __name__ == "__main__":
    main()
