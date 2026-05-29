from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from competition_agents.restricted_python_codeact import run_restricted_python


def run_task(task: dict[str, Any], memory: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """执行一条受限 Python CodeAct 任务。"""

    code = task["reference_code"]
    execution_result = run_restricted_python(
        code,
        inputs=task.get("inputs", {}),
        memory=memory,
        allowed_imports=task.get("allowed_imports", []),
        timeout_ms=task.get("timeout_ms", 3000),
    )
    expected = task["expected"]
    exact_match = execution_result["result"] == expected
    memory[task["id"]] = {
        "task_id": task["id"],
        "result": execution_result["result"],
        "tags": task.get("tags", []),
        "execution_mode": execution_result["execution_mode"],
        "sandbox_status": execution_result["sandbox_status"],
    }
    return {
        "id": task["id"],
        "task": task["task"],
        "expected": expected,
        "execution_result": {
            **execution_result,
            "code_source": "reference_code",
            "code": code,
        },
        "exact_match": exact_match,
        "memory_hit": "memory" in code,
        "tags": task.get("tags", []),
    }


def load_tasks(path: Path, limit: int) -> list[dict[str, Any]]:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    return tasks[:limit] if limit else tasks


def print_case(result: dict[str, Any]) -> None:
    execution = result["execution_result"]
    print(f"\ncase: {result['id']}")
    print(f"task: {result['task']}")
    print(f"allowed_imports: {execution['allowed_imports']}")
    print(f"code_source: {execution['code_source']}")
    print(f"code:\n{execution['code']}")
    print(f"result: {json.dumps(execution['result'], ensure_ascii=False, sort_keys=True)}")
    print(f"expected: {json.dumps(result['expected'], ensure_ascii=False, sort_keys=True)}")
    print(
        "execution: "
        f"mode={execution['execution_mode']}, "
        f"sandbox_status={execution['sandbox_status']}, "
        f"exit_code={execution['exit_code']}, "
        f"elapsed_ms={execution['elapsed_ms']:.4f}, "
        f"exact_match={result['exact_match']}"
    )
    if execution.get("blocked_reason"):
        print(f"blocked_reason: {execution['blocked_reason']}")


def print_summary(results: list[dict[str, Any]], dataset_file: str, limit: int) -> None:
    exact = sum(1 for result in results if result["exact_match"])
    total_elapsed = sum(result["execution_result"]["elapsed_ms"] for result in results)
    memory_hits = sum(1 for result in results if result["memory_hit"])
    metrics = {
        "task_count": len(results),
        "exact_match": exact / len(results) if results else None,
        "memory_hits": memory_hits,
        "memory_hit_rate": memory_hits / len(results) if results else 0.0,
        "elapsed_ms": total_elapsed,
        "message_count": len(results),
        "text_chars": sum(len(result["task"]) for result in results),
        "latent_transfer_count": 0,
        "latent_bytes": 0,
        "tool_execution_count": len(results),
        "blocked_count": sum(1 for result in results if result["execution_result"]["sandbox_status"] == "blocked"),
    }
    print("\n=== codeact summary ===")
    print(
        "tasks | exact_match | tool_execution_count | memory_hits | blocked | elapsed_ms\n"
        f"{len(results)} | {metrics['exact_match']:.2%} | {len(results)} | "
        f"{memory_hits} | {metrics['blocked_count']} | {total_elapsed:.4f}"
    )
    payload = {
        "schema_version": 1,
        "experiment_mode": "codeact_tools",
        "dataset_type": "codeact",
        "dataset_file": dataset_file,
        "limit": limit,
        "task_count": len(results),
        "runs": [
            {
                "mode": "codeact_tools",
                "experiment_mode": "codeact_tools",
                "dataset_type": "codeact",
                "dataset_file": dataset_file,
                "limit": limit,
                "tasks": len(results),
                "task_count": len(results),
                "metrics": metrics,
                "memory_hits": memory_hits,
                "memory_hit_rate": metrics["memory_hit_rate"],
                "elapsed_ms": total_elapsed,
                "exact_match": metrics["exact_match"],
            }
        ],
    }
    print("\n=== run_result_json ===")
    print(json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run restricted Python CodeAct samples.")
    parser.add_argument(
        "--dataset-file",
        default="examples/competition_agents/data/codeact_tasks.json",
        help="CodeAct task dataset.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit task count. 0 means all tasks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_tasks(Path(args.dataset_file), args.limit)
    memory: dict[str, dict[str, Any]] = {}
    print("=== restricted Python CodeAct samples ===")
    print(f"dataset_file: {args.dataset_file}")
    results = [run_task(task, memory) for task in tasks]
    for result in results:
        print_case(result)
    print_summary(results, args.dataset_file, args.limit)


if __name__ == "__main__":
    main()
