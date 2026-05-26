from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from competition_agents.graph import build_graph
from competition_agents.llm.client import get_llm_provider
from competition_agents.state import initial_state


BUILTIN_TASKS = [
    "Design a LangGraph multi-agent protocol with shared memory",
    "Reuse the previous LangGraph protocol memory for openEuler deployment",
    "Compare text-mode and protocol-mode communication cost",
    "Design latent state exchange with embedding references",
    "Reuse latent state exchange memory for retrieval planning",
    "Design shared memory metadata for competition evaluation",
    "Reuse shared memory metadata for report generation",
    "Plan CodeAct tool execution in a lightweight sandbox",
    "Reuse CodeAct execution memory for performance analysis",
    "Summarize the complete multi-agent competition architecture",
]


def normalize_answer(text: str) -> str:
    """把答案标准化，方便计算 exact_match。"""

    text = text.lower().strip()
    text = re.sub(r"^(answer|final answer)\s*:\s*", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def build_multihop_task(sample: dict[str, Any]) -> str:
    """把 HotpotQA-style 样例转换成 Agent 可读取的任务文本。"""

    return (
        "HotpotQA-style multi-hop QA task.\n"
        "Answer using only the provided Context. Return the shortest answer string.\n\n"
        f"Question: {sample['question']}\n"
        f"Context: {json.dumps(sample.get('context', []), ensure_ascii=False)}\n"
    )


def load_builtin_tasks(limit: int | None) -> list[dict[str, Any]]:
    tasks = [
        {
            "task_id": f"builtin-{index:03d}",
            "group_id": "builtin",
            "turn_id": f"turn-{index}",
            "task": task,
            "expected_answer": None,
            "dataset_type": "builtin",
        }
        for index, task in enumerate(BUILTIN_TASKS, start=1)
    ]
    return tasks[:limit] if limit else tasks


def load_multihop_tasks(path: Path, limit: int | None) -> list[dict[str, Any]]:
    samples = json.loads(path.read_text(encoding="utf-8"))
    if limit:
        samples = samples[:limit]
    return [
        {
            "task_id": sample.get("id", f"multihop-{index:03d}"),
            "group_id": sample.get("id", f"multihop-{index:03d}"),
            "turn_id": "turn-1",
            "task": build_multihop_task(sample),
            "question": sample.get("question", ""),
            "expected_answer": sample.get("answer"),
            "context": sample.get("context", []),
            "supporting_facts": sample.get("supporting_facts", []),
            "dataset_type": "multihop",
        }
        for index, sample in enumerate(samples, start=1)
    ]


def load_continuous_tasks(path: Path, limit: int | None) -> list[dict[str, Any]]:
    groups = json.loads(path.read_text(encoding="utf-8"))
    if limit:
        groups = groups[:limit]

    tasks: list[dict[str, Any]] = []
    for group in groups:
        for turn in group.get("tasks", []):
            task_text = (
                "Continuous HotpotQA-style task. Reuse prior memory in the same group when helpful.\n\n"
                f"Group topic: {group.get('topic', '')}\n"
                f"Task type: {turn.get('task_type', '')}\n"
                f"Question: {turn.get('question', '')}\n"
                f"Context: {json.dumps(group.get('context', []), ensure_ascii=False)}\n"
                f"Supporting facts: {json.dumps(group.get('supporting_facts', []), ensure_ascii=False)}\n"
            )
            tasks.append(
                {
                    "task_id": f"{group.get('group_id', 'group')}:{turn.get('turn_id', 'turn')}",
                    "group_id": group.get("group_id", "group"),
                    "turn_id": turn.get("turn_id", "turn"),
                    "task": task_text,
                    "question": turn.get("question", ""),
                    "expected_answer": turn.get("expected_answer"),
                    "context": group.get("context", []),
                    "supporting_facts": group.get("supporting_facts", []),
                    "dataset_type": "continuous",
                    "expected_reuse": turn.get("expected_reuse", []),
                }
            )
    return tasks


def load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dataset_type == "builtin":
        return load_builtin_tasks(args.limit)
    if not args.dataset_file:
        raise ValueError("--dataset-file is required when dataset type is multihop or continuous")
    path = Path(args.dataset_file)
    if args.dataset_type == "multihop":
        return load_multihop_tasks(path, args.limit)
    if args.dataset_type == "continuous":
        return load_continuous_tasks(path, args.limit)
    raise ValueError(f"Unsupported dataset type: {args.dataset_type}")


def parse_modes(raw_modes: str) -> list[str]:
    modes = [mode.strip().lower() for mode in raw_modes.split(",") if mode.strip()]
    invalid = [mode for mode in modes if mode not in {"text", "protocol"}]
    if invalid:
        raise ValueError(f"Unsupported mode(s): {invalid}. Use text and/or protocol.")
    return modes or ["text", "protocol"]


def prediction_from_result(result: dict[str, Any]) -> str:
    execution_result = result.get("execution_result", {})
    return str(execution_result.get("answer_candidate") or result.get("final_answer", ""))


def run_mode(mode: str, tasks: list[dict[str, Any]], *, dataset_type: str) -> dict[str, Any]:
    """运行一种协作模式，并汇总该模式的指标。"""

    graph = build_graph()
    global_memory: list[dict[str, Any]] = []
    group_memory: dict[str, list[dict[str, Any]]] = {}
    total_elapsed = 0.0
    total_memory_hits = 0
    correct = 0
    comparable = 0
    total_metrics = {
        "message_count": 0,
        "text_chars": 0,
        "latent_transfer_count": 0,
        "latent_bytes": 0,
    }

    print(f"\n=== {mode} mode ===")
    for index, task_item in enumerate(tasks, start=1):
        group_id = task_item.get("group_id", "default")
        memory = group_memory.get(group_id, []) if dataset_type == "continuous" else global_memory
        started = time.perf_counter()
        result = graph.invoke(initial_state(task_item["task"], mode, memory))
        elapsed = time.perf_counter() - started
        total_elapsed += elapsed

        if dataset_type == "continuous":
            group_memory[group_id] = result["memory"]
        else:
            global_memory = result["memory"]

        metrics = result["metrics"]
        memory_hits = len(result["memory_hits"])
        total_memory_hits += memory_hits
        for key in total_metrics:
            total_metrics[key] += metrics[key]

        expected_answer = task_item.get("expected_answer")
        prediction = prediction_from_result(result)
        exact_match = None
        if expected_answer:
            comparable += 1
            exact_match = normalize_answer(prediction) == normalize_answer(str(expected_answer))
            correct += int(exact_match)

        print(f"task {index}: {task_item['task_id']}")
        if task_item.get("question"):
            print(f"question: {task_item['question']}")
        print(f"answer: {result['final_answer']}")
        if expected_answer:
            print(f"gold: {expected_answer}")
            print(f"exact_match: {exact_match}")
        if task_item.get("expected_reuse"):
            print(f"expected_reuse: {', '.join(task_item['expected_reuse'])}")
        print("message path:")
        for message in result["messages"]:
            print(
                f"  {message['from']} -> {message['to']} "
                f"action={message['action']} chars={message['char_cost']}"
            )
        print(
            "metrics: "
            f"messages={metrics['message_count']}, "
            f"text_chars={metrics['text_chars']}, "
            f"latent_transfers={metrics['latent_transfer_count']}, "
            f"latent_bytes={metrics['latent_bytes']}, "
            f"memory_hits={memory_hits}, "
            f"elapsed_ms={elapsed * 1000:.2f}"
        )

    memory_hit_rate = total_memory_hits / len(tasks) if tasks else 0.0
    exact_match_rate = correct / comparable if comparable else None
    summary = (
        "summary: "
        f"dataset={dataset_type}, "
        f"tasks={len(tasks)}, "
        f"messages={total_metrics['message_count']}, "
        f"text_chars={total_metrics['text_chars']}, "
        f"latent_transfers={total_metrics['latent_transfer_count']}, "
        f"latent_bytes={total_metrics['latent_bytes']}, "
        f"memory_hits={total_memory_hits}, "
        f"memory_hit_rate={memory_hit_rate:.2f}, "
        f"elapsed_ms={total_elapsed * 1000:.2f}"
    )
    if exact_match_rate is not None:
        summary += f", exact_match={exact_match_rate:.2%}"
    print(summary)

    return {
        "mode": mode,
        "dataset_type": dataset_type,
        "tasks": len(tasks),
        "metrics": total_metrics,
        "memory_hits": total_memory_hits,
        "memory_hit_rate": memory_hit_rate,
        "elapsed_ms": total_elapsed * 1000,
        "exact_match": exact_match_rate,
    }


def print_comparison(results: list[dict[str, Any]]) -> None:
    """打印 text/protocol 两种模式的对比结果。"""

    by_mode = {result["mode"]: result for result in results}
    if "text" not in by_mode or "protocol" not in by_mode:
        return

    text_result = by_mode["text"]
    protocol_result = by_mode["protocol"]
    text_chars = text_result["metrics"]["text_chars"]
    protocol_chars = protocol_result["metrics"]["text_chars"]
    text_elapsed = text_result["elapsed_ms"]
    protocol_elapsed = protocol_result["elapsed_ms"]
    char_saving = (text_chars - protocol_chars) / text_chars if text_chars else 0.0
    speedup = (text_elapsed - protocol_elapsed) / text_elapsed if text_elapsed else 0.0

    print("\n=== comparison ===")
    print(f"text chars: {text_chars}")
    print(f"protocol chars: {protocol_chars}")
    print(f"average character saving: {char_saving:.2%}")
    print(f"text elapsed ms: {text_elapsed:.2f}")
    print(f"protocol elapsed ms: {protocol_elapsed:.2f}")
    print(f"elapsed improvement: {speedup:.2%}")
    print(f"text memory hit rate: {text_result['memory_hit_rate']:.2f}")
    print(f"protocol memory hit rate: {protocol_result['memory_hit_rate']:.2f}")
    if text_result.get("exact_match") is not None and protocol_result.get("exact_match") is not None:
        print(f"text exact match: {text_result['exact_match']:.2%}")
        print(f"protocol exact match: {protocol_result['exact_match']:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run competition multi-agent experiments.")
    parser.add_argument(
        "--dataset-type",
        choices=["builtin", "multihop", "continuous"],
        default="builtin",
        help="Dataset type. Default keeps the original 10 competition tasks.",
    )
    parser.add_argument("--dataset-file", default="", help="Path to multihop or continuous JSON data.")
    parser.add_argument("--limit", type=int, default=0, help="Limit samples or continuous groups. 0 means no limit.")
    parser.add_argument("--modes", default="text,protocol", help="Comma-separated modes: text,protocol")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_tasks(args)
    modes = parse_modes(args.modes)
    print(
        "LLM providers: "
        f"default={get_llm_provider()}, "
        f"planner={get_llm_provider('planner')}, "
        f"reasoner={get_llm_provider('reasoner')}, "
        f"summarizer={get_llm_provider('summarizer')}"
    )
    print(
        "dataset: "
        f"type={args.dataset_type}, "
        f"file={args.dataset_file or '<builtin>'}, "
        f"tasks={len(tasks)}, "
        f"modes={','.join(modes)}"
    )
    results = [run_mode(mode, tasks, dataset_type=args.dataset_type) for mode in modes]
    print_comparison(results)


if __name__ == "__main__":
    main()
