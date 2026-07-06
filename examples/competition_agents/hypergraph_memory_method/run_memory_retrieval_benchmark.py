from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hypergraph_memory_method.benchmark import evaluate_retrieval_modes
from hypergraph_memory_method.memory_model import normalize_memory_record


def _query_for_turn(group_id: str, turn: dict[str, Any]) -> str:
    expected_reuse = [str(value) for value in turn.get("expected_reuse", []) or [] if str(value).strip()]
    artifacts = []
    for value in expected_reuse:
        _, _, artifact = value.partition(".")
        if artifact and artifact not in artifacts:
            artifacts.append(artifact)
    lines = [
        str(turn.get("question") or ""),
        "Continuous task reuse hints:",
        f"- Group id: {group_id}",
        f"- Current turn: {turn.get('turn_id')}",
        f"- Task type: {turn.get('task_type') or 'continuous'}",
    ]
    if expected_reuse:
        lines.append("- Expected reuse: " + ", ".join(expected_reuse))
        lines.append("- Required artifacts: " + ", ".join(artifacts))
    return "\n".join(line for line in lines if line)


def build_offline_cases(
    dataset_file: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """用历史轮次标准产物构造隔离生成质量的离线检索评测。"""

    groups = json.loads(dataset_file.read_text(encoding="utf-8"))
    memory: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    sequence = 0
    for group in groups:
        group_id = str(group.get("group_id") or "group")
        available_ids: list[str] = []
        for turn in group.get("tasks", []):
            sequence += 1
            turn_id = str(turn.get("turn_id") or f"turn-{sequence}")
            memory_id = f"{group_id}:{turn_id}"
            if available_ids:
                expected_ids = [
                    f"{group_id}:{str(spec).split('.', 1)[0]}"
                    for spec in turn.get("expected_reuse", [])
                    if str(spec).startswith("turn-")
                ]
                cases.append(
                    {
                        "case_id": memory_id,
                        "query": _query_for_turn(group_id, turn),
                        "expected_memory_ids": expected_ids,
                        "available_memory_ids": list(available_ids),
                    }
                )

            supporting_passage = str(turn.get("supporting_passage") or "")
            answer = str(turn.get("expected_answer") or "")
            record = normalize_memory_record(
                {
                    "memory_id": memory_id,
                    "memory_type": "evidence",
                    "source_agent": "summarizer",
                    "created_at": f"2026-01-{sequence:02d}T00:00:00+00:00",
                    "task_topic": str(turn.get("question") or ""),
                    "summary": answer,
                    "answer": answer,
                    "tags": [str(turn.get("task_type") or "").lower()],
                    "trace": {
                        "task_meta": {
                            "task_id": memory_id,
                            "group_id": group_id,
                            "turn_id": turn_id,
                            "dataset_type": "continuous",
                            "question": str(turn.get("question") or ""),
                            "supporting_facts": [
                                [str(turn.get("topic") or ""), supporting_passage]
                            ],
                        },
                        "evidence": [supporting_passage] if supporting_passage else [],
                    },
                }
            )
            memory.append(record)
            available_ids.append(memory_id)
    return memory, cases


def main() -> None:
    parser = argparse.ArgumentParser(description="公平比较五种共享记忆检索模式。")
    parser.add_argument("dataset_file", type=Path)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    memory, cases = build_offline_cases(args.dataset_file)
    report = evaluate_retrieval_modes(memory, cases, top_k=args.top_k)
    summary = {mode: values["summary"] for mode, values in report["modes"].items()}
    print(
        json.dumps(
            {"summary": summary, "comparison": report["comparison"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
