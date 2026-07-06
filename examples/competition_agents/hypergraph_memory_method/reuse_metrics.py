from __future__ import annotations

from typing import Any

from .memory_model import normalize_memory_record

ARTIFACT_ALIASES = {
    "answer": "answer",
    "final-answer": "answer",
    "supporting-fact": "supporting-facts",
    "supporting-facts": "supporting-facts",
    "evidence": "evidence",
    "evidence-chain": "evidence",
    "plan": "plan",
    "state": "state",
    "trajectory": "trajectory",
    "trace": "trajectory",
    "experience": "experience",
    "strategy": "strategy",
    "outcome": "outcome",
    "metrics": "outcome",
}


def _normalize_artifact(value: Any) -> str:
    label = str(value or "").lower().strip().replace("_", "-")
    return ARTIFACT_ALIASES.get(label, label)


def expected_reuse_specs(
    expected_reuse: list[Any] | None,
) -> tuple[set[tuple[str, str]], set[str]]:
    """Split concrete turn artifacts from group-level context requirements."""

    scored: set[tuple[str, str]] = set()
    unscored: set[str] = set()
    for raw_value in expected_reuse or []:
        value = str(raw_value).strip()
        source, separator, artifact = value.partition(".")
        if source.startswith("turn-") and separator and artifact:
            scored.add((source, _normalize_artifact(artifact)))
        elif value:
            unscored.add(value)
    return scored, unscored


def expected_reuse_turns(expected_reuse: list[Any] | None) -> set[str]:
    """Backward-compatible helper returning concrete prior turn ids."""

    specs, _ = expected_reuse_specs(expected_reuse)
    return {turn for turn, _ in specs}


def _source_turn(hit: dict[str, Any], question_to_turn: dict[str, str]) -> str:
    trace = hit.get("trace")
    if isinstance(trace, dict):
        task_meta = trace.get("task_meta")
        if isinstance(task_meta, dict) and task_meta.get("turn_id"):
            return str(task_meta["turn_id"])

    source_case = hit.get("source_case")
    if isinstance(source_case, dict):
        if source_case.get("turn_id"):
            return str(source_case["turn_id"])
        question = str(source_case.get("question") or "").strip().lower()
        if question:
            source_turn = question_to_turn.get(question, "")
            if source_turn:
                return source_turn

    task_topic = str(hit.get("task_topic") or "").strip()
    if task_topic:
        question = (
            task_topic.split("Question:", 1)[1]
            if "Question:" in task_topic
            else task_topic
        )
        if "\nContext:" in question:
            question = question.split("\nContext:", 1)[0]
        source_turn = question_to_turn.get(question.strip().lower(), "")
        if source_turn:
            return source_turn
    return ""


def _hit_artifacts(hit: dict[str, Any]) -> set[str]:
    """Infer which reusable products a memory record actually carries."""

    artifacts: set[str] = set()
    memory_type = str(hit.get("memory_type", "evidence") or "evidence")
    if memory_type == "experience":
        artifacts.update({"experience", "strategy"})
    else:
        # Summarizer evidence records expose the prior final answer as summary.
        artifacts.add("answer")

    trace = hit.get("trace") if isinstance(hit.get("trace"), dict) else {}
    task_meta = (
        trace.get("task_meta") if isinstance(trace.get("task_meta"), dict) else {}
    )
    if (
        hit.get("supporting_evidence")
        or hit.get("supporting_facts")
        or task_meta.get("supporting_facts")
    ):
        artifacts.update({"supporting-facts", "evidence"})
    if hit.get("evidence") or trace.get("evidence"):
        artifacts.add("evidence")
    if trace.get("messages") or trace.get("agent_outputs"):
        artifacts.add("trajectory")
    if trace.get("metrics") or trace.get("execution_result"):
        artifacts.add("outcome")
    state_refs = (
        trace.get("state_refs") if isinstance(trace.get("state_refs"), dict) else {}
    )
    artifacts.update(
        _normalize_artifact(role) for role, ref in state_refs.items() if ref
    )

    hypergraph = (
        hit.get("hypergraph") if isinstance(hit.get("hypergraph"), dict) else {}
    )
    for node in hypergraph.get("structured_nodes", []):
        node_text = str(node)
        if node_text.startswith("artifact:"):
            artifacts.add(_normalize_artifact(node_text.partition(":")[2]))
    return {artifact for artifact in artifacts if artifact}


def _used_memory_ids(task_result: dict[str, Any]) -> set[str]:
    """只统计推理或执行阶段显式声明采用的记忆，候选上下文不算采用。"""

    used: set[str] = set()
    execution_result = task_result.get("execution_result", {})
    if isinstance(execution_result, dict):
        for key in (
            "memory_used_ids",
            "evidence_used_memory_ids",
            "experience_used_memory_ids",
            "cited_memory_ids",
        ):
            used.update(
                str(memory_id)
                for memory_id in execution_result.get(key, []) or []
                if memory_id
            )
        for citation in execution_result.get("memory_citations", []) or []:
            if isinstance(citation, dict) and citation.get("memory_id"):
                used.add(str(citation["memory_id"]))
    return used


def evaluate_task_reuse(
    task_result: dict[str, Any],
    *,
    question_to_turn: dict[str, str],
) -> dict[str, Any]:
    """Evaluate source-turn and artifact correctness for one completed task."""

    expected_specs, unscored_expected = expected_reuse_specs(
        task_result.get("expected_reuse")
    )
    hits = [hit for hit in task_result.get("memory_hits", []) if isinstance(hit, dict)]
    hit_details: list[dict[str, Any]] = []
    relevant_hit_ids: set[str] = set()
    matched_specs: set[tuple[str, str]] = set()
    chain_matched_specs: set[tuple[str, str]] = set()
    chain_relevant_hit_ids: set[str] = set()

    for hit in hits:
        memory_id = str(hit.get("memory_id") or "")
        source_turn = _source_turn(hit, question_to_turn)
        artifacts = _hit_artifacts(hit)
        matches = {
            (turn, artifact)
            for turn, artifact in expected_specs
            if turn == source_turn and artifact in artifacts
        }
        normalized_hit = normalize_memory_record(hit)
        chain = normalized_hit.get("evidence_chain", {})
        chain_qualified = (
            bool(chain.get("valid"))
            and float(chain.get("completeness", 0.0) or 0.0) >= 0.8
        )
        if memory_id and matches:
            relevant_hit_ids.add(memory_id)
            matched_specs.update(matches)
            if chain_qualified:
                chain_relevant_hit_ids.add(memory_id)
                chain_matched_specs.update(matches)
        hit_details.append(
            {
                "memory_id": memory_id,
                "memory_type": str(hit.get("memory_type", "evidence") or "evidence"),
                "source_turn": source_turn,
                "artifacts": sorted(artifacts),
                "matched_expected_reuse": [
                    f"{turn}.{artifact}" for turn, artifact in sorted(matches)
                ],
                "relevant": bool(matches),
                "evidence_chain_id": chain.get("chain_id", ""),
                "evidence_chain_valid": bool(chain.get("valid")),
                "evidence_chain_completeness": float(
                    chain.get("completeness", 0.0) or 0.0
                ),
                "evidence_chain_qualified": chain_qualified,
            }
        )

    used_memory_ids = _used_memory_ids(task_result)
    for detail in hit_details:
        detail["used"] = detail["memory_id"] in used_memory_ids
    relevant_used_ids = relevant_hit_ids & used_memory_ids
    requires_reuse = bool(expected_specs)
    answer_correct = task_result.get("exact_match") is True
    retrieval_correct = requires_reuse and bool(matched_specs)
    complete_retrieval = requires_reuse and matched_specs == expected_specs
    correct_reuse = requires_reuse and answer_correct and bool(relevant_used_ids)
    chain_precision = (
        len(chain_relevant_hit_ids) / len(hits) if requires_reuse and hits else 0.0
    )
    chain_recall = (
        len(chain_matched_specs) / len(expected_specs) if expected_specs else 0.0
    )
    chain_f1 = (
        2 * chain_precision * chain_recall / (chain_precision + chain_recall)
        if chain_precision + chain_recall
        else 0.0
    )

    return {
        "task_id": task_result.get("task_id"),
        "turn_id": task_result.get("turn_id"),
        "requires_memory_reuse": requires_reuse,
        "expected_turns": sorted({turn for turn, _ in expected_specs}),
        "expected_artifacts": sorted({artifact for _, artifact in expected_specs}),
        "scored_expected_reuse": [
            f"{turn}.{artifact}" for turn, artifact in sorted(expected_specs)
        ],
        "unscored_expected_reuse": sorted(unscored_expected),
        "matched_expected_reuse": [
            f"{turn}.{artifact}" for turn, artifact in sorted(matched_specs)
        ],
        "expected_artifact_recall": (
            len(matched_specs) / len(expected_specs) if expected_specs else 0.0
        ),
        "memory_hit_count": len(hits),
        "relevant_memory_hit_count": len(relevant_hit_ids),
        "memory_precision": (
            len(relevant_hit_ids) / len(hits) if requires_reuse and hits else 0.0
        ),
        "evidence_chain_reuse_precision": chain_precision,
        "evidence_chain_reuse_recall": chain_recall,
        "evidence_chain_reuse_f1": chain_f1,
        "memory_retrieval_correct": retrieval_correct,
        "memory_retrieval_complete": complete_retrieval,
        "hit_details": hit_details,
        "used_memory_ids": sorted(used_memory_ids),
        "relevant_used_memory_ids": sorted(relevant_used_ids),
        "answer_correct": answer_correct,
        "correct_reuse": correct_reuse,
    }


def evaluate_run_reuse(run_result: dict[str, Any]) -> dict[str, Any]:
    """Aggregate artifact-aware memory precision and correct reuse metrics."""

    task_results = [
        task for task in run_result.get("task_results", []) if isinstance(task, dict)
    ]
    question_to_turn = {
        str(task.get("question") or "").strip().lower(): str(task.get("turn_id") or "")
        for task in task_results
        if task.get("question") and task.get("turn_id")
    }
    rows = [
        evaluate_task_reuse(task, question_to_turn=question_to_turn)
        for task in task_results
    ]
    expected_rows = [row for row in rows if row["requires_memory_reuse"]]

    total_hits = sum(row["memory_hit_count"] for row in expected_rows)
    relevant_hits = sum(row["relevant_memory_hit_count"] for row in expected_rows)
    total_used = sum(len(row["used_memory_ids"]) for row in expected_rows)
    relevant_used = sum(len(row["relevant_used_memory_ids"]) for row in expected_rows)
    expected_specs = sum(len(row["scored_expected_reuse"]) for row in expected_rows)
    matched_specs = sum(len(row["matched_expected_reuse"]) for row in expected_rows)
    expected_count = len(expected_rows)
    correct_reuse_count = sum(1 for row in expected_rows if row["correct_reuse"])
    chain_precision_values = [
        row["evidence_chain_reuse_precision"] for row in expected_rows
    ]
    chain_recall_values = [row["evidence_chain_reuse_recall"] for row in expected_rows]
    chain_f1_values = [row["evidence_chain_reuse_f1"] for row in expected_rows]

    summary = {
        "expected_reuse_task_count": expected_count,
        "memory_hit_count": total_hits,
        "relevant_memory_hit_count": relevant_hits,
        "memory_precision": relevant_hits / total_hits if total_hits else 0.0,
        "expected_artifact_recall": (
            matched_specs / expected_specs if expected_specs else 0.0
        ),
        "memory_retrieval_accuracy": (
            sum(1 for row in expected_rows if row["memory_retrieval_correct"])
            / expected_count
            if expected_count
            else 0.0
        ),
        "memory_complete_retrieval_rate": (
            sum(1 for row in expected_rows if row["memory_retrieval_complete"])
            / expected_count
            if expected_count
            else 0.0
        ),
        "memory_use_precision": relevant_used / total_used if total_used else 0.0,
        "correct_reuse_count": correct_reuse_count,
        "correct_reuse_rate": (
            correct_reuse_count / expected_count if expected_count else 0.0
        ),
        "evidence_chain_reuse_precision": (
            sum(chain_precision_values) / expected_count if expected_count else 0.0
        ),
        "evidence_chain_reuse_recall": (
            sum(chain_recall_values) / expected_count if expected_count else 0.0
        ),
        "evidence_chain_reuse_f1": (
            sum(chain_f1_values) / expected_count if expected_count else 0.0
        ),
    }
    return {"summary": summary, "rows": rows}


def evaluate_paired_memory_gain(
    full_memory_run: dict[str, Any],
    no_memory_run: dict[str, Any],
) -> dict[str, Any]:
    """计算相同任务下有记忆相对无记忆的因果净增益和退化。"""

    full_tasks = {
        str(task.get("task_id")): task
        for task in full_memory_run.get("task_results", [])
        if isinstance(task, dict) and task.get("task_id")
    }
    no_memory_tasks = {
        str(task.get("task_id")): task
        for task in no_memory_run.get("task_results", [])
        if isinstance(task, dict) and task.get("task_id")
    }
    rows: list[dict[str, Any]] = []
    for task_id in sorted(set(full_tasks) & set(no_memory_tasks)):
        full_task = full_tasks[task_id]
        baseline_task = no_memory_tasks[task_id]
        if not full_task.get("expected_reuse"):
            continue
        full_correct = full_task.get("exact_match") is True
        baseline_correct = baseline_task.get("exact_match") is True
        rows.append(
            {
                "task_id": task_id,
                "full_memory_correct": full_correct,
                "no_memory_correct": baseline_correct,
                "memory_improved": full_correct and not baseline_correct,
                "memory_degraded": baseline_correct and not full_correct,
            }
        )

    improved = sum(1 for row in rows if row["memory_improved"])
    degraded = sum(1 for row in rows if row["memory_degraded"])
    count = len(rows)
    return {
        "summary": {
            "paired_reuse_task_count": count,
            "memory_improvement_count": improved,
            "memory_degradation_count": degraded,
            "memory_causal_net_gain_rate": (
                (improved - degraded) / count if count else 0.0
            ),
        },
        "rows": rows,
    }
