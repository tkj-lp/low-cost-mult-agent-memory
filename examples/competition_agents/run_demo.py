from __future__ import annotations

import argparse
import json
import os
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


def load_codeact_tasks(path: Path, limit: int | None) -> list[dict[str, Any]]:
    """读取 CodeAct 数据集，并保留主图 Executor 所需的 task_meta 字段。"""

    samples = json.loads(path.read_text(encoding="utf-8"))
    if limit:
        samples = samples[:limit]
    return [
        {
            "task_id": sample["id"],
            "group_id": "codeact",
            "turn_id": sample["id"],
            "task": sample["task"],
            "task_type": sample.get("task_type", ""),
            "inputs": sample.get("inputs", {}),
            "expected": sample.get("expected"),
            "expected_answer": None,
            "allowed_imports": sample.get("allowed_imports", []),
            "reference_code": sample.get("reference_code", ""),
            "timeout_ms": sample.get("timeout_ms", 3000),
            "tags": sample.get("tags", []),
            "dataset_type": "codeact",
        }
        for sample in samples
    ]


def load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dataset_type == "builtin":
        return load_builtin_tasks(args.limit)
    if args.dataset_type == "codeact" and not args.dataset_file:
        args.dataset_file = "examples/competition_agents/data/codeact_tasks.json"
    if not args.dataset_file:
        raise ValueError("--dataset-file is required when dataset type is multihop, continuous, or codeact")
    path = Path(args.dataset_file)
    if args.dataset_type == "multihop":
        return load_multihop_tasks(path, args.limit)
    if args.dataset_type == "continuous":
        return load_continuous_tasks(path, args.limit)
    if args.dataset_type == "codeact":
        return load_codeact_tasks(path, args.limit)
    raise ValueError(f"Unsupported dataset type: {args.dataset_type}")


def load_env_file(path: Path) -> None:
    """加载 PowerShell/Unix 通用的 KEY=VALUE 环境变量文件。"""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ[name.strip()] = value.strip().strip('"').strip("'")


def configure_experiment_mode(args: argparse.Namespace) -> list[str]:
    """把正式实验模式映射为底层协作模式和 provider 配置。"""

    if not args.experiment_mode:
        return parse_modes(args.modes)

    if args.experiment_mode == "codeact_tools":
        os.environ["LLM_PROVIDER"] = "mock"
        os.environ["LLM_PLANNER_PROVIDER"] = "mock"
        os.environ["LLM_REASONER_PROVIDER"] = "mock"
        os.environ["LLM_SUMMARIZER_PROVIDER"] = "mock"
        os.environ["EMBEDDING_PROVIDER"] = "hash"
        return ["protocol"]

    load_env_file(Path(args.env_file))
    os.environ["LLM_PROVIDER"] = "openai_compatible"
    os.environ["LLM_PLANNER_PROVIDER"] = "openai_compatible"
    os.environ["LLM_REASONER_PROVIDER"] = "openai_compatible"
    os.environ["LLM_SUMMARIZER_PROVIDER"] = "openai_compatible"
    os.environ["LLM_REQUIRE_REAL"] = "1"

    if args.experiment_mode == "baseline_text":
        os.environ["EMBEDDING_PROVIDER"] = "hash"
        return ["text"]
    if args.experiment_mode == "protocol_api":
        os.environ["EMBEDDING_PROVIDER"] = "hash"
        return ["protocol"]
    if args.experiment_mode == "protocol_api_embedding":
        os.environ["EMBEDDING_PROVIDER"] = "openai_compatible"
        return ["protocol"]
    raise ValueError(f"Unsupported experiment mode: {args.experiment_mode}")


def parse_modes(raw_modes: str) -> list[str]:
    modes = [mode.strip().lower() for mode in raw_modes.split(",") if mode.strip()]
    invalid = [mode for mode in modes if mode not in {"text", "protocol"}]
    if invalid:
        raise ValueError(f"Unsupported mode(s): {invalid}. Use text and/or protocol.")
    return modes or ["text", "protocol"]


def prediction_from_result(result: dict[str, Any]) -> str:
    execution_result = result.get("execution_result", {})
    return str(execution_result.get("answer_candidate") or result.get("final_answer", ""))


def task_meta_from_item(task_item: dict[str, Any]) -> dict[str, Any]:
    """从数据集条目中提取稳定任务元信息，写入 state 和记忆 trace。"""

    return {
        key: task_item.get(key)
        for key in [
            "task_id",
            "group_id",
            "turn_id",
            "dataset_type",
            "expected_reuse",
            "question",
            "supporting_facts",
            "task_type",
            "inputs",
            "expected",
            "allowed_imports",
            "reference_code",
            "timeout_ms",
            "tags",
        ]
        if key in task_item
    }


def _shorten(value: Any, limit: int = 240) -> str:
    """把展示字段压短，避免 trace 输出太长。"""

    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "..."


def _payload_refs(payload: Any) -> tuple[list[str], list[str]]:
    """从结构化 payload 中提取 state_refs 和 memory_refs。"""

    if not isinstance(payload, dict):
        return [], []
    state_refs = payload.get("state_refs") or payload.get("refs") or payload.get("s") or []
    memory_refs = payload.get("memory_refs") or payload.get("m") or []
    return list(state_refs), list(memory_refs)


def _format_list(values: list[Any], limit: int = 12) -> str:
    """限制列表展示长度，避免超图节点过多时刷屏。"""

    shown = values[:limit]
    suffix = f" ... (+{len(values) - limit} more)" if len(values) > limit else ""
    return f"{shown}{suffix}"


def _format_json(value: Any, limit: int = 700) -> str:
    """把结构化字段压成单行 JSON，便于在终端里检查 state 内容。"""

    return _shorten(json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":")), limit)


FRONTEND_OMIT_KEYS = {"embedding", "hypergraph", "pooled_vector", "source_text", "tensor", "vector"}


def _frontend_safe(value: Any, *, max_string: int = 3000) -> Any:
    """清理前端 JSON 中不需要的向量和超长文本，保留可解释字段。"""

    if isinstance(value, dict):
        return {
            str(key): _frontend_safe(item, max_string=max_string)
            for key, item in value.items()
            if str(key) not in FRONTEND_OMIT_KEYS
        }
    if isinstance(value, list):
        return [_frontend_safe(item, max_string=max_string) for item in value]
    if isinstance(value, tuple):
        return [_frontend_safe(item, max_string=max_string) for item in value]
    if isinstance(value, str) and len(value) > max_string:
        return value[:max_string] + "..."
    return value


def _frontend_memory_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """给页面展示的检索命中，去掉完整 trace 和 embedding。"""

    compact_hits = []
    for hit in hits:
        compact_hits.append(
            {
                key: _frontend_safe(value)
                for key, value in hit.items()
                if key not in {"embedding", "hypergraph", "trace", "vector"}
            }
        )
    return compact_hits


def build_task_result(
    *,
    task_index: int,
    task_item: dict[str, Any],
    result: dict[str, Any],
    elapsed_ms: float,
    exact_match: bool | None,
    prediction: str,
) -> dict[str, Any]:
    """构造前端可直接读取的单任务明细。"""

    metrics = result.get("metrics", {})
    memory_hits = result.get("memory_hits", [])
    latest_memory = result.get("memory", [])[-1] if result.get("memory") else {}
    execution_result = result.get("execution_result", {})
    task_metrics = {
        **metrics,
        "memory_hits": len(memory_hits),
        "memory_hit_rate": 1.0 if memory_hits else 0.0,
        "elapsed_ms": elapsed_ms,
        "exact_match": exact_match,
    }
    task_result = {
        "task_index": task_index,
        "task_id": task_item.get("task_id"),
        "group_id": task_item.get("group_id"),
        "turn_id": task_item.get("turn_id"),
        "dataset_type": task_item.get("dataset_type"),
        "task_type": task_item.get("task_type"),
        "task": task_item.get("task"),
        "question": task_item.get("question", ""),
        "expected_answer": task_item.get("expected_answer"),
        "gold": task_item.get("expected_answer"),
        "expected": task_item.get("expected"),
        "expected_reuse": task_item.get("expected_reuse", []),
        "prediction": prediction,
        "final_answer": result.get("final_answer", ""),
        "exact_match": exact_match,
        "messages": _frontend_safe(result.get("messages", [])),
        "metrics": _frontend_safe(task_metrics),
        "memory_hits": _frontend_memory_hits(memory_hits),
        "memory_hit_count": len(memory_hits),
        "memory_context": _frontend_safe(result.get("memory_context", [])),
        "latent_states": _frontend_safe(result.get("latent_states", {})),
        "state_refs": _frontend_safe(result.get("state_refs", {})),
        "evidence": _frontend_safe(result.get("evidence", [])),
        "execution_result": _frontend_safe(execution_result),
        "cognitive_gaps": _frontend_safe(result.get("cognitive_gaps", [])),
        "agent_outputs": _frontend_safe(result.get("agent_outputs", []), max_string=1200),
        "latest_memory_write": _frontend_safe(
            {
                "memory_id": latest_memory.get("memory_id"),
                "source_agent": latest_memory.get("source_agent"),
                "summary": latest_memory.get("summary"),
                "tags": latest_memory.get("tags"),
                "state_refs": latest_memory.get("state_refs"),
                "has_trace": "trace" in latest_memory,
            }
        ),
        "trace": _frontend_safe(latest_memory.get("trace", {}), max_string=1800),
    }
    if task_item.get("dataset_type") == "codeact":
        task_result.update(
            {
                "inputs": _frontend_safe(task_item.get("inputs", {})),
                "allowed_imports": task_item.get("allowed_imports", []),
                "tags": task_item.get("tags", []),
                "expected_result": task_item.get("expected"),
                "actual_result": execution_result.get("result"),
                "sandbox_status": execution_result.get("sandbox_status"),
            }
        )
    return task_result


def print_agent_trace(result: dict[str, Any]) -> None:
    """展示每个 Agent 的主要输出，方便调试协作链路。"""

    print("agent outputs:")
    if result.get("plan"):
        print("  PlannerAgent:")
        for index, step in enumerate(result["plan"], start=1):
            print(f"    {index}. {_shorten(step, 120)}")
    if result.get("evidence"):
        print("  RetrieverAgent evidence:")
        for item in result["evidence"][:5]:
            print(f"    - {_shorten(item, 160)}")
    if result.get("execution_result"):
        print("  ReasonerAgent execution_result:")
        for key in ["answer_candidate", "reasoning_summary", "retrieval_mode", "plan_ref", "evidence_ref", "answer_ref"]:
            if key in result["execution_result"]:
                print(f"    {key}: {_shorten(result['execution_result'][key], 180)}")
    if result.get("final_answer"):
        print(f"  SummarizerAgent final_answer: {_shorten(result['final_answer'], 240)}")
    for output in result.get("agent_outputs", []):
        agent = output.get("agent", "unknown")
        raw_output = output.get("raw_output", "")
        print(f"  raw {agent}: {_shorten(raw_output, 180)}")


def print_memory_trace(result: dict[str, Any]) -> None:
    """展示超图记忆命中、分数和可解释路径。"""

    print("memory retrieval trace:")
    hits = result.get("memory_hits", [])
    if not hits:
        print("  no memory hits")
        return
    for index, hit in enumerate(hits, start=1):
        print(f"  hit {index}:")
        print(f"    memory_id: {hit.get('memory_id')}")
        print(f"    summary: {_shorten(hit.get('summary', ''), 220)}")
        print(f"    tags: {hit.get('tags', [])}")
        print(f"    retrieval_score: {hit.get('retrieval_score', '<none>')}")
        print(f"    hypergraph_score: {hit.get('hypergraph_score', '<none>')}")
        print(f"    embedding_score: {hit.get('embedding_score', '<none>')}")
        print(f"    matched_nodes: {_format_list(hit.get('matched_nodes', []))}")
        print(f"    matched_hyperedges: {_format_list(hit.get('matched_hyperedges', []))}")
        reasoning_path = hit.get("reasoning_path", [])
        if reasoning_path:
            print("    reasoning_path:")
            for path in reasoning_path[:5]:
                print(
                    "      "
                    f"{path.get('query_node')} -> {path.get('hyperedge_id')} -> {path.get('memory_id')}"
                )


def print_latent_trace(result: dict[str, Any]) -> None:
    """展示状态引用和 latent state 元数据。"""

    print("message refs:")
    for message in result.get("messages", []):
        state_refs, memory_refs = _payload_refs(message.get("payload"))
        if state_refs or memory_refs:
            print(
                f"  {message['from']} -> {message['to']} "
                f"action={message['action']} state_refs={state_refs} memory_refs={memory_refs}"
            )

    print("latent states:")
    latent_states = result.get("latent_states", {})
    if not latent_states:
        print("  no latent states")
        return
    for state_id, state in latent_states.items():
        print(
            f"  {state_id}: producer={state.get('producer')}, "
            f"content_type={state.get('content_type')}, "
            f"provider={state.get('embedding_provider', state.get('model', '<none>'))}, "
            f"shape={state.get('shape')}, checksum={state.get('checksum')}"
        )


def print_state_trace(task_item: dict[str, Any], result: dict[str, Any]) -> None:
    """展示超图建图需要的 state 快照字段，不改变任何记忆写入逻辑。"""

    print("state snapshot trace:")
    task_meta = task_meta_from_item(task_item)
    print(f"  task_meta_from_runner: {_format_json(task_meta, 900)}")
    print(f"  task_meta_from_state: {_format_json(result.get('task_meta', {}), 900)}")
    print(f"  top_level_state_refs: {_format_json(result.get('state_refs', {}), 500)}")

    print("  message payloads:")
    for index, message in enumerate(result.get("messages", []), start=1):
        print(
            f"    {index}. {message.get('from')} -> {message.get('to')} "
            f"action={message.get('action')} chars={message.get('char_cost')}"
        )
        print(f"       payload={_format_json(message.get('payload'), 900)}")

    print("  cognitive_gaps:")
    gaps = result.get("cognitive_gaps", [])
    if not gaps:
        print("    []")
    for index, gap in enumerate(gaps, start=1):
        print(f"    {index}. {_format_json(gap, 700)}")

    print("  execution_result:")
    print(f"    {_format_json(result.get('execution_result', {}), 900)}")

    print("  memory_context:")
    print(f"    {_format_json(result.get('memory_context', []), 900)}")

    print("  metrics:")
    print(f"    {_format_json(result.get('metrics', {}), 500)}")

    latest_memory = result.get("memory", [])[-1] if result.get("memory") else {}
    print("  latest_memory_write:")
    if not latest_memory:
        print("    <none>")
        return
    print(
        "    "
        + _format_json(
            {
                "memory_id": latest_memory.get("memory_id"),
                "source_agent": latest_memory.get("source_agent"),
                "task_topic": _shorten(latest_memory.get("task_topic", ""), 160),
                "summary": _shorten(latest_memory.get("summary", ""), 180),
                "tags": latest_memory.get("tags"),
                "state_refs": latest_memory.get("state_refs"),
                "has_trace": "trace" in latest_memory,
                "trace_keys": list(latest_memory.get("trace", {}).keys()),
            },
            1000,
        )
    )
    trace = latest_memory.get("trace", {})
    if not trace:
        print("    trace: <not stored by current add_memory call>")
    else:
        print(
            "    trace_counts: "
            + _format_json(
                {
                    "messages": len(trace.get("messages", [])),
                    "evidence": len(trace.get("evidence", [])),
                    "memory_hits": len(trace.get("memory_hits", [])),
                    "latent_states": len(trace.get("latent_states", {})),
                    "cognitive_gaps": len(trace.get("cognitive_gaps", [])),
                    "agent_outputs": len(trace.get("agent_outputs", [])),
                },
                500,
            )
        )


def run_mode(
    mode: str,
    tasks: list[dict[str, Any]],
    *,
    dataset_type: str,
    dataset_file: str = "",
    limit: int = 0,
    experiment_mode: str | None = None,
    show_agents: bool = False,
    show_memory: bool = False,
    show_latent: bool = False,
    show_state: bool = False,
) -> dict[str, Any]:
    """运行一种协作模式，并汇总该模式的指标。"""

    graph = build_graph()
    global_memory: list[dict[str, Any]] = []
    group_memory: dict[str, list[dict[str, Any]]] = {}
    total_elapsed = 0.0
    total_memory_hits = 0
    correct = 0
    comparable = 0
    task_results: list[dict[str, Any]] = []
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
        result = graph.invoke(
            initial_state(task_item["task"], mode, memory, task_meta=task_meta_from_item(task_item))
        )
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
        if dataset_type == "codeact":
            comparable += 1
            expected_result = task_item.get("expected")
            actual_result = result.get("execution_result", {}).get("result")
            exact_match = actual_result == expected_result
            correct += int(exact_match)
        elif expected_answer:
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
        if dataset_type == "codeact":
            execution_result = result.get("execution_result", {})
            print(f"expected_result: {json.dumps(task_item.get('expected'), ensure_ascii=False, sort_keys=True)}")
            print(f"actual_result: {json.dumps(execution_result.get('result'), ensure_ascii=False, sort_keys=True)}")
            print(f"sandbox_status: {execution_result.get('sandbox_status')}")
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
        if show_agents:
            print_agent_trace(result)
        if show_memory:
            print_memory_trace(result)
        if show_latent:
            print_latent_trace(result)
        if show_state:
            print_state_trace(task_item, result)

        task_results.append(
            build_task_result(
                task_index=index,
                task_item=task_item,
                result=result,
                elapsed_ms=elapsed * 1000,
                exact_match=exact_match,
                prediction=prediction,
            )
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

    elapsed_ms = total_elapsed * 1000
    aggregate_metrics = {
        **total_metrics,
        "memory_hits": total_memory_hits,
        "memory_hit_rate": memory_hit_rate,
        "elapsed_ms": elapsed_ms,
        "exact_match": exact_match_rate,
        "task_count": len(tasks),
    }

    run_result = {
        "mode": mode,
        "experiment_mode": experiment_mode or mode,
        "dataset_type": dataset_type,
        "dataset_file": dataset_file,
        "limit": limit,
        "tasks": len(tasks),
        "task_count": len(tasks),
        "metrics": aggregate_metrics,
        "memory_hits": total_memory_hits,
        "memory_hit_rate": memory_hit_rate,
        "elapsed_ms": elapsed_ms,
        "exact_match": exact_match_rate,
        "task_results": task_results,
    }
    return run_result


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


def print_run_result_json(args: argparse.Namespace, tasks: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    """输出前端可直接解析的汇总 JSON，避免页面从终端文本里猜字段。"""

    payload = {
        "schema_version": 1,
        "experiment_mode": args.experiment_mode or "<developer-modes>",
        "dataset_type": args.dataset_type,
        "dataset_file": args.dataset_file,
        "limit": args.limit,
        "task_count": len(tasks),
        "runs": results,
    }
    print("\n=== run_result_json ===")
    print(json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run competition multi-agent experiments.")
    parser.add_argument(
        "--experiment-mode",
        choices=["baseline_text", "protocol_api", "protocol_api_embedding", "codeact_tools"],
        default="",
        help=(
            "Formal experiment mode for dashboards. "
            "baseline_text uses text collaboration; protocol_api uses structured protocol; "
            "protocol_api_embedding additionally uses configured embedding API."
        ),
    )
    parser.add_argument(
        "--env-file",
        default="examples/competition_agents/.env.local",
        help="Environment file loaded only when --experiment-mode is set.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["builtin", "multihop", "continuous", "codeact"],
        default="builtin",
        help="Dataset type. Default keeps the original 10 competition tasks.",
    )
    parser.add_argument("--dataset-file", default="", help="Path to multihop or continuous JSON data.")
    parser.add_argument("--limit", type=int, default=0, help="Limit samples or continuous groups. 0 means no limit.")
    parser.add_argument(
        "--modes",
        default="text,protocol",
        help="Developer/self-test only. Comma-separated modes: text,protocol. Ignored when --experiment-mode is set.",
    )
    parser.add_argument("--show-agents", action="store_true", help="Print Planner/Retriever/Reasoner/Summarizer outputs.")
    parser.add_argument("--show-memory", action="store_true", help="Print hypergraph memory hits and reasoning paths.")
    parser.add_argument("--show-latent", action="store_true", help="Print state refs and latent state metadata.")
    parser.add_argument("--show-state", action="store_true", help="Print state snapshot fields needed by hypergraph tracing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = configure_experiment_mode(args)
    tasks = load_tasks(args)
    print(
        "LLM providers: "
        f"default={get_llm_provider()}, "
        f"planner={get_llm_provider('planner')}, "
        f"reasoner={get_llm_provider('reasoner')}, "
        f"summarizer={get_llm_provider('summarizer')}"
    )
    print(
        "dataset: "
        f"experiment_mode={args.experiment_mode or '<developer-modes>'}, "
        f"type={args.dataset_type}, "
        f"file={args.dataset_file or '<builtin>'}, "
        f"tasks={len(tasks)}, "
        f"modes={','.join(modes)}"
    )
    results = [
        run_mode(
            mode,
            tasks,
            dataset_type=args.dataset_type,
            dataset_file=args.dataset_file,
            limit=args.limit,
            experiment_mode=args.experiment_mode or None,
            show_agents=args.show_agents,
            show_memory=args.show_memory,
            show_latent=args.show_latent,
            show_state=args.show_state,
        )
        for mode in modes
    ]
    print_comparison(results)
    print_run_result_json(args, tasks, results)


if __name__ == "__main__":
    main()
