from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from competition_agents.capabilities import build_capability_registry
from competition_agents.graph import build_graph
from competition_agents.llm.client import get_llm_provider
from competition_agents.runtime_events import (
    DEFAULT_EVENT_LOG_FILE,
    configure_event_log,
    emit_runtime_event,
    emit_task_result_events,
)
from competition_agents.state_store import clear_state_store
from competition_agents.state import initial_state
from hypergraph_memory_method import (
    RETRIEVAL_MODE_VALUES,
    memory_organization_for_mode,
)
from hypergraph_memory_method.reuse_metrics import evaluate_run_reuse


DEFAULT_MULTIHOP_DATASET = "examples/competition_agents/data/hotpot_multihop_sample.json"
DEFAULT_CONTINUOUS_DATASET = "examples/competition_agents/data/hotpot_continuous_tasks.json"
DEFAULT_CODEACT_DATASET = "examples/competition_agents/data/codeact_tasks.json"
DEFAULT_CONTEXT_SOURCE = "inline"


def configure_stdout() -> None:
    """尽量避免 Windows 控制台因单个 Unicode 字符中断整轮实验。"""

    stream = getattr(sys, "stdout", None)
    if stream is None:
        return
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            # 保留当前控制台编码，只把不可编码字符替换掉，避免 print 直接抛异常。
            reconfigure(errors="replace")
        except Exception:
            pass


def force_utf8_stdout() -> None:
    """强制子进程 stdout 使用 utf-8，避免前端读取实验结果时出现乱码。"""

    stream = getattr(sys, "stdout", None)
    if stream is None:
        return
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def normalize_answer(text: str) -> str:
    """把答案标准化，方便计算 exact_match。"""

    text = text.lower().strip()
    text = re.sub(r"^(answer|final answer)\s*:\s*", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


MONTH_ALIASES = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}


ANSWER_ALIASES = {
    "uk": "united kingdom",
    "u k": "united kingdom",
    "u s": "united states",
    "usa": "united states",
    "u s a": "united states",
    "fishing": "fishery",
}


def canonicalize_answer(text: str) -> str:
    """把常见别名折叠成统一写法，便于宽松匹配。"""

    normalized = normalize_answer(text)
    return ANSWER_ALIASES.get(normalized, normalized)


def normalize_date_answer(text: str) -> str | None:
    """把常见日期写法压成 `yyyy-mm-dd` 或 `mm-dd` 形式。"""

    normalized = canonicalize_answer(text)
    month_pattern = "|".join(MONTH_ALIASES.keys())

    match = re.search(rf"\b(\d{{1,2}})\s+({month_pattern})(?:\s+(\d{{4}}))?\b", normalized)
    if match:
        day = int(match.group(1))
        month = MONTH_ALIASES[match.group(2)]
        year = match.group(3)
        if year:
            return f"{year}-{month}-{day:02d}"
        return f"{month}-{day:02d}"

    match = re.search(rf"\b({month_pattern})\s+(\d{{1,2}})(?:\s+(\d{{4}}))?\b", normalized)
    if match:
        month = MONTH_ALIASES[match.group(1)]
        day = int(match.group(2))
        year = match.group(3)
        if year:
            return f"{year}-{month}-{day:02d}"
        return f"{month}-{day:02d}"

    return None


def split_answer_list(text: str) -> list[str]:
    """把枚举型答案拆成若干项，便于判断列表覆盖关系。"""

    normalized = canonicalize_answer(text)
    normalized = re.sub(r"\band\b", ",", normalized)
    parts = [part.strip() for part in normalized.split(",")]
    return [part for part in parts if part]


def answers_match(prediction: str, gold: str) -> bool:
    """用更宽松的语义近似规则判断答案是否等价。"""

    pred = canonicalize_answer(prediction)
    expected = canonicalize_answer(gold)
    if not pred or not expected:
        return False
    if pred == expected:
        return True
    if expected in pred:
        return True

    pred_date = normalize_date_answer(prediction)
    expected_date = normalize_date_answer(gold)
    if pred_date and expected_date and pred_date == expected_date:
        return True

    pred_items = split_answer_list(prediction)
    expected_items = split_answer_list(gold)
    if len(expected_items) > 1 and pred_items:
        pred_joined = " ".join(pred_items)
        if all(item in pred_joined for item in expected_items):
            return True

    pred_tokens = [token for token in pred.split() if token]
    expected_tokens = [token for token in expected.split() if token]
    if not pred_tokens or not expected_tokens:
        return False
    if pred in expected:
        if expected_date and not pred_date:
            return False
        if len(expected_tokens) <= 2:
            return True
        return len(pred_tokens) >= max(2, int(len(expected_tokens) * 0.8))

    expected_set = set(expected_tokens)
    pred_set = set(pred_tokens)
    overlap = expected_set & pred_set
    if len(expected_set) <= 3:
        return overlap == expected_set

    coverage = len(overlap) / len(expected_set)
    return coverage >= 0.8


def build_multihop_task(sample: dict[str, Any], *, context_source: str = DEFAULT_CONTEXT_SOURCE) -> str:
    """构建 multi-hop 任务文本，可在 `inline` 和 `hypergraph` 两种上下文模式间切换。"""

    if context_source == "hypergraph":
        return (
            "HotpotQA-style multi-hop QA task.\n"
            "Answer the question using evidence retrieved by RetrieverAgent.\n\n"
            f"Question: {sample['question']}\n"
        )

    return (
        "HotpotQA-style multi-hop QA task.\n"
        "Answer using only the provided Context. Return a short answer string that best matches the question.\n\n"
        f"Question: {sample['question']}\n"
        f"Context: {json.dumps(sample.get('context', []), ensure_ascii=False)}\n"
    )


def load_multihop_tasks(
    path: Path,
    limit: int | None,
    *,
    context_source: str = DEFAULT_CONTEXT_SOURCE,
) -> list[dict[str, Any]]:
    samples = json.loads(path.read_text(encoding="utf-8"))
    if limit:
        samples = samples[:limit]
    return [
        {
            "task_id": sample.get("id", f"multihop-{index:03d}"),
            "group_id": sample.get("id", f"multihop-{index:03d}"),
            "turn_id": "turn-1",
            "task": build_multihop_task(sample, context_source=context_source),
            "question": sample.get("question", ""),
            "expected_answer": sample.get("answer"),
            "context": sample.get("context", []),
            "supporting_facts": sample.get("supporting_facts", []),
            "dataset_type": "multihop",
            "context_source": context_source,
        }
        for index, sample in enumerate(samples, start=1)
    ]


def _build_continuous_task_text(
    group: dict[str, Any],
    turn: dict[str, Any],
    *,
    context_source: str = DEFAULT_CONTEXT_SOURCE,
) -> str:
    """构建连续任务文本，默认保留旧逻辑，可切换到只给问题的实验模式。"""

    if context_source == "hypergraph":
        return (
            "Continuous HotpotQA-style task. Reuse prior memory in the same group when helpful.\n\n"
            f"Question: {turn.get('question', '')}\n"
        )

    return (
        "Continuous HotpotQA-style task. Reuse prior memory in the same group when helpful.\n\n"
        f"Group topic: {group.get('topic', '')}\n"
        f"Task type: {turn.get('task_type', '')}\n"
        f"Question: {turn.get('question', '')}\n"
        f"Context: {json.dumps(group.get('context', []), ensure_ascii=False)}\n"
    )


def load_continuous_tasks(
    path: Path,
    limit: int | None,
    *,
    context_source: str = DEFAULT_CONTEXT_SOURCE,
) -> list[dict[str, Any]]:
    groups = json.loads(path.read_text(encoding="utf-8"))
    if limit:
        groups = groups[:limit]

    tasks: list[dict[str, Any]] = []
    for group in groups:
        for turn in group.get("tasks", []):
            task_text = _build_continuous_task_text(group, turn, context_source=context_source)
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
                    "source_dataset": group.get("source_dataset"),
                    "topic": turn.get("topic") or group.get("topic"),
                    "topic_section": turn.get("topic_section"),
                    "conversation_context": turn.get("conversation_context", []),
                    "supporting_passage": turn.get("supporting_passage"),
                    "context_source": context_source,
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
            "question": sample["task"],
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
    if args.dataset_type == "multihop" and not args.dataset_file:
        args.dataset_file = DEFAULT_MULTIHOP_DATASET
    if args.dataset_type == "continuous" and not args.dataset_file:
        args.dataset_file = DEFAULT_CONTINUOUS_DATASET
    if args.dataset_type == "codeact" and not args.dataset_file:
        args.dataset_file = DEFAULT_CODEACT_DATASET
    if not args.dataset_file:
        raise ValueError("--dataset-file is required when dataset type is multihop, continuous, or codeact")
    path = Path(args.dataset_file)
    if args.dataset_type == "multihop":
        return load_multihop_tasks(path, args.limit, context_source=args.context_source)
    if args.dataset_type == "continuous":
        return load_continuous_tasks(path, args.limit, context_source=args.context_source)
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
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


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
    llm_provider = "local_hf" if args.llm_backend == "local" else "openai_compatible"
    os.environ["LLM_PROVIDER"] = llm_provider
    local_fast_mode = args.llm_backend == "local" and os.environ.get(
        "LOCAL_FAST_MODE", ""
    ).lower() in {"1", "true", "yes"}
    if local_fast_mode:
        os.environ["LLM_PLANNER_PROVIDER"] = "mock"
        os.environ["LLM_REASONER_PROVIDER"] = "local_hf"
        os.environ["LLM_SUMMARIZER_PROVIDER"] = "mock"
        os.environ["LLM_REQUIRE_REAL"] = "0"
    else:
        os.environ["LLM_PLANNER_PROVIDER"] = llm_provider
        os.environ["LLM_REASONER_PROVIDER"] = llm_provider
        os.environ["LLM_SUMMARIZER_PROVIDER"] = llm_provider
        os.environ["LLM_REQUIRE_REAL"] = "1"

    if args.experiment_mode == "baseline_text":
        os.environ["EMBEDDING_PROVIDER"] = "hash"
        return ["text"]
    if args.experiment_mode == "protocol_api":
        os.environ["EMBEDDING_PROVIDER"] = "hash"
        return ["protocol"]
    if args.experiment_mode == "protocol_api_embedding":
        if args.llm_backend == "local":
            os.environ["EMBEDDING_PROVIDER"] = "local"
            os.environ.setdefault("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        else:
            os.environ["EMBEDDING_PROVIDER"] = "openai_compatible"
        return ["protocol"]
    if args.experiment_mode == "hidden_api":
        os.environ["EMBEDDING_PROVIDER"] = "hash"
        if args.llm_backend == "local":
            os.environ.setdefault("HIDDEN_RETRIEVER_STATE_ENCODER", "embedding")
            os.environ.setdefault("HIDDEN_RETRIEVER_EMBEDDING_PROVIDER", "hash")
            os.environ.setdefault("LOCAL_HIDDEN_POOLED_DIM", "128")
            os.environ.setdefault("LOCAL_LLM_PLANNER_MAX_NEW_TOKENS", "48")
            os.environ.setdefault("LOCAL_LLM_REASONER_MAX_NEW_TOKENS", "72")
        return ["hidden"]
    raise ValueError(f"Unsupported experiment mode: {args.experiment_mode}")


def parse_modes(raw_modes: str) -> list[str]:
    modes = [mode.strip().lower() for mode in raw_modes.split(",") if mode.strip()]
    invalid = [mode for mode in modes if mode not in {"text", "protocol", "hidden"}]
    if invalid:
        raise ValueError(f"Unsupported mode(s): {invalid}. Use text, protocol, and/or hidden.")
    return modes or ["text", "protocol"]


def prediction_from_result(result: dict[str, Any]) -> str:
    execution_result = result.get("execution_result", {})
    if execution_result.get("result") is not None:
        return json.dumps(execution_result.get("result"), ensure_ascii=False, sort_keys=True)
    candidate = str(execution_result.get("answer_candidate") or "").strip()
    if candidate:
        return candidate
    final_answer = str(result.get("final_answer", "") or "")
    match = re.search(
        r"Answer:\s*(.+?)(?:\n\s*Implementation notes:|\n\s*Run diagnostics:|$)",
        final_answer,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return " ".join(match.group(1).strip().split())
    return final_answer


def codeact_result_text(execution_result: dict[str, Any]) -> str:
    """把 CodeAct 执行结果压成稳定的前端展示文本。"""

    result = execution_result.get("result")
    if result is None:
        return ""
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


def task_meta_from_item(task_item: dict[str, Any]) -> dict[str, Any]:
    """从数据集条目中提取稳定任务元信息，写入 state 和记忆 trace。"""

    return {
        key: task_item.get(key)
        for key in [
            "task_id",
            "group_id",
            "turn_id",
            "dataset_type",
            "source_dataset",
            "expected_answer",
            "expected_reuse",
            "question",
            "context",
            "supporting_facts",
            "topic",
            "topic_section",
            "conversation_context",
            "supporting_passage",
            "context_source",
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


def dashboard_task_title(task_item: dict[str, Any]) -> str:
    """给运行面板展示的真实任务标题，只来自已加载数据集。"""

    return str(task_item.get("question") or task_item.get("task_type") or task_item.get("task_id") or "任务")


def dashboard_task_queue(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把已加载任务队列压成前端可展示字段。"""

    return [
        {
            "task_index": index,
            "task_id": task_item.get("task_id") or str(index),
            "group_id": task_item.get("group_id"),
            "turn_id": task_item.get("turn_id"),
            "dataset_type": task_item.get("dataset_type"),
            "task_type": task_item.get("task_type"),
            "task_title": dashboard_task_title(task_item),
            "expected_reuse": task_item.get("expected_reuse", []),
        }
        for index, task_item in enumerate(tasks, start=1)
    ]


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


def _agent_output_map(outputs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """把 agent_outputs 列表整理成按 agent 名称索引的字典。"""

    indexed: dict[str, dict[str, Any]] = {}
    for item in outputs:
        if not isinstance(item, dict):
            continue
        agent = str(item.get("agent", "")).strip()
        if not agent:
            continue
        indexed[agent] = item
    return indexed


def _state_display(latent_states: dict[str, Any], state_ref: str) -> dict[str, Any]:
    """提取 state_ref 对应的可展示信息，不暴露完整向量。"""

    if not state_ref:
        return {}
    state = latent_states.get(state_ref, {})
    if not isinstance(state, dict):
        return {}
    return _frontend_safe(
        {
            "state_ref": state_ref,
            "content_type": state.get("content_type"),
            "summary": state.get("summary"),
            "source_text": state.get("source_text"),
            "shape": state.get("shape"),
            "pooled_shape": state.get("pooled_shape"),
            "token_count": state.get("token_count"),
            "embedding_provider": state.get("embedding_provider"),
            "embedding_model": state.get("embedding_model"),
            "model": state.get("model"),
            "layer": state.get("layer"),
            "checksum": state.get("checksum"),
        },
        max_string=1200,
    )


def _build_agent_views(task_item: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """构造统一的 Agent 展示视图，供前端直接渲染。"""

    latent_states = result.get("latent_states", {})
    state_refs = result.get("state_refs", {})
    outputs = _agent_output_map(result.get("agent_outputs", []))
    execution_result = result.get("execution_result", {})
    memory_context = result.get("memory_context", [])
    memory_hits = result.get("memory_hits", [])
    experience_hits = result.get("experience_hits", [])
    is_codeact = task_item.get("dataset_type") == "codeact"
    codeact_display = codeact_result_text(execution_result)

    planner_raw = outputs.get("PlannerAgent", {}).get("raw_output", "")
    retriever_output = outputs.get("RetrieverAgent", {})
    reasoner_raw = outputs.get("ReasonerAgent", {}).get("raw_output", "")
    summarizer_raw = outputs.get("SummarizerAgent", {}).get("raw_output", "")
    executor_raw = outputs.get("ExecutorAgent", {}).get("raw_output", "")
    executor_display_text = executor_raw or execution_result.get("sandbox_status", "")
    executor_structured = execution_result if is_codeact else {}

    views = {
        "planner": _frontend_safe(
            {
                "content_type": "text" if result.get("mode") == "text" else "latent_ref",
                "display_text": "\n".join(result.get("plan", [])) or planner_raw,
                "structured": {
                    "plan": result.get("plan", []),
                    "raw_output": planner_raw,
                },
                "state_ref": state_refs.get("plan"),
                "state_display": _state_display(latent_states, state_refs.get("plan", "")),
            },
            max_string=1200,
        ),
        "retriever": _frontend_safe(
            {
                "content_type": "structured",
                "display_text": (
                    "\n".join(
                        [
                            f"retrieval_query: {retriever_output.get('retrieval_query', '')}",
                            (
                                "planner_focus: "
                                + " | ".join(retriever_output.get("planner_focus", []))
                                if retriever_output.get("planner_focus")
                                else ""
                            ),
                        ]
                    ).strip()
                ),
                "structured": {
                    "retrieval_query": retriever_output.get("retrieval_query", ""),
                    "planner_focus": retriever_output.get("planner_focus", []),
                    "context_source": retriever_output.get("context_source", "inline"),
                    "context_hit_ids": retriever_output.get("context_hit_ids", []),
                    "context_hits": retriever_output.get("context_hits", []),
                    "memory_hit_ids": retriever_output.get("memory_hit_ids", []),
                    "evidence_hits": retriever_output.get("evidence_hits", []),
                    "experience_hit_ids": retriever_output.get("experience_hit_ids", []),
                    "experience_hits": retriever_output.get("experience_hits", []),
                    "experience_rules": retriever_output.get("experience_rules", []),
                    "memory_context": memory_context,
                },
                "state_ref": state_refs.get("evidence"),
                "state_display": _state_display(latent_states, state_refs.get("evidence", "")),
            },
            max_string=1200,
        ),
        "reasoner": _frontend_safe(
            {
                "content_type": "structured",
                "display_text": (
                    (
                        "reference_code execution request"
                        if execution_result.get("execution_request")
                        else ""
                    )
                    if is_codeact
                    else (
                        execution_result.get("answer_candidate")
                        or execution_result.get("reasoning_summary")
                        or reasoner_raw
                    )
                ),
                "structured": {
                    "answer_candidate": None if is_codeact else execution_result.get("answer_candidate"),
                    "reasoning_summary": None if is_codeact else execution_result.get("reasoning_summary"),
                    "limitations": [] if is_codeact else execution_result.get("limitations", []),
                    "experience_rules": execution_result.get("experience_rules", []),
                    "experience_used_count": execution_result.get("experience_used_count", 0),
                    "execution_request": execution_result.get("execution_request"),
                },
                "state_ref": state_refs.get("answer"),
                "state_display": _state_display(latent_states, state_refs.get("answer", "")),
            },
            max_string=1200,
        ),
        "summarizer": _frontend_safe(
            {
                "content_type": "text",
                "display_text": (codeact_display or summarizer_raw) if is_codeact else (result.get("final_answer", "") or summarizer_raw),
                "structured": {
                    "final_answer": execution_result.get("result") if is_codeact else result.get("final_answer", ""),
                    "latest_memory_summary": (result.get("memory", [])[-1] or {}).get("summary", "")
                    if result.get("memory")
                    else "",
                },
                "state_ref": execution_result.get("latent_answer_ref") or state_refs.get("answer"),
                "state_display": _state_display(
                    latent_states,
                    execution_result.get("latent_answer_ref") or state_refs.get("answer", ""),
                ),
            },
            max_string=1200,
        ),
        "memory": _frontend_safe(
            {
                "memory_hit_count": len(memory_hits),
                "experience_hit_count": len(experience_hits),
                "memory_context_count": len(memory_context),
            }
        ),
    }

    if executor_display_text or executor_structured:
        views["executor"] = _frontend_safe(
            {
                "content_type": "structured",
                "display_text": executor_display_text,
                "structured": executor_structured,
                "state_ref": "",
                "state_display": {},
            },
            max_string=1200,
        )

    return views


def _build_retrieval_debug(result: dict[str, Any]) -> dict[str, Any]:
    """统一暴露 Retriever 到超图记忆检索的输入与解释字段。"""

    outputs = _agent_output_map(result.get("agent_outputs", []))
    retriever_output = outputs.get("RetrieverAgent", {})
    return _frontend_safe(
        {
            "retrieval_query": retriever_output.get("retrieval_query", result.get("task", "")),
            "planner_focus": retriever_output.get("planner_focus", []),
            "context_source": retriever_output.get("context_source", "inline"),
            "context_hit_ids": retriever_output.get("context_hit_ids", []),
            "context_hits": retriever_output.get("context_hits", []),
            "memory_hit_ids": retriever_output.get("memory_hit_ids", []),
            "evidence_hits": retriever_output.get("evidence_hits", []),
            "evidence_hit_ids": retriever_output.get("evidence_hit_ids", []),
            "experience_hit_ids": retriever_output.get("experience_hit_ids", []),
            "experience_hits": retriever_output.get("experience_hits", []),
            "experience_rules": retriever_output.get("experience_rules", []),
            "memory_context": result.get("memory_context", []),
            "memory_hits": _frontend_memory_hits(result.get("memory_hits", [])),
        },
        max_string=1500,
    )


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
    experience_hits = result.get("experience_hits", [])
    all_memory = result.get("memory", [])
    latest_memory = all_memory[-1] if all_memory else {}
    latest_evidence_write = result.get("latest_evidence_write", {}) or {}
    latest_experience_write = result.get("latest_experience_write", {}) or {}
    if not latest_evidence_write:
        latest_evidence_write = next(
            (item for item in reversed(all_memory) if item.get("memory_type", "evidence") == "evidence"),
            {},
        )
    has_latest_experience_write = bool(latest_experience_write and latest_experience_write.get("memory_id"))
    execution_result = result.get("execution_result", {})
    is_codeact = task_item.get("dataset_type") == "codeact"
    codeact_display = codeact_result_text(execution_result)
    evidence_memory_count = sum(1 for item in all_memory if item.get("memory_type", "evidence") == "evidence")
    experience_memory_count = sum(1 for item in all_memory if item.get("memory_type") == "experience")
    task_metrics = {
        **metrics,
        "memory_hits": len(memory_hits),
        "memory_hit_rate": 1.0 if memory_hits else 0.0,
        "experience_hit_count": len(experience_hits),
        "experience_used_count": int(result.get("execution_result", {}).get("experience_used_count", 0)),
        "memory_write_count": result.get("memory_write_count", 0),
        "experience_write_count": result.get("experience_write_count", 0),
        "memory_growth": len(all_memory),
        "evidence_memory_count": evidence_memory_count,
        "experience_memory_count": experience_memory_count,
        "elapsed_ms": elapsed_ms,
        "adjusted_elapsed_ms": max(0.0, elapsed_ms - float(metrics.get("state_generation_ms", 0.0) or 0.0)),
        "exact_match": exact_match,
    }
    task_result = {
        "task_index": task_index,
        "task_id": task_item.get("task_id"),
        "group_id": task_item.get("group_id"),
        "turn_id": task_item.get("turn_id"),
        "dataset_type": task_item.get("dataset_type"),
        "memory_retrieval_mode": os.environ.get("MEMORY_RETRIEVAL_MODE", "hypergraph"),
        "memory_organization": os.environ.get("MEMORY_ORGANIZATION", "hypergraph"),
        "task_type": task_item.get("task_type"),
        "task": task_item.get("task"),
        "question": task_item.get("question", ""),
        "expected_answer": task_item.get("expected_answer"),
        "gold": codeact_display if is_codeact else task_item.get("expected_answer"),
        "expected": task_item.get("expected"),
        "expected_reuse": task_item.get("expected_reuse", []),
        "prediction": prediction,
        "final_answer": codeact_display if is_codeact else result.get("final_answer", ""),
        "exact_match": exact_match,
        "messages": _frontend_safe(result.get("messages", [])),
        "metrics": _frontend_safe(task_metrics),
        "memory_hits": _frontend_memory_hits(memory_hits),
        "memory_hit_count": len(memory_hits),
        "experience_hits": _frontend_memory_hits(experience_hits),
        "experience_hit_count": len(experience_hits),
        "experience_rules": _frontend_safe(result.get("experience_rules", [])),
        "memory_context": _frontend_safe(result.get("memory_context", [])),
        "latent_states": _frontend_safe(result.get("latent_states", {})),
        "state_refs": _frontend_safe(result.get("state_refs", {})),
        "evidence": _frontend_safe(result.get("evidence", [])),
        "execution_result": _frontend_safe(execution_result),
        "cognitive_gaps": _frontend_safe(result.get("cognitive_gaps", [])),
        "agent_outputs": _frontend_safe(result.get("agent_outputs", []), max_string=1200),
        "agent_views": _build_agent_views(task_item, result),
        "retrieval_debug": _build_retrieval_debug(result),
        "latest_evidence_write": _frontend_safe(
            {
                "memory_id": latest_evidence_write.get("memory_id"),
                "memory_type": latest_evidence_write.get("memory_type", "evidence"),
                "source_agent": latest_evidence_write.get("source_agent"),
                "summary": latest_evidence_write.get("summary"),
                "tags": latest_evidence_write.get("tags"),
                "state_refs": latest_evidence_write.get("state_refs"),
                "has_trace": "trace" in latest_evidence_write,
            }
        ),
        "latest_experience_write": _frontend_safe(
            (
                {
                    "memory_id": latest_experience_write.get("memory_id"),
                    "memory_type": latest_experience_write.get("memory_type", "experience"),
                    "source_agent": latest_experience_write.get("source_agent"),
                    "summary": latest_experience_write.get("summary"),
                    "tags": latest_experience_write.get("tags"),
                    "state_refs": latest_experience_write.get("state_refs"),
                    "experience_source": latest_experience_write.get("experience_source", ""),
                    "abstraction_note": latest_experience_write.get("abstraction_note", ""),
                    "reuse_count": latest_experience_write.get("reuse_count", 0),
                    "reuse_success_count": latest_experience_write.get("reuse_success_count", 0),
                    "merge_count": latest_experience_write.get("merge_count", 0),
                    "source_case": latest_experience_write.get("source_case", {}),
                    "experience_rule_card": latest_experience_write.get("experience_rule_card", {}),
                    "has_trace": "trace" in latest_experience_write,
                }
                if has_latest_experience_write
                else {}
            )
        ),
        "latest_memory_write": _frontend_safe(
            {
                "memory_id": latest_memory.get("memory_id"),
                "memory_type": latest_memory.get("memory_type", "evidence"),
                "source_agent": latest_memory.get("source_agent"),
                "summary": latest_memory.get("summary"),
                "tags": latest_memory.get("tags"),
                "state_refs": latest_memory.get("state_refs"),
                "experience_source": latest_memory.get("experience_source", ""),
                "abstraction_note": latest_memory.get("abstraction_note", ""),
                "reuse_count": latest_memory.get("reuse_count", 0),
                "reuse_success_count": latest_memory.get("reuse_success_count", 0),
                "source_case": latest_memory.get("source_case", {}),
                "experience_rule_card": latest_memory.get("experience_rule_card", {}),
                "has_trace": "trace" in latest_memory,
            }
        ),
        "trace": _frontend_safe(latest_memory.get("trace", {}), max_string=1800),
        "memory_refine_case": _frontend_safe(result.get("memory_refine_case", {}), max_string=1800),
        "memory_refine_debug": _frontend_safe(result.get("memory_refine_debug", {}), max_string=1800),
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


def warm_local_runtime() -> None:
    """在任务计时前加载本地模型，避免一次性启动成本污染单任务耗时。"""

    providers = {
        os.environ.get("LLM_PROVIDER", ""),
        os.environ.get("LLM_PLANNER_PROVIDER", ""),
        os.environ.get("LLM_REASONER_PROVIDER", ""),
        os.environ.get("LLM_SUMMARIZER_PROVIDER", ""),
    }
    if "local_hf" in providers:
        from competition_agents.latent_hidden.local_model import load_local_hidden_model

        load_local_hidden_model()
    if os.environ.get("EMBEDDING_PROVIDER", "").lower() == "local":
        from competition_agents.embedding import embed_text

        embed_text("__warmup_embedding__")
    if os.environ.get("HIDDEN_RETRIEVER_EMBEDDING_PROVIDER", "").lower() == "local":
        from competition_agents.embedding import embed_text

        previous = os.environ.get("EMBEDDING_PROVIDER")
        os.environ["EMBEDDING_PROVIDER"] = "local"
        try:
            embed_text("__warmup_hidden_retriever_embedding__")
        finally:
            if previous is None:
                os.environ.pop("EMBEDDING_PROVIDER", None)
            else:
                os.environ["EMBEDDING_PROVIDER"] = previous


def warm_codeact_runtime(dataset_type: str) -> None:
    """Warm the long-lived CodeAct sandbox outside per-task timing."""

    if dataset_type != "codeact":
        return
    from competition_agents.restricted_python_codeact import warm_persistent_codeact_worker

    warm_persistent_codeact_worker()


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

    os.environ["LOCAL_CAPTURE_HIDDEN"] = "0"
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
        "protocol_bytes": 0,
        "communication_bytes": 0,
        "latent_transfer_count": 0,
        "latent_bytes": 0,
        "transfer_latency_ms": 0.0,
        "state_generation_ms": 0.0,
        "planner_llm_ms": 0.0,
        "retriever_context_ms": 0.0,
        "retriever_memory_ms": 0.0,
        "reasoner_llm_ms": 0.0,
        "summarizer_llm_ms": 0.0,
        "memory_refine_ms": 0.0,
        "memory_write_ms": 0.0,
        "memory_write_count": 0,
        "experience_write_count": 0,
        "experience_hit_count": 0,
        "experience_used_count": 0,
        "reuse_success_count": 0,
    }

    print(f"\n=== {mode} mode ===")
    clear_state_store()
    warm_local_runtime()
    warm_codeact_runtime(dataset_type)
    for index, task_item in enumerate(tasks, start=1):
        group_id = task_item.get("group_id", "default")
        memory = group_memory.get(group_id, []) if dataset_type == "continuous" else global_memory
        task_meta = {**task_meta_from_item(task_item), "task_index": index}
        emit_runtime_event(
            {
                "event_type": "task_start",
                "mode": mode,
                "task_index": index,
                "task_id": task_meta.get("task_id"),
                "group_id": task_meta.get("group_id"),
                "turn_id": task_meta.get("turn_id"),
                "dataset_type": dataset_type,
                "task_title": dashboard_task_title(task_item),
                "from_agent": "runtime",
                "to_agent": "planner",
                "action_type": "start_task",
                "state_refs": [],
                "memory_refs": [],
                "text_chars": 0,
                "elapsed_ms": 0,
            }
        )
        emit_runtime_event(
            {
                "event_type": "handshake",
                "mode": mode,
                "task_index": index,
                "task_id": task_meta.get("task_id"),
                "group_id": task_meta.get("group_id"),
                "turn_id": task_meta.get("turn_id"),
                "dataset_type": dataset_type,
                "task_title": dashboard_task_title(task_item),
                "from_agent": "runtime",
                "to_agent": "planner",
                "action_type": "open_session",
                "input_params": {
                    "mode": mode,
                    "dataset_type": dataset_type,
                    "dataset_file": dataset_file,
                    "limit": limit,
                },
                "return_result": {
                    "accepted": True,
                    "protocol_version": "0.2" if mode != "text" else "text-v1",
                    "task_index": index,
                    "task_id": task_meta.get("task_id"),
                },
                "state_refs": [],
                "memory_refs": [],
                "text_chars": 0,
                "elapsed_ms": 0,
            }
        )
        capability_registry = build_capability_registry()
        emit_runtime_event(
            {
                "event_type": "capability_discovery",
                "mode": mode,
                "task_index": index,
                "task_id": task_meta.get("task_id"),
                "group_id": task_meta.get("group_id"),
                "turn_id": task_meta.get("turn_id"),
                "dataset_type": dataset_type,
                "task_title": dashboard_task_title(task_item),
                "from_agent": "planner",
                "to_agent": "runtime",
                "action_type": "discover_capability",
                "input_params": {
                    "required_roles": ["planner", "retriever", "reasoner", "executor", "summarizer"],
                    "task_type": task_item.get("task_type"),
                },
                "return_result": capability_registry,
                "capability": {
                    agent: profile.get("capabilities", [])
                    for agent, profile in capability_registry.items()
                },
                "state_refs": [],
                "memory_refs": [],
                "text_chars": 0,
                "elapsed_ms": 0,
            }
        )
        emit_runtime_event(
            {
                "event_type": "protocol_mapping",
                "mode": mode,
                "task_index": index,
                "task_id": task_meta.get("task_id"),
                "group_id": task_meta.get("group_id"),
                "turn_id": task_meta.get("turn_id"),
                "dataset_type": dataset_type,
                "task_title": dashboard_task_title(task_item),
                "from_agent": "runtime",
                "to_agent": "planner",
                "action_type": "map_protocol",
                "input_params": {
                    "task_title": dashboard_task_title(task_item),
                    "mode": mode,
                },
                "return_result": {
                    "first_action": "retrieve",
                    "route": ["planner", "retriever", "reasoner", "executor?", "summarizer"],
                    "message_contract": ["action_type", "input_params", "return_result", "capability", "state_refs", "memory_refs"],
                },
                "state_refs": [],
                "memory_refs": [],
                "text_chars": 0,
                "elapsed_ms": 0,
            }
        )
        print(
            f"[progress] task_start index={index} id={task_item.get('task_id', index)}",
            flush=True,
        )
        started = time.perf_counter()
        result = graph.invoke(
            initial_state(task_item["task"], mode, memory, task_meta=task_meta)
        )
        elapsed = time.perf_counter() - started
        print(
            f"[progress] task_done index={index} elapsed_ms={elapsed * 1000:.2f}",
            flush=True,
        )
        total_elapsed += elapsed

        if dataset_type == "continuous":
            group_memory[group_id] = result["memory"]
        else:
            global_memory = result["memory"]

        metrics = result["metrics"]
        memory_hits = len(result["memory_hits"])
        total_memory_hits += memory_hits
        for key in total_metrics:
            if key in metrics:
                total_metrics[key] += metrics[key]
        total_metrics["memory_write_count"] += int(result.get("memory_write_count", 0))
        total_metrics["experience_write_count"] += int(result.get("experience_write_count", 0))
        total_metrics["experience_hit_count"] += len(result.get("experience_hits", []))
        total_metrics["experience_used_count"] += int(result.get("execution_result", {}).get("experience_used_count", 0))

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
            exact_match = answers_match(prediction, str(expected_answer))
            correct += int(exact_match)
        if exact_match and int(result.get("execution_result", {}).get("experience_used_count", 0)) > 0:
            total_metrics["reuse_success_count"] += 1

        emit_task_result_events(
            mode=mode,
            task_index=index,
            task_item=task_item,
            result=result,
            elapsed_ms=elapsed * 1000,
            exact_match=exact_match,
        )

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
            f"protocol_bytes={metrics.get('protocol_bytes', 0)}, "
            f"communication_bytes={metrics.get('communication_bytes', 0)}, "
            f"latent_transfers={metrics['latent_transfer_count']}, "
            f"latent_bytes={metrics['latent_bytes']}, "
            f"transfer_latency_ms={metrics['transfer_latency_ms']:.2f}, "
            f"memory_hits={memory_hits}, "
            f"experience_hits={len(result.get('experience_hits', []))}, "
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

    hit_task_count = sum(1 for task_result in task_results if task_result.get("memory_hit_count", 0) > 0)
    memory_hit_rate = hit_task_count / len(tasks) if tasks else 0.0
    exact_match_rate = correct / comparable if comparable else None
    summary = (
        "summary: "
        f"dataset={dataset_type}, "
        f"tasks={len(tasks)}, "
        f"messages={total_metrics['message_count']}, "
        f"text_chars={total_metrics['text_chars']}, "
        f"protocol_bytes={total_metrics.get('protocol_bytes', 0)}, "
        f"communication_bytes={total_metrics.get('communication_bytes', 0)}, "
        f"latent_transfers={total_metrics['latent_transfer_count']}, "
        f"latent_bytes={total_metrics['latent_bytes']}, "
        f"transfer_latency_ms={total_metrics['transfer_latency_ms']:.2f}, "
        f"memory_hits={total_memory_hits}, "
        f"experience_hits={total_metrics['experience_hit_count']}, "
        f"experience_used={total_metrics['experience_used_count']}, "
        f"memory_writes={total_metrics['memory_write_count']}, "
        f"memory_hit_rate={memory_hit_rate:.2f}, "
        f"elapsed_ms={total_elapsed * 1000:.2f}"
    )
    if exact_match_rate is not None:
        summary += f", exact_match={exact_match_rate:.2%}"
    print(summary)

    elapsed_ms = total_elapsed * 1000
    adjusted_elapsed_ms = max(0.0, elapsed_ms - float(total_metrics.get("state_generation_ms", 0.0) or 0.0))
    aggregate_metrics = {
        **total_metrics,
        "memory_hits": total_memory_hits,
        "memory_hit_rate": memory_hit_rate,
        "elapsed_ms": elapsed_ms,
        "adjusted_elapsed_ms": adjusted_elapsed_ms,
        "exact_match": exact_match_rate,
        "task_count": len(tasks),
    }
    reuse_report = evaluate_run_reuse({"task_results": task_results})

    run_result = {
        "mode": mode,
        "experiment_mode": experiment_mode or mode,
        "memory_retrieval_mode": os.environ.get("MEMORY_RETRIEVAL_MODE", "hypergraph"),
        "memory_organization": os.environ.get("MEMORY_ORGANIZATION", "hypergraph"),
        "dataset_type": dataset_type,
        "dataset_file": dataset_file,
        "limit": limit,
        "tasks": len(tasks),
        "task_count": len(tasks),
        "metrics": aggregate_metrics,
        "reuse_report": reuse_report,
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


def print_readable_run_summaries(results: list[dict[str, Any]]) -> None:
    """打印更适合人工查看的运行汇总表。"""

    print("\n=== run summaries ===")
    print(
        "mode | experiment_mode | tasks | exact_match | messages | "
        "text_chars | latent_bytes | transfer_latency_ms | memory_hits | elapsed_ms"
    )
    for result in results:
        metrics = result.get("metrics", {})
        exact_match = result.get("exact_match")
        exact_match_text = f"{exact_match:.2%}" if isinstance(exact_match, (int, float)) else "n/a"
        print(
            f"{result.get('mode', '')} | "
            f"{result.get('experiment_mode', '')} | "
            f"{result.get('task_count', result.get('tasks', 0))} | "
            f"{exact_match_text} | "
            f"{metrics.get('message_count', 0)} | "
            f"{metrics.get('text_chars', 0)} | "
            f"{metrics.get('latent_bytes', 0)} | "
            f"{metrics.get('transfer_latency_ms', 0.0):.2f} | "
            f"{metrics.get('memory_hits', 0)} | "
            f"{metrics.get('elapsed_ms', 0.0):.2f}"
        )


def print_run_result_json(args: argparse.Namespace, tasks: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    """输出前端可直接解析的汇总 JSON，避免页面从终端文本里猜字段。"""

    payload = {
        "schema_version": 1,
        "experiment_mode": args.experiment_mode or "<developer-modes>",
        "memory_retrieval_mode": args.memory_retrieval_mode,
        "memory_organization": memory_organization_for_mode(args.memory_retrieval_mode),
        "dataset_type": args.dataset_type,
        "dataset_file": args.dataset_file,
        "context_source": args.context_source,
        "limit": args.limit,
        "task_count": len(tasks),
        "runs": results,
    }
    print("\n=== run_result_json ===")
    text = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
    try:
        print(text)
    except UnicodeEncodeError:
        import sys

        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run competition multi-agent experiments.")
    parser.add_argument(
        "--experiment-mode",
        choices=["baseline_text", "protocol_api", "protocol_api_embedding", "hidden_api", "codeact_tools"],
        default="",
        help=(
            "Formal experiment mode for dashboards. "
            "baseline_text uses text collaboration; protocol_api uses structured protocol; "
            "protocol_api_embedding additionally uses configured embedding API; "
            "hidden_api keeps real multi-agent reasoning but replaces agent-to-agent text exchange with hidden-state refs."
        ),
    )
    parser.add_argument(
        "--env-file",
        default="examples/competition_agents/.env.local",
        help="Environment file loaded only when --experiment-mode is set.",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["api", "local"],
        default="api",
        help="api uses OpenAI-compatible LLMs; local uses LOCAL_HIDDEN_MODEL through HuggingFace.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["multihop", "continuous", "codeact"],
        default="multihop",
        help="Dataset type. Formal QA experiments use Hotpot-style multihop or continuous tasks.",
    )
    parser.add_argument("--dataset-file", default="", help="Path to multihop, continuous, or codeact JSON data.")
    parser.add_argument("--limit", type=int, default=0, help="Limit samples or continuous groups. 0 means no limit.")
    parser.add_argument(
        "--context-source",
        choices=["inline", "hypergraph"],
        default=DEFAULT_CONTEXT_SOURCE,
        help=(
            "Context feeding mode. inline keeps dataset context inside each task; "
            "hypergraph sends only the question and expects RetrieverAgent to fetch context."
        ),
    )
    parser.add_argument(
        "--memory-retrieval-mode",
        choices=RETRIEVAL_MODE_VALUES,
        default=os.environ.get("MEMORY_RETRIEVAL_MODE", "hypergraph"),
        help="Shared-memory retrieval mode. keyword/tag/semantic use text memory; hypergraph/fusion use hypergraph memory.",
    )
    parser.add_argument(
        "--modes",
        default="text,protocol",
        help="Developer/self-test only. Comma-separated modes: text,protocol,hidden. Ignored when --experiment-mode is set.",
    )
    parser.add_argument("--show-agents", action="store_true", help="Print Planner/Retriever/Reasoner/Summarizer outputs.")
    parser.add_argument("--show-memory", action="store_true", help="Print hypergraph memory hits and reasoning paths.")
    parser.add_argument("--show-latent", action="store_true", help="Print state refs and latent state metadata.")
    parser.add_argument("--show-state", action="store_true", help="Print state snapshot fields needed by hypergraph tracing.")
    parser.add_argument(
        "--event-log-file",
        default=os.environ.get("AGENT_EVENT_LOG_PATH", DEFAULT_EVENT_LOG_FILE),
        help="JSONL file for dashboard runtime events. Empty string disables event logging.",
    )
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    force_utf8_stdout()
    args = parse_args()
    os.environ["MEMORY_RETRIEVAL_MODE"] = args.memory_retrieval_mode
    os.environ["MEMORY_ORGANIZATION"] = memory_organization_for_mode(args.memory_retrieval_mode)
    modes = configure_experiment_mode(args)
    tasks = load_tasks(args)
    configure_event_log(
        args.event_log_file,
        {
            "experiment_mode": args.experiment_mode or "<developer-modes>",
            "dataset_type": args.dataset_type,
            "dataset_file": args.dataset_file,
            "context_source": args.context_source,
            "memory_retrieval_mode": args.memory_retrieval_mode,
            "memory_organization": memory_organization_for_mode(args.memory_retrieval_mode),
            "task_count": len(tasks),
            "modes": modes,
            "from_agent": "runtime",
            "to_agent": "dashboard",
            "action_type": "start_run",
            "state_refs": [],
            "memory_refs": [],
            "text_chars": 0,
            "elapsed_ms": 0,
        },
    )
    emit_runtime_event(
        {
            "event_type": "task_queue",
            "experiment_mode": args.experiment_mode or "<developer-modes>",
            "dataset_type": args.dataset_type,
            "dataset_file": args.dataset_file,
            "task_count": len(tasks),
            "tasks": dashboard_task_queue(tasks),
            "from_agent": "runtime",
            "to_agent": "dashboard",
            "action_type": "load_task_queue",
            "state_refs": [],
            "memory_refs": [],
            "text_chars": 0,
            "elapsed_ms": 0,
        }
    )
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
        f"file={args.dataset_file}, "
        f"context_source={args.context_source}, "
        f"memory_retrieval_mode={args.memory_retrieval_mode}, "
        f"memory_organization={memory_organization_for_mode(args.memory_retrieval_mode)}, "
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
    emit_runtime_event(
        {
            "event_type": "run_done",
            "experiment_mode": args.experiment_mode or "<developer-modes>",
            "dataset_type": args.dataset_type,
            "dataset_file": args.dataset_file,
            "memory_retrieval_mode": args.memory_retrieval_mode,
            "memory_organization": memory_organization_for_mode(args.memory_retrieval_mode),
            "task_count": len(tasks),
            "modes": modes,
            "from_agent": "runtime",
            "to_agent": "dashboard",
            "action_type": "finish_run",
            "state_refs": [],
            "memory_refs": [],
            "text_chars": sum((result.get("metrics", {}) or {}).get("text_chars", 0) for result in results),
            "elapsed_ms": sum((result.get("metrics", {}) or {}).get("elapsed_ms", 0) for result in results),
            "return_result": {
                "runs": len(results),
                "messages": sum((result.get("metrics", {}) or {}).get("message_count", 0) for result in results),
                "memory_hits": sum((result.get("metrics", {}) or {}).get("memory_hits", 0) for result in results),
            },
        }
    )
    print_readable_run_summaries(results)
    print_comparison(results)
    print_run_result_json(args, tasks, results)


if __name__ == "__main__":
    main()
