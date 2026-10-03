from __future__ import annotations

import json
import os
import signal
import re
import subprocess
import sys
import time
import hashlib
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from uuid import uuid4
from urllib.parse import urlparse

from hypergraph_memory_method.reuse_metrics import evaluate_run_reuse
from hypergraph_memory_method.index import clear_hypergraph_index_cache
from hypergraph_memory_method.layers import memory_layer_for_record
from hypergraph_memory_method.store import SQLiteHypergraphMemoryStore


ROOT = Path(__file__).resolve().parents[2]
FRONTEND_PUBLIC = ROOT / "examples" / "competition_agents" / "frontend" / "public"
RUNTIME_DIR = FRONTEND_PUBLIC / "runtime"
EVENT_LOG_FILE = RUNTIME_DIR / "run_events.jsonl"
STDOUT_LOG_FILE = RUNTIME_DIR / "latest_run_stdout.log"
RUN_RESULT_DIR = ROOT / "run_result"
RUN_ARCHIVE_DIR = RUN_RESULT_DIR
HOST = "127.0.0.1"
PORT = 8765
RUN_RESULT_MARKER = "=== run_result_json ==="
METRICS_CACHE_TTL_SECONDS = 30
METRICS_CACHE_AT = 0.0
METRICS_CACHE: list[dict[str, Any]] = []
FILE_RECORD_CACHE: dict[str, tuple[tuple[int, ...], list[dict[str, Any]]]] = {}


from competition_agents.methods.catalog import CATALOG, DATASETS as METHOD_DATASETS, METHODS, validate

DATASETS = {key: {**value, 'task_count': value['taskCount'], 'dataset_type': value['datasetType']}
            for key, value in METHOD_DATASETS.items()}
EXPERIMENT_MODES = {key: value['experimentMode'] for key, value in METHODS.items()}


class RunnerState:
    process: subprocess.Popen[bytes] | None = None
    stdout_handle: Any = None
    started_at: float | None = None
    config: dict[str, Any] = {}
    command: list[str] = []
    last_error: str = ""
    archived_pid: int | None = None
    run_id: str = ''
    stopped: bool = False
    finished_at: float | None = None


STATE = RunnerState()


def normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    collaboration = str(raw.get('collaborationMode') or 'plain')
    method = METHODS.get(collaboration)
    if method is None:
        raise ValueError('不支持的协作方式。')
    dataset_key = str(raw.get('dataset') or CATALOG['defaultDataset'])
    dataset = DATASETS.get(dataset_key)
    if dataset is None:
        raise ValueError('所选数据集没有绑定已准备好的知识库。')
    config = {
        'dataset': dataset_key,
        'taskCount': dataset.get('runTaskCount', dataset['taskCount']),
        'collaborationMode': collaboration,
        'memoryMode': str(raw.get('memoryMode') or method['memory']),
        'retrievalMode': str(raw.get('retrievalMode') or method['retrieval']),
        'backend': str(raw.get('backend') or 'local'),
        'localModel': str(raw.get('localModel') or CATALOG['defaultLocalModel']),
        'baseUrl': str(raw.get('baseUrl') or ''),
        'remoteModel': str(raw.get('remoteModel') or CATALOG['defaultRemoteModel']),
        'apiKey': str(raw.get('apiKey') or ''),
        'maxTokens': raw.get('maxTokens', 1024),
        'maxSearches': raw.get('maxSearches', CATALOG.get('defaultMaxSearches', 2)),
        'memoryEnabled': raw.get('memoryEnabled', True),
        'outputRepeatNgramSize': raw.get('outputRepeatNgramSize',
            CATALOG.get('defaultOutputRepeatNgramSize', 0) if str(raw.get('backend') or 'local') == 'local' else 0),
    }
    validate(config)
    return config


def build_command(config: dict[str, Any]) -> list[str]:
    validate(config)
    command = [sys.executable, 'examples/competition_agents/run_competition.py',
            '--method', config['collaborationMode'], '--dataset', config['dataset'],
            '--backend', config['backend'], '--local-model', config['localModel'],
            '--remote-model', config['remoteModel'], '--limit', str(config['taskCount']),
            '--max-tokens', str(config.get('maxTokens', 1024)),
            '--max-searches', str(config.get('maxSearches', CATALOG.get('defaultMaxSearches', 2))),
            '--output-repeat-ngram-size', str(config.get('outputRepeatNgramSize',
                CATALOG.get('defaultOutputRepeatNgramSize', 0) if config['backend'] == 'local' else 0)),
            '--event-log-file', str(EVENT_LOG_FILE)]
    if not config.get('memoryEnabled', True):
        command.append('--disable-memory')
    return command


def build_env(config: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    for key in ('TEXT_COMM_STYLE', 'LOCAL_FAST_MODE', 'LOCAL_CAPTURE_HIDDEN',
                'AGENT_EXECUTION_CONTEXT', 'HANDOFF_EMBEDDING_PROVIDER'):
        env.pop(key, None)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if config["backend"] == "local":
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("LOCAL_HIDDEN_LOCAL_FILES_ONLY", "1")
        env.setdefault("LLM_PROVIDER", "local_hf")
        env.setdefault("PLANNER_LLM_PROVIDER", "local_hf")
        env.setdefault("REASONER_LLM_PROVIDER", "local_hf")
        env.setdefault("SUMMARIZER_LLM_PROVIDER", "local_hf")
        env["LOCAL_HIDDEN_MODEL"] = config["localModel"]
    else:
        if config["baseUrl"]:
            env["LLM_BASE_URL"] = config["baseUrl"]
        if config["remoteModel"]:
            env["LLM_MODEL"] = config["remoteModel"]
        if config["apiKey"]:
            env["LLM_API_KEY"] = config["apiKey"]
    long_term_enabled = str(env.get("COMPETITION_LONG_TERM_MEMORY", "0")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if long_term_enabled:
        memory_root = Path(
            env.get("COMPETITION_MEMORY_ROOT")
            or (RUN_RESULT_DIR / "memory_store")
        )
        memory_root.mkdir(parents=True, exist_ok=True)
        env["HYPERGRAPH_MEMORY_DB"] = str(
            memory_root / memory_database_name(config)
        )
        env["HYPERGRAPH_MEMORY_AUTOPERSIST"] = "1"
        env["HYPERGRAPH_MEMORY_AUTOLOAD"] = "1"
    else:
        env.pop("HYPERGRAPH_MEMORY_DB", None)
    return env


def start_run(config: dict[str, Any]) -> dict[str, Any]:
    global METRICS_CACHE_AT, METRICS_CACHE
    refresh_process_state()
    if STATE.process and STATE.process.poll() is None:
        raise RuntimeError("已有实验正在运行")

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    for path in (EVENT_LOG_FILE, STDOUT_LOG_FILE):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    STATE.run_id = uuid4().hex
    STATE.stopped = False
    STATE.finished_at = None
    command = build_command(config)
    stdout_handle = STDOUT_LOG_FILE.open("wb")
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**build_env(config), 'COMPETITION_RUN_ID': STATE.run_id},
            stdout=stdout_handle,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
    except Exception:
        stdout_handle.close()
        raise

    STATE.process = process
    STATE.stdout_handle = stdout_handle
    STATE.started_at = time.time()
    STATE.config = {**config, "apiKey": ""}
    STATE.command = command
    STATE.last_error = ""
    STATE.archived_pid = None
    METRICS_CACHE = []
    METRICS_CACHE_AT = 0.0
    return status_payload()


def stop_run() -> dict[str, Any]:
    refresh_process_state()
    process = STATE.process
    if process and process.poll() is None:
        STATE.stopped = True
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    refresh_process_state()
    return status_payload()


def refresh_process_state() -> None:
    global METRICS_CACHE_AT, METRICS_CACHE
    process = STATE.process
    if process and process.poll() is not None:
        if STATE.finished_at is None:
            STATE.finished_at = time.time()
        if STATE.stdout_handle:
            STATE.stdout_handle.close()
            STATE.stdout_handle = None
        if STATE.archived_pid != process.pid:
            if archive_latest_run_result():
                METRICS_CACHE = []
                METRICS_CACHE_AT = 0.0
            STATE.archived_pid = process.pid
        if STATE.stopped:
            STATE.last_error = ''
        elif process.returncode and process.returncode != 0:
            STATE.last_error = tail_text(STDOUT_LOG_FILE, 1600) or f"进程退出码：{process.returncode}"
        elif process.returncode == 0:
            STATE.last_error = ""


def read_events() -> list[dict[str, Any]]:
    if not EVENT_LOG_FILE.exists():
        return []
    events = []
    for line in EVENT_LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return [e for e in events if e.get('run_id') == STATE.run_id] if STATE.run_id else []


def has_archived_records() -> bool:
    if not RUN_ARCHIVE_DIR.exists():
        return False
    return any(path.is_file() for path in RUN_ARCHIVE_DIR.rglob("*.json"))


def has_running_process() -> bool:
    return bool(STATE.process and STATE.process.poll() is None)


def clear_runtime_files(*, include_stdout: bool = False) -> None:
    paths = (EVENT_LOG_FILE, STDOUT_LOG_FILE) if include_stdout else (EVENT_LOG_FILE,)
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def read_run_result_payload(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if path.suffix.lower() == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    index = text.rfind(RUN_RESULT_MARKER)
    if index < 0:
        return None
    raw = text[index + len(RUN_RESULT_MARKER) :].strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        first_line = raw.splitlines()[0].strip() if raw.splitlines() else ""
        if not first_line:
            return None
        try:
            return json.loads(first_line)
        except json.JSONDecodeError:
            return None


def safe_folder_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def result_folder_name(config: dict[str, Any] | None, payload: dict[str, Any] | None = None) -> str:
    config = config or {}
    payload = payload or {}
    backend = str(config.get("backend") or "").lower()
    if backend == "local":
        model_name = str(config.get("localModel") or os.environ.get("LOCAL_HIDDEN_MODEL") or "local")
        return safe_folder_name(model_name.split("/")[-1])
    if backend == "openai-compatible":
        model_name = str(config.get("remoteModel") or os.environ.get("LLM_MODEL") or "api")
        return f"api_{safe_folder_name(model_name)}"
    model_name = str(payload.get("model") or os.environ.get("LOCAL_HIDDEN_MODEL") or os.environ.get("LLM_MODEL") or "unknown")
    return safe_folder_name(model_name.split("/")[-1])


def archive_latest_run_result() -> bool:
    payload = read_run_result_payload(STDOUT_LOG_FILE)
    if not payload or not payload.get("runs"):
        return False
    archived_at = datetime.now(timezone.utc).isoformat()
    archive_hash = hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]
    dataset = dataset_key_for_file(str(payload.get("dataset_file") or ""))
    mode = str(payload.get("experiment_mode") or "experiment").replace("/", "_")
    retrieval = str(payload.get("memory_retrieval_mode") or "retrieval").replace("/", "_")
    fallback_variant = (
        "fallback-on"
        if payload.get("extractive_fallback_enabled", True)
        else "fallback-off"
    )
    archive_id = f"{dataset}_{mode}_{retrieval}_{fallback_variant}_{archive_hash}"
    archived_payload = {
        **payload,
        "archive_id": archive_id,
        "archived_at": archived_at,
    }
    output_dir = RUN_ARCHIVE_DIR / result_folder_name(STATE.config, payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"frontend_{mode}_{dataset}_{retrieval}_{fallback_variant}.json"
    path.write_text(json.dumps(archived_payload, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
    return True


def delete_archived_record(record_id: str) -> bool:
    target = str(record_id or "").strip()
    if not target or not RUN_ARCHIVE_DIR.exists():
        return False
    archive_target = target.split("|", 1)[0]
    deleted = False
    deleted_fingerprints: set[str] = set()
    for path in RUN_ARCHIVE_DIR.rglob("*.json"):
        payload = read_run_result_payload(path)
        if not payload:
            continue
        if str(payload.get("archive_id") or "") == archive_target:
            for run in payload.get("runs", []) or []:
                deleted_fingerprints.add(run_fingerprint(payload, run))
            path.unlink()
            deleted = True
    if deleted_fingerprints:
        latest_payload = read_run_result_payload(STDOUT_LOG_FILE)
        latest_fingerprints = {
            run_fingerprint(latest_payload, run)
            for run in (latest_payload or {}).get("runs", []) or []
        }
        if latest_fingerprints & deleted_fingerprints:
            try:
                STDOUT_LOG_FILE.unlink()
            except FileNotFoundError:
                pass
    return deleted


def dataset_key_for_file(dataset_file: str) -> str:
    normalized = str(dataset_file).replace('\\', '/')
    for key, dataset in DATASETS.items():
        if normalized.endswith(dataset['file']):
            return key
    return normalized or "unknown"


def collaboration_key_for_mode(mode: str) -> str:
    normalized = str(mode or "").lower()
    mapping = {
        "baseline_text": "plain",
        "autogen_text": "autogen",
        "camel_text": "camel",
        "autogen_camel_text": "autogen",
        "autoform_text": "autoform",
        "llmlingua_text": "llmlingua",
        "text": "plain",
        "plain": "plain",
        "protocol_api": "structured",
        "protocol": "structured",
        "protocol_api_embedding": "protocol-vector",
        "hidden_api": "latent",
        "hidden": "latent",
        "latent_hidden": "latent",
        "codeact_tools": "structured",
    }
    return mapping.get(normalized, normalized or "unknown")


def memory_key_for_value(value: str) -> str:
    normalized = str(value or "").lower()
    if normalized in {"hypergraph", "fusion"}:
        return "hypergraph"
    if normalized in {"text", "keyword", "tag", "semantic"}:
        return "text"
    return normalized or "unknown"


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def task_hit_ids(task_result: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for key in ("memory_hit_ids", "evidence_hit_ids", "experience_hit_ids"):
        for item in task_result.get(key, []) or []:
            ids.add(str(item))
    for hit in task_result.get("memory_hits", []) or []:
        if isinstance(hit, dict):
            for key in ("memory_id", "id"):
                if hit.get(key):
                    ids.add(str(hit[key]))
    return ids


def compute_memory_recall(task_results: list[dict[str, Any]]) -> float | None:
    expected_total = 0
    recalled = 0
    for task_result in task_results:
        expected = {str(item) for item in task_result.get("expected_reuse", []) or [] if str(item)}
        if not expected:
            continue
        hits = task_hit_ids(task_result)
        expected_total += len(expected)
        recalled += len(expected & hits)
    if expected_total == 0:
        return None
    return recalled / expected_total


def compute_effective_reuse(run: dict[str, Any], task_results: list[dict[str, Any]], task_count: int) -> float | None:
    metrics = run.get("metrics", {}) or {}
    success_count = safe_int(metrics.get("reuse_success_count"))
    if "reuse_success_count" in metrics and task_count:
        return min(1.0, success_count / task_count)
    comparable = [
        task
        for task in task_results
        if task.get("exact_match") is not None and safe_int(task.get("memory_hit_count")) > 0
    ]
    if comparable and task_count:
        success = sum(1 for task in comparable if bool(task.get("exact_match")))
        return min(1.0, success / task_count)
    used_count = safe_int(metrics.get("experience_used_count"))
    if "experience_used_count" in metrics and task_count:
        return min(1.0, used_count / task_count)
    return None


def reuse_summary_from_run(run: dict[str, Any], task_results: list[dict[str, Any]]) -> dict[str, Any]:
    report = run.get("reuse_report")
    if isinstance(report, dict) and isinstance(report.get("summary"), dict):
        return report["summary"]
    if not task_results:
        return {}
    try:
        return evaluate_run_reuse({"task_results": task_results}).get("summary", {})
    except Exception:
        return {}


def per_task_series(task_results: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for task_result in sorted(task_results, key=lambda item: safe_int(item.get("task_index"))):
        metrics = task_result.get("metrics", {}) or {}
        if key == "text_chars":
            values.append(float(safe_int(metrics.get("text_chars"))))
        elif key == "actual_message_bytes":
            values.append(metrics['actual_message_bytes'] if 'actual_message_bytes' in metrics else
                float(
                    safe_int(
                        metrics.get("logical_protocol_bytes")
                        or metrics.get("protocol_bytes")
                        or metrics.get("text_chars")
                    )
                )
            )
        elif key == "elapsed_seconds":
            elapsed = safe_float(metrics.get("elapsed_ms")) or 0.0
            values.append(round(float(elapsed) / 1000, 3))
        elif key == "memory_reuse":
            hit_rate = safe_float(metrics.get("memory_hit_rate"))
            if hit_rate is None:
                hit_rate = 1.0 if safe_int(task_result.get("memory_hit_count")) > 0 else 0.0
            values.append(round(hit_rate * 100, 2))
        elif key == 'handoff_payload_bytes':
            values.append(safe_float(metrics.get(key)))
    return values


def answer_token_f1_from_run(run: dict[str, Any], task_results: list[dict[str, Any]]) -> float | None:
    provided = safe_float(run.get("answer_token_f1"))
    if provided is not None:
        return provided
    values = [
        safe_float(task.get("answer_token_f1") or (task.get("metrics") or {}).get("answer_token_f1"))
        for task in task_results
    ]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _metric_tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def reference_evidence_coverage(task_results: list[dict[str, Any]]) -> float | None:
    """Post-hoc coverage of reference-answer terms in retrieved evidence.

    This is an evidence-coverage diagnostic, not an online retrieval feature and
    never changes Agent inputs or rankings.
    """

    values: list[float] = []
    for task in task_results:
        evidence_tokens = _metric_tokens(" ".join(str(item) for item in task.get("evidence", []) or []))
        references = task.get("reference_answers") or [task.get("expected_answer")]
        scores = []
        for reference in references:
            reference_tokens = _metric_tokens(reference)
            if reference_tokens:
                scores.append(len(evidence_tokens & reference_tokens) / len(reference_tokens))
        if scores:
            values.append(max(scores))
    return sum(values) / len(values) if values else None


def planner_execution_rate(task_results: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for task in task_results:
        debug = task.get("retrieval_debug", {}) or {}
        policy = debug.get("retrieval_policy", {}) or {}
        planned = len(policy.get("knowledge_steps", []) or [])
        executed = safe_int((task.get("metrics", {}) or {}).get("knowledge_query_count"))
        if not planned:
            # Older run records preserve the Planner focus but not the structured
            # step list. Treat that focus as one retrieval plan for compatibility.
            planned = 1 if (debug.get("planner_focus") or debug.get("retrieval_query")) else 0
        if not planned:
            continue
        if not executed and debug.get("context_source"):
            executed = 1
        values.append(min(1.0, executed / planned))
    return sum(values) / len(values) if values else None


def run_fingerprint(payload: dict[str, Any], run: dict[str, Any]) -> str:
    compact = {
        "dataset_file": run.get("dataset_file") or payload.get("dataset_file"),
        "experiment_mode": run.get("experiment_mode") or payload.get("experiment_mode") or run.get("mode"),
        "memory_retrieval_mode": run.get("memory_retrieval_mode") or payload.get("memory_retrieval_mode"),
        "memory_organization": run.get("memory_organization") or payload.get("memory_organization"),
        "task_count": run.get("task_count") or run.get("tasks"),
        "metrics": run.get("metrics", {}),
    }
    return hashlib.sha1(json.dumps(compact, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def method_record_from_run(
    payload: dict[str, Any],
    run: dict[str, Any],
    source: str,
    run_index: int = 0,
    *,
    include_payload: bool = False,
) -> dict[str, Any]:
    metrics = run.get("metrics", {}) or {}
    task_results = run.get("task_results", []) or []
    task_count = safe_int(run.get("task_count") or run.get("tasks") or metrics.get("task_count") or payload.get("task_count"))
    experiment_mode = str(run.get("experiment_mode") or payload.get("experiment_mode") or run.get("mode") or "")
    collaboration = collaboration_key_for_mode(experiment_mode)
    retrieval = str(run.get("memory_retrieval_mode") or payload.get("memory_retrieval_mode") or "")
    memory = memory_key_for_value(run.get("memory_organization") or payload.get("memory_organization") or retrieval)
    dataset = dataset_key_for_file(str(run.get("dataset_file") or payload.get("dataset_file") or ""))
    exact_match = safe_float(run.get("exact_match"))
    if exact_match is None:
        exact_match = safe_float(metrics.get("exact_match"))
    if exact_match is None:
        comparable_exact = [
            bool(task.get("exact_match"))
            for task in task_results
            if isinstance(task.get("exact_match"), bool)
        ]
        if comparable_exact:
            exact_match = sum(1 for value in comparable_exact if value) / len(comparable_exact)
    reuse_summary = reuse_summary_from_run(run, task_results)
    expected_reuse_count = safe_int(reuse_summary.get("expected_reuse_task_count"))
    memory_hit_rate = safe_float(run.get("memory_hit_rate"))
    if memory_hit_rate is None:
        memory_hit_rate = safe_float(metrics.get("memory_hit_rate"))
    recall = safe_float(reuse_summary.get("expected_artifact_recall")) if expected_reuse_count else None
    expected_reuse_hit_rate = (
        safe_float(reuse_summary.get("memory_retrieval_accuracy"))
        if expected_reuse_count
        else None
    )
    memory_precision = (
        safe_float(reuse_summary.get("memory_precision")) if expected_reuse_count else None
    )
    correct_reuse_rate = (
        safe_float(reuse_summary.get("correct_reuse_rate")) if expected_reuse_count else None
    )
    complete_retrieval_rate = (
        safe_float(reuse_summary.get("memory_complete_retrieval_rate"))
        if expected_reuse_count
        else None
    )
    elapsed_ms = safe_float(run.get("elapsed_ms")) or safe_float(metrics.get("elapsed_ms")) or 0.0
    state_generation_ms = safe_float(metrics.get("state_generation_ms")) or 0.0
    comparable_elapsed_ms = safe_float(metrics.get("elapsed_excluding_state_generation_ms"))
    if comparable_elapsed_ms is None:
        comparable_elapsed_ms = safe_float(metrics.get("adjusted_elapsed_ms"))
    if comparable_elapsed_ms is None:
        comparable_elapsed_ms = max(0.0, elapsed_ms - state_generation_ms)
    text_only_collaboration = collaboration in {"plain", "autogen", "camel", "autoform", "llmlingua"}
    state_transfer_count = 0 if text_only_collaboration else safe_int(metrics.get("latent_transfer_count"))
    state_bytes = 0 if text_only_collaboration else safe_int(metrics.get("latent_bytes"))
    protocol_bytes = safe_int(metrics.get("logical_protocol_bytes"))
    if not protocol_bytes:
        protocol_bytes = safe_int(metrics.get("protocol_bytes"))
    if not protocol_bytes:
        protocol_bytes = safe_int(metrics.get("text_chars"))
    communication_bytes = safe_int(metrics.get("reference_wire_bytes"))
    if not communication_bytes:
        communication_bytes = safe_int(metrics.get("communication_bytes"))
    if not communication_bytes:
        communication_bytes = protocol_bytes
    if 'actual_message_bytes' in metrics:
        protocol_bytes = safe_float(metrics['actual_message_bytes'])
        communication_bytes = protocol_bytes
    answer_token_f1 = answer_token_f1_from_run(run, task_results)
    evidence_coverage = reference_evidence_coverage(task_results)
    plan_execution = planner_execution_rate(task_results)
    # The primary cross-method latency metric is always the complete task loop.
    # State generation is retained as a separate diagnostic, never subtracted.
    comparable_elapsed_ms = elapsed_ms
    archive_id = str(payload.get("archive_id") or "").strip()
    fingerprint = run_fingerprint(payload, run)
    record_id = f"{archive_id or source}|{run_index}|{fingerprint}"

    record = {
        "recordId": record_id,
        "fingerprint": fingerprint,
        "archivedAt": payload.get("archived_at"),
        "source": source,
        "runIndex": run_index,
        "dataset": dataset,
        "collaborationMode": collaboration,
        "experimentMode": experiment_mode,
        "memoryMode": memory,
        "retrievalMode": retrieval or "unknown",
        "taskCount": task_count,
        "runId": payload.get('run_id'),
        "model": payload.get('model'),
        "backend": payload.get('backend'),
        "baselineVersion": payload.get('baseline_version'),
        "datasetHash": payload.get('dataset_sha256'),
        "knowledgeHash": payload.get('knowledge_corpus_sha256'),
        "timingValid": payload.get('full_workflow_timing_valid') is True,
        "runStatus": payload.get('status', 'legacy'),
        "handoffPayloadBytes": safe_float(metrics.get('handoff_payload_bytes')),
        "handoffMeasurement": metrics.get('handoff_measurement'),
        "inputTokens": safe_float(metrics.get('input_tokens')),
        "outputTokens": safe_float(metrics.get('output_tokens')),
        "modelCalls": safe_float(metrics.get('model_calls')),
        "modelServiceMs": safe_float(metrics.get('model_service_ms')),
        "knowledgeSearchMs": safe_float(metrics.get('knowledge_search_ms')),
        "knowledgeSearches": safe_float(metrics.get('knowledge_searches')),
        "memorySearchMs": safe_float(metrics.get('memory_search_ms')),
        "executionMs": safe_float(metrics.get('execution_ms')),
        "messageCount": safe_int(metrics.get("message_count")),
        "textChars": safe_int(metrics.get("text_chars")),
        "protocolBytes": protocol_bytes,
        "actualMessageBytes": protocol_bytes,
        "stateTransferCount": state_transfer_count,
        "stateBytes": state_bytes,
        "statePeakBytes": safe_int(metrics.get('state_peak_bytes')),
        "stateReuseCount": safe_int(metrics.get('state_reuse_count')),
        "stateTransmissionMeasured": protocol_bytes is not None,
        "answerCoverage": safe_float(run.get('answer_accuracy')),
        "communicationCost": communication_bytes,
        "referenceControlBytes": communication_bytes,
        "stateGenerationMs": state_generation_ms,
        "elapsedMs": elapsed_ms,
        "comparableElapsedMs": comparable_elapsed_ms,
        "fullElapsedMs": elapsed_ms,
        "stateStoreWriteCount": safe_int(metrics.get("state_store_write_count")),
        "stateStoreReadCount": safe_int(metrics.get("state_store_read_count")),
        "stateStoreReadBytes": safe_int(metrics.get("state_store_read_bytes")),
        "stateStoreBytes": safe_int(metrics.get("state_store_bytes")),
        "plannerExecutionRate": plan_execution,
        "referenceEvidenceCoverage": evidence_coverage,
        "memoryHitRate": memory_hit_rate,
        "expectedReuseHitRate": expected_reuse_hit_rate,
        "memoryReuseRate": correct_reuse_rate,
        "effectiveMemoryReuseRate": None,
        "answerAccuracy": safe_float(run.get('answer_accuracy')) if 'answer_accuracy' in run else exact_match,
        "answerMetric": payload.get('evaluation_metric') or ('reference_coverage' if 'answer_accuracy' in run else 'legacy_exact_match'),
        "answerTokenF1": answer_token_f1,
        "memoryRecallRate": recall,
        "memoryPrecisionRate": memory_precision,
        "memoryCompleteRetrievalRate": complete_retrieval_rate,
        "expectedReuseTaskCount": expected_reuse_count,
        "series": {
            "textChars": per_task_series(task_results, "text_chars"),
            "actualMessageBytes": per_task_series(task_results, "actual_message_bytes"),
            "handoffPayloadBytes": per_task_series(task_results, "handoff_payload_bytes"),
            "elapsedSeconds": per_task_series(task_results, "elapsed_seconds"),
            "memoryReuse": per_task_series(task_results, "memory_reuse"),
        },
    }
    if include_payload:
        record["payload"] = {
            **{key: payload.get(key) for key in ('run_id', 'model', 'backend', 'baseline_version', 'dataset_sha256',
                                               'knowledge_corpus_sha256', 'full_workflow_timing_valid', 'evaluation_metric', 'status')},
            "schema_version": payload.get("schema_version", 1),
            "experiment_mode": experiment_mode,
            "memory_retrieval_mode": run.get("memory_retrieval_mode") or payload.get("memory_retrieval_mode"),
            "memory_organization": run.get("memory_organization") or payload.get("memory_organization"),
            "dataset_type": run.get("dataset_type") or payload.get("dataset_type"),
            "dataset_file": run.get("dataset_file") or payload.get("dataset_file"),
            "context_source": payload.get("context_source"),
            "task_count": task_count,
            "archived_at": payload.get("archived_at"),
            "runs": [run],
        }
    return record


def safe_result_file_path(relative_path: str) -> Path:
    relative = Path(str(relative_path).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"无效结果文件路径：{relative_path}")
    path = (RUN_RESULT_DIR / relative).resolve()
    result_root = RUN_RESULT_DIR.resolve()
    if result_root != path and result_root not in path.parents:
        raise ValueError(f"结果文件不在 run_result 目录内：{relative_path}")
    if path.suffix.lower() != ".json" or not path.is_file():
        raise ValueError(f"结果文件不存在或不是 JSON：{relative_path}")
    return path


def result_files_payload() -> dict[str, Any]:
    RUN_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    for path in sorted(RUN_RESULT_DIR.rglob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        stat = path.stat()
        relative_path = path.relative_to(RUN_RESULT_DIR)
        files.append(
            {
                "name": path.name,
                "path": relative_path.as_posix(),
                "folder": relative_path.parent.as_posix() if str(relative_path.parent) != "." else "",
                "modifiedTime": stat.st_mtime,
                "modifiedLabel": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "size": stat.st_size,
            }
        )
    return {"directory": str(RUN_RESULT_DIR), "files": files}


def memory_database_name(config: dict[str, Any]) -> str:
    method_scope = safe_folder_name("_".join(
        str(config[key]) for key in ("collaborationMode", "memoryMode", "retrievalMode")
    ))
    return f"{result_folder_name(config)}__{method_scope}.sqlite3"


def clear_experience_memory(config: dict[str, Any]) -> dict[str, Any]:
    if has_running_process():
        raise ValueError("请先停止正在运行的实验，再清空经验缓存。")
    memory_root = Path(os.environ.get("COMPETITION_MEMORY_ROOT") or RUN_RESULT_DIR / "memory_store").resolve()
    if memory_root == ROOT.resolve() or memory_root == Path(memory_root.anchor):
        raise ValueError("经验存储目录配置无效。")
    target = memory_root / memory_database_name(config)
    paths = [target] if target.is_file() else []
    for path in paths:
        if memory_root not in path.resolve().parents:
            raise ValueError("经验数据库路径超出了配置的存储目录。")
    deleted = 0
    for path in paths:
        with SQLiteHypergraphMemoryStore(path) as store:
            experiences = [record for record in store.list()
                           if memory_layer_for_record(record) == "experience"
                           or record.get("memory_type") == "failed_experience"]
            with store.transaction() as connection:
                for record in experiences:
                    cursor = connection.execute(
                        "DELETE FROM memories WHERE global_memory_id = ?",
                        (str(record["global_memory_id"]),),
                    )
                    deleted += cursor.rowcount
    clear_hypergraph_index_cache()
    return {"cleared": True, "deletedCount": deleted, "databaseCount": len(paths)}


def import_result_records(file_paths: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in file_paths:
        path = safe_result_file_path(str(item))
        records.extend(records_for_file(path))
    return records


def records_for_file(path: Path) -> list[dict[str, Any]]:
    stat = path.stat()
    evaluation_path = path.with_suffix(".reuse-evaluation")
    evaluation_stat = evaluation_path.stat() if evaluation_path.is_file() else None
    stamp = (stat.st_mtime_ns, stat.st_size,
             evaluation_stat.st_mtime_ns if evaluation_stat else 0,
             evaluation_stat.st_size if evaluation_stat else 0)
    key = str(path.resolve())
    cached = FILE_RECORD_CACHE.get(key)
    if cached and cached[0] == stamp:
        return cached[1]
    payload = read_run_result_payload(path)
    if payload and evaluation_stat:
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation.get("source_size") == stat.st_size and evaluation.get("source_mtime_ns") == stat.st_mtime_ns:
            for run, report in zip(payload.get("runs", []), evaluation.get("reports", [])):
                if not isinstance(run, dict) or not isinstance(report, dict):
                    continue
                # Offline QA-reuse sidecar only replaces the report when it is
                # available; missing groups keep the original run metrics.
                if report.get("available"):
                    run["reuse_report"] = report
    records = [
        method_record_from_run(payload, run, str(path.relative_to(ROOT)), index)
        for index, run in enumerate((payload or {}).get("runs", []) or [])
    ]
    FILE_RECORD_CACHE[key] = (stamp, records)
    return records


def import_result_payloads(file_paths: list[Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for item in file_paths:
        path = safe_result_file_path(str(item))
        payload = read_run_result_payload(path)
        if not payload:
            continue
        payloads.append(
            {
                "source": str(path.relative_to(ROOT)),
                "payload": payload,
            }
        )
    return payloads


def trace_payload_for_source(source: str) -> dict[str, Any] | None:
    if source == "latest":
        return read_run_result_payload(STDOUT_LOG_FILE)
    relative = Path(str(source).replace("\\", "/"))
    path = (ROOT / relative).resolve()
    result_root = RUN_RESULT_DIR.resolve()
    if result_root != path and result_root not in path.parents:
        return None
    return read_run_result_payload(path)


def collect_metric_records() -> list[dict[str, Any]]:
    global METRICS_CACHE_AT, METRICS_CACHE
    refresh_process_state()
    now = time.time()
    if METRICS_CACHE and now - METRICS_CACHE_AT < METRICS_CACHE_TTL_SECONDS:
        return METRICS_CACHE

    records: list[dict[str, Any]] = []
    archived_fingerprints: set[str] = set()

    if RUN_ARCHIVE_DIR.exists():
        for path in sorted(RUN_ARCHIVE_DIR.rglob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            for record in records_for_file(path):
                records.append(record)
                archived_fingerprints.add(record["fingerprint"])

    latest_payload = read_run_result_payload(STDOUT_LOG_FILE)
    if latest_payload:
        latest_records = [
            method_record_from_run(latest_payload, run, "latest", index)
            for index, run in enumerate(latest_payload.get("runs", []) or [])
        ]
        missing_latest_records = [
            record for record in latest_records if record["fingerprint"] not in archived_fingerprints
        ]
        if missing_latest_records and not latest_payload.get("archive_id") and archive_latest_run_result():
            METRICS_CACHE = []
            METRICS_CACHE_AT = 0.0
            return collect_metric_records()
        records.extend(missing_latest_records)

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for record in records:
        key = (
            record["dataset"],
            record["collaborationMode"],
            record["memoryMode"],
            record["retrievalMode"],
            record["taskCount"],
            record["textChars"],
            record["elapsedMs"],
            record["source"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    METRICS_CACHE = deduped
    METRICS_CACHE_AT = now
    return deduped


def event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    task_count = 0
    run_start = next((event for event in events if event.get("event_type") == "run_start"), {})
    if run_start:
        task_count = int(run_start.get("task_count") or 0)
    task_done = [event for event in events if event.get("event_type") == "task_done"
                 and not (event.get('return_result') or {}).get('error') and event.get('status') != 'failed']
    failed = [event for event in events if event.get('event_type') == 'task_failed'
              or (event.get('event_type') == 'task_done' and (event.get('return_result') or {}).get('error'))]
    task_start = [event for event in events if event.get("event_type") == "task_start"]
    latest = next((event for event in reversed(events) if event.get("task_index")), {})
    latest_agent = latest.get("to_agent") or latest.get("from_agent") or "未开始"
    text_chars = sum(int(event.get("text_chars") or 0) for event in task_done)
    if not text_chars:
        text_chars = sum(int(event.get("text_chars") or 0) for event in events if event.get("event_type") == "agent_message")
    return {
        "taskCount": task_count or len(task_start),
        "currentTask": int(latest.get("task_index") or len(task_done) or 0),
        "completedTasks": len({event.get('task_id', event.get('task_index')) for event in task_done}),
        "failedTasks": len({event.get('task_id', event.get('task_index')) for event in failed}),
        "currentAgent": normalize_agent(latest_agent),
        "messageCount": len([event for event in events if event.get("event_type") == "agent_message"]),
        "textChars": text_chars,
        "stateTransfers": len([event for event in events if event.get("event_type") == "state_transfer"]),
        "memoryHits": len([event for event in events if event.get("event_type") == "memory_hit"]),
        "runDone": any(event.get("event_type") == "run_done" for event in events),
    }


def normalize_agent(agent: str) -> str:
    mapping = {
        "planner": "Planner",
        "retriever": "Retriever",
        "reasoner": "Reasoner",
        "executor": "Executor",
        "summarizer": "Summarizer",
        "memory": "Memory",
        "runtime": "Runtime",
        "dashboard": "Dashboard",
    }
    return mapping.get(str(agent).lower(), str(agent))




def latest_run_result_payload() -> dict[str, Any] | None:
    refresh_process_state()
    latest_payload = read_run_result_payload(STDOUT_LOG_FILE)
    if latest_payload and latest_payload.get('run_id') == STATE.run_id:
        return latest_payload
    return None

def status_payload() -> dict[str, Any]:
    refresh_process_state()
    process = STATE.process
    running = bool(process and process.poll() is None)
    returncode = process.poll() if process else None
    events = read_events()
    summary = event_summary(events)
    status = ('running' if running else 'stopped' if STATE.stopped else 'failed'
              if STATE.last_error or summary['failedTasks'] else 'completed'
              if summary['runDone'] and summary['completedTasks'] == summary['taskCount']
              else 'failed' if process is not None else 'idle')
    elapsed = int((STATE.finished_at or time.time()) - STATE.started_at) if STATE.started_at else 0
    return {
        "status": status,
        "runId": STATE.run_id,
        "running": running,
        "returnCode": returncode,
        "startedAt": datetime.fromtimestamp(STATE.started_at, timezone.utc).isoformat() if STATE.started_at else "",
        "elapsedSeconds": elapsed,
        "config": STATE.config,
        "command": STATE.command,
        "summary": summary,
        "eventLog": str(EVENT_LOG_FILE),
        "stdoutLog": str(STDOUT_LOG_FILE),
        "lastError": STATE.last_error,
    }


def tail_text(path: Path, limit: int) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self.send_empty(204)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/runner/methods":
            self.send_json(CATALOG)
            return
        if path == "/api/runner/status":
            self.send_json(status_payload())
            return
        if path == "/api/runner/events":
            refresh_process_state()
            if not has_running_process() and not has_archived_records() and not read_run_result_payload(STDOUT_LOG_FILE):
                clear_runtime_files()
            self.send_json({"events": read_events()})
            return
        if path == "/api/runner/metrics":
            self.send_json({"records": collect_metric_records()})
            return
        if path == "/api/runner/records":
            self.send_json({"records": collect_metric_records()})
            return
        if path == "/api/runner/result-files":
            self.send_json(result_files_payload())
            return
        if path == "/api/runner/result":
            self.send_json({"payload": latest_run_result_payload()})
            return
        self.send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self.read_json()
            if path == "/api/runner/start":
                self.send_json(start_run(normalize_config(body.get("config") or body)))
                return
            if path == "/api/runner/stop":
                self.send_json(stop_run())
                return
            if path == "/api/runner/clear-experience":
                if body.get("confirm") is not True:
                    raise ValueError("请确认清空经验缓存。")
                config = body.get("config")
                if not isinstance(config, dict) or not all(
                    config.get(key) for key in ("backend", "collaborationMode", "memoryMode", "retrievalMode")
                ) or not config.get("localModel" if config.get("backend") == "local" else "remoteModel"):
                    raise ValueError("请选择要清空经验的模型和方法。")
                self.send_json(clear_experience_memory(normalize_config(config)))
                return
            if path == "/api/runner/delete-record":
                global METRICS_CACHE_AT, METRICS_CACHE
                deleted = delete_archived_record(str(body.get("recordId") or ""))
                METRICS_CACHE = []
                METRICS_CACHE_AT = 0.0
                records = collect_metric_records()
                if not records:
                    clear_runtime_files(include_stdout=True)
                self.send_json({"deleted": deleted, "records": records})
                return
            if path == "/api/runner/import-results":
                file_paths = body.get("files")
                if not isinstance(file_paths, list) or not file_paths:
                    raise ValueError("请选择至少一个结果文件。")
                records = import_result_records(file_paths)
                METRICS_CACHE = []
                METRICS_CACHE_AT = 0.0
                self.send_json({"records": records, "payloads": []})
                return
            if path == "/api/runner/trace":
                source = str(body.get("source") or "")
                payload = trace_payload_for_source(source)
                if not payload:
                    raise ValueError("Trace payload was not found.")
                index = int(body.get("runIndex") or 0)
                runs = payload.get("runs", [])
                if index < 0 or index >= len(runs):
                    raise ValueError("Run index was not found.")
                payload = {**payload, "runs": [runs[index]]}
                self.send_json({"payload": payload})
                return
            if path == "/api/runner/clear-runtime":
                clear_runtime_files(include_stdout=bool(body.get("includeStdout")))
                self.send_json({"cleared": True, "events": read_events()})
                return
            self.send_json({"error": "not found"}, status=404)
        except Exception as exc:
            STATE.last_error = str(exc)
            self.send_json({"error": str(exc), **status_payload()}, status=500)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if not length:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def send_empty(self, status: int) -> None:
        self.send_response(status)
        self.send_cors()
        self.end_headers()

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return

    def send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"runner server: http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
