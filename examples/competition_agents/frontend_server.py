from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
RESULT_MARKER = "=== run_result_json ==="
DEFAULT_ENV_FILE = "examples/competition_agents/.env.local"
FALLBACK_ENV_FILE = "examples/competition_agents/.env.example"
ECHO_RUN_OUTPUT = False
DEFAULT_RUN_TIMEOUT_SECONDS = 3600
RUN_RESULT_DIR = REPO_ROOT / "run_result"
RUN_LOCK = threading.Lock()
CURRENT_RUN_PROCESS: subprocess.Popen[str] | None = None
CURRENT_RUN_META: dict[str, Any] | None = None
STOP_REQUESTED = False

TOPIOCQA_RESULT_FILES = {
    "baseline_text": "baseline_topiocqa_run.txt",
    "protocol_api": "api_topiocqa_run.txt",
    "protocol_api_embedding": "api_embedding_TopiOCQA_run.txt",
    "latent_hidden": "hidden_pooled_only_TopiOCQA_run.txt",
}


DATASETS: dict[str, dict[str, Any]] = {
    "hotpot_multihop": {
        "label": "HotpotQA 多跳问答",
        "script_dataset_type": "multihop",
        "dataset_file": "examples/competition_agents/data/hotpot_multihop_sample.json",
        "default_limit": 5,
        "preview_task_count": 5,
    },
    "hotpot_continuous": {
        "label": "continuous HotpotQA",
        "script_dataset_type": "continuous",
        "dataset_file": "examples/competition_agents/data/hotpot_continuous_tasks.json",
        "default_limit": 1,
        "preview_task_count": 10,
    },
    "hotpot_continuous_3turn": {
        "label": "continuous HotpotQA 3-turn",
        "script_dataset_type": "continuous",
        "dataset_file": "examples/competition_agents/data/hotpot_continuous_tasks_3turn.json",
        "default_limit": 1,
        "preview_task_count": 3,
    },
    "topiocqa_continuous": {
        "label": "TopiOCQA 连续追问",
        "script_dataset_type": "continuous",
        "dataset_file": "examples/competition_agents/data/topiocqa_two_chains_10turns_refined.json",
        "default_limit": 1,
        "preview_task_count": 10,
    },
    "hidden_hotpot_5": {
        "label": "Hidden Hotpot 样例",
        "script_dataset_type": "samples",
        "dataset_file": "",
        "default_limit": 0,
        "preview_task_count": 5,
    },
    "hidden_continuous": {
        "label": "Hidden 连续任务",
        "script_dataset_type": "continuous",
        "dataset_file": "examples/competition_agents/data/hotpot_continuous_tasks.json",
        "default_limit": 1,
        "preview_task_count": 10,
    },
    "codeact_20": {
        "label": "CodeAct 20 题",
        "script_dataset_type": "codeact",
        "dataset_file": "examples/competition_agents/data/codeact_tasks.json",
        "default_limit": 10,
        "preview_task_count": 10,
    },
}


EXPERIMENT_MODES: dict[str, dict[str, Any]] = {
    "baseline_text": {
        "label": "纯文本协作基线",
        "script": "run_demo.py",
        "datasets": {"hotpot_multihop", "hotpot_continuous", "hotpot_continuous_3turn", "topiocqa_continuous", "codeact_20"},
    },
    "protocol_api": {
        "label": "结构化协议协作",
        "script": "run_demo.py",
        "datasets": {"hotpot_multihop", "hotpot_continuous", "hotpot_continuous_3turn", "topiocqa_continuous", "codeact_20"},
    },
    "protocol_api_embedding": {
        "label": "协议 + API embedding",
        "script": "run_demo.py",
        "datasets": {"hotpot_multihop", "hotpot_continuous", "hotpot_continuous_3turn", "topiocqa_continuous", "codeact_20"},
    },
    "latent_hidden": {
        "label": "Hidden state 引用传递",
        "script": "run_hidden_latent_samples.py",
        "datasets": {"hidden_hotpot_5", "hidden_continuous", "topiocqa_continuous", "codeact_20"},
    },
    "codeact_tools": {
        "label": "受限工具执行",
        "script": "run_codeact_samples.py",
        "datasets": {"codeact_20"},
    },
}


def _read_json_request(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    raw_length = handler.headers.get("Content-Length", "0")
    try:
        length = int(raw_length)
    except ValueError as exc:
        raise ValueError("Content-Length must be an integer.") from exc
    if length <= 0:
        return {}
    if length > 64 * 1024:
        raise ValueError("Request body is too large.")
    body = handler.rfile.read(length).decode("utf-8")
    if not body.strip():
        return {}
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Request body is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    return payload


def _json_response(
    handler: BaseHTTPRequestHandler,
    status: int,
    payload: dict[str, Any],
) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(encoded)


def _stdout_tail(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _extract_run_result(stdout: str) -> dict[str, Any]:
    marker_index = stdout.rfind(RESULT_MARKER)
    if marker_index < 0:
        raise RuntimeError(f"Could not find `{RESULT_MARKER}` in experiment output.")
    after_marker = stdout[marker_index + len(RESULT_MARKER) :].strip()
    if not after_marker:
        raise RuntimeError("Experiment output marker was found, but JSON payload is missing.")
    json_line = after_marker.splitlines()[0].strip()
    try:
        result = json.loads(json_line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Experiment result JSON is invalid: {exc.msg}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("Experiment result JSON must be an object.")
    return result


def _result_file_stem(experiment_mode: str, dataset: str) -> str:
    if dataset == "topiocqa_continuous" and experiment_mode in TOPIOCQA_RESULT_FILES:
        return Path(TOPIOCQA_RESULT_FILES[experiment_mode]).stem
    return f"frontend_{experiment_mode}_{dataset}"


def _result_output_paths(experiment_mode: str, dataset: str) -> tuple[Path, Path]:
    stem = _result_file_stem(experiment_mode, dataset)
    stdout_path = RUN_RESULT_DIR / f"{stem}.txt"
    json_path = RUN_RESULT_DIR / f"{stem}.json"
    return stdout_path, json_path


def _persist_run_output(
    *,
    experiment_mode: str,
    dataset: str,
    stdout: str,
    result: dict[str, Any] | None = None,
) -> dict[str, str]:
    RUN_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path, json_path = _result_output_paths(experiment_mode, dataset)
    stdout_path.write_text(stdout, encoding="utf-8")
    saved = {"stdout_path": str(stdout_path.relative_to(REPO_ROOT))}
    if result is not None:
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        saved["json_path"] = str(json_path.relative_to(REPO_ROOT))
    return saved


def _set_current_run(
    process: subprocess.Popen[str],
    *,
    experiment_mode: str,
    dataset: str,
    command: list[str],
) -> None:
    global CURRENT_RUN_PROCESS, CURRENT_RUN_META, STOP_REQUESTED
    with RUN_LOCK:
        CURRENT_RUN_PROCESS = process
        CURRENT_RUN_META = {
            "experiment_mode": experiment_mode,
            "dataset": dataset,
            "command": command,
            "pid": process.pid,
        }
        STOP_REQUESTED = False


def _clear_current_run(process: subprocess.Popen[str] | None = None) -> None:
    global CURRENT_RUN_PROCESS, CURRENT_RUN_META
    with RUN_LOCK:
        if process is not None and CURRENT_RUN_PROCESS is not process:
            return
        CURRENT_RUN_PROCESS = None
        CURRENT_RUN_META = None


def stop_current_run() -> dict[str, Any]:
    global STOP_REQUESTED
    with RUN_LOCK:
        process = CURRENT_RUN_PROCESS
        meta = dict(CURRENT_RUN_META or {})
        if process is None or process.poll() is not None:
            CURRENT_RUN_PROCESS = None
            CURRENT_RUN_META = None
            STOP_REQUESTED = False
            return {
                "ok": True,
                "stopped": False,
                "message": "当前没有正在运行的实验进程。",
            }
        STOP_REQUESTED = True

    try:
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    finally:
        _clear_current_run(process)

    return {
        "ok": True,
        "stopped": True,
        "message": "已请求停止后端实验进程。",
        "pid": meta.get("pid"),
        "experiment_mode": meta.get("experiment_mode"),
        "dataset": meta.get("dataset"),
    }


def _summarize_task_result(task: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    for key in (
        "task_id",
        "group_id",
        "turn_id",
        "dataset_type",
        "question",
        "final_answer",
        "gold",
    ):
        if key in task:
            summary[key] = task.get(key)

    messages = task.get("messages")
    if isinstance(messages, list):
        summary["messages"] = [
            {
                "from": message.get("from"),
                "to": message.get("to"),
                "action": message.get("action"),
                "char_cost": message.get("char_cost"),
            }
            for message in messages
            if isinstance(message, dict)
        ]

    metrics = task.get("metrics")
    if isinstance(metrics, dict):
        summary["metrics"] = metrics

    if "exact_match" in task:
        summary["exact_match"] = task.get("exact_match")

    if "execution_result" in task:
        execution_result = task.get("execution_result")
        summary["execution_result"] = execution_result if isinstance(execution_result, dict) else {}

    for key in (
        "agent_views",
        "agent_outputs",
        "retrieval_debug",
        "memory_hits",
        "memory_context",
        "latest_evidence_write",
        "latest_experience_write",
        "latest_memory_write",
        "latent_states",
        "state_refs",
    ):
        value = task.get(key)
        if value is not None:
            summary[key] = value

    return summary


def _summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    for key in ("mode", "experiment_mode", "memory_hit_rate", "exact_match", "elapsed_ms"):
        if key in run:
            summary[key] = run.get(key)

    metrics = run.get("metrics")
    if isinstance(metrics, dict):
        summary["metrics"] = metrics

    task_results = run.get("task_results")
    if isinstance(task_results, list):
        summary["task_results"] = [
            _summarize_task_result(task) for task in task_results if isinstance(task, dict)
        ]

    return summary


def _summarize_result_for_frontend(result: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    for key in ("schema_version", "experiment_mode", "dataset_type"):
        if key in result:
            summary[key] = result.get(key)

    runs = result.get("runs")
    if isinstance(runs, list):
        summary["runs"] = [_summarize_run(run) for run in runs if isinstance(run, dict)]

    return summary


def _validate_run_request(payload: dict[str, Any]) -> tuple[str, str, int | None]:
    experiment_mode = str(payload.get("experiment_mode") or "").strip()
    dataset = str(payload.get("dataset") or payload.get("dataset_key") or "").strip()
    limit_value = payload.get("limit")

    if experiment_mode not in EXPERIMENT_MODES:
        raise ValueError(f"Unsupported experiment_mode: {experiment_mode!r}.")
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset!r}.")
    allowed_datasets = EXPERIMENT_MODES[experiment_mode]["datasets"]
    if dataset not in allowed_datasets:
        raise ValueError(f"Unsupported combination: {experiment_mode} + {dataset}.")

    limit: int | None = None
    if limit_value not in (None, ""):
        try:
            limit = int(limit_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("limit must be an integer.") from exc
        if limit < 0:
            raise ValueError("limit must be greater than or equal to 0.")
        if limit > 50:
            raise ValueError("limit is too large for the dashboard API; use 50 or less.")
    return experiment_mode, dataset, limit


PLACEHOLDER_CONFIG_VALUES = {
    "https://api.example.com/v1",
    "api.example.com",
    "sk-competition-console",
    "replace_with_your_api_key",
    "your_api_key",
    "你的 key",
}


def _payload_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    text = str(value).strip() if value not in (None, "") else ""
    return "" if text in PLACEHOLDER_CONFIG_VALUES else text


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def _repo_path(path: str) -> Path:
    parsed = Path(path)
    return parsed if parsed.is_absolute() else REPO_ROOT / parsed


def _resolve_env_file(env_file: str) -> str:
    if _repo_path(env_file).exists():
        return env_file
    if env_file == DEFAULT_ENV_FILE and _repo_path(FALLBACK_ENV_FILE).exists():
        return FALLBACK_ENV_FILE
    return env_file


def _is_missing_config_value(value: str | None) -> bool:
    return not value or value.strip() in PLACEHOLDER_CONFIG_VALUES


def _validate_real_llm_config(
    experiment_mode: str,
    env: dict[str, str],
    env_file: str,
    *,
    developer_mode: bool,
) -> None:
    if developer_mode or EXPERIMENT_MODES[experiment_mode]["script"] != "run_demo.py":
        return
    if _is_missing_config_value(env.get("LLM_API_KEY")):
        raise ValueError(
            "LLM_API_KEY is not configured. Fill api_key in the dashboard or create "
            f"{env_file} from examples/competition_agents/.env.example with a real key."
        )


def _build_env(
    env_file: str | None,
    payload: dict[str, Any],
    *,
    developer_mode: bool = False,
) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(REPO_ROOT / "libs" / "langgraph"), str(EXAMPLE_DIR)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if env_file:
        env["COMPETITION_AGENTS_ENV_FILE"] = env_file
        for name, value in _read_env_file(_repo_path(env_file)).items():
            env.setdefault(name, value)

    base_url = _payload_string(payload, "base_url") or _payload_string(payload, "llm_base_url")
    model = _payload_string(payload, "model") or _payload_string(payload, "llm_model")
    api_key = _payload_string(payload, "api_key") or _payload_string(payload, "llm_api_key")
    if base_url:
        env["LLM_BASE_URL"] = base_url
        env.setdefault("EMBEDDING_BASE_URL", base_url)
    if model:
        env["LLM_MODEL"] = model
    if api_key:
        env["LLM_API_KEY"] = api_key
        env.setdefault("EMBEDDING_API_KEY", api_key)

    if developer_mode:
        env["LLM_PROVIDER"] = "mock"
        env["LLM_PLANNER_PROVIDER"] = "mock"
        env["LLM_REASONER_PROVIDER"] = "mock"
        env["LLM_SUMMARIZER_PROVIDER"] = "mock"
        env["EMBEDDING_PROVIDER"] = "hash"
        env.pop("LLM_REQUIRE_REAL", None)
    return env


def _build_command(
    experiment_mode: str,
    dataset: str,
    limit: int | None,
    env_file: str,
    *,
    developer_mode: bool = False,
) -> list[str]:
    mode_config = EXPERIMENT_MODES[experiment_mode]
    dataset_config = DATASETS[dataset]
    effective_limit = dataset_config["default_limit"] if limit is None else limit

    if mode_config["script"] == "run_codeact_samples.py":
        command = [
            sys.executable,
            "examples/competition_agents/run_codeact_samples.py",
        ]
        if dataset_config["dataset_file"]:
            command.extend(["--dataset-file", dataset_config["dataset_file"]])
        if effective_limit:
            command.extend(["--limit", str(effective_limit)])
        return command

    if mode_config["script"] == "run_hidden_latent_samples.py":
        command = [
            sys.executable,
            "examples/competition_agents/run_hidden_latent_samples.py",
            "--dataset-type",
            dataset_config["script_dataset_type"],
        ]
        if dataset_config["dataset_file"]:
            command.extend(["--dataset-file", dataset_config["dataset_file"]])
        if effective_limit:
            command.extend(["--limit", str(effective_limit)])
        if dataset == "topiocqa_continuous":
            command.extend(["--context-source", "hypergraph", "--answer-mode", "generate"])
        return command

    command = [sys.executable, "examples/competition_agents/run_demo.py"]
    if developer_mode:
        mode = "text" if experiment_mode == "baseline_text" else "protocol"
        command.extend(["--modes", mode])
    else:
        command.extend(["--experiment-mode", experiment_mode, "--env-file", env_file])
    command.extend(["--dataset-type", dataset_config["script_dataset_type"]])
    if dataset_config["dataset_file"]:
        command.extend(["--dataset-file", dataset_config["dataset_file"]])
    if effective_limit:
        command.extend(["--limit", str(effective_limit)])
    if experiment_mode == "protocol_api_embedding" and dataset == "topiocqa_continuous":
        command.extend(["--context-source", "hypergraph"])
    if dataset in {"hotpot_continuous", "topiocqa_continuous", "codeact_20"} or experiment_mode in {
        "protocol_api",
        "protocol_api_embedding",
        "codeact_tools",
    }:
        command.extend(["--show-memory", "--show-latent", "--show-state"])
    return command


def _mask_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def run_experiment(payload: dict[str, Any]) -> dict[str, Any]:
    global STOP_REQUESTED
    experiment_mode, dataset, limit = _validate_run_request(payload)
    env_file = _resolve_env_file(str(payload.get("env_file") or DEFAULT_ENV_FILE))
    developer_mode = bool(payload.get("developer_mode", False))
    timeout_value = payload.get("timeout_seconds", DEFAULT_RUN_TIMEOUT_SECONDS)
    try:
        timeout_seconds = int(timeout_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout_seconds must be an integer.") from exc
    if timeout_seconds < 1 or timeout_seconds > 3600:
        raise ValueError("timeout_seconds must be between 1 and 3600.")

    command = _build_command(
        experiment_mode,
        dataset,
        limit,
        env_file,
        developer_mode=developer_mode,
    )
    env = _build_env(env_file, payload, developer_mode=developer_mode)
    print(
        "[frontend_server] received run request: "
        f"mode={experiment_mode}, dataset={dataset}, limit={limit}, developer_mode={developer_mode}",
        flush=True,
    )
    print(
        "[frontend_server] effective llm config: "
        f"provider={env.get('LLM_PROVIDER', '')}, "
        f"planner={env.get('LLM_PLANNER_PROVIDER', '')}, "
        f"reasoner={env.get('LLM_REASONER_PROVIDER', '')}, "
        f"summarizer={env.get('LLM_SUMMARIZER_PROVIDER', '')}, "
        f"base_url={env.get('LLM_BASE_URL', '')}, "
        f"model={env.get('LLM_MODEL', '')}, "
        f"api_key={_mask_secret(env.get('LLM_API_KEY'))}",
        flush=True,
    )
    print("[frontend_server] launching command:", " ".join(command), flush=True)
    _validate_real_llm_config(
        experiment_mode,
        env,
        env_file,
        developer_mode=developer_mode,
    )
    with RUN_LOCK:
        if CURRENT_RUN_PROCESS is not None and CURRENT_RUN_PROCESS.poll() is None:
            raise RuntimeError("已有实验正在运行，请先等待完成或点击停止。")

    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    _set_current_run(
        process,
        experiment_mode=experiment_mode,
        dataset=dataset,
        command=command,
    )

    try:
        stdout, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        timeout_stdout, _ = process.communicate()
        output = timeout_stdout or ((exc.stdout or "") if isinstance(exc.stdout, str) else "")
        _clear_current_run(process)
        STOP_REQUESTED = False
        raise subprocess.TimeoutExpired(exc.cmd, exc.timeout, output, exc.stderr) from exc
    finally:
        _clear_current_run(process)

    stdout = stdout or ""
    saved_paths = _persist_run_output(
        experiment_mode=experiment_mode,
        dataset=dataset,
        stdout=stdout,
    )
    if ECHO_RUN_OUTPUT:
        print("\n[frontend_server] experiment command:", " ".join(command), flush=True)
        print(stdout, end="" if stdout.endswith("\n") else "\n", flush=True)
    if process.returncode != 0:
        stopped_by_user = STOP_REQUESTED
        STOP_REQUESTED = False
        return {
            "ok": False,
            "canceled": stopped_by_user,
            "error": "实验已被停止。"
            if stopped_by_user
            else f"Experiment process exited with code {process.returncode}.",
            "command": command,
            **saved_paths,
            "stdout_tail": _stdout_tail(stdout),
        }

    try:
        result = _extract_run_result(stdout)
    except RuntimeError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "command": command,
            **saved_paths,
            "stdout_tail": _stdout_tail(stdout),
        }
    saved_paths = _persist_run_output(
        experiment_mode=experiment_mode,
        dataset=dataset,
        stdout=stdout,
        result=result,
    )
    STOP_REQUESTED = False
    return {
        "ok": True,
        "command": command,
        "developer_mode": developer_mode,
        "result": _summarize_result_for_frontend(result),
        **saved_paths,
        "stdout_tail": _stdout_tail(stdout),
    }


def options_payload() -> dict[str, Any]:
    return {
        "ok": True,
        "experiment_modes": [
            {
                "value": value,
                "label": config["label"],
                "script": config["script"],
                "datasets": sorted(config["datasets"]),
            }
            for value, config in EXPERIMENT_MODES.items()
        ],
        "datasets": [
            {
                "value": value,
                "label": config["label"],
                "default_limit": config["default_limit"],
                "preview_task_count": config["preview_task_count"],
                "script_dataset_type": config["script_dataset_type"],
                "dataset_file": config["dataset_file"],
            }
            for value, config in DATASETS.items()
        ],
    }


class CompetitionAgentsHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path in {"/health", "/api/health"}:
            _json_response(self, 200, {"ok": True, "service": "competition_agents_frontend_server"})
            return
        if self.path in {"/api/options", "/options"}:
            _json_response(self, 200, options_payload())
            return
        _json_response(self, 404, {"ok": False, "error": f"Not found: {self.path}"})

    def do_POST(self) -> None:
        if self.path in {"/api/stop", "/stop"}:
            response = stop_current_run()
            _json_response(self, 200, response)
            return
        if self.path not in {"/api/run", "/run"}:
            _json_response(self, 404, {"ok": False, "error": f"Not found: {self.path}"})
            return
        try:
            payload = _read_json_request(self)
            response = run_experiment(payload)
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            _json_response(
                self,
                504,
                {
                    "ok": False,
                    "error": "Experiment timed out.",
                    "stdout_tail": _stdout_tail(output),
                },
            )
            return
        except (ValueError, RuntimeError) as exc:
            _json_response(self, 400, {"ok": False, "error": str(exc)})
            return
        status = 200 if response.get("ok") else 502
        _json_response(self, status, response)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[frontend_server] {self.address_string()} - {format % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTTP API for competition agent dashboard runs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--echo-run-output",
        action="store_true",
        help="Print captured experiment stdout, including run_result_json, to this server terminal.",
    )
    return parser.parse_args()


def main() -> None:
    global ECHO_RUN_OUTPUT
    args = parse_args()
    ECHO_RUN_OUTPUT = args.echo_run_output
    server = ThreadingHTTPServer((args.host, args.port), CompetitionAgentsHandler)
    print(f"competition agents frontend server listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down frontend server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
