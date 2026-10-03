from __future__ import annotations

import ast
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


DEFAULT_ALLOWED_IMPORTS = {
    "collections",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "itertools",
    "json",
    "math",
    "re",
    "statistics",
}

BLOCKED_IMPORTS = {
    "builtins",
    "ctypes",
    "http",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "threading",
    "urllib",
}

BLOCKED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "setattr",
    "vars",
}

BLOCKED_NODES = (
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
    ast.While,
    ast.With,
)


EXECUTION_SOURCE = r"""
import contextlib
import json
import traceback

class BoundedOutput:
    def __init__(self):
        self.parts, self.remaining = [], 4000
    def write(self, text):
        if self.remaining:
            self.parts.append(text[:self.remaining])
        self.remaining = max(0, self.remaining-len(text))
        return len(text)
    def flush(self):
        pass
    def value(self):
        return ''.join(self.parts)

def execute_request(request):
    code = request.get("code", "")
    inputs = request.get("inputs", {})
    memory = request.get("memory", {})
    namespace = {"inputs": inputs, "memory": memory}
    captured = BoundedOutput()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            exec(code, namespace, namespace)
        result = namespace.get("result", {})
        return {"ok": True, "result": result, "program_stdout": captured.value()}
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=3),
            "program_stdout": captured.value(),
        }
"""

WORKER_SOURCE = EXECUTION_SOURCE + r"""
while True:
    request = json.loads(input())
    print(json.dumps(execute_request(request), ensure_ascii=False, sort_keys=True), flush=True)
"""


class _PersistentCodeActWorker:
    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()
        self.responses: queue.Queue[str] = queue.Queue()
        self.reader: threading.Thread | None = None

    def _start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        self.process = subprocess.Popen(
            [sys.executable, "-u", "-c", WORKER_SOURCE],
            cwd=tempfile.gettempdir(),
            env=_safe_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader.start()

    def _read_stdout(self) -> None:
        process = self.process
        if not process or not process.stdout:
            return
        for line in process.stdout:
            self.responses.put(line)

    def kill(self) -> None:
        process = self.process
        self.process = None
        if process and process.poll() is None:
            process.kill()

    def run(
        self,
        *,
        code: str,
        inputs: dict[str, Any],
        memory: dict[str, Any],
        timeout_ms: int,
    ) -> dict[str, Any]:
        with self.lock:
            self._start()
            if not self.process or not self.process.stdin:
                raise RuntimeError("persistent worker did not start")
            while not self.responses.empty():
                self.responses.get_nowait()
            self.process.stdin.write(json.dumps({"code": code, "inputs": inputs, "memory": memory}, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            try:
                raw = self.responses.get(timeout=max(timeout_ms / 1000, 0.001))
            except queue.Empty as exc:
                self.kill()
                raise TimeoutError("persistent worker timeout") from exc
            return json.loads(raw)


_WORKER = _PersistentCodeActWorker()


def warm_persistent_codeact_worker() -> None:
    """Start the persistent sandbox worker before per-task timing begins."""

    worker_enabled = os.environ.get("CODEACT_PERSISTENT_WORKER", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    if not worker_enabled:
        return
    _WORKER._start()


def _name_of_call(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _root_module(name: str) -> str:
    return name.split(".", 1)[0]


def check_code_safety(code: str, *, allowed_imports: list[str] | None = None) -> dict[str, Any]:
    """用 AST 静态检查限制 LLM 生成代码的导入、调用和语法形态。"""

    allowed = set(DEFAULT_ALLOWED_IMPORTS if allowed_imports is None else allowed_imports)
    blocked_imports: list[str] = []
    blocked_calls: list[str] = []
    blocked_nodes: list[str] = []
    blocked_attributes: list[str] = []

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "passed": False,
            "blocked_reason": f"syntax_error: {exc.msg}",
            "blocked_imports": [],
            "blocked_calls": [],
            "blocked_nodes": [],
            "blocked_attributes": [],
        }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _root_module(alias.name)
                if root in BLOCKED_IMPORTS or root not in allowed:
                    blocked_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = _root_module(module)
            if root in BLOCKED_IMPORTS or root not in allowed:
                blocked_imports.append(module)
        elif isinstance(node, ast.Call):
            call_name = _name_of_call(node)
            if call_name in BLOCKED_CALLS:
                blocked_calls.append(call_name)
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                blocked_attributes.append(node.attr)
        elif isinstance(node, BLOCKED_NODES):
            blocked_nodes.append(type(node).__name__)

    blocked_reason = ""
    if blocked_imports:
        blocked_reason = f"blocked import: {sorted(set(blocked_imports))[0]}"
    elif blocked_calls:
        blocked_reason = f"blocked call: {sorted(set(blocked_calls))[0]}"
    elif blocked_nodes:
        blocked_reason = f"blocked syntax: {sorted(set(blocked_nodes))[0]}"
    elif blocked_attributes:
        blocked_reason = f"blocked attribute: {sorted(set(blocked_attributes))[0]}"

    return {
        "passed": not blocked_reason,
        "blocked_reason": blocked_reason,
        "blocked_imports": sorted(set(blocked_imports)),
        "blocked_calls": sorted(set(blocked_calls)),
        "blocked_nodes": sorted(set(blocked_nodes)),
        "blocked_attributes": sorted(set(blocked_attributes)),
    }


def _runner_source(code: str, *, inputs: dict[str, Any], memory: dict[str, Any]) -> str:
    request = json.dumps({'code': code, 'inputs': inputs, 'memory': memory}, ensure_ascii=False)
    return EXECUTION_SOURCE + f'\nprint(json.dumps(execute_request(json.loads({request!r})), ensure_ascii=False, sort_keys=True))\n'


def _safe_env() -> dict[str, str]:
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    if "SystemRoot" in os.environ:
        env["SystemRoot"] = os.environ["SystemRoot"]
    return env


def _run_subprocess_once(
    code: str,
    *,
    inputs: dict[str, Any],
    memory: dict[str, Any],
    timeout_ms: int,
    stdout_limit: int,
    stderr_limit: int,
) -> tuple[str, str, dict[str, Any], str, int]:
    with tempfile.TemporaryDirectory(prefix="codeact_") as temp_dir:
        runner = Path(temp_dir) / "runner.py"
        runner.write_text(
            _runner_source(code, inputs=inputs, memory=memory),
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(
                [sys.executable, str(runner)],
                cwd=temp_dir,
                env=_safe_env(),
                capture_output=True,
                text=True,
                timeout=max(timeout_ms / 1000, 0.001),
                check=False,
            )
            stdout = completed.stdout[:stdout_limit]
            stderr = completed.stderr[:stderr_limit]
            try:
                response = json.loads(completed.stdout.strip()) if completed.returncode == 0 else {}
            except json.JSONDecodeError:
                response = {}
            result = response.get('result', {})
            stdout = (response.get('program_stdout') or json.dumps(result, ensure_ascii=False))[:stdout_limit]
            status = 'ok' if completed.returncode == 0 and response.get('ok') else 'error'
            stderr = (stderr or response.get('error', ''))[:stderr_limit]
            exit_code = 0 if status == 'ok' else (completed.returncode or 1)
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or "")[:stdout_limit]
            stderr = "timeout"[:stderr_limit]
            result = {}
            status = "timeout"
            exit_code = 124
    return stdout, stderr, result, status, exit_code


def _run_persistent_worker(
    code: str,
    *,
    inputs: dict[str, Any],
    memory: dict[str, Any],
    timeout_ms: int,
    stdout_limit: int,
    stderr_limit: int,
) -> tuple[str, str, dict[str, Any], str, int]:
    response = _WORKER.run(code=code, inputs=inputs, memory=memory, timeout_ms=timeout_ms)
    stdout = (response.get('program_stdout') or json.dumps(response.get("result", {}), ensure_ascii=False, sort_keys=True))[:stdout_limit]
    if response.get("ok"):
        return stdout, "", response.get("result", {}), "ok", 0
    stderr = (response.get("error") or response.get("traceback") or "worker error")[:stderr_limit]
    return "", stderr, {}, "error", 1


def run_restricted_python(
    code: str,
    *,
    inputs: dict[str, Any] | None = None,
    memory: dict[str, Any] | None = None,
    allowed_imports: list[str] | None = None,
    timeout_ms: int = 3000,
    stdout_limit: int = 4000,
    stderr_limit: int = 4000,
) -> dict[str, Any]:
    """在临时目录子进程中运行通过安全检查的 Python CodeAct 代码。"""

    started = time.perf_counter()
    safety = check_code_safety(code, allowed_imports=allowed_imports)
    if not safety["passed"]:
        return {
            "execution_mode": "restricted_python_codeact",
            "sandbox_status": "blocked",
            "blocked_reason": safety["blocked_reason"],
            "allowed_imports": sorted(DEFAULT_ALLOWED_IMPORTS if allowed_imports is None else allowed_imports),
            "safety_checks": safety,
            "stdout": "",
            "stderr": "",
            "exit_code": 1,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "result": {},
        }

    worker_enabled = os.environ.get("CODEACT_PERSISTENT_WORKER", "1").lower() not in {"0", "false", "no"}
    runner = _run_persistent_worker if worker_enabled else _run_subprocess_once
    try:
        stdout, stderr, result, status, exit_code = runner(
            code,
            inputs=inputs or {},
            memory=memory or {},
            timeout_ms=timeout_ms,
            stdout_limit=stdout_limit,
            stderr_limit=stderr_limit,
        )
        execution_backend = "persistent_worker" if worker_enabled else "subprocess_once"
    except Exception as exc:
        _WORKER.kill()
        stdout, stderr, result, status, exit_code = _run_subprocess_once(
            code,
            inputs=inputs or {},
            memory=memory or {},
            timeout_ms=timeout_ms,
            stdout_limit=stdout_limit,
            stderr_limit=stderr_limit,
        )
        execution_backend = f"subprocess_fallback:{type(exc).__name__}"

    return {
        "execution_mode": "restricted_python_codeact",
        "execution_backend": execution_backend,
        "sandbox_status": status,
        "blocked_reason": "",
        "allowed_imports": sorted(DEFAULT_ALLOWED_IMPORTS if allowed_imports is None else allowed_imports),
        "safety_checks": safety,
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "result": result,
    }
