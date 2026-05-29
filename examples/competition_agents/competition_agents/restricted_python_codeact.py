from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
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
    "print",
    "quit",
    "setattr",
    "vars",
}

BLOCKED_NODES = (
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Lambda,
    ast.Nonlocal,
    ast.While,
    ast.With,
)


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
    inputs_json = json.dumps(inputs, ensure_ascii=False)
    memory_json = json.dumps(memory, ensure_ascii=False)
    return (
        "import json\n"
        f"inputs = json.loads({inputs_json!r})\n"
        f"memory = json.loads({memory_json!r})\n"
        f"{code}\n"
        "print(json.dumps(result, ensure_ascii=False, sort_keys=True))\n"
    )


def _safe_env() -> dict[str, str]:
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    if "SystemRoot" in os.environ:
        env["SystemRoot"] = os.environ["SystemRoot"]
    return env


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

    with tempfile.TemporaryDirectory(prefix="codeact_") as temp_dir:
        runner = Path(temp_dir) / "runner.py"
        runner.write_text(
            _runner_source(code, inputs=inputs or {}, memory=memory or {}),
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
                result = json.loads(stdout.strip()) if completed.returncode == 0 else {}
            except json.JSONDecodeError:
                result = {}
            status = "ok" if completed.returncode == 0 and result != {} else "error"
            exit_code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or "")[:stdout_limit]
            stderr = "timeout"[:stderr_limit]
            result = {}
            status = "timeout"
            exit_code = 124

    return {
        "execution_mode": "restricted_python_codeact",
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
