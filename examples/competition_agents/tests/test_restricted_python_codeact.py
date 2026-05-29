from __future__ import annotations

from competition_agents.restricted_python_codeact import run_restricted_python


def test_restricted_python_runs_reference_code() -> None:
    result = run_restricted_python(
        "values = inputs['values']\nresult = {'sum': sum(values), 'max': max(values)}",
        inputs={"values": [3, 5, 8]},
        allowed_imports=[],
    )

    assert result["sandbox_status"] == "ok"
    assert result["result"] == {"sum": 16, "max": 8}


def test_restricted_python_blocks_dangerous_import() -> None:
    result = run_restricted_python(
        "import os\nresult = {'cwd': os.getcwd()}",
        allowed_imports=["os"],
    )

    assert result["sandbox_status"] == "blocked"
    assert result["blocked_reason"] == "blocked import: os"


def test_restricted_python_blocks_open_call() -> None:
    result = run_restricted_python(
        "handle = open('x.txt', 'w')\nresult = {'ok': True}",
        allowed_imports=[],
    )

    assert result["sandbox_status"] == "blocked"
    assert result["blocked_reason"] == "blocked call: open"


def test_restricted_python_blocks_while_loop() -> None:
    result = run_restricted_python(
        "while True:\n    pass\nresult = {'ok': True}",
        allowed_imports=[],
    )

    assert result["sandbox_status"] == "blocked"
    assert result["blocked_reason"] == "blocked syntax: While"


def test_restricted_python_blocks_print_call() -> None:
    result = run_restricted_python(
        "result = {'ok': True}\nprint(result)",
        allowed_imports=[],
    )

    assert result["sandbox_status"] == "blocked"
    assert result["blocked_reason"] == "blocked call: print"
