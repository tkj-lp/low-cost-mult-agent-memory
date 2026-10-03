from __future__ import annotations

from competition_agents.restricted_python_codeact import run_restricted_python
import pytest


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


@pytest.mark.parametrize('persistent', ['0', '1'])
def test_print_is_captured_without_corrupting_executor_protocol(monkeypatch, persistent) -> None:
    monkeypatch.setenv('CODEACT_PERSISTENT_WORKER', persistent)
    result = run_restricted_python(
        "result = {'ok': True}\nprint('not a protocol response')\nprint(result)",
        allowed_imports=[],
    )

    assert result['sandbox_status'] == 'ok'
    assert result['result'] == {'ok': True}
    assert 'not a protocol response' in result['stdout']


def test_lambda_sorting_keeps_call_and_import_checks():
    result = run_restricted_python("result={'items':sorted(inputs['items'],key=lambda x:x[1])}",
                                  inputs={'items': [['a', 2], ['b', 1]]}, allowed_imports=[])
    assert result['sandbox_status'] == 'ok'
    assert result['result']['items'] == [['b', 1], ['a', 2]]
    blocked = run_restricted_python("f=lambda:open('secret'); result={'x':f()}", allowed_imports=[])
    assert blocked['sandbox_status'] == 'blocked'
