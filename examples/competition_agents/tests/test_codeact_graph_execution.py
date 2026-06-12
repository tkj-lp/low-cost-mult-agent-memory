from __future__ import annotations

from competition_agents.graph import build_graph
from competition_agents.state import initial_state


def _invoke_codeact(reference_code: str, *, allowed_imports: list[str] | None = None) -> dict:
    graph = build_graph()
    state = initial_state(
        "CodeAct task: calculate a deterministic result.",
        "protocol",
        task_meta={
            "task_id": "codeact-test-001",
            "dataset_type": "codeact",
            "inputs": {"values": [2, 3, 5]},
            "reference_code": reference_code,
            "allowed_imports": allowed_imports or [],
        },
    )
    return graph.invoke(state)


def test_codeact_task_enters_executor_agent() -> None:
    """CodeAct 任务应经过 Reasoner -> Executor -> Summarizer。"""

    result = _invoke_codeact("values = inputs['values']\nresult = {'sum': sum(values)}")

    assert any(message["from"] == "reasoner" and message["to"] == "executor" for message in result["messages"])
    assert any(message["from"] == "executor" and message["to"] == "summarizer" for message in result["messages"])


def test_regular_task_skips_executor_agent() -> None:
    """普通任务没有 execution_request，不应误进入 ExecutorAgent。"""

    graph = build_graph()
    result = graph.invoke(initial_state("Summarize the protocol architecture.", "protocol"))

    assert not any(message["from"] == "executor" for message in result["messages"])
    assert result["execution_result"]["execution_mode"] == "model_reasoning"


def test_executor_outputs_restricted_python_result() -> None:
    """ExecutorAgent 应输出真实 restricted_python_codeact 结果。"""

    result = _invoke_codeact("values = inputs['values']\nresult = {'sum': sum(values), 'max': max(values)}")

    execution_result = result["execution_result"]
    assert execution_result["execution_mode"] == "restricted_python_codeact"
    assert execution_result["sandbox_status"] == "ok"
    assert execution_result["result"] == {"sum": 10, "max": 5}
    assert execution_result["return_value"] == {"sum": 10, "max": 5}


def test_execution_result_is_written_to_memory_trace() -> None:
    """Summarizer 写入的 memory trace 应保留真实 execution_result。"""

    result = _invoke_codeact("values = inputs['values']\nresult = {'sum': sum(values)}")

    evidence_memory = next(record for record in result["memory"] if "trace" in record)
    trace_execution = evidence_memory["trace"]["execution_result"]
    assert trace_execution["execution_mode"] == "restricted_python_codeact"
    assert trace_execution["sandbox_status"] == "ok"
    assert trace_execution["result"] == {"sum": 10}


def test_summarizer_writes_evidence_and_experience_memory() -> None:
    """当前主图在任务结束后应同时写 evidence memory 和 experience memory。"""

    result = _invoke_codeact("values = inputs['values']\nresult = {'sum': sum(values)}")

    memory_types = [record.get("memory_type", "evidence") for record in result["memory"]]
    assert "evidence" in memory_types
    assert "experience" in memory_types
    assert result["memory"][-1]["memory_type"] == "experience"


def test_blocked_codeact_graph_task_does_not_crash() -> None:
    """主图中的危险 CodeAct 代码应被拦截，并以结构化错误进入 trace。"""

    result = _invoke_codeact("import os\nresult = {'cwd': os.getcwd()}", allowed_imports=["os"])

    execution_result = result["execution_result"]
    assert execution_result["execution_mode"] == "restricted_python_codeact"
    assert execution_result["sandbox_status"] == "blocked"
    assert execution_result["blocked_reason"] == "blocked import: os"
    assert execution_result["error"]["type"] == "SandboxExecutionError"
    assert result["final_answer"]
    evidence_memory = next(record for record in result["memory"] if "trace" in record)
    assert evidence_memory["trace"]["execution_result"]["sandbox_status"] == "blocked"
