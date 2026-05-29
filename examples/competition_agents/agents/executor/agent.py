from __future__ import annotations

from typing import Any

from competition_agents.protocol import make_cognitive_gap, make_packet
from competition_agents.restricted_python_codeact import run_restricted_python
from competition_agents.state import CompetitionState, record_message


def _codeact_memory(memory: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """从共享记忆 trace 中提取 CodeAct 历史结果，供后续任务复用。"""

    indexed: dict[str, dict[str, Any]] = {}
    for record in memory:
        trace = record.get("trace", {})
        if not isinstance(trace, dict):
            continue
        task_meta = trace.get("task_meta", {})
        execution_result = trace.get("execution_result", {})
        task_id = task_meta.get("task_id") if isinstance(task_meta, dict) else None
        if not task_id or not isinstance(execution_result, dict):
            continue
        if execution_result.get("execution_mode") != "restricted_python_codeact":
            continue
        indexed[task_id] = {
            "task_id": task_id,
            "result": execution_result.get("result", {}),
            "sandbox_status": execution_result.get("sandbox_status", ""),
            "execution_mode": execution_result.get("execution_mode", ""),
        }
    return indexed


def _normalize_error(execution_result: dict[str, Any]) -> dict[str, str] | None:
    if execution_result.get("sandbox_status") == "ok":
        return None
    blocked_reason = execution_result.get("blocked_reason") or execution_result.get("stderr") or "execution failed"
    return {
        "type": "SandboxExecutionError",
        "message": str(blocked_reason),
    }


def executor_agent(state: CompetitionState) -> dict:
    """执行 ReasonerAgent 生成的受限 Python CodeAct 请求。"""

    request = state.get("execution_result", {}).get("execution_request")
    if not request:
        result = {
            "execution_mode": "restricted_python_codeact",
            "sandbox_status": "skipped",
            "blocked_reason": "missing execution_request",
            "result": {},
            "error": {"type": "MissingExecutionRequest", "message": "missing execution_request"},
        }
    else:
        result = run_restricted_python(
            request.get("code", ""),
            inputs=state.get("task_meta", {}).get("inputs", {}),
            memory=_codeact_memory(state.get("memory", [])),
            allowed_imports=request.get("allowed_imports", []),
            timeout_ms=request.get("timeout_ms", 3000),
        )
        result = {
            **result,
            "code_source": request.get("code_source", "reference_code"),
            "code": request.get("code", ""),
            "language": request.get("language", "python"),
            "return_value": result.get("result"),
            "error": _normalize_error(result),
        }

    cognitive_gap = make_cognitive_gap(
        agent="executor",
        known=["execution_request", "sandbox_policy", "codeact_memory"],
        unknown=[] if result.get("sandbox_status") == "ok" else ["successful_execution"],
        needed=["summary_generation", "memory_write"],
        gap_type="tool_execution_complete" if result.get("sandbox_status") == "ok" else "tool_execution_failed",
        confidence=0.88 if result.get("sandbox_status") == "ok" else 0.52,
    )

    if state["mode"] == "text":
        payload: str | dict[str, Any] = (
            "ExecutorAgent ran a restricted Python CodeAct request. "
            f"Sandbox status: {result.get('sandbox_status')}. "
            f"Result: {result.get('result')}. "
            f"Error: {result.get('error')}. "
            "The summarizer should store this structured execution_result in trace memory."
        )
    else:
        payload = make_packet(
            sender="executor",
            receiver="summarizer",
            action="summarize",
            content_type="protocol.execution_result",
            schema_ref="competition.protocol.execution_result.v0.2",
            params={
                "execution_mode": result.get("execution_mode"),
                "sandbox_status": result.get("sandbox_status"),
                "code_source": result.get("code_source", ""),
            },
            result={
                "result": result.get("result", {}),
                "error": result.get("error"),
                "safety_checks": result.get("safety_checks", {}),
            },
            capability_required=["summary_generation", "memory_write"],
            receiver_capability="summary_generation",
            cognitive_gap=cognitive_gap,
            confidence=0.88 if result.get("sandbox_status") == "ok" else 0.52,
        )

    messages, metrics = record_message(
        state,
        sender="executor",
        receiver="summarizer",
        action="summarize",
        payload=payload,
    )
    return {
        "execution_result": result,
        "messages": messages,
        "metrics": metrics,
        "agent_outputs": [
            *state.get("agent_outputs", []),
            {
                "agent": "ExecutorAgent",
                "raw_output": f"{result.get('sandbox_status')}:{result.get('result') or result.get('error')}",
            },
        ],
        "cognitive_gaps": [*state["cognitive_gaps"], cognitive_gap],
    }
