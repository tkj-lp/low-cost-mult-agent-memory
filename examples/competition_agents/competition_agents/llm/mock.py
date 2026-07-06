from __future__ import annotations

from typing import Any


class MockLLMClient:
    """用于演示和队友开发的确定性本地模型替身。"""

    def generate(self, prompt: str) -> str:
        # 返回可预测结果，保证实验输出稳定。
        return f"Mock model response for: {prompt[:120]}"

    def plan(self, task: str) -> list[str]:
        # mock 规划保持简短稳定，方便复现实验指标。
        return [
            "identify reusable memory",
            "retrieve supporting evidence",
            "execute a tool step",
            "summarize and store memory",
        ]

    def plan_with_raw(self, task: str) -> tuple[list[str], str]:
        plan = self.plan(task)
        return plan, "\n".join(plan)

    def reason(
        self,
        *,
        task: str,
        plan: list[str],
        evidence: list[str],
        memory_hits: int,
        mode: str,
        memory_hit_details: list[dict[str, Any]] | None = None,
        experience_rules: list[dict[str, Any]] | None = None,
        answer_rules: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "answer_candidate": (
                "Mock ReasonerAgent placeholder. Configure LLM_REASONER_PROVIDER=openai_compatible "
                "to get a real model-generated answer."
            ),
            "reasoning_summary": (
                "Mock reasoner did not perform real model reasoning. "
                f"Received {len(memory_hit_details or [])} memory hit detail records and "
                f"{len(experience_rules or [])} experience rules."
            ),
            "limitations": ["mock_reasoner"],
            "raw_model_output": "",
        }

    def summarize(
        self,
        *,
        task: str,
        memory_hits: int,
        latent_states: int,
        plan: list[str] | None = None,
        evidence: list[str] | None = None,
        execution_result: dict[str, Any] | None = None,
        cognitive_gaps: list[dict[str, Any]] | None = None,
        mode: str | None = None,
    ) -> str:
        # mock 总结要先回答任务本身，运行指标只作为最后的诊断信息。
        plan_text = " -> ".join(plan or self.plan(task))
        evidence_text = "; ".join(evidence or ["LangGraph can connect agent nodes through shared state."])
        gap_types = sorted({gap.get("gap_type", "unknown") for gap in cognitive_gaps or []})
        gap_text = ", ".join(gap_types) if gap_types else "none"
        execution = execution_result or {}
        answer_candidate = str(execution.get("answer_candidate") or "").strip()
        task_lower = task.lower()
        evidence_text = evidence_text.rstrip(".")

        if answer_candidate:
            task_answer = answer_candidate
        elif "performance" in task_lower:
            task_answer = (
                "CodeAct execution memory can be reused for performance analysis by storing structured execution "
                "metadata from previous runs, including execution_mode, sandbox_status, elapsed_ms, message_count, "
                "latent_state_count, memory_hit_count, exit_code, and failure category. A later analysis task can "
                "retrieve these records, compare protocol mode with text mode, identify repeated bottlenecks, and "
                "summarize whether communication overhead, sandbox execution, or retrieval quality dominates the "
                "runtime. In the current prototype, CodeAct execution is simulated, so the analysis should be treated "
                "as a metrics-pipeline demonstration rather than a real code-runtime benchmark."
            )
        elif "codeact" in task_lower or "sandbox" in task_lower:
            task_answer = (
                "A lightweight CodeAct sandbox should be designed around a strict execution contract. "
                "The ExecutorAgent receives a structured request containing the code intent, allowed tool type, "
                "inputs, timeout, and expected output schema. Before execution, the sandbox validates the request "
                "against an allowlist of safe Python operations and blocks filesystem deletion, network access, "
                "shell commands, and unbounded loops. The sandbox then runs the code in an isolated process with "
                "a timeout and returns stdout, stderr, exit_code, elapsed_ms, and safety_status. In the current "
                "prototype this is still simulated by a deterministic CodeAct statistics step; the real sandbox "
                "is a planned extension."
            )
        elif "communication cost" in task_lower or "compare" in task_lower:
            task_answer = (
                "The communication-cost experiment should run the same task sequence in text mode and protocol "
                "mode, then compare message_count, text_chars, latent_transfer_count, latent_bytes, elapsed_ms, "
                "and memory_hit_rate. Text mode serializes repeated natural-language context, while protocol mode "
                "sends compact action, parameter, result, capability, memory, and state reference fields. The key "
                "claim is not that protocol mode changes the task result, but that it preserves the same workflow "
                "while reducing repeated context transfer."
            )
        elif "latent" in task_lower or "embedding" in task_lower:
            task_answer = (
                "The latent-state exchange design stores semantic evidence once as a latent_state_ref and passes "
                "only state_refs between agents. Each latent state records producer, consumers, content_type, "
                "embedding_provider, vector shape, checksum, scope, and ttl. Downstream agents use the reference "
                "to recover or inspect the non-text representation without receiving the full evidence text again. "
                "The current vector is a hash-based placeholder; a real embedding provider can replace it later "
                "without changing the protocol fields."
            )
        elif "memory metadata" in task_lower or "shared memory" in task_lower:
            task_answer = (
                "The shared-memory design stores each reusable result as a MemoryUnit with memory_id, source_agent, "
                "created_at, task_topic, summary, tags, and embedding. Protocol messages carry memory_refs so future "
                "tasks can reuse prior summaries without copying the full memory body through every agent message. "
                "The MemoryUnit schema is intentionally stable so a later HyperGraphMemory implementation can replace "
                "the in-memory list behind add_memory and search_memory."
            )
        elif "openeuler" in task_lower or "deployment" in task_lower:
            task_answer = (
                "The openEuler deployment plan should keep the demo dependency-light, run first with LLM_PROVIDER=mock, "
                "set PYTHONPATH to include libs/langgraph and examples/competition_agents, and record the exact Python "
                "version, environment variables, command line, and protocol/text comparison output for reproducibility."
            )
        elif "architecture" in task_lower or "summarize" in task_lower:
            task_answer = (
                "The complete architecture is a LangGraph-style pipeline with PlannerAgent for decomposition, "
                "RetrieverAgent for evidence and latent-state creation, ExecutorAgent for tool or CodeAct-style "
                "execution, and SummarizerAgent for final answer generation and memory writing. The protocol layer "
                "adds capability discovery, CKM cognitive gaps, memory_refs, and state_refs. The memory layer is "
                "currently a simple replaceable module, while the protocol and latent-state interfaces are already "
                "stable enough for a future hypergraph memory backend."
            )
        else:
            task_answer = (
                "The proposed result is a LangGraph multi-agent protocol where PlannerAgent decomposes the task, "
                "RetrieverAgent gathers evidence and creates latent state references, ExecutorAgent performs a "
                "tool-style step, and SummarizerAgent produces the final answer and writes reusable memory."
            )

        return (
            f"Answer:\n{task_answer}\n\n"
            "Implementation notes:\n"
            f"- Plan used by the agents: {plan_text}.\n"
            f"- Evidence available in this run: {evidence_text}.\n\n"
            "Run diagnostics:\n"
            f"- mode={mode or 'unknown'}\n"
            f"- memory_hits={memory_hits}\n"
            f"- latent_states={latent_states}\n"
            f"- messages_so_far={execution.get('message_count_so_far', 'unknown')}\n"
            f"- cognitive_gap_types={gap_text}"
        )
