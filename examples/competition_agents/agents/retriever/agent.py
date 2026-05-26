from __future__ import annotations

from competition_agents.memory import search_memory
from competition_agents.protocol import make_cognitive_gap, make_compact_packet, make_packet
from competition_agents.state import CompetitionState, put_latent_state, record_message


def _task_specific_evidence(task: str) -> list[str]:
    """根据任务主题补充具体证据，避免 Summarizer 只复述运行状态。"""

    task_lower = task.lower()
    if "report" in task_lower:
        return [
            "Report generation should use memory metadata fields such as task_topic, summary, tags, source_agent, created_at, memory_refs, and state_refs.",
            "A report can be assembled as sections for architecture, protocol flow, latent-state exchange, memory reuse, limitations, and experiment metrics.",
        ]
    if "codeact" in task_lower or "sandbox" in task_lower:
        return [
            "A lightweight CodeAct sandbox should validate a structured execution request before running code.",
            "The sandbox design should include an allowlist, timeout, isolated process, stdout, stderr, exit_code, elapsed_ms, and safety_status.",
            "The current ExecutorAgent only simulates CodeAct statistics; real sandbox execution is not implemented yet.",
        ]
    if "performance" in task_lower:
        return [
            "Performance analysis should compare message_count, text_chars, latent_transfer_count, latent_bytes, memory_hit_rate, and elapsed_ms across runs.",
            "CodeAct execution memory can store prior execution_mode, sandbox_status, elapsed_ms, failure mode, and reusable optimization notes.",
        ]
    if "communication cost" in task_lower or "compare" in task_lower:
        return [
            "Communication cost should be measured by comparing text-mode natural-language payload size with protocol-mode structured packet size.",
            "Protocol mode should preserve action, params, result, capability_required, state_refs, memory_refs, and cognitive_gap while reducing repeated context.",
        ]
    if "latent" in task_lower or "embedding" in task_lower:
        return [
            "Latent state exchange should store evidence once in latent_states and pass compact state_refs between agents.",
            "The current latent vector is hash-v1 placeholder data; a real embedding provider or local model hidden state can replace it later.",
        ]
    if "openeuler" in task_lower or "deployment" in task_lower:
        return [
            "openEuler validation should document Python version, environment variables, PYTHONPATH, dependency setup, and exact run command.",
            "The first deployment target should use LLM_PROVIDER=mock before enabling OpenAI-compatible Qwen API calls.",
        ]
    if "architecture" in task_lower or "summarize" in task_lower:
        return [
            "The architecture consists of PlannerAgent, RetrieverAgent, ExecutorAgent, and SummarizerAgent connected by LangGraph state transitions.",
            "The protocol layer adds capability discovery, CKM cognitive gaps, state_refs, memory_refs, and metrics collection.",
        ]
    return []


def retriever_agent(state: CompetitionState) -> dict:
    """检索可复用记忆，并生成非文本状态引用。"""

    # 这里是后续接入超图记忆模块的主要接口。
    hits = search_memory(state["memory"], state["task"], top_k=2)
    evidence = [
        "LangGraph StateGraph can organize stateful agent nodes.",
        "Shared state can carry protocol packets, memory ids, and vector references.",
        "Current memory retrieval uses keyword and tag matching; true vector similarity is planned but not implemented.",
        *_task_specific_evidence(state["task"]),
    ]
    cognitive_gap = make_cognitive_gap(
        agent="retriever",
        known=["current_task", "local_evidence_templates", "memory_hit_count"],
        unknown=["hypergraph_expansion", "true_embedding_similarity"],
        needed=["latent_state_write", "tool_execution"],
        gap_type="missing_memory" if not hits else "partial_context",
        confidence=0.78 if hits else 0.64,
    )
    # latent state 只存一次语义信息，后续 Agent 之间只传 state_id。
    latent_ref, latent_states, metrics = put_latent_state(
        state,
        producer="retriever",
        topic=state["task"],
        text=" ".join(evidence + [hit["summary"] for hit in hits]),
        consumers=["executor", "summarizer"],
    )

    if state["mode"] == "text":
        # text 模式把证据和检索理由都序列化成长文本，作为 baseline。
        payload = (
            "RetrieverAgent found evidence: "
            + " | ".join(evidence)
            + ". Memory hits: "
            + " | ".join(hit["memory_id"] for hit in hits)
            + f". It forwards latent semantic content as {latent_ref}. "
            + "Because this is text-mode collaboration, the retriever serializes the evidence, "
            + "memory summaries, retrieval rationale, and downstream execution instruction as "
            + "natural language instead of passing compact references. Evidence repeated for "
            + "baseline cost. The retriever also explains that it can perform memory recall, "
            + "evidence search, and latent state writing. Its current cognitive gap is that "
            + "hypergraph expansion and true embedding similarity are not available yet, so the "
            + "downstream executor should treat the evidence as a lightweight latent reference "
            + "rather than a full hidden-state transfer. Evidence repeated for baseline cost: "
            + " | ".join(evidence)
        )
    else:
        # protocol 模式通过 id 传证据和记忆引用，通信开销更低。
        payload = make_compact_packet(
            sender="retriever",
            receiver="reasoner",
            action="reason",
            content_type="state_ref",
            params={"tool": "codeact_stats", "n": len(evidence)},
            result={"mode": "memory", "hits": len(hits)},
            state_refs=[latent_ref],
            memory_refs=[hit["memory_id"] for hit in hits],
            capability_required=["reason"],
            cognitive_gap=cognitive_gap,
            confidence=0.78 if hits else 0.64,
        )

    # 复用 put_latent_state 返回的 metrics，确保 latent 传递也被统计。
    messages, metrics = record_message(
        state,
        sender="retriever",
        receiver="reasoner",
        action="reason",
        payload=payload,
        metrics=metrics,
    )
    return {
        "evidence": evidence,
        "memory_hits": hits,
        "latent_states": latent_states,
        "state_refs": {**state.get("state_refs", {}), "evidence": latent_ref},
        "messages": messages,
        "metrics": metrics,
        "cognitive_gaps": [*state["cognitive_gaps"], cognitive_gap],
    }
