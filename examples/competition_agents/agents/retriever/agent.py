from __future__ import annotations

from competition_agents.memory import search_memory
from competition_agents.protocol import make_cognitive_gap, make_packet
from competition_agents.state import CompetitionState, put_latent_state, record_message


def retriever_agent(state: CompetitionState) -> dict:
    """检索可复用记忆，并生成非文本状态引用。"""

    # 这里是后续接入超图记忆模块的主要接口。
    hits = search_memory(state["memory"], state["task"], top_k=2)
    evidence = [
        "LangGraph StateGraph can organize stateful agent nodes.",
        "Shared state can carry protocol packets, memory ids, and vector references.",
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
        payload = make_packet(
            sender="retriever",
            receiver="executor",
            action="execute",
            content_type="protocol.state_ref",
            schema_ref="competition.protocol.state_ref.v0.2",
            params={"tool": "codeact_stats", "evidence_count": len(evidence)},
            result={"evidence_refs": [latent_ref], "memory_hits": [hit["memory_id"] for hit in hits]},
            state_refs=[latent_ref],
            memory_refs=[hit["memory_id"] for hit in hits],
            capability_required=["tool_execution"],
            receiver_capability="tool_execution",
            cognitive_gap=cognitive_gap,
            confidence=0.78 if hits else 0.64,
        )

    # 复用 put_latent_state 返回的 metrics，确保 latent 传递也被统计。
    messages, metrics = record_message(
        state,
        sender="retriever",
        receiver="executor",
        action="execute",
        payload=payload,
        metrics=metrics,
    )
    return {
        "evidence": evidence,
        "memory_hits": hits,
        "latent_states": latent_states,
        "messages": messages,
        "metrics": metrics,
        "cognitive_gaps": [*state["cognitive_gaps"], cognitive_gap],
    }
