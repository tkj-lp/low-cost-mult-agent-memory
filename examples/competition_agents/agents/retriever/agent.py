from __future__ import annotations

from competition_agents.memory import search_memory
from competition_agents.protocol import make_packet
from competition_agents.state import CompetitionState, put_latent_state, record_message


def retriever_agent(state: CompetitionState) -> dict:
    """检索可复用记忆，并生成非文本状态引用。"""

    # 这里是后续接入超图记忆模块的主要接口。
    hits = search_memory(state["memory"], state["task"], top_k=2)
    evidence = [
        "LangGraph StateGraph can organize stateful agent nodes.",
        "Shared state can carry protocol packets, memory ids, and vector references.",
    ]
    # latent state 只存一次语义信息，后续 Agent 之间只传 state_id。
    latent_ref, latent_states, metrics = put_latent_state(
        state,
        producer="retriever",
        topic=state["task"],
        text=" ".join(evidence + [hit["summary"] for hit in hits]),
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
            + "baseline cost: "
            + " | ".join(evidence)
        )
    else:
        # protocol 模式通过 id 传证据和记忆引用，通信开销更低。
        payload = make_packet(
            sender="retriever",
            receiver="executor",
            action="execute",
            params={"tool": "codeact_stats", "evidence_count": len(evidence)},
            result={"evidence_refs": [latent_ref], "memory_hits": [hit["memory_id"] for hit in hits]},
            state_refs=[latent_ref],
            capability_required=["tool_execution"],
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
    }
