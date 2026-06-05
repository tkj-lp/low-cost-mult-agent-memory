from __future__ import annotations

import json

from competition_agents.llm import get_llm_client
from competition_agents.memory import add_memory
from competition_agents.protocol import make_cognitive_gap
from competition_agents.state import CompetitionState


def _summarizer_task_text(state: CompetitionState) -> str:
    """在 hypergraph 模式下只把问题文本传给 Summarizer，不直接暴露原始上下文。"""

    task_meta = state.get("task_meta", {})
    if str(task_meta.get("context_source") or "") == "hypergraph":
        question = str(task_meta.get("question") or "").strip()
        if question:
            return f"Question: {question}"
    return state["task"]


def summarizer_agent(state: CompetitionState) -> dict:
    """生成最终答案，并沉淀为可复用共享记忆。"""

    hit_count = len(state.get("memory_hits", []))
    latent_count = len(state.get("latent_states", {}))
    execution_result = state.get("execution_result", {})
    task_text = _summarizer_task_text(state)
    print(
        "[summarizer] "
        f"task chars={len(task_text)} memory_hits={hit_count} latent_states={latent_count} "
        f"memory_context chars={len(json.dumps(state.get('memory_context', []), ensure_ascii=False))}",
        flush=True,
    )
    if state["mode"] in {"protocol", "hidden"}:
        answer_ref = state.get("state_refs", {}).get("answer", "")
        answer_state = state.get("latent_states", {}).get(answer_ref, {})
        if answer_state:
            execution_result = {
                **execution_result,
                "latent_answer_ref": answer_ref,
                "latent_answer_summary": answer_state.get("summary", ""),
                "latent_embedding_provider": answer_state.get("embedding_provider", ""),
                "latent_embedding_model": answer_state.get("embedding_model", ""),
                "latent_model": answer_state.get("model", ""),
                "latent_layer": answer_state.get("layer", ""),
                "latent_shape": answer_state.get("shape", []),
            }
    # Summarizer 可单独接真实 API，其他 Agent 保持 mock 以便快速验证。
    summary = get_llm_client("summarizer").summarize(
        task=task_text,
        memory_hits=hit_count,
        latent_states=latent_count,
        plan=state.get("plan", []),
        evidence=state.get("evidence", []),
        execution_result=execution_result,
        cognitive_gaps=state.get("cognitive_gaps", []),
        mode=state["mode"],
    )
    # 这里是共享记忆写入路径，也是后续接超图存储的钩子。
    trace = {
        "task_meta": state.get("task_meta", {}),
        "plan": state.get("plan", []),
        "messages": state.get("messages", []),
        "evidence": state.get("evidence", []),
        "memory_hits": state.get("memory_hits", []),
        "memory_context": state.get("memory_context", []),
        "latent_states": state.get("latent_states", {}),
        "state_refs": state.get("state_refs", {}),
        "execution_result": state.get("execution_result", {}),
        "cognitive_gaps": state.get("cognitive_gaps", []),
        "agent_outputs": state.get("agent_outputs", []),
        "metrics": state.get("metrics", {}),
    }
    memory = add_memory(
        state["memory"],
        source_agent="summarizer",
        task_topic=task_text,
        summary=summary,
        tags=["langgraph", "multi-agent", "protocol", state["mode"]],
        trace=trace,
    )
    cognitive_gap = make_cognitive_gap(
        agent="summarizer",
        known=["plan", "evidence", "execution_result", "latent_state_refs"],
        unknown=[],
        needed=[],
        gap_type="closed",
        confidence=0.9,
    )
    return {
        "final_answer": summary,
        "memory": memory,
        "agent_outputs": [
            *state.get("agent_outputs", []),
            {"agent": "SummarizerAgent", "raw_output": summary},
        ],
        "cognitive_gaps": [*state["cognitive_gaps"], cognitive_gap],
    }
