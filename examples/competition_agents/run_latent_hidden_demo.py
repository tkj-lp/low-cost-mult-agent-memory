from __future__ import annotations

import json
import os
import time

from competition_agents.latent_hidden import extract_last_hidden_state, put_hidden_tensor
from competition_agents.latent_hidden.adapter import cosine_similarity


TASK = "Design a low-cost protocol for multi-agent shared memory and state transfer."


def main() -> None:
    """演示参考 LatentMAS 的本地 LLM hidden-state 传递。"""

    started = time.perf_counter()
    latent_states: dict[str, dict] = {}

    planner_prompt = (
        "PlannerAgent task:\n"
        f"{TASK}\n"
        "Create a short internal plan. Do not write a long final answer."
    )
    planner_hidden, planner_tensor = extract_last_hidden_state(planner_prompt)
    planner_ref, latent_states = put_hidden_tensor(
        latent_states,
        producer="planner",
        topic=TASK,
        result=planner_hidden,
        tensor=planner_tensor,
        consumers=["reasoner"],
    )

    reasoner_prompt = (
        "ReasonerAgent receives a latent state reference instead of a long plan.\n"
        f"latent_state_ref={planner_ref}\n"
        "Summarize how this reference can reduce communication overhead."
    )
    reasoner_hidden, reasoner_tensor = extract_last_hidden_state(reasoner_prompt)
    reasoner_ref, latent_states = put_hidden_tensor(
        latent_states,
        producer="reasoner",
        topic=TASK,
        result=reasoner_hidden,
        tensor=reasoner_tensor,
        consumers=["summarizer"],
    )

    similarity = cosine_similarity(
        latent_states[planner_ref]["pooled_vector"],
        latent_states[reasoner_ref]["pooled_vector"],
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    message = {
        "from": "planner",
        "to": "reasoner",
        "action": "reason_from_hidden_state",
        "payload": {
            "state_refs": [planner_ref],
            "content_type": "latent.llm_hidden_state",
            "note": "消息只传 state_ref，完整 hidden tensor 保存在进程内共享 latent memory。",
        },
    }
    serialized_message = json.dumps(message, ensure_ascii=False, separators=(",", ":"))

    print("=== hidden-state latent collaboration demo ===")
    print(f"LOCAL_HIDDEN_MODEL={os.environ.get('LOCAL_HIDDEN_MODEL', 'Qwen/Qwen2.5-1.5B-Instruct')}")
    print("message:")
    print(json.dumps(message, ensure_ascii=False, indent=2))
    print("latent states:")
    for state_id, state in latent_states.items():
        print(
            f"  {state_id}: producer={state['producer']}, "
            f"model={state['model']}, layer={state['layer']}, "
            f"shape={state['shape']}, pooled_shape={state['pooled_shape']}, "
            f"checksum={state['checksum']}"
        )
    print(
        "metrics: "
        f"messages=1, "
        f"text_chars={len(serialized_message)}, "
        f"hidden_state_refs=2, "
        f"planner_hidden_bytes={planner_hidden.shape[0] * planner_hidden.shape[1] * planner_hidden.shape[2] * 4}, "
        f"reasoner_hidden_bytes={reasoner_hidden.shape[0] * reasoner_hidden.shape[1] * reasoner_hidden.shape[2] * 4}, "
        f"pooled_similarity={similarity:.4f}, "
        f"elapsed_ms={elapsed_ms:.2f}"
    )


if __name__ == "__main__":
    main()
