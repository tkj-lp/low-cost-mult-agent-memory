from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from competition_agents.latent_hidden.hidden_state import HiddenStateResult


_HIDDEN_TENSOR_STORE: dict[str, object] = {}


def put_hidden_tensor(
    latent_states: dict[str, dict[str, Any]],
    *,
    producer: str,
    topic: str,
    result: HiddenStateResult,
    tensor: object,
    consumers: list[str] | None = None,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """把完整 hidden tensor 放到进程内存，消息里只传 state_id。"""

    state_id = f"hidden-{len(latent_states) + 1:04d}"
    _HIDDEN_TENSOR_STORE[state_id] = tensor
    latent_state = {
        "state_id": state_id,
        "type": "latent_hidden_state",
        "content_type": "latent.llm_hidden_state",
        "producer": producer,
        "consumers": consumers or [],
        "topic": topic,
        "model": result.model_name,
        "layer": result.layer,
        "shape": result.shape,
        "pooling": result.pooling,
        "pooled_vector": result.pooled_vector,
        "pooled_shape": [len(result.pooled_vector)],
        "token_count": result.token_count,
        "dtype": result.dtype,
        "device": result.device,
        "checksum": result.checksum,
        "storage": "process_memory",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return state_id, {**latent_states, state_id: latent_state}


def get_hidden_tensor(state_id: str) -> object | None:
    """根据 state_id 读取完整 hidden tensor。"""

    return _HIDDEN_TENSOR_STORE.get(state_id)
