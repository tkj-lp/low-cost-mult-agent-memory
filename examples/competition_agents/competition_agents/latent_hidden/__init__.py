"""本地 LLM hidden state 传递实验模块。"""

from competition_agents.latent_hidden.hidden_state import (
    extract_last_hidden_state,
    generate_text_with_last_hidden_state,
    hidden_storage_mode,
    should_store_full_hidden_tensor,
    stored_hidden_bytes,
    stored_hidden_shape,
)
from competition_agents.latent_hidden.latent_memory import (
    get_hidden_tensor,
    put_hidden_tensor,
)

__all__ = [
    "extract_last_hidden_state",
    "generate_text_with_last_hidden_state",
    "hidden_storage_mode",
    "get_hidden_tensor",
    "put_hidden_tensor",
    "should_store_full_hidden_tensor",
    "stored_hidden_bytes",
    "stored_hidden_shape",
]
