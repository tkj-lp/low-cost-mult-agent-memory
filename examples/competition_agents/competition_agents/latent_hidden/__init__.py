"""本地 LLM hidden state 传递实验模块。"""

from competition_agents.latent_hidden.hidden_state import extract_last_hidden_state
from competition_agents.latent_hidden.latent_memory import (
    get_hidden_tensor,
    put_hidden_tensor,
)

__all__ = ["extract_last_hidden_state", "get_hidden_tensor", "put_hidden_tensor"]
