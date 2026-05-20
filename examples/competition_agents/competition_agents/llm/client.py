from __future__ import annotations

import os

from .base import LLMClient
from .mock import MockLLMClient
from .openai_compatible import OpenAICompatibleClient


def get_llm_client(role: str | None = None) -> LLMClient:
    """返回当前配置的模型客户端。

    默认使用 mock，保证没有 API key 时仓库也能运行。
    """

    # 按角色配置 provider，可以在快速验证时减少 API 调用。
    role_key = f"LLM_{role.upper()}_PROVIDER" if role else ""
    provider = os.environ.get(role_key) or os.environ.get("LLM_PROVIDER", "mock")
    provider = provider.lower()
    if provider in {"openai", "openai_compatible", "api"}:
        # openai_compatible 也覆盖阿里云 DashScope 兼容模式接口。
        return OpenAICompatibleClient()
    if provider == "mock":
        # mock 不依赖网络，便于演示和可复现实验。
        return MockLLMClient()
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
