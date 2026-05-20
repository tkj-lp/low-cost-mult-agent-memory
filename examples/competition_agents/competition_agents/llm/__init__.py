from .client import get_llm_client

# 统一导出模型选择入口，后续可扩展 mock/API/local。
__all__ = ["get_llm_client"]
