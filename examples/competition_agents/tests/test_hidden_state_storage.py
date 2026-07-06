from __future__ import annotations

from types import SimpleNamespace

import torch

from competition_agents.latent_hidden import hidden_state


class _FakeTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return torch.arange(5, dtype=torch.long).reshape(1, 5)


class _FakeModel:
    def __call__(self, **kwargs):
        return SimpleNamespace(
            hidden_states=(torch.arange(35, dtype=torch.float32).reshape(1, 5, 7),),
        )


def test_extract_last_hidden_state_returns_pooled_tensor_by_default(monkeypatch) -> None:
    """默认只返回最后 token 的最后一层向量，避免把整段 hidden 搬回 CPU。"""

    hidden_state._extract_last_hidden_state_cached.cache_clear()
    monkeypatch.delenv("LOCAL_HIDDEN_STORAGE_MODE", raising=False)
    monkeypatch.setattr(
        hidden_state,
        "load_local_hidden_model",
        lambda: SimpleNamespace(
            model_name="fake-hidden-model",
            tokenizer=_FakeTokenizer(),
            model=_FakeModel(),
            torch=torch,
            device="cpu",
        ),
    )

    result, tensor = hidden_state.extract_last_hidden_state("question")

    assert result.shape == [1, 5, 7]
    assert hidden_state.stored_hidden_shape(result) == [7]
    assert hidden_state.stored_hidden_bytes(result) == 28
    assert list(tensor.shape) == [1, 7]
    assert result.pooled_vector == [float(value) for value in torch.arange(28, 35).tolist()]


def test_extract_last_hidden_state_can_return_full_tensor(monkeypatch) -> None:
    """显式 full 模式保留旧行为，方便需要完整 tensor 的实验复核。"""

    hidden_state._extract_last_hidden_state_cached.cache_clear()
    monkeypatch.setenv("LOCAL_HIDDEN_STORAGE_MODE", "full")
    monkeypatch.setattr(
        hidden_state,
        "load_local_hidden_model",
        lambda: SimpleNamespace(
            model_name="fake-hidden-model",
            tokenizer=_FakeTokenizer(),
            model=_FakeModel(),
            torch=torch,
            device="cpu",
        ),
    )

    result, tensor = hidden_state.extract_last_hidden_state("question")

    assert result.shape == [1, 5, 7]
    assert hidden_state.stored_hidden_shape(result) == [1, 5, 7]
    assert hidden_state.stored_hidden_bytes(result) == 140
    assert list(tensor.shape) == [1, 5, 7]


def test_extract_last_hidden_state_caches_pooled_results(monkeypatch) -> None:
    """相同文本的轻量 hidden 抽取应复用缓存，减少重复本地模型前向。"""

    hidden_state._extract_last_hidden_state_cached.cache_clear()
    calls = {"count": 0}

    class CountingModel:
        def __call__(self, **kwargs):
            calls["count"] += 1
            return SimpleNamespace(
                hidden_states=(torch.arange(35, dtype=torch.float32).reshape(1, 5, 7),),
            )

    monkeypatch.delenv("LOCAL_HIDDEN_STORAGE_MODE", raising=False)
    monkeypatch.setattr(
        hidden_state,
        "load_local_hidden_model",
        lambda: SimpleNamespace(
            model_name="fake-hidden-model",
            tokenizer=_FakeTokenizer(),
            model=CountingModel(),
            torch=torch,
            device="cpu",
        ),
    )

    first, _ = hidden_state.extract_last_hidden_state("same hidden input")
    second, _ = hidden_state.extract_last_hidden_state("same hidden input")

    assert calls["count"] == 1
    assert first.checksum == second.checksum
