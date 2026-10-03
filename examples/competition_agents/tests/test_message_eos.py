import copy
from types import SimpleNamespace

import torch

from competition_agents.llm.generation import generation_eos_token_ids, standard_generate


def test_released_message_eos_is_recognized_when_generation_config_is_missing():
    bundle = SimpleNamespace(model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2)),
                             tokenizer=SimpleNamespace(eos_token_id=3))
    assert generation_eos_token_ids(bundle) == [2, 3]
    bundle.model.generation_config.eos_token_id = [2, 3]
    assert generation_eos_token_ids(bundle) == [2, 3]


def test_official_generation_stops_at_message_end_without_modifying_model_configuration():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(29)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        eos_token_id=2, pad_token_id=0)).eval()
    # 只允许生成消息结束词元。旧实现会继续生成至上限，因为模型配置只认另一种结束。
    model.generation_config.suppress_tokens = [i for i in range(16) if i != 3]
    configuration = copy.deepcopy(model.generation_config.to_dict())
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    tokenizer = SimpleNamespace(eos_token_id=3, pad_token_id=0,
        decode=lambda ids, **kwargs: str(ids.tolist()))
    bundle = SimpleNamespace(model=model, torch=torch, tokenizer=tokenizer)
    inputs = torch.tensor([[4, 5, 6]])
    result = standard_generate(bundle, inputs, max_tokens=8)
    assert result['output_token_ids'] == [3]
    assert result['finish_reason'] == 'stop'
    assert result['effective_eos_token_ids'] == [2, 3]
    assert model.generation_config.to_dict() == configuration
    assert all(torch.equal(value, weights[key]) for key, value in model.state_dict().items())
