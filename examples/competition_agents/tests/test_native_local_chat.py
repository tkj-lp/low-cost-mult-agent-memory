import copy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from competition_agents.llm.native_local_chat import NativeLocalChat


class Tokenizer:
    def __init__(self, torch, response=None):
        self.torch, self.response = torch, response
        self.seen_messages = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True}
        self.seen_messages.append(copy.deepcopy(messages))
        return 'original-template:' + str(messages)

    def __call__(self, text, **kwargs):
        assert kwargs == {'return_tensors': 'pt', 'add_special_tokens': False}
        return {'input_ids': self.torch.tensor([[3, 4, 5]]),
                'attention_mask': self.torch.ones((1, 3), dtype=self.torch.long)}

    def decode(self, ids, **kwargs):
        assert kwargs == {'skip_special_tokens': True}
        return self.response if self.response is not None else str(ids.tolist())


def test_native_call_preserves_messages_and_invalid_raw_output_without_cache_or_repair(monkeypatch):
    import torch
    monkeypatch.setenv('LOCAL_LLM_COMPACT_TASK', '1')
    monkeypatch.setenv('LOCAL_LLM_MAX_INPUT_TOKENS', '1')
    raw = '```json\n{"task":"Unseen question?","lookup":"false","reuse":[]}\n```\n'
    tokenizer = Tokenizer(torch, raw)

    class Model:
        generation_config = SimpleNamespace(eos_token_id=[9])

        def __init__(self):
            self.received = []

        def generate(self, **kwargs):
            self.received.append(kwargs)
            assert set(kwargs) == {'input_ids', 'attention_mask', 'do_sample', 'max_new_tokens'}
            assert kwargs['do_sample'] is False and kwargs['max_new_tokens'] == 1024
            return torch.cat((kwargs['input_ids'], torch.tensor([[7, 9]])), dim=1)

    model = Model()
    client = NativeLocalChat(SimpleNamespace(model=model, tokenizer=tokenizer, torch=torch,
                                             model_name='test', device='cpu'))
    messages = [{'role': 'system', 'content': 'Unchanged protocol instructions'},
                {'role': 'user', 'content': '{"question":"Unseen question?"}'}]
    before = copy.deepcopy(messages)
    for role in ('planner', 'retriever', 'reasoner', 'summarizer'):
        assert client.call(role, messages)['content'] == raw
    assert messages == before
    assert tokenizer.seen_messages == [before] * 4
    assert all(call['messages'] == call['effective_messages'] == before for call in client.calls)
    assert all(call['normalizations'] == [] and call['cached_input_tokens'] == 0 for call in client.calls)
    assert all(call['input_tokens'] == 3 and call['output_tokens'] == 2 for call in client.calls)
    assert not hasattr(client, 'past') and not hasattr(client, 'delta')


def test_native_outputs_match_direct_standard_generation_and_model_stays_unchanged():
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(17)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, eos_token_id=2, pad_token_id=0)).eval()
    tokenizer = Tokenizer(torch)
    client = NativeLocalChat(SimpleNamespace(model=model, tokenizer=tokenizer, torch=torch,
                                             model_name='tiny', device='cpu'), max_tokens=5)
    messages = [{'role': 'user', 'content': 'A new fact'}]
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    configuration = copy.deepcopy(model.generation_config.to_dict())
    forward = model.forward.__func__
    inputs = tokenizer('', return_tensors='pt', add_special_tokens=False)
    with torch.inference_mode():
        expected = model.generate(**inputs, do_sample=False, max_new_tokens=5)[0, 3:]
    for _ in range(2):
        assert client.call('reasoner', messages)['content'] == str(expected.tolist())
    assert model.forward.__func__ is forward
    assert model.generation_config.to_dict() == configuration
    assert all(torch.equal(value, weights[name]) for name, value in model.state_dict().items())
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_native_module_has_no_experimental_imports_in_a_fresh_process():
    source = (
        'import sys; import competition_agents.methods.protocol; '
        'from competition_agents.llm.native_local_chat import NativeLocalChat; '
        'assert not any(k.startswith(("lmformatenforcer", "competition_agents.latent_hidden")) '
        'or k in ("competition_agents.llm.experimental_state_chat", "competition_agents.llm.local_hf", '
        '"competition_agents.llm.field_output_schema") for k in sys.modules)'
    )
    result = subprocess.run([sys.executable, '-c', source],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
