from types import SimpleNamespace

import torch

from competition_agents.llm.experimental_state_chat import cache_bytes
from competition_agents.llm.native_local_chat import local_model_source, state_message, text_decoder


def test_deployment_manifest_keeps_fixed_model_identity(tmp_path, monkeypatch):
    import json
    model = tmp_path / 'weights'
    model.mkdir()
    manifest = tmp_path / 'models.json'
    manifest.write_text(json.dumps({'small': {
        'repo': 'Qwen/example', 'path': str(model), 'complete': True}}))
    monkeypatch.setenv('COMPETITION_MODEL_MANIFEST', str(manifest))
    assert local_model_source('Qwen/example') == str(model)
    assert local_model_source('another-model') == 'another-model'


def test_mixed_cache_counts_full_and_recurrent_state_without_double_counting():
    keys = torch.zeros(1, 2, 5, 4)
    values = torch.zeros_like(keys)
    conv = torch.zeros(1, 16, 4, dtype=torch.bfloat16)
    recurrent = torch.zeros(1, 2, 8, 8)
    cache = SimpleNamespace(layers=[
        SimpleNamespace(keys=keys, values=values),
        SimpleNamespace(conv_states=conv, recurrent_states=recurrent),
        SimpleNamespace(conv_states=conv),
    ])
    assert cache_bytes(cache) == sum(t.numel() * t.element_size() for t in (keys, values, conv, recurrent))


def test_strict_official_template_fragment_preserves_instructions_without_a_fake_query():
    class StrictTemplate:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
            assert any(m['role'] == 'user' for m in messages)
            assert enable_thinking is False
            text = ''.join('<' + m['role'] + '>' + m['content'] + '</end>\n' for m in messages)
            return text + ('<assistant><think>\n\n</think>\n\n' if add_generation_prompt else '')
    bundle = SimpleNamespace(tokenizer=StrictTemplate(), state_template_requires_user=True,
                             template_kwargs={'enable_thinking': False})
    messages = [{'role': 'system', 'content': 'Keep all returned source relations.'}]
    assert state_message(bundle, messages, open_assistant=False) == (
        '<system>Keep all returned source relations.</end>\n')
    fragment = state_message(bundle, messages, open_assistant=True)
    assert fragment == '<system>Keep all returned source relations.</end>\n<assistant><think>\n\n</think>\n\n'
    assert '__COMPETITION_TEMPLATE_RENDER_ONLY__' not in fragment


def test_official_hybrid_decoder_continues_every_layer_without_replaying_prefix():
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from competition_agents.llm.latent_only_chat import LatentOnlyChat
    from test_latent_only import bundle_with_tokenizer

    torch.manual_seed(13)
    config = Qwen3_5TextConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=2,
        layer_types=['linear_attention', 'full_attention'],
        eos_token_id=2, pad_token_id=0, partial_rotary_factor=1.0)
    model = Qwen3_5ForCausalLM(config).eval()
    bundle = bundle_with_tokenizer()
    # 模拟官方多模态外壳，语言解码器与权重仍是官方实现。
    bundle.model = SimpleNamespace(
        model=SimpleNamespace(language_model=model.model), lm_head=model.lm_head,
        generation_config=model.generation_config,
        get_input_embeddings=model.get_input_embeddings)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    chat = LatentOnlyChat(bundle, latent_steps=0, feedback_mode='cache_only',
                          cache_policy='consumer_lifetime')
    chat.call('planner', question='Where does the person live?', scope='one')
    chat.call('retriever')
    chat.call('reasoner')
    assert text_decoder(bundle) is model.model
    assert all(c['layer_positions'] == [c['forward_positions']] * 2 for c in chat.calls)
    assert all(c['cached_prefix_positions'] > 0 for c in chat.calls[1:])
    assert chat.calls[-1]['lifetime']['removed_positions'] > 0
    assert chat.calls[-1]['lifetime']['compacted_attention_layers'] == 1
    assert chat.calls[-1]['lifetime']['preserved_recurrent_layers'] == 1
    assert sum(b.length for b in chat.lifecycle.blocks) == chat.past.get_seq_length()
    assert chat.past.layers[0].recurrent_states is not None
    assert chat.state_bytes() > sum(t.numel() * t.element_size()
                                  for t in (chat.past.layers[1].keys, chat.past.layers[1].values))
    assert bundle.tokenizer.decode_calls == 0
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
