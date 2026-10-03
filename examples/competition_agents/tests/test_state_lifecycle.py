"""核验混合缓存回收、来源保留及历史回答的显式期限。"""
from types import SimpleNamespace

import torch
from transformers.cache_utils import DynamicLayer, LinearAttentionLayer

from competition_agents.latent_hidden.state_lifecycle import StateLifecycle
from competition_agents.llm.latent_only_chat import ROLES
from competition_agents.field_collaboration import FIELD_ROLES
from competition_agents.plain_text_collaboration import FINAL_ANSWER_REQUIREMENTS, TEXT_ROLES


def cache_of_length(length):
    attention = DynamicLayer()
    keys = torch.arange(length * 4, dtype=torch.float32).reshape(1, 1, length, 4)
    attention.update(keys, keys + 100)
    recurrent = LinearAttentionLayer()
    recurrent.update_conv_state(torch.arange(16, dtype=torch.float32).reshape(1, 4, 4))
    recurrent.update_recurrent_state(torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4))
    return SimpleNamespace(layers=[recurrent, attention], get_seq_length=attention.get_seq_length)


def test_mixed_cache_removes_only_expired_rows_and_preserves_recurrent_state():
    lifecycle = StateLifecycle()
    lifecycle.turn = 1
    lifecycle.append(3, 'external_facts', 'planner')
    lifecycle.append(2, 'role_instruction', 'planner')
    lifecycle.append(4, 'question', 'planner')
    cache = cache_of_length(9)
    recurrent, attention = cache.layers
    keys, values = attention.keys.clone(), attention.values.clone()
    conv, state = recurrent.conv_states.clone(), recurrent.recurrent_states.clone()
    pointers = (recurrent.conv_states.data_ptr(), recurrent.recurrent_states.data_ptr())
    assert lifecycle.reclaim(cache, 'planner', torch)['removed_positions'] == 0
    report = lifecycle.reclaim(cache, 'retriever', torch)
    keep = torch.tensor([0, 1, 2, 5, 6, 7, 8])
    assert torch.equal(attention.keys, keys.index_select(-2, keep))
    assert torch.equal(attention.values, values.index_select(-2, keep))
    assert torch.equal(recurrent.conv_states, conv)
    assert torch.equal(recurrent.recurrent_states, state)
    assert pointers == (recurrent.conv_states.data_ptr(), recurrent.recurrent_states.data_ptr())
    assert report['removed_by_kind'] == {'role_instruction': 2}
    assert report['after_bytes'] < report['before_bytes'] <= report['peak_bytes']
    assert lifecycle.next_position == 9
    assert sum(b.length for b in lifecycle.blocks) == cache.get_seq_length() == 7


def test_previous_answer_survives_next_turn_then_expires_without_erasing_sources():
    lifecycle = StateLifecycle()
    lifecycle.turn = 1
    lifecycle.append(4, 'external_facts', 'retriever')
    lifecycle.append(3, 'answer', 'summarizer')
    cache = cache_of_length(7)
    lifecycle.turn = 2
    assert lifecycle.reclaim(cache, 'planner', torch)['removed_positions'] == 0
    lifecycle.turn = 3
    report = lifecycle.reclaim(cache, 'planner', torch)
    assert report['removed_by_kind'] == {'answer': 3}
    assert [(b.kind, b.length) for b in lifecycle.blocks] == [('external_facts', 4)]
    assert cache.get_seq_length() == 4
    assert lifecycle.next_position == 7


def test_unknown_layer_layout_is_rejected_before_mutating_any_attention_layer():
    lifecycle = StateLifecycle()
    lifecycle.turn = 1
    lifecycle.append(3, 'external_facts', 'planner')
    lifecycle.append(2, 'role_instruction', 'planner')
    cache = cache_of_length(5)
    cache.layers.append(SimpleNamespace(keys=torch.ones(1), values=torch.ones(1)))
    original = cache.layers[1].keys
    report = lifecycle.reclaim(cache, 'retriever', torch)
    assert report['skipped'] == 'unsupported_cache_layout_keep_full_state'
    assert cache.layers[1].keys is original
    assert sum(b.length for b in lifecycle.blocks) == 5


def test_all_three_methods_share_exact_final_answer_requirements():
    assert TEXT_ROLES['summarizer'] == FIELD_ROLES['summarizer'] == ROLES['summarizer'] == FINAL_ANSWER_REQUIREMENTS
