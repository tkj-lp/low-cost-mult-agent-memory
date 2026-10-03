"""验证删除边界和绝对位置；小型随机网络不作为任务准确率或速度数据。"""

import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from role_private_cache_probe import PrivateInstructionManager, RolePrivateCache, history_metadata_spans
from competition_agents.llm.native_local_chat import text_decoder
from runtime_fixtures import cache_bundle, scripted_logits


@pytest.mark.parametrize('architecture', ['attention', 'hybrid', 'multimodal'])
def test_only_marked_rows_are_removed_and_absolute_positions_match_full_masked_cache(architecture):
    bundle = cache_bundle(architecture)
    decoder = text_decoder(bundle)
    initial = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    suffix = torch.tensor([[11, 12]])
    with torch.inference_mode():
        full = decoder(input_ids=initial, use_cache=True).past_key_values
        preserved = copy.deepcopy(full)
        compact = RolePrivateCache(decoder.config)
        compact.layers = copy.deepcopy(full.layers)
        compact.absolute_length = 8
        compact.retained_positions = list(range(8))
        compact.private_positions = {i: {'owner': 'planner', 'kind': 'old_role_duty',
                                        'expires_after': 1} for i in (2, 3)}
        assert compact.retire('planner')['removed_positions'] == 0
        report = compact.retire('retriever')
        assert report['removed_positions'] == 2
        assert compact.get_seq_length() == 8
        keep = torch.tensor([0, 1, 4, 5, 6, 7])
        for left, right in zip(preserved.layers, compact.layers, strict=True):
            for name in ('keys', 'values'):
                value = getattr(left, name, None)
                if value is not None:
                    assert torch.equal(getattr(right, name), value.index_select(-2, keep))
            for name in ('conv_states', 'recurrent_states'):
                value = getattr(left, name, None)
                if value is not None:
                    assert torch.equal(getattr(right, name), value)

        # 独立参考保持十个物理位置，仅在注意力掩码中屏蔽同样两行。
        queries = torch.tensor([8, 9])
        keys = torch.arange(10)
        allowed = (keys[None, :] <= queries[:, None]) & (keys[None, :] != 2) & (keys[None, :] != 3)
        mask = torch.zeros((1, 1, 2, 10))
        mask.masked_fill_(~allowed[None, None], torch.finfo(mask.dtype).min)
        reference = decoder(input_ids=suffix, past_key_values=full,
                            attention_mask=mask, use_cache=True).last_hidden_state
        with PrivateInstructionManager(bundle):
            result = decoder(input_ids=suffix, past_key_values=compact,
                             attention_mask=torch.ones((1, 10)), use_cache=True)
        assert torch.allclose(result.last_hidden_state, reference, atol=2e-5, rtol=1e-4)
        assert result.past_key_values.get_seq_length() == 10
        assert result.past_key_values.retained_positions == [0, 1, 4, 5, 6, 7, 8, 9]


def test_private_prompt_marking_preserves_current_task_and_message_boundaries():
    bundle = cache_bundle()
    manager = PrivateInstructionManager(bundle)
    past = RolePrivateCache(text_decoder(bundle).config)
    text = '<user>PRIVATE DUTY\nCurrent task: Alice left in 1990.[end]<assistant>'
    manager.mark(past, text, 100, 'PRIVATE DUTY', 'planner', 'old_role_duty', 1)
    offsets = bundle.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    selected = [(a,b) for i,(a,b) in enumerate(offsets) if 100+i in past.private_positions]
    assert selected
    assert all(6 <= a < b <= 18 for a,b in selected)


def test_consumed_steps_keep_delivery_rows_and_do_not_expire_before_consumer_completes():
    bundle = cache_bundle()
    decoder = text_decoder(bundle)
    with torch.inference_mode():
        past = RolePrivateCache(decoder.config)
        with PrivateInstructionManager(bundle):
            decoder(input_ids=torch.tensor([[3] * 40]), past_key_values=past, use_cache=True)
        past.private_positions = {i: {'owner': 'planner', 'kind': 'consumed_internal_steps',
                                     'expires_after': 1} for i in range(32)}
        assert past.retire('planner')['removed_positions'] == 0
        assert past.retire('retriever')['removed_positions'] == 32
        assert past.retained_positions == list(range(32, 40))
        assert past.absolute_length == 40


def test_compaction_peak_includes_one_replacement_at_a_time():
    bundle = cache_bundle()
    decoder = text_decoder(bundle)
    with torch.inference_mode(), PrivateInstructionManager(bundle):
        past = decoder(input_ids=torch.tensor([[3] * 8]), use_cache=True).past_key_values
    before = cache_bytes_for_test(past)
    replacement = past.layers[0].keys.numel() * past.layers[0].keys.element_size() * 6 // 8
    past.private_positions = {i: {'kind': 'old_role_duty', 'expires_after': 0} for i in (2, 3)}
    report = past.retire('planner')
    assert report['compaction_peak_bytes'] == before + replacement
    for layer in past.layers:
        for name in ('keys', 'values'):
            value = getattr(layer, name)
            assert value.untyped_storage().nbytes() == value.numel() * value.element_size()


def cache_bytes_for_test(past):
    return sum(getattr(layer, name).numel() * getattr(layer, name).element_size()
               for layer in past.layers for name in ('keys', 'values'))


def test_repeated_retrieval_role_can_consume_previous_work_without_waiting_for_reasoner():
    bundle = cache_bundle()
    decoder = text_decoder(bundle)
    with torch.inference_mode(), PrivateInstructionManager(bundle):
        past = decoder(input_ids=torch.tensor([[3] * 40]), use_cache=True).past_key_values
    past.private_positions = {i: {'kind': 'consumed_internal_steps', 'expires_after': 0,
                                 'after_internal_call': 2} for i in range(32)}
    assert past.retire('retriever', completed_internal_calls=1)['removed_positions'] == 0
    assert past.retire('retriever', completed_internal_calls=2)['removed_positions'] == 32
    assert past.retained_positions == list(range(32, 40))


def test_role_duty_and_routes_survive_until_its_own_tool_decision_finishes():
    bundle = cache_bundle()
    decoder = text_decoder(bundle)
    with torch.inference_mode(), PrivateInstructionManager(bundle):
        past = decoder(input_ids=torch.tensor([[3] * 8]), use_cache=True).past_key_values
    past.private_positions = {i: {'kind': 'old_role_duty', 'expires_after': 1,
                                 'after_tool_call': 1} for i in (2, 3)}
    assert past.retire('retriever', completed_internal_calls=2,
                       completed_tool_calls=0)['removed_positions'] == 0
    assert past.retire('retriever', completed_internal_calls=2,
                       completed_tool_calls=1)['removed_positions'] == 2


def test_history_provenance_survives_the_planners_first_internal_call():
    bundle = cache_bundle()
    decoder = text_decoder(bundle)
    with torch.inference_mode(), PrivateInstructionManager(bundle):
        past = decoder(input_ids=torch.tensor([[3] * 8]), use_cache=True).past_key_values
    past.private_positions[2] = {'kind': 'duplicate_history_source_id', 'expires_after': 0,
                                'requires_fact_ids': ['E1'], 'after_internal_call': 1}
    assert past.retire('planner', available_facts=['E1'],
                       completed_internal_calls=0)['removed_positions'] == 0
    assert past.retire('planner', available_facts=['E1'],
                       completed_internal_calls=1)['removed_positions'] == 1


def test_history_source_ids_require_actual_fact_rows_and_body_is_preserved():
    bundle = cache_bundle()
    content = {'current_task': {'question': 'What is the answer?'}, 'new_conversation': [
        {'question': 'Who?', 'answer': 'Literal "evidence_ids": ["E1"] is part of this answer.',
         'evidence_ids': ['E1', 'MISSING'], 'verified': False, 'completed': True}]}
    payload, spans = history_metadata_spans(content)
    decoder = text_decoder(bundle)
    manager = PrivateInstructionManager(bundle, policy='consumed_controls')
    with torch.inference_mode(), manager:
        past = decoder(input_ids=bundle.tokenizer(payload, return_tensors='pt')['input_ids'],
                       use_cache=True).past_key_values
    past.protected_positions.update(past.retained_positions)
    for left, right, kind, required in spans:
        manager.mark_span(past, payload, 0, left, right, 'planner', kind, 0, requires_fact_ids=required)
    offsets = bundle.tokenizer(payload, return_offsets_mapping=True)['offset_mapping']
    before = set(past.retained_positions)
    report = past.retire('planner', available_facts=['E1'])
    removed = before - set(past.retained_positions)
    assert report['removed_by_kind']['duplicate_history_source_id'] == len('"E1"')
    assert report['removed_by_kind']['consumed_history_status'] > 0
    missing_left = payload.index('"MISSING"')
    assert all(not (a >= missing_left and b <= missing_left + len('"MISSING"'))
               for i, (a, b) in enumerate(offsets) if i in removed)
    assert not removed & past.protected_positions
    # 所有删除都局限于结构化元数据字段；正文中的同名字符串和事实仍在原位置。
    for i in removed:
        a, b = offsets[i]
        assert any(a >= left and b <= right for left, right, _, _ in spans)


def test_protected_evidence_is_rejected_before_any_tensor_changes():
    bundle = cache_bundle()
    decoder = text_decoder(bundle)
    with torch.inference_mode(), PrivateInstructionManager(bundle):
        past = decoder(input_ids=torch.tensor([[3] * 8]), use_cache=True).past_key_values
    original = past.layers[0].keys.clone()
    past.protected_positions.add(2)
    past.private_positions[2] = {'kind': 'old_role_duty', 'expires_after': 0}
    with pytest.raises(ValueError, match='原始证据'):
        past.retire('planner')
    assert torch.equal(original, past.layers[0].keys)
    assert past.retained_positions == list(range(8))


def test_official_generation_continues_sparse_cache_without_reencoding_old_positions():
    bundle = cache_bundle()
    decoder = text_decoder(bundle)
    manager = PrivateInstructionManager(bundle, policy='consumed_controls')
    initial = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    with torch.inference_mode(), manager:
        past = decoder(input_ids=initial, use_cache=True).past_key_values
        past.private_positions = {i: {'kind': 'old_role_duty', 'expires_after': 0} for i in (2, 3)}
        past.retire('planner')
        manager.forward_records = []
        score = scripted_logits(bundle, 'OK')
        try:
            logical = torch.cat((initial, torch.tensor([[11, 12]])), dim=1)
            output = bundle.model.generate(input_ids=logical, attention_mask=torch.ones_like(logical),
                past_key_values=past, max_new_tokens=4, return_dict_in_generate=True, do_sample=False)
        finally:
            score.remove()
    assert bundle.tokenizer.decode(output.sequences[0, 10:], skip_special_tokens=True) == 'OK'
    assert [r['new_forward_positions'] for r in manager.forward_records] == [2, 1, 1]
    assert manager.forward_records[0]['logical_prefix_positions'] == 8
    assert manager.forward_records[0]['retained_prefix_positions'] == 6
