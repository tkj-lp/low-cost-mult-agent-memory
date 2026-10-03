import copy
import json
from types import SimpleNamespace

import pytest
import torch

from competition_agents.llm.context_budget import ContextCapacityError, generation_budget
from competition_agents.llm.decision_contract import parse_decision, parse_hidden_header
from competition_agents.llm.generation import standard_generate
from competition_agents.llm.native_local_chat import NativeLocalChat
from competition_agents.llm.resident_chat import ResidentChat
from competition_agents.collaboration_transport import CollaborationTransport
from competition_agents.role_prompts import role_prompt
from runtime_fixtures import cache_bundle, scripted_logits


def test_numbered_prose_partial_json_wrong_types_and_unknown_actions_never_execute_tools():
    invalid = ['1. First inspect the entity.', '{"action":"lookup","query":"A date"',
               '{"action":"lookup","query":true,"work":null}',
               '{"action":"lookup","query":" ","work":null}',
               '{"action":"delete","query":null,"work":null}',
               '{"action":"deliver","query":null,"work":{"candidate":true,"basis":[]}}']
    for raw in invalid:
        result = parse_decision(raw, role='reasoner')
        assert result['action'] is None and result['lookup_query'] is None
        assert result['content'] == raw and result['decision_error']
    assert parse_decision('{"action":"lookup","query":"A","work":null}', role='planner')['action'] is None


@pytest.mark.parametrize('architecture', ['attention', 'hybrid'])
@pytest.mark.parametrize('body', [
    {'action': 'lookup', 'query': 'Alice appointment date, not birth date', 'work': None},
    {'action': 'deliver', 'query': None, 'work': {'candidate': '1990', 'basis': ['Original relation.']}},
    {'action': 'reuse', 'query': None, 'work': None},
])
def test_official_generation_obeys_full_tool_syntax_on_two_cache_architectures(architecture, body):
    bundle = cache_bundle(architecture)
    weights = {key: value.clone() for key, value in bundle.model.state_dict().items()}
    configuration = copy.deepcopy(bundle.model.generation_config.to_dict())
    hook = scripted_logits(bundle, body, tempting_number=True)
    try:
        result = standard_generate(bundle, torch.tensor([[3, 4]]), max_tokens=256,
                                   role='reasoner', control=True, allow_reuse=True)
    finally:
        hook.remove()
    assert result['action'] == body['action']
    assert result['lookup_query'] == body['query']
    assert result['decision_error'] is None
    assert result['control_stop_reason'] == 'complete_decision_object'
    assert json.loads(bundle.tokenizer.decode(result['output_token_ids'])) == body
    assert bundle.model.generation_config.to_dict() == configuration
    assert all(torch.equal(v, weights[k]) for k, v in bundle.model.state_dict().items())
    assert not bundle.model._forward_hooks and not bundle.model._forward_pre_hooks


def test_truncated_tool_parameters_are_not_executed_and_context_never_overflows():
    bundle = cache_bundle()
    hook = scripted_logits(bundle, {'action': 'lookup', 'query': 'Alice date', 'work': None})
    bundle.model.config.max_position_embeddings = 12
    try:
        result = standard_generate(bundle, torch.tensor([[3, 4]]), max_tokens=128, control=True)
    finally:
        hook.remove()
    assert result['context_budget_reduced'] and result['effective_output_budget'] == 10
    assert result['output_tokens'] <= 10
    assert result['action'] is None and result['decision_error']
    with pytest.raises(ContextCapacityError):
        generation_budget(bundle, 12, 128)


def test_hidden_header_does_not_decode_work_or_confuse_escaped_query_with_a_field():
    bundle = cache_bundle()
    body = {'action': 'lookup', 'query': 'find "work": literal relation', 'work': None}
    ids = bundle.tokenizer.encode(json.dumps(body), add_special_tokens=False)
    assert parse_hidden_header(bundle.tokenizer, ids, role='retriever', allow_reuse=False)['lookup_query'] == body['query']
    body = {'action': 'deliver', 'query': None, 'work': {'candidate': 'A' * 2000, 'basis': []}}
    ids = bundle.tokenizer.encode(json.dumps(body), add_special_tokens=False)
    original, lengths = bundle.tokenizer.decode, []
    def observe(values, **kwargs):
        lengths.append(len(values))
        return original(values, **kwargs)
    bundle.tokenizer.decode = observe
    parsed = parse_hidden_header(bundle.tokenizer, ids, role='reasoner', allow_reuse=False)
    assert parsed['action'] == 'deliver' and parsed['content'] == ''
    assert max(lengths) < 100


def task_snapshot():
    return {'task': {'question': 'When did Alice leave?'}, 'history': [],
            'evidence': {'f': {'title': 'Source', 'fact_fields': {
                'document_id': 'doc', 'subject': 'Alice', 'relation': 'left', 'object': 'Council',
                'polarity': 'positive', 'qualifiers': [{'date': 1990}], 'sentences': [1]}}},
            'work': [], 'experience': None, 'execution': None,
            'diagnostics': [{'kind': 'lookup_budget_exhausted', 'message': 'Not the question.'}]}


@pytest.mark.parametrize('architecture', ['attention', 'hybrid'])
def test_task_lifecycle_keeps_all_active_state_restores_exact_prefix_and_reencodes_recalled_facts(architecture):
    bundle = cache_bundle(architecture)
    resident = ResidentChat(NativeLocalChat(bundle, max_tokens=256))
    data = task_snapshot()
    transport = CollaborationTransport('latent', None, SimpleNamespace(source_aliases=[]))
    packet = transport.packet(data)
    plan = {'action': 'deliver', 'query': None,
            'work': {'entities': ['Alice'], 'target': 'departure date', 'constraints': []}}
    for role, output in [('planner', plan), ('retriever', {'action': 'reuse', 'query': None, 'work': None})]:
        hook = scripted_logits(bundle, output)
        try:
            resident.call_role(role, role_prompt('continuous', role), data, scope='one', turn_key='t1',
                               packet=packet, identifiers=transport.identifiers)
        finally:
            hook.remove()
    first, second = resident.calls
    assert first['facts_encoded'] == 1 and second['facts_reused_without_encoding'] == 1
    assert second['cached_input_tokens'] > first['cached_input_tokens']
    assert not first['state_issues'] and not second['state_issues']
    categories = [p['category'] for p in first['inputs']]
    assert categories.index('workflow_status') < categories.index('role_instruction')
    request = next(p['text'] for p in first['inputs'] if p['category'] == 'role_instruction')
    assert data['task']['question'] in request and 'Current role: planner' in request
    length = len(resident.ids)
    assert not resident.finish_task(persisted=False)['released']
    assert len(resident.ids) == length
    report = resident.finish_task(persisted=True)
    assert report['removed_positions'] > 0 and report['after_bytes'] < report['before_bytes']
    assert resident.past.get_seq_length() == len(resident.ids) == len(resident.lifecycle.prefix_ids)
    assert resident.peak == report['after_bytes']
    assert not resident.encoded_facts and not resident.encoded_history
    suffix = torch.tensor([[3, 4]])
    prefix = torch.tensor([resident.ids])
    with torch.inference_mode():
        full = bundle.model(torch.cat((prefix, suffix), dim=1)).logits[:, -1]
        resumed = bundle.model(suffix, past_key_values=copy.deepcopy(resident.past)).logits[:, -1]
    assert torch.allclose(full, resumed, atol=1e-5, rtol=1e-4)
    hook = scripted_logits(bundle, plan)
    try:
        resident.call_role('planner', role_prompt('continuous', 'planner'), data, scope='one', turn_key='t2',
                           packet=packet, identifiers=transport.identifiers)
    finally:
        hook.remove()
    assert resident.calls[-1]['facts_encoded'] == 1
    assert resident.calls[-1]['shared_instruction_reused'] is True
    assert resident.calls[-1]['cache_recovery_positions'] == 0
    resident.close()
