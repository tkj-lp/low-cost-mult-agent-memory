import copy
import json
import re
from types import SimpleNamespace

import torch

from competition_agents.collaboration_fields import projected_work, work_fields
from competition_agents.collaboration_transport import CollaborationTransport
from competition_agents.llm.generation import standard_generate
from competition_agents.llm.native_local_chat import NativeLocalChat
from competition_agents.llm.resident_chat import ResidentChat
from competition_agents.role_prompts import role_prompt, role_prompt_parts
from runtime_fixtures import cache_bundle, scripted_logits


def snapshot():
    return {'task': {'question': 'When did she leave that post?'},
            'history': [{'question': 'Earlier question', 'answer': 'UNABRIDGED HISTORY SENTINEL',
                         'evidence_ids': ['fact'], 'verified': False}],
            'evidence': {'fact': {'title': 'Full source', 'fact_fields': {
                'subject': 'Alice', 'relation': 'left', 'object': 'Council', 'document_id': 'doc',
                'polarity': 'negative', 'qualifiers': [{'date': 1990, 'condition': 'during winter'}],
                'sentences': [2, 7]}}},
            'work': [{'role': 'planner', 'action': 'deliver', 'content': json.dumps({
                'entities': ['Alice', 'Council'], 'target': 'departure date',
                'constraints': ['not an appointment date'], 'new_unknown_field': 'preserve me'})}],
            'diagnostics': [], 'experience': None, 'execution': None}


def test_protocol_routes_history_to_planner_and_preserves_complete_evidence_and_constraints():
    data = snapshot()
    original = copy.deepcopy(data)
    transport = CollaborationTransport('structured', None, SimpleNamespace(source_aliases=[]))
    assert 'UNABRIDGED HISTORY SENTINEL' in transport.render(data, receiver='planner')
    for receiver in ('retriever', 'reasoner', 'summarizer'):
        body = json.loads(transport.render(data, receiver=receiver))
        assert 'conversation' not in body
        assert 'UNABRIDGED HISTORY SENTINEL' not in json.dumps(body)
        assert body['role_work']['fields_by_role']['planner']['new_unknown_field'] == 'preserve me'
        assert body['evidence']['facts'][0][1:6] == [
            'Alice', 'left', 'Council', 'negative', [{'date': 1990, 'condition': 'during winter'}]]
        assert body['evidence']['facts'][0][7] == [2, 7]
        assert list(body['evidence']['sources'].values()) == ['Full source']
    assert data == original


def test_unparsed_work_keeps_original_and_history_without_blocking_or_claiming_compression():
    data = snapshot()
    data['work'][0]['content'] = 'An unusual but complete plan, including a condition.'
    transport = CollaborationTransport('structured', None, SimpleNamespace(source_aliases=[]))
    body = json.loads(transport.render(data, receiver='retriever'))
    assert body['role_work']['unparsed_work'][0]['unparsed_work'] == data['work'][0]['content']
    assert body['conversation'][0]['answer'] == 'UNABRIDGED HISTORY SENTINEL'
    assert body['history_fallback'] == 'reference_plan_unavailable'
    assert work_fields('["not a field object"]') == ({}, '["not a field object"]')


def test_revisions_keep_qualifications_and_mark_previous_values_instead_of_silent_overwrite():
    rows = [{'role': 'reasoner', 'content': json.dumps({'candidate': 'Alice', 'constraints': ['after 1990']})},
            {'role': 'reasoner', 'content': json.dumps({'candidate': 'Bob', 'conditions': ['not before 1992']})},
            {'role': 'reasoner', 'content': '', 'action': 'reuse'}]
    fields = projected_work(rows, 'summarizer')['fields_by_role']['reasoner']
    assert fields['candidate'] == {'current': 'Bob', 'previous': ['Alice']}
    assert fields['constraints'] == ['after 1990']
    assert fields['conditions'] == ['not before 1992']


class WordTokenizer:
    eos_token_id, pad_token_id, bos_token = 2, 0, None

    def __init__(self):
        self.vocab = {'0': 10, '1': 11, '2': 12}

    def __call__(self, text, **kwargs):
        values = []
        for word in re.findall(r'\s+|[^\s]+', text):
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab) + 32
            values.append(self.vocab[word])
        if kwargs.get('return_tensors') == 'pt':
            return {'input_ids': torch.tensor([values])}
        return {'input_ids': values}

    def decode(self, values, **kwargs):
        reverse = {value: key for key, value in self.vocab.items()}
        return ''.join(reverse.get(int(value), '') for value in values)

    def apply_chat_template(self, messages, *, add_generation_prompt=False, **kwargs):
        value = ''.join('<' + m['role'] + '>\n' + m['content'] + '\n</end>\n' for m in messages)
        return value + ('<assistant>\n' if add_generation_prompt else '')


def tiny_bundle():
    return cache_bundle()


def test_official_generation_reuse_stops_after_complete_decision_and_hooks_are_removed():
    bundle = tiny_bundle()
    weights = {key: value.clone() for key, value in bundle.model.state_dict().items()}
    generation = copy.deepcopy(bundle.model.generation_config.to_dict())
    decision = {'action': 'reuse', 'query': None, 'work': None}
    hook = scripted_logits(bundle, decision)
    try:
        result = standard_generate(bundle, torch.tensor([[20, 21]]), max_tokens=128, control=True)
    finally:
        hook.remove()
    assert result['action'] == 'reuse'
    assert json.loads(bundle.tokenizer.decode(result['output_token_ids'])) == decision
    assert result['finish_reason'] == 'stop'
    assert result['control_stop_reason'] == 'complete_decision_object'
    assert result['generation_forward_calls'] == result['output_tokens']
    assert not bundle.model._forward_hooks and not bundle.model._forward_pre_hooks
    assert bundle.model.generation_config.to_dict() == generation
    assert all(torch.equal(value, weights[key]) for key, value in bundle.model.state_dict().items())


def test_resident_encodes_shared_rules_once_and_reuses_unmodified_evidence_with_real_cache():
    bundle = tiny_bundle()
    native = NativeLocalChat(bundle, max_tokens=256)
    resident = ResidentChat(native)
    data = snapshot()
    transport = CollaborationTransport('latent', native, SimpleNamespace(source_aliases=[]))
    packet = transport.packet(data)
    for role in ('planner', 'retriever'):
        body = ({'action': 'deliver', 'query': None, 'work': {'entities': ['Alice'],
                 'target': 'departure', 'constraints': []}} if role == 'planner' else
                {'action': 'reuse', 'query': None, 'work': None})
        hook = scripted_logits(bundle, body)
        try:
            resident.call_role(role, role_prompt('continuous', role), data, scope='one', turn_key='turn1',
                               packet=packet, identifiers=transport.identifiers)
        finally:
            hook.remove()
    first, second = resident.calls
    assert first['shared_instruction_reused'] is False
    assert second['shared_instruction_reused'] is True
    assert sum(part['category'] == 'shared_instruction' for c in resident.calls for part in c['inputs']) == 1
    assert all(part['category'] != 'shared_instruction' for part in second['inputs'])
    assert first['facts_encoded'] == 1 and second['facts_encoded'] == 0
    assert second['facts_reused_without_encoding'] == 1
    assert second['cache_recovery_positions'] == 0
    assert second['cached_input_tokens'] > 0
    assert resident.past.get_seq_length() == len(resident.ids)
    assert all(c['output_tokens'] > 1 and not c['state_issues'] for c in resident.calls)
    assert second['role_prompt'] == role_prompt('continuous', 'retriever')
    common, duties = role_prompt_parts('continuous', 'retriever')
    assert common + duties == second['role_prompt']
    resident.close()
    assert resident.shared_instruction_signature is None


def test_shared_duties_and_final_answer_contract_do_not_depend_on_transport():
    for kind in ('continuous', 'codeact'):
        for role in ('planner', 'retriever', 'reasoner', 'summarizer'):
            assert role_prompt(kind, role) == ''.join(role_prompt_parts(kind, role))
    from competition_agents.role_prompts import FINAL_ANSWER_REQUIREMENTS
    assert role_prompt('continuous', 'summarizer').endswith(FINAL_ANSWER_REQUIREMENTS)
    assert 'runtime decision interface' not in role_prompt('continuous', 'summarizer')
    assert 'JSON object of semantic fields' not in role_prompt('continuous', 'summarizer')


def test_reuse_requires_existing_work_and_never_replaces_planning_or_executable_code():
    from competition_agents.collaboration_fields import can_reuse_work
    data = snapshot()
    assert not can_reuse_work('planner', data, 'continuous')
    assert can_reuse_work('retriever', data, 'continuous')
    assert not can_reuse_work('reasoner', data, 'continuous')
    data['work'].append({'role': 'reasoner', 'action': 'deliver', 'content': '{"candidate":"1990"}'})
    assert can_reuse_work('reasoner', data, 'continuous')
    assert not can_reuse_work('reasoner', data, 'codeact')
    data['evidence'] = {}
    assert not can_reuse_work('retriever', data, 'continuous')
