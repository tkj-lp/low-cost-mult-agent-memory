"""连续计算机制回归；固定输出仅验证接口，不代表预训练模型答题成绩。"""
import copy
from types import SimpleNamespace

import pytest
import torch

from competition_agents.llm.continuous_state_chat import ContinuousStateChat
from competition_agents.llm.decision_contract import parse_tool_decision
from competition_agents.llm.generation import standard_generate
from competition_agents.llm.native_local_chat import NativeLocalChat
from competition_agents.collaboration_transport import CollaborationTransport
from competition_agents.role_prompts import COMMON, role_prompt, role_duty_prompt, QUESTION_ROLES, CODE_ROLES
from competition_agents.unified_collaboration import UnifiedCollaboration
from runtime_fixtures import cache_bundle, scripted_logits
from test_shared_runtime_end_to_end import ObservedMemory
from test_typed_decision_lifecycle import task_snapshot


@pytest.mark.parametrize('kind,roles', [('continuous', QUESTION_ROLES), ('codeact', CODE_ROLES)])
def test_business_instructions_are_unchanged_and_only_text_interface_is_removed(kind, roles):
    for role, duties in roles.items():
        actual = role_duty_prompt(kind, role)
        assert actual.startswith(COMMON)
        assert actual.endswith(duties)
        assert role_prompt(kind, role).startswith(actual)
        if role == 'summarizer':
            assert actual == role_prompt(kind, role)


def test_tool_interface_has_no_work_field_and_rejects_invalid_or_truncated_parameters():
    for text in ['1. retrieve', '{"action":"lookup","query":',
                 '{"action":"lookup","query":1990}', '{"query":"Alice"}',
                 '{"action":"deliver","query":null,"work":"new prose"}',
                 '{"action":"deliver","query":"Alice"}']:
        result = parse_tool_decision(text)
        assert result['action'] is None and result['decision_error']
    result = parse_tool_decision('{"action":"lookup","query":"Alice date 1990"}')
    assert result['lookup_query'] == 'Alice date 1990' and result['content'] == ''
    assert parse_tool_decision('{"action":"deliver","query":null}')['action'] == 'deliver'
    assert parse_tool_decision('{"action":"reuse","query":null}')['decision_error']
    assert parse_tool_decision('{"action":"reuse","query":null}', allow_reuse=True)['action'] == 'reuse'


@pytest.mark.parametrize('architecture', ['attention', 'hybrid'])
def test_planning_reuses_layer_cache_without_token_generation_or_last_layer_reinjection(architecture, monkeypatch):
    bundle = cache_bundle(architecture)
    chat = ContinuousStateChat(NativeLocalChat(bundle))
    transport = CollaborationTransport('latent', None, SimpleNamespace(source_aliases=[]))
    data = task_snapshot()
    weights = {k: v.clone() for k, v in bundle.model.state_dict().items()}
    configuration = copy.deepcopy(bundle.model.generation_config.to_dict())
    def forbidden(*args, **kwargs):
        raise AssertionError('规划内部计算不允许调用文字生成。')
    monkeypatch.setattr(bundle.model, 'generate', forbidden)
    monkeypatch.setattr(bundle.tokenizer, 'decode', forbidden)
    monkeypatch.setattr(bundle.model.get_output_embeddings(), 'forward', forbidden)
    decoder = bundle.model.model
    forward = decoder.forward
    def no_reinjection(*args, **kwargs):
        assert kwargs.get('inputs_embeds') is None
        return forward(*args, **kwargs)
    monkeypatch.setattr(decoder, 'forward', no_reinjection)
    result = chat.call_role('planner', role_prompt('continuous', 'planner'), data,
                           scope='s', turn_key='t', packet=transport.packet(data))
    assert result['content'] == '' and result['action'] == 'deliver'
    call = chat.calls[-1]
    assert call['latent_steps'] == 0 and call['output_tokens'] == 0
    assert call['role_prefill_positions'] > 0
    assert call['external_generation_calls'] == 0 and call['intermediate_work_output_tokens'] == 0
    assert call['facts_encoded'] == 1
    original_ids = chat.ids[:]
    original_cache = copy.deepcopy(chat.past)
    chat.past = None
    chat.metrics = {k: 0 for k in ('appended_forward_ms', 'forward_positions', 'cache_recovery_positions')}
    chat._ensure_state()
    assert chat.ids == original_ids and chat.metrics['cache_recovery_positions'] == len(original_ids)
    suffix = torch.tensor([[3, 4]])
    with torch.inference_mode():
        expected = bundle.model.model(suffix, past_key_values=original_cache).last_hidden_state
        actual = bundle.model.model(suffix, past_key_values=copy.deepcopy(chat.past)).last_hidden_state
    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-4)
    assert not chat.finish_task(persisted=False)['released']
    assert len(chat.ids) == len(original_ids)
    assert chat.finish_task(persisted=True)['released']
    assert len(chat.ids) == len(chat.lifecycle.prefix_ids)
    assert bundle.model.generation_config.to_dict() == configuration
    assert all(torch.equal(v, weights[k]) for k, v in bundle.model.state_dict().items())


@pytest.mark.parametrize('architecture', ['attention', 'hybrid'])
@pytest.mark.parametrize('kind', ['continuous', 'codeact'])
def test_real_generation_tools_memory_and_final_answer_with_no_intermediate_work_text(architecture, kind, monkeypatch):
    bundle = cache_bundle(architecture)
    client = NativeLocalChat(bundle, max_tokens=256)
    runtime = UnifiedCollaboration.__new__(UnifiedCollaboration)
    runtime.method, runtime.task_kind, runtime.max_searches = 'latent', kind, 2
    runtime.memory, runtime.client = ObservedMemory(), client
    runtime.transport = CollaborationTransport('latent', client, SimpleNamespace(source_aliases=[]))
    runtime.resident = ContinuousStateChat(client)
    seen = []
    def generated(*args, **kwargs):
        role = kwargs['role']
        if kwargs['tool_only']:
            body = ({'action': 'lookup', 'query': 'Alice joined Council date'} if not runtime.memory.records
                    else {'action': 'deliver', 'query': None})
        elif role == 'reasoner':
            body = '```python\nresult = sum(inputs["values"])\n```'
        else:
            body = '6' if kind == 'codeact' else '1990'
        seen.append((role, kwargs['tool_only']))
        handle = scripted_logits(bundle, body)
        try:
            return standard_generate(*args, **kwargs)
        finally:
            handle.remove()
    monkeypatch.setattr('competition_agents.llm.continuous_state_chat.standard_generate', generated)
    rows = []
    for question in ('When did Alice join Council?', 'What was that year?'):
        task = {'task': question, 'inputs': {'values': [1, 2, 3]}, 'allowed_imports': [],
                'output_schema': {'type': 'number'}, 'output_fields': []}
        rows.append(runtime.run_task(task, history_scope='s') if kind == 'codeact' else
                    runtime.run(question, history_scope='s'))
    first, second = rows
    assert first['workflow_complete'] and second['workflow_complete']
    assert not first['error'] and not second['error']
    assert len(first['calls']) == 5 and len(second['calls']) == 4
    assert second['metrics']['memory_only_tasks'] == 1 and len(runtime.memory.queries) == 1
    assert not any(role == 'planner' for role, _ in seen)
    for row in rows:
        assert row['prediction'] == ('6' if kind == 'codeact' else '1990')
        assert not row['workflow_issues']
        assert row['cache_lifecycle']['released']
        assert row['source_evidence']['f']['fact_fields'] == runtime.memory.hit['fact_fields']
        assert all(c['intermediate_work_output_tokens'] == 0 for c in row['calls'])
        assert all(c['latent_steps'] == 0 for c in row['calls'])
        assert all(c['role_prefill_positions'] > 0 for c in row['calls'] if c['role'] != 'summarizer')
        assert all(c['output_tokens'] == 0 for c in row['calls'] if c['role'] == 'planner')
        if kind == 'codeact':
            assert row['execution_result']['sandbox_status'] == 'ok'
            assert row['execution_result']['result'] == 6
    assert second['calls'][0]['reused_input_tokens'] == first['cache_lifecycle']['retained_prefix_positions']
    assert second['calls'][0]['facts_encoded'] == 1
    runtime.resident.close()
