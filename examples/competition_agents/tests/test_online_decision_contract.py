import json
from contextlib import contextmanager

import pytest

from competition_agents.llm.audited_chat import AuditedChat
from competition_agents.role_prompts import role_prompt


@pytest.mark.parametrize('valid', [True, False])
def test_online_and_local_use_identical_typed_decisions_without_extra_calls(monkeypatch, valid):
    monkeypatch.setenv('LLM_BASE_URL', 'https://example.invalid')
    monkeypatch.setenv('LLM_API_KEY', 'unit-test-only')
    requests = []

    @contextmanager
    def open_request(request, **kwargs):
        payload = json.loads(request.data)
        requests.append(payload)
        assert payload['messages'][0]['role'] == 'system'
        assert payload['messages'][-1]['content'].endswith(role_prompt('continuous', 'retriever').split('Current role:', 1)[1])
        for tool in payload['tools']:
            parameters = tool['function']['parameters']
            assert set(parameters['required']) == set(parameters['properties'])
            work = parameters['properties']['work']
            if work['type'] == 'object':
                assert set(work['required']) == set(work['properties'])
        message = {'tool_calls': [{'function': {'name': 'lookup', 'arguments': json.dumps({
            'query': 'Alice appointment date' if valid else True, 'work': None})}}]}
        body = {'model': 'same-model', 'usage': {'prompt_tokens': 3, 'completion_tokens': 4},
                'choices': [{'finish_reason': 'tool_calls', 'message': message}]}
        class Response:
            def read(self):
                return json.dumps(body).encode()
        yield Response()

    monkeypatch.setattr('urllib.request.urlopen', open_request)
    client = AuditedChat('same-model')
    result = client.call('retriever', [
        {'role': 'system', 'content': role_prompt('continuous', 'retriever')},
        {'role': 'user', 'content': 'Current task and complete original evidence.'}],
        role_control={'task_kind': 'continuous', 'allow_reuse': True}, shared_roles=True)
    assert len(requests) == len(client.calls) == 1
    assert result['action'] == ('lookup' if valid else None)
    assert result['lookup_query'] == ('Alice appointment date' if valid else None)
    assert bool(result['decision_error']) is not valid
