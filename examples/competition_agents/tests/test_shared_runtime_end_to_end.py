"""固定模型输出的流程回归：检验真实生成接口、交接、记忆和执行器，不计模型成绩。"""
import copy
from types import SimpleNamespace

import pytest

from competition_agents.unified_collaboration import UnifiedCollaboration
from competition_agents.collaboration_transport import CollaborationTransport
from competition_agents.llm.native_local_chat import NativeLocalChat
from competition_agents.llm.resident_chat import ResidentChat
from runtime_fixtures import cache_bundle, scripted_logits


class ObservedMemory:
    enabled = True

    def __init__(self):
        self.records, self.episodes, self.queries = {}, [], []
        self.hit = {'context_id': 'f', 'title': 'Original source',
                    'text': 'Alice joined Council in 1990. Preserve the entire source.',
                    'fact_fields': {'document_id': 'd', 'subject': 'Alice', 'relation': 'joined',
                                    'object': 'Council', 'polarity': 'positive',
                                    'qualifiers': [{'kind': 'date', 'value': '1990'}], 'sentences': [0]}}

    def recall(self, query, scope, kind):
        return {'history': copy.deepcopy(self.episodes),
                'evidence': [dict(copy.deepcopy(hit), from_memory=True) for hit in self.records.values()],
                'text': self.hit['text'] if self.records else '', 'experience': None}

    def search(self, query, scope, *, expanded):
        self.queries.append(query)
        return [copy.deepcopy(self.hit)]

    def observe(self, question, hits, scope):
        self.records.update({h['context_id']: copy.deepcopy(h) for h in hits})
        return len(hits)

    def commit(self, question, answer, snapshot, scope, kind, *, completed):
        self.observe(question, snapshot['evidence'].values(), scope)
        self.episodes.append({'question': question, 'answer': answer, 'evidence_ids': list(self.records)})
        return {'written': list(self.records), 'experience': None, 'experience_writes': 0}


@pytest.mark.parametrize('method', ['plain', 'structured', 'latent'])
@pytest.mark.parametrize('kind', ['continuous', 'codeact'])
def test_two_tasks_four_roles_preserve_evidence_reuse_memory_and_execute_code(method, kind):
    bundle = cache_bundle()
    client = NativeLocalChat(bundle, max_tokens=256)
    runtime = UnifiedCollaboration.__new__(UnifiedCollaboration)
    runtime.method, runtime.task_kind, runtime.max_searches = method, kind, 2
    runtime.memory, runtime.client = ObservedMemory(), client
    runtime.transport = CollaborationTransport(method, client, SimpleNamespace(source_aliases=[]))
    runtime.resident = ResidentChat(client) if method == 'latent' else None
    target = runtime.resident if runtime.resident else client
    original = target.call_role if runtime.resident else target.call

    def call(role, *args, **kwargs):
        if role == 'summarizer':
            output = '6' if kind == 'codeact' else '1990'
        elif role == 'retriever':
            output = ({'action': 'reuse', 'query': None, 'work': None} if runtime.memory.records else
                      {'action': 'lookup', 'query': 'Alice joined Council date', 'work': None})
        else:
            work = ({'entities': ['Alice'], 'target': 'appointment date', 'constraints': []}
                    if role == 'planner' else {'candidate': '1990', 'basis': ['Original dated relation.']})
            if kind == 'codeact':
                work = ('Sum the provided values.' if role == 'planner' else
                        '```python\nresult = sum(inputs["values"])\n```')
            output = {'action': 'deliver', 'query': None, 'work': work}
        hook = scripted_logits(bundle, output)
        try:
            return original(role, *args, **kwargs)
        finally:
            hook.remove()
    if runtime.resident:
        target.call_role = call
    else:
        target.call = call
    rows = []
    for question in ('When did Alice join Council?', 'What was that year?'):
        task = {'task': question, 'inputs': {'values': [1, 2, 3]}, 'allowed_imports': [],
                'output_schema': {'type': 'number'}, 'output_fields': []}
        rows.append(runtime.run_task(task, history_scope='same-session') if kind == 'codeact' else
                    runtime.run(question, history_scope='same-session'))
    first, second = rows
    assert first['workflow_complete'] and second['workflow_complete']
    assert len(first['calls']) == 5 and len(second['calls']) == 4
    assert len(runtime.memory.queries) == 1
    assert first['metrics']['knowledge_searches'] == 1
    assert second['metrics']['knowledge_searches'] == 0
    assert second['metrics']['memory_only_tasks'] == 1
    for row in rows:
        assert not row['error'] and not row['workflow_issues']
        assert row['source_evidence']['f']['fact_fields'] == runtime.memory.hit['fact_fields']
        assert row['prediction'] == ('6' if kind == 'codeact' else '1990')
        if kind == 'codeact':
            assert row['execution_result']['sandbox_status'] == 'ok'
            assert row['execution_result']['result'] == 6
        if method == 'latent':
            assert row['cache_lifecycle']['released']
            assert row['metrics']['cache_reclaimed_positions'] > 0
            assert row['calls'][1]['cached_input_tokens'] > 0
