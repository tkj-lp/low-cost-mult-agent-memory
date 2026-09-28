import hashlib
import json
from types import SimpleNamespace

import pytest

from competition_agents.plain_text_collaboration import PlainTextKnowledge, PlainTextMemory, TEXT_ROLES
from competition_agents.plain_text_collaboration import PlainTextCollaboration
from hypergraph_memory_method.prepared_knowledge import digest


SOURCE = 'Ron was Director.\n\nHe started in 2001. The garden gate was blue.  '
OTHER = 'An unrelated complete document about the weather.'


def knowledge(*bodies):
    return SimpleNamespace(documents=[{'id': str(i), 'title': 'Document ' + str(i), 'sentences': [body]}
                                      for i, body in enumerate(bodies or (SOURCE,))])


class Client:
    def __init__(self, *, truncate=None):
        self.calls = []
        self.truncate = truncate

    def call(self, role, messages):
        if role == 'planner':
            content = 'Find the requested fact about Ron. Resolve the follow-up using the conversation.'
        elif role == 'retriever':
            content = 'The personnel record describes Ron and when he started.'
        elif role == 'reasoner':
            content = 'The source states his job and starting year.'
        else:
            content = '2001' if 'When did he start?' in messages[-1]['content'] else 'Director'
        self.calls.append({'role': role, 'messages': messages, 'input_tokens': 1, 'output_tokens': 1,
                           'elapsed_ms': 1., 'request_bytes': len(json.dumps(messages).encode()),
                           'response_bytes': len(content.encode()),
                           'finish_reason': 'length' if role == self.truncate else 'stop'})
        return {'content': content}


def runtime(client=None, corpus=None):
    def forbidden(*args, **kwargs):
        pytest.fail('纯文本链路不应编码查询或事实。')
    return PlainTextCollaboration(PlainTextKnowledge.from_knowledge(corpus or knowledge()), client or Client())


def test_active_text_entry_preserves_whole_sources_in_every_downstream_handoff():
    app = runtime(corpus=knowledge(SOURCE, OTHER))
    result = app.run('What position did Ron hold?', history_scope='a')
    assert result['error'] is None
    assert result['implementation'] == 'full-text-collaboration-v1'
    assert len(result['calls']) == 4
    assert isinstance(app.memory, PlainTextMemory)
    assert SOURCE not in result['calls'][0]['messages'][-1]['content']
    assert SOURCE not in result['messages'][0]['content']
    for call in result['calls'][1:]:
        incoming = call['messages'][-1]['content']
        assert incoming.count(SOURCE) == 1
        assert incoming.count(OTHER) == 1
    for message in result['messages'][1:]:
        assert SOURCE in message['content'] and OTHER in message['content']
        assert 'The garden gate was blue.' in message['content']
    assert result['searches'][0]['documents_read'] == 2
    assert result['metrics']['communication_bytes'] == sum(m['wire_bytes'] for m in result['messages'])


def test_handoffs_contain_received_text_and_all_unmodified_role_outputs():
    result = runtime().run('What position did Ron hold?')
    for message, call in zip(result['messages'], result['calls'][1:]):
        incoming = call['messages'][-1]['content']
        # 检索角色会加入刚读取的原文，后两个角色直接读取实际交接正文。
        assert incoming.startswith(message['content'])
        assert message['content_bytes'] == len(message['content'].encode('utf-8'))
    assert result['messages'][2]['content'].startswith(result['messages'][1]['content'])
    for role in ('planner', 'retriever', 'reasoner'):
        assert result['outputs'][role] in result['messages'][2]['content']


def test_followup_reads_all_history_and_sources_without_quote_selection_or_index():
    app = runtime()
    first = app.run('What position did Ron hold?', history_scope='same')
    second = app.run('When did he start?', history_scope='same')
    assert first['prediction'] == 'Director' and second['prediction'] == '2001'
    assert first['metrics']['knowledge_searches'] == 1
    assert second['metrics']['knowledge_searches'] == 0
    assert second['knowledge_access']['source'] == 'session_memory'
    assert SOURCE in first['memory_written']['text']
    assert 'The garden gate was blue.' in second['memory_read']['text']
    for call in second['calls']:
        incoming = call['messages'][-1]['content']
        assert SOURCE in incoming
        assert first['question'] in incoming
        for output in first['outputs'].values():
            assert output in incoming
    assert second['memory_read']['text'] == first['memory_written']['text']
    after, _, turns = app.memory.read_all('same')
    assert after == first['memory_written']['text'] + second['memory_written']['text']
    assert after.count(SOURCE) == 1 and turns == 2


def test_memory_reads_older_turns_without_recent_turn_cutoff_and_isolates_sessions():
    app = runtime()
    for index in range(4):
        app.run('Old question ' + str(index), history_scope='a')
    fifth = app.run('Repeat his position?', history_scope='a')
    assert 'Old question 0' in fifth['memory_read']['text']
    assert fifth['metrics']['memory_turns_read'] == 4
    other = app.run('First question in a different session', history_scope='b')
    assert other['memory_read']['text'] == ''
    assert other['metrics']['knowledge_searches'] == 1
    assert all('Old question' not in c['messages'][-1]['content'] for c in other['calls'])


def test_changed_corpus_is_read_in_full_instead_of_treating_old_memory_as_complete():
    app = runtime()
    app.run('His job?', history_scope='a')
    app.knowledge = PlainTextKnowledge.from_knowledge(knowledge(SOURCE, OTHER))
    result = app.run('What about the weather?', history_scope='a')
    assert result['metrics']['knowledge_searches'] == 1
    assert SOURCE in result['searches'][0]['result'] and OTHER in result['searches'][0]['result']


def test_truncated_generation_does_not_silently_create_successful_memory():
    app = runtime(Client(truncate='reasoner'))
    result = app.run('His job?', history_scope='a')
    assert result['error'] and '截断' in result['error']
    assert result['memory_written']['committed'] is False
    assert app.memory.read_all('a')[0] == ''
    assert len(result['calls']) == 3


def test_plain_loader_never_opens_graph_or_builds_text_index(tmp_path):
    docs = knowledge().documents
    raw = json.dumps(docs, ensure_ascii=False).encode('utf-8')
    (tmp_path / 'text_documents.json').write_bytes(raw)
    manifest = {'status': 'ready', 'text_retrieval': 'full-context', 'corpus_sha256': digest(docs),
                'files': {'text_documents.json': hashlib.sha256(raw).hexdigest(),
                          'hypergraph.sqlite': 'intentionally absent'}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    lib = PlainTextKnowledge(tmp_path)
    assert SOURCE in lib.read_all()
    assert sorted(p.name for p in tmp_path.iterdir()) == ['manifest.json', 'text_documents.json']


def test_plain_loader_reads_all_original_source_parts_without_selecting_segments():
    knowledge = PlainTextKnowledge.from_knowledge(SimpleNamespace(documents=[
        {'title': 'Original source', 'sentences': ['First paragraph.', 'Second paragraph.']}]))
    assert 'First paragraph. Second paragraph.' in knowledge.read_all()
    assert len(knowledge.documents) == 1


def test_plain_roles_do_not_require_fields_source_ids_or_minimal_evidence():
    assert list(TEXT_ROLES) == ['planner', 'retriever', 'reasoner', 'summarizer']
    result = runtime().run('His job?')
    for call in result['calls']:
        prompt = call['messages'][0]['content']
        for obsolete in ('"take"', '"support"', '"missing"', 'source ids', 'minimal supporting facts'):
            assert obsolete not in prompt
