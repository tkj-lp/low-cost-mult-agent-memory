import json

import pytest

from competition_agents.field_collaboration import FieldCollaboration


def fact(key, subject, relation, value, *, title='Story', qualifiers=None):
    return {'context_id': key, 'title': title, 'text': subject + ' ' + relation + ' ' + value,
            'fact_fields': {'id': key, 'document_id': 'doc', 'subject': subject, 'relation': relation,
                            'object': value, 'polarity': 'positive', 'qualifiers': qualifiers or []}}


class Knowledge:
    documents, text_vectors, source_aliases = [], [], []

    def __init__(self):
        self.queries = []

    def search_graph(self, query, limit):
        self.queries.append(query)
        return [fact('color', 'Cotton', 'is a', 'little white kitten'),
                fact('background', 'farmer', 'stored', 'a large barrel of unrelated paint')]

    def search_text(self, *args):
        pytest.fail('协议链路不应读取原始文本知识库。')


class Client:
    def __init__(self, *, bad_source=None, retry=False, missing_only=False):
        self.calls, self.bad_source, self.retry, self.missing_only = [], bad_source, retry, missing_only

    def call(self, role, messages):
        payload = json.loads(messages[-1]['content'])
        fields = [['s1', 'Cotton', 'color', 'white']]
        if role == 'planner':
            content = {'task': 'What color was Cotton?', 'lookup': True, 'reuse': []}
        elif role == 'retriever':
            content = {'facts': fields}
            if self.bad_source == role:
                content['facts'][0][0] = 's99'
            if self.retry and not any(c['role'] == role for c in self.calls):
                content.update(missing=['fur pattern'], query='Cotton fur pattern')
            if self.missing_only:
                content = {'facts': [], 'missing': ['unknown value']}
        elif role == 'reasoner':
            content = {'answer': 'white', 'support': [row[0] for row in payload['facts']]}
            if self.bad_source == role:
                content['support'] = ['s99']
        else:
            content = 'Cotton was white.'
        raw = content if isinstance(content, str) else json.dumps(content)
        self.calls.append({'role': role, 'messages': messages, 'input_tokens': 1, 'output_tokens': 1,
                           'elapsed_ms': 1., 'request_bytes': len(json.dumps(messages)), 'response_bytes': len(raw)})
        return {'content': raw}


def test_native_fenced_array_and_source_wrappers_need_no_extra_model_call():
    class ArrayClient(Client):
        def call(self, role, messages):
            result = super().call(role, messages)
            if role == 'retriever':
                result['content'] = '```json\n[["[s1]","Cotton","color","white",[],"d1"]]\n```'
            return result
    result = FieldCollaboration(Knowledge(), ArrayClient()).run('What color was Cotton?')
    assert result['error'] is None
    assert len(result['calls']) == 4
    row = json.loads(result['messages'][1]['content'])['facts'][0]
    assert row[0] == 's1' and row[3] == 'little white kitten'
    assert row[5] == 'd1'


def test_named_document_alias_and_qualifier_list_preserve_source_validation():
    app = FieldCollaboration(Knowledge(), Client())
    app.evidence = {'s1': fact('first', 'Cotton', 'color', 'white',
                              qualifiers=[{'kind': 'condition', 'value': 'in winter'}])}
    rows = [{'id': '1', 'subject': 'Cotton', 'relation': 'color', 'value': 'white',
             'qualifiers': ['condition=in winter'], 'source_title': 'd1'}]
    normalized = app._normalize_fields('retriever', rows, [])['facts']
    assert normalized[0][0] == 's1'
    assert 'condition=in winter' in normalized[0][4]
    rows[0]['source_title'] = 'invented-document'
    with pytest.raises(ValueError, match='文档与所引用来源不一致'):
        app._normalize_fields('retriever', rows, [])


def test_retrieved_fields_are_forwarded_without_model_rewriting_or_full_text_expansion():
    library = Knowledge()
    app = FieldCollaboration(library, Client())
    result = app.run('What color was Cotton?')
    assert result['error'] is None
    assert result['implementation'] == 'native-field-collaboration-v4'
    assert len(result['calls']) == 4 and len(library.queries) == 1
    planner, retriever, reasoner = [json.loads(m['content']) for m in result['messages']]
    assert set(planner) == {'task', 'lookup'}
    assert set(retriever) == {'task', 'facts', 'sources'}
    assert set(reasoner) == {'task', 'answer', 'facts', 'sources'}
    assert [row[:4] for row in retriever['facts']] == [
        ['s1', 'Cotton', 'is a', 'little white kitten'],
        ['s2', 'farmer', 'stored', 'a large barrel of unrelated paint']]
    for message in result['messages'][1:]:
        assert 'little white kitten' in message['content']
        assert 'unrelated paint' in message['content']
        assert json.loads(message['content'])['sources'] == {'d1': 'Story'}
        assert 'conversation' not in message['content']
        assert message['content_bytes'] == len(message['content'].encode('utf-8'))
    assert result['calls'][2]['messages'][-1]['content'] == result['messages'][1]['content']
    assert result['calls'][3]['messages'][-1]['content'] == result['messages'][2]['content']
    assert 'facts' not in json.loads(result['outputs']['reasoner'])
    assert reasoner['facts'] == retriever['facts']
    assert {hit['context_id'] for hit in result['memory_written']} == {'color', 'background'}
    assert [hit['context_id'] for hit in result['memory_support_used']] == ['color', 'background']
    assert result['metrics']['memory_facts_added'] == 2


@pytest.mark.parametrize('role', ['retriever', 'reasoner'])
def test_fields_cannot_cite_sources_not_received_by_that_role(role):
    runtime = FieldCollaboration(Knowledge(), Client(bad_source=role))
    result = runtime.run('Color?')
    assert result['error'] is None
    assert all('s99' not in m['content'] for m in result['messages'])
    assert json.loads(result['messages'][-1]['content'])['facts'] == json.loads(result['messages'][1]['content'])['facts']
    assert {hit['context_id'] for hit in result['memory_written']} == {'color', 'background'}
    if role == 'reasoner':
        assert result['memory_support_used'] == []
    assert len(runtime.memory.episodes) == 1


def test_retrieval_interpretation_remains_in_audit_without_duplicating_source_facts():
    class FindingsClient(Client):
        def call(self, role, messages):
            result = super().call(role, messages)
            if role == 'retriever':
                result['content'] = json.dumps({'findings': 'Cotton has white fur.',
                    'facts': [['s1', 'Wrong person', 'has', 'blue fur']]})
            return result
    result = FieldCollaboration(Knowledge(), FindingsClient()).run('What color was Cotton?')
    assert result['error'] is None and len(result['calls']) == 4
    packets = [json.loads(m['content']) for m in result['messages'][1:]]
    assert all('retrieval_findings' not in packet for packet in packets)
    assert result['retrieval_decisions'][0]['findings'] == 'Cotton has white fur.'
    assert packets[0]['facts'] == packets[1]['facts']
    assert len(packets[0]['facts']) == 2
    assert all('Wrong person' not in m['content'] for m in result['messages'])


def test_compact_fields_keep_qualifiers_and_relation_chain_verbatim():
    app = FieldCollaboration(Knowledge(), Client())
    rows = [['s1', 'Ron', 'held position', 'Director', 'since=2001;until=2003'],
            ['s2', 'Director', 'department', 'Protocol Department']]
    assert app._facts(rows, {'s1', 's2'}) == rows
    projected = app._facts([['s1', 'Ron', 'NOT held position', 'Minister']], {'s1'})
    assert projected[0][2] == 'NOT held position'
    assert app._support(['s1', 's2'], rows) == rows
    assert app._support(['s2'], rows) == [rows[1]]


def test_reasoner_selection_does_not_delete_received_relations():
    class TwoRelations(Knowledge):
        def search_graph(self, query, limit):
            return [fact('publisher', 'River', 'released by', 'North Records',
                         qualifiers=[{'kind': 'time', 'value': '2001'}]),
                    fact('artist', 'River', 'album by', 'Mira')]

    class SelectOne(Client):
        def call(self, role, messages):
            response = super().call(role, messages)
            if role == 'retriever':
                response['content'] = json.dumps({'facts': [
                    ['s1', 'River', 'released by', 'North Records', 'time=2001'],
                    ['s2', 'River', 'album by', 'Mira']]})
            elif role == 'reasoner':
                response['content'] = json.dumps({'answer': 'Mira', 'support': ['s2']})
            return response

    runtime = FieldCollaboration(TwoRelations(), SelectOne())
    try:
        result = runtime.run('Who released River?')
        assert result['error'] is None
        received = json.loads(result['messages'][1]['content'])['facts']
        forwarded = json.loads(result['messages'][2]['content'])['facts']
        assert forwarded == received
        assert len(result['calls']) == 4
        # 保留的关系不是模型引用记录，不得据此虚增采用量。
        assert [hit['context_id'] for hit in result['memory_support_used']] == ['artist']
        assert runtime.memory.episodes[-1]['evidence_ids'] == ['artist']
    finally:
        runtime.memory.close()


@pytest.mark.parametrize('generated', [
    {'facts': [['s1', 'Singles', 'released from', 'the album']]},
    {'facts': [{'wrong': 'shape'}]},
    {'facts': []},
    {'notes': 'Use the supplied count and release relation.'},
    {'notes': {'relations': ['count and release'], 'ambiguities': []}},
    {'notes': ['count and release']},
])
def test_retriever_cannot_remove_counts_negation_qualifiers_or_competing_relations(generated):
    class Source(Knowledge):
        def search_graph(self, query, limit):
            one = fact('count', 'Six singles', 'were released from', 'the album',
                       title='River', qualifiers=[{'kind': 'time', 'value': '2001'}])
            two = fact('other', 'River', 'released by', 'South Records', title='River')
            two['fact_fields']['polarity'] = 'negative'
            return [one, two]

    class Rewrite(Client):
        def call(self, role, messages):
            result = super().call(role, messages)
            if role == 'retriever':
                result['content'] = json.dumps(generated)
            if role == 'reasoner':
                result['content'] = json.dumps({'answer': 'Six', 'support': ['s1']})
            return result

    runtime = FieldCollaboration(Source(), Rewrite())
    try:
        result = runtime.run('How many singles were released?')
        assert result['error'] is None and len(result['calls']) == 4
        rows = json.loads(result['messages'][1]['content'])['facts']
        assert rows[0][:4] == ['s1', 'Six singles', 'were released from', 'the album']
        assert 'time=2001' in rows[0][4] and rows[0][5] == 'd1'
        assert json.loads(result['messages'][1]['content'])['sources']['d1'] == 'River'
        assert rows[1][2] == 'NOT released by'
        assert json.loads(result['messages'][2]['content'])['facts'] == rows
        assert all('retrieval_notes' not in json.loads(m['content']) for m in result['messages'])
        assert [r['context_id'] for r in result['memory_support_used']] == ['count']
    finally:
        runtime.memory.close()


def test_malformed_retrieval_notes_do_not_block_original_fields_or_execute_guessed_query():
    class MalformedNotes(Client):
        def call(self, role, messages):
            response = super().call(role, messages)
            if role == 'retriever':
                response['content'] = '{"notes":{"relations":[{"Cotton", "is white"}]}, "query": "invented search"}'
            return response

    runtime = FieldCollaboration(Knowledge(), MalformedNotes())
    try:
        result = runtime.run('What color was Cotton?')
        assert result['error'] is None and len(result['calls']) == 4
        assert result['metrics']['knowledge_searches'] == 1
        sent = json.loads(result['messages'][1]['content'])
        assert sent['facts'][0][1:4] == ['Cotton', 'is a', 'little white kitten']
        assert 'retrieval_notes' not in sent
        assert any(change.get('action') == 'audit_unparsed_retrieval_notes_no_tool_command'
                   for parse in result['protocol_parsing'] for change in parse['normalizations'])
    finally:
        runtime.memory.close()


def test_bare_source_identifiers_are_repaired_without_regeneration_or_answer_changes():
    class BareIds(Client):
        def call(self, role, messages):
            response = super().call(role, messages)
            if role == 'reasoner':
                response['content'] = '{"answer":"s2 is printed on the box; not s1", "support":[s1,s2]}'
            return response
    app = FieldCollaboration(Knowledge(), BareIds())
    try:
        result = app.run('Which label?')
        assert result['error'] is None and len(result['calls']) == 4
        final = json.loads(result['messages'][-1]['content'])
        assert final['answer'] == 's2 is printed on the box; not s1'
        assert len(final['facts']) == 2
        assert len(result['memory_support_used']) == 2
    finally:
        app.memory.close()


@pytest.mark.parametrize('reuse', [['s999'], 's999', {'source': 's999'}])
def test_unused_planner_rewrite_and_advisory_ids_never_replace_question_or_block_sources(reuse):
    class WrongPlan(Client):
        def call(self, role, messages):
            response = super().call(role, messages)
            if role == 'planner':
                response['content'] = json.dumps({'task': 'Different question about someone else',
                                                  'lookup': True, 'reuse': reuse})
            return response
    app = FieldCollaboration(Knowledge(), WrongPlan())
    try:
        result = app.run('What color was Cotton?')
        assert result['error'] is None and len(result['calls']) == 4
        assert result['resolved_question'] == 'What color was Cotton?'
        assert all(json.loads(m['content'])['task'] == result['question'] for m in result['messages'])
        assert len(json.loads(result['messages'][-1]['content'])['facts']) == 2
        assert result['planning_decision']['reuse'] == reuse
    finally:
        app.memory.close()


def test_followup_object_and_previous_turn_reach_every_downstream_role(tmp_path):
    from test_relation_coverage import _compiled_graph
    from test_prepared_knowledge import _assertion, _focused_embed
    graph = _compiled_graph(tmp_path, [('River', ['River is a studio album. River was released by North Records.'], [
        _assertion('River', 'is', 'studio album'),
        _assertion('River', 'was released by', 'North Records')])])
    graph.prepare_focused_search(_focused_embed)
    graph.documents, graph.text_vectors = [], []
    class Followup(Client):
        def call(self, role, messages):
            response = super().call(role, messages)
            payload = json.loads(messages[-1]['content'])
            if role == 'planner':
                response['content'] = json.dumps({'task': payload['question'], 'lookup': True})
            elif role == 'summarizer':
                response['content'] = 'River'
            return response
    app = FieldCollaboration(graph, Followup())
    try:
        first = app.run('Which album is River?', history_scope='one')
        assert first['error'] is None
        second = app.run('Who released the album?', history_scope='one')
        assert second['error'] is None
        for call in second['calls'][1:]:
            received = json.loads(call['messages'][-1]['content'])
            assert received['question_referents'] == ['river']
            assert 'conversation_context' not in received
        next_scope = app.run('Who released River?', history_scope='different')
        assert all('conversation_context' not in json.loads(m['content']) for m in next_scope['messages'])
    finally:
        app.memory.close()


def test_numbering_new_sources_does_not_overwrite_sparse_memory_labels():
    app = FieldCollaboration(Knowledge(), Client())
    app.evidence = {'s1': fact('a', 'A', 'r', 'v'), 's3': fact('b', 'B', 'r', 'v')}
    app.metrics = {'knowledge_search_ms': 0., 'knowledge_searches': 0}
    app.searches = []
    app._search('Cotton')
    assert app.evidence['s3']['context_id'] == 'b'
    assert app.evidence['s4']['context_id'] == 'color'
    assert app.evidence['s5']['context_id'] == 'background'


@pytest.mark.parametrize('raw,expected', [('true', True), ('false', False), ('lookup=false', False),
    ('lookup: true', True), ('```json\nfalse\n```', False), ('"false"', False)])
def test_direct_planner_control_never_requires_another_model_call(raw, expected):
    class Direct(Client):
        def call(self, role, messages):
            response = super().call(role, messages)
            if role == 'planner':
                response['content'] = raw
            return response
    app = FieldCollaboration(Knowledge(), Direct())
    try:
        result = app.run('Color?')
        assert result['error'] is None and len(result['calls']) == 4
        assert result['planning_decision']['lookup'] is expected
        assert result['outputs']['planner'] == raw
    finally:
        app.memory.close()


@pytest.mark.parametrize('raw', ['"white"', '["white"]', 'white', '{"answer": "white", BROKEN}'])
def test_reasoner_format_does_not_trigger_regeneration_or_remove_source_fields(raw):
    class Direct(Client):
        def call(self, role, messages):
            response = super().call(role, messages)
            if role == 'reasoner':
                response['content'] = raw
            return response
    app = FieldCollaboration(Knowledge(), Direct())
    try:
        result = app.run('Color?')
        assert result['error'] is None and len(result['calls']) == 4
        before, after = [json.loads(m['content']) for m in result['messages'][1:]]
        assert before['facts'] == after['facts']
        if 'BROKEN' in raw:
            assert 'answer' not in after and after['candidate_status'] == 'missing'
        assert result['outputs']['reasoner'] == raw
    finally:
        app.memory.close()


def test_unread_related_source_is_fetched_despite_model_stop_suggestion(tmp_path):
    from test_relation_coverage import _compiled_graph
    from test_prepared_knowledge import _assertion, _focused_embed
    graph = _compiled_graph(tmp_path, [('River', ['River is an album. River was released by North Records.'], [
        _assertion('River', 'is', 'album'), _assertion('River', 'was released by', 'North Records')])])
    graph.documents, graph.text_vectors = [], []
    graph.prepare_focused_search(_focused_embed)
    class Stop(Client):
        def call(self, role, messages):
            response = super().call(role, messages)
            if role == 'planner':
                response['content'] = 'false'
            return response
    app = FieldCollaboration(graph, Stop())
    try:
        identity = next(h for h in graph.search_graph_focused('River') if h['fact_fields']['object'] == 'album')
        app.memory.remember('What is River?', 'An album', [identity], 'one', observed_evidence=[identity])
        result = app.run('Who released the album?', history_scope='one')
        assert result['error'] is None
        assert result['planning_decision']['lookup'] is False
        assert result['planning_control']['unread_related_sources'] > 0
        assert result['metrics']['knowledge_searches'] == 1
        assert any(row[3] == 'North Records' for row in json.loads(result['messages'][-1]['content'])['facts'])
    finally:
        app.memory.close()


def test_only_missing_evidence_can_trigger_an_extra_search():
    app = FieldCollaboration(Knowledge(), Client())
    app.task, app.searches = 'Color?', []
    assert app._gap({'query': 'redundant check'}) == ([], None, 'invalid_followup_keep_sources')
    result = FieldCollaboration(Knowledge(), Client(retry=True)).run('Color?')
    assert result['error'] is None
    assert result['metrics']['knowledge_searches'] == 2
    assert len(result['calls']) == 5


def test_memory_only_completion_does_not_require_generated_source_citations():
    class NoCitations(Client):
        def call(self, role, messages):
            response = super().call(role, messages)
            payload = json.loads(messages[-1]['content'])
            if role == 'planner':
                response['content'] = json.dumps({'lookup': not payload['memory_evidence']['facts']})
            elif role == 'reasoner':
                response['content'] = '{"answer":"white"}'
            return response
    app = FieldCollaboration(Knowledge(), NoCitations())
    try:
        assert app.run('What color was Cotton?', history_scope='one')['error'] is None
        second = app.run('What color was Cotton?', history_scope='one')
        assert second['error'] is None
        assert second['metrics']['knowledge_searches'] == 0
        assert second['metrics']['memory_evidence_used'] == 0
        assert second['metrics']['memory_only_tasks'] == 1
        assert all(h.get('from_memory') for h in second['source_evidence'].values())
    finally:
        app.memory.close()


def test_empty_evidence_cannot_be_declared_complete():
    class EmptyKnowledge(Knowledge):
        def search_graph(self, query, limit):
            return []
    class Empty(Client):
        def call(self, role, messages):
            response = super().call(role, messages)
            if role == 'retriever':
                response['content'] = '{"facts":[]}'
            return response
    result = FieldCollaboration(EmptyKnowledge(), Empty()).run('Color?')
    assert '空证据' in result['error']


def test_final_role_has_same_answer_requirement_as_plain_text():
    from competition_agents.field_collaboration import FIELD_ROLES
    from competition_agents.plain_text_collaboration import TEXT_ROLES
    assert FIELD_ROLES['summarizer'] == TEXT_ROLES['summarizer']


@pytest.mark.parametrize('empty', [None, '', []])
def test_empty_optional_query_does_not_reject_sufficient_fields(empty):
    app = FieldCollaboration(Knowledge(), Client())
    app.task, app.searches = 'Color?', [{'query': 'Color?'}]
    assert app._gap({'missing': [], 'query': empty}) == ([], None, 'complete')


@pytest.mark.parametrize('answer', [False, True, 0, 2.5, ['white'], ['white', 'black'],
                                    [{'name': 'Cotton', 'color': 'white'}]])
def test_native_boolean_and_numeric_answers_reach_summarizer_unchanged(answer):
    class Typed(Client):
        def call(self, role, messages):
            response = super().call(role, messages)
            if role == 'reasoner':
                response['content'] = json.dumps({'answer': answer, 'support': ['s1']})
            return response
    result = FieldCollaboration(Knowledge(), Typed()).run('Question?')
    assert result['error'] is None
    assert len(result['calls']) == 4
    received = json.loads(result['calls'][-1]['messages'][-1]['content'])['answer']
    assert received == answer and type(received) is type(answer)


@pytest.mark.parametrize('answer', [None, '', [], {}, float('nan')])
def test_invalid_answer_values_are_not_forwarded_as_candidates(answer):
    class Invalid(Client):
        def call(self, role, messages):
            response = super().call(role, messages)
            if role == 'reasoner':
                response['content'] = json.dumps({'answer': answer, 'support': ['s1']})
            return response
    result = FieldCollaboration(Knowledge(), Invalid()).run('Question?')
    assert result['error'] is None and len(result['calls']) == 4
    sent = json.loads(result['messages'][-1]['content'])
    assert 'answer' not in sent and sent['candidate_status'] == 'missing'
    assert sent['facts'] == json.loads(result['messages'][1]['content'])['facts']
    assert {hit['context_id'] for hit in result['memory_written']} == {'color', 'background'}
    assert result['memory_state']['conversation_committed'] is True


def test_observed_source_is_forwarded_stored_and_reused_later_without_regeneration():
    class ObservingClient(Client):
        turn = 0

        def call(self, role, messages):
            payload = json.loads(messages[-1]['content'])
            if role == 'planner':
                self.turn += 1
                entity = 'Cotton' if self.turn == 1 else 'farmer'
                reuse = [row[0] for row in payload['memory_evidence']['facts'] if row[1] == entity]
                content = {'task': 'Color?' if self.turn == 1 else 'What did the farmer store?',
                           'lookup': not reuse, 'reuse': reuse}
            elif role == 'retriever':
                entity = 'Cotton' if self.turn == 1 else 'farmer'
                content = {'facts': [row[:4] for row in payload['evidence']['facts'] if row[1] == entity]}
            elif role == 'reasoner':
                chosen = next(row for row in payload['facts'] if row[1] == ('Cotton' if self.turn == 1 else 'farmer'))
                content = {'answer': chosen[3], 'support': [chosen[0]]}
            else:
                content = payload['answer']
            raw = content if isinstance(content, str) else json.dumps(content)
            self.calls.append({'role': role, 'messages': messages, 'input_tokens': 1, 'output_tokens': 1,
                               'elapsed_ms': 1., 'request_bytes': len(json.dumps(messages)), 'response_bytes': len(raw)})
            return {'content': raw}

    knowledge = Knowledge()
    runtime = FieldCollaboration(knowledge, ObservingClient())
    try:
        first = runtime.run('Color?', history_scope='same')
        assert first['error'] is None
        assert all('paint' in m['content'] for m in first['messages'][1:])
        assert first['memory_state']['fact_count'] == 2
        second = runtime.run('What did the farmer store?', history_scope='same')
        assert second['error'] is None
        assert 'paint' in second['prediction']
        assert len(knowledge.queries) == 1
        assert second['metrics']['knowledge_searches'] == 0
        assert second['metrics']['memory_evidence_used'] == 1
        assert second['metrics']['memory_only_tasks'] == 1
        assert second['metrics']['memory_facts_added'] == 0
        assert len(second['calls']) == 4
        assert runtime.memory.episodes[0]['evidence_ids'] == ['color']
        assert set(runtime.memory.episodes[0]['observed_evidence_ids']) == {'color', 'background'}
    finally:
        runtime.memory.close()


def test_repeated_search_does_not_duplicate_source_memory():
    runtime = FieldCollaboration(Knowledge(), Client(retry=True))
    try:
        result = runtime.run('Color?')
        assert result['error'] is None
        assert result['metrics']['knowledge_searches'] == 2
        assert result['memory_state']['fact_count'] == 2
        assert result['metrics']['memory_facts_added'] == 2
        assert runtime.memory.db.execute('SELECT count(*) FROM facts').fetchone()[0] == 2
    finally:
        runtime.memory.close()


@pytest.mark.parametrize('literal', ['true', 'false', 'False', ' false '])
def test_lookup_literal_conversion_keeps_raw_output_and_uses_no_extra_model_call(literal):
    class StringBoolean(Client):
        def call(self, role, messages):
            response = super().call(role, messages)
            if role == 'planner':
                value = json.loads(response['content'])
                value['lookup'] = literal
                response['content'] = json.dumps(value)
            return response

    runtime = FieldCollaboration(Knowledge(), StringBoolean())
    try:
        result = runtime.run('Color?')
        assert result['error'] is None
        assert len(result['calls']) == 4
        assert json.loads(result['outputs']['planner'])['lookup'] == literal
        assert result['planning_decision']['lookup'] is (literal.strip().lower() == 'true')
        # 原有无证据首查规则保持不变，类型兼容不覆盖模型的原始决定。
        assert result['planning_control']['lookup'] is True
        assert result['protocol_parsing'][0]['normalizations'] == [
            {'field': 'lookup', 'before': literal, 'after': literal.strip().lower() == 'true'}]
        assert result['metrics']['protocol_parse_ms'] >= result['metrics']['field_normalization_ms'] >= 0
    finally:
        runtime.memory.close()


@pytest.mark.parametrize('invalid', ['yes', '', 0, 1, None, [], {}])
def test_unknown_lookup_controls_are_audited_and_retrieve_instead_of_discarding_sources(invalid):
    class InvalidBoolean(Client):
        def call(self, role, messages):
            response = super().call(role, messages)
            if role == 'planner':
                value = json.loads(response['content'])
                value['lookup'] = invalid
                response['content'] = json.dumps(value)
            return response

    runtime = FieldCollaboration(Knowledge(), InvalidBoolean())
    try:
        result = runtime.run('Color?')
        assert result['error'] is None and len(result['calls']) == 4
        assert result['planning_control']['lookup'] is True
        assert result['protocol_parsing'][0]['normalizations'][0]['action'] == 'unparsed_control_conservative_lookup'
        assert len(result['source_evidence']) == 2
    finally:
        runtime.memory.close()


@pytest.mark.parametrize('raw', [
    [{'id': '1', 'subject': 'Cotton', 'relation': 'color', 'value': 'white'}],
    [['s1', 'Cotton', 'color', 'white']],
    [['s1', 'Cotton', 'color', 'white', None, 'd1']],
    {'facts': [{'id': 's1', 'subject': 'Cotton', 'relation': 'color', 'value': 'white'}]},
])
def test_equivalent_fact_shapes_reach_next_role_with_provenance_and_no_extra_call(raw):
    class CompatibleFacts(Client):
        def call(self, role, messages):
            response = super().call(role, messages)
            if role == 'retriever':
                response['content'] = json.dumps(raw)
            return response

    runtime = FieldCollaboration(Knowledge(), CompatibleFacts())
    try:
        result = runtime.run('Color?')
        assert result['error'] is None
        assert len(result['calls']) == 4
        assert json.loads(result['outputs']['retriever']) == raw
        row = json.loads(result['calls'][2]['messages'][-1]['content'])['facts'][0]
        assert row[:4] == ['s1', 'Cotton', 'is a', 'little white kitten']
        if isinstance(raw, list) and isinstance(raw[0], list) and len(raw[0]) == 6:
            assert row[5] == 'd1'
            assert json.loads(result['calls'][2]['messages'][-1]['content'])['sources']['d1'] == 'Story'
    finally:
        runtime.memory.close()


def test_scalar_fact_values_preserve_false_and_zero_without_guessing_negation():
    runtime = FieldCollaboration(Knowledge(), Client())
    runtime.evidence = {}
    changes = []
    value = runtime._normalize_fields('retriever', [
        {'id': 's1', 'subject': 'Cotton', 'relation': 'lived alone', 'value': False},
        ['s2', 'Cotton', 'count', 0]], changes)
    assert value['facts'] == [['s1', 'Cotton', 'lived alone', 'false'], ['s2', 'Cotton', 'count', '0']]


def test_fact_compatibility_rejects_unknown_keys_and_mismatched_document_sources():
    runtime = FieldCollaboration(Knowledge(), Client())
    runtime.evidence = {'s1': fact('color', 'Cotton', 'color', 'white')}
    with pytest.raises(ValueError, match='文档与所引用来源不一致'):
        runtime._normalize_fields('retriever', [['s1', 'Cotton', 'color', 'white', '', 'd99']], [])
    with pytest.raises(ValueError, match='无法归一化'):
        runtime._normalize_fields('retriever', [{'id': 's1', 'subject': 'Cotton',
            'relation': 'color', 'value': 'white', 'negated': True}], [])


def test_support_row_conversion_preserves_received_qualifiers_and_rejects_changed_facts():
    runtime = FieldCollaboration(Knowledge(), Client())
    received = [['s1', 'Cotton', 'color', 'white', 'time=before painting']]
    changes = []
    decision = runtime._normalize_fields('reasoner', {'answer': 'white', 'support': [received[0][:4]]},
                                         changes, {'facts': received})
    assert decision == {'answer': 'white', 'support': ['s1']}
    assert runtime._support(decision['support'], received) == received
    with pytest.raises(ValueError, match='与已收到的字段不一致'):
        runtime._normalize_fields('reasoner', {'answer': 'orange', 'support': [
            ['s1', 'Cotton', 'color', 'orange']]}, [], {'facts': received})


def test_document_id_support_requires_unique_unchanged_received_fact():
    runtime = FieldCollaboration(Knowledge(), Client())
    received = [['s1', 'Cotton', 'color', 'white', 'document=d1; time=before painting']]
    raw = {'answer': 'white', 'support': [['d1', 'Cotton', 'color', 'white']]}
    value = runtime._normalize_fields('reasoner', raw, [], {'facts': received})
    assert value['support'] == ['s1']
    assert runtime._support(value['support'], received) == received
    for bad in (['d99', 'Cotton', 'color', 'white'], ['d1', 'white', 'color', 'Cotton']):
        with pytest.raises(ValueError, match='与已收到的字段不一致'):
            runtime._normalize_fields('reasoner', {'support': [bad]}, [], {'facts': received})
    with pytest.raises(ValueError, match='与已收到的字段不一致'):
        runtime._normalize_fields('reasoner', raw, [], {'facts': received + [['s2', *received[0][1:]]]})
