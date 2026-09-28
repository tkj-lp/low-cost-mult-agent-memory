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
                content['support'] = ['s2']
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
    assert row[0] == 's1' and row[3] == 'white'
    assert 'document=d1' in row[4]


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


def test_native_projection_is_transmitted_without_expanding_original_records_again():
    library = Knowledge()
    app = FieldCollaboration(library, Client())
    result = app.run('What color was Cotton?')
    assert result['error'] is None
    assert result['implementation'] == 'native-field-collaboration-v4'
    assert len(result['calls']) == 4 and len(library.queries) == 1
    planner, retriever, reasoner = [json.loads(m['content']) for m in result['messages']]
    assert set(planner) == {'task', 'lookup'}
    assert set(retriever) == {'task', 'facts'}
    assert set(reasoner) == {'task', 'answer', 'facts'}
    assert retriever['facts'] == [['s1', 'Cotton', 'color', 'white']]
    for message in result['messages'][1:]:
        assert 'little white kitten' not in message['content']
        assert 'unrelated paint' not in message['content']
        assert 'sources' not in message['content']
        assert 'conversation' not in message['content']
        assert message['content_bytes'] == len(message['content'].encode('utf-8'))
    assert result['calls'][2]['messages'][-1]['content'] == result['messages'][1]['content']
    assert result['calls'][3]['messages'][-1]['content'] == result['messages'][2]['content']
    assert 'facts' not in json.loads(result['outputs']['reasoner'])
    assert reasoner['facts'] == retriever['facts']
    assert {hit['context_id'] for hit in result['memory_written']} == {'color', 'background'}
    assert [hit['context_id'] for hit in result['memory_support_used']] == ['color']
    assert result['metrics']['memory_facts_added'] == 2


@pytest.mark.parametrize('role', ['retriever', 'reasoner'])
def test_fields_cannot_cite_sources_not_received_by_that_role(role):
    runtime = FieldCollaboration(Knowledge(), Client(bad_source=role))
    result = runtime.run('Color?')
    assert '未提供的来源' in result['error']
    assert {hit['context_id'] for hit in result['memory_written']} == {'color', 'background'}
    assert result['memory_support_used'] == []
    assert runtime.memory.episodes == []


def test_compact_fields_keep_qualifiers_and_relation_chain_verbatim():
    app = FieldCollaboration(Knowledge(), Client())
    rows = [['s1', 'Ron', 'held position', 'Director', 'since=2001;until=2003'],
            ['s2', 'Director', 'department', 'Protocol Department']]
    assert app._facts(rows, {'s1', 's2'}) == rows
    projected = app._facts([['s1', 'Ron', 'NOT held position', 'Minister']], {'s1'})
    assert projected[0][2] == 'NOT held position'
    assert app._support(['s1', 's2'], rows) == rows
    assert app._support(['s2'], rows) == [rows[1]]


def test_numbering_new_sources_does_not_overwrite_sparse_memory_labels():
    app = FieldCollaboration(Knowledge(), Client())
    app.evidence = {'s1': fact('a', 'A', 'r', 'v'), 's3': fact('b', 'B', 'r', 'v')}
    app.metrics = {'knowledge_search_ms': 0., 'knowledge_searches': 0}
    app.searches = []
    app._search('Cotton')
    assert app.evidence['s3']['context_id'] == 'b'
    assert app.evidence['s4']['context_id'] == 'color'
    assert app.evidence['s5']['context_id'] == 'background'


def test_only_missing_evidence_can_trigger_an_extra_search():
    app = FieldCollaboration(Knowledge(), Client())
    app.task, app.searches = 'Color?', []
    with pytest.raises(ValueError, match='没有证据缺口'):
        app._gap({'query': 'redundant check'})
    result = FieldCollaboration(Knowledge(), Client(retry=True)).run('Color?')
    assert result['error'] is None
    assert result['metrics']['knowledge_searches'] == 2
    assert len(result['calls']) == 5


def test_empty_evidence_cannot_be_declared_complete():
    class Empty(Client):
        def call(self, role, messages):
            response = super().call(role, messages)
            if role == 'retriever':
                response['content'] = '{"facts":[]}'
            return response
    result = FieldCollaboration(Knowledge(), Empty()).run('Color?')
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


@pytest.mark.parametrize('answer', [False, True, 0, 2.5])
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
    assert '缺少有效答案候选' in result['error']
    assert {hit['context_id'] for hit in result['memory_written']} == {'color', 'background'}
    assert result['memory_state']['conversation_committed'] is False


def test_unselected_source_is_remembered_without_being_transmitted_and_reused_later():
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
                content = {'answer': payload['facts'][0][3], 'support': [payload['facts'][0][0]]}
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
        assert all('paint' not in m['content'] for m in first['messages'])
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


@pytest.mark.parametrize('literal', ['true', 'false'])
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
        assert result['planning_decision']['lookup'] is (literal == 'true')
        # 原有无证据首查规则保持不变，类型兼容不覆盖模型的原始决定。
        assert result['planning_control']['lookup'] is True
        assert result['protocol_parsing'][0]['normalizations'] == [
            {'field': 'lookup', 'before': literal, 'after': literal == 'true'}]
        assert result['metrics']['protocol_parse_ms'] >= result['metrics']['field_normalization_ms'] >= 0
    finally:
        runtime.memory.close()


@pytest.mark.parametrize('invalid', ['False', ' false ', 'yes', '', 0, 1, None, [], {}])
def test_lookup_compatibility_does_not_guess_other_types_or_spellings(invalid):
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
        assert '必须为布尔值' in result['error']
        assert len(result['calls']) == 1
        assert result['protocol_parsing'][0]['normalizations'] == []
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
        assert row[:4] == ['s1', 'Cotton', 'color', 'white']
        if isinstance(raw, list) and isinstance(raw[0], list) and len(raw[0]) == 6:
            assert 'document=d1' in row[4] and 'title=Story' in row[4]
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
