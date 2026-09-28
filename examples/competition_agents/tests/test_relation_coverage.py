import json

import pytest

from hypergraph_memory_method.source_context import bind_source_parts
from tools.prepare_ranked_continuous import source_sections
from test_prepared_knowledge import _compiled_graph as compile_graph, _assertion, _focused_embed


def _compiled_graph(path, rows):
    graph = compile_graph(path, rows)
    graph._prepare_graph_runtime()
    return graph


def test_owned_parts_remain_reachable_without_merging_different_owners(tmp_path):
    graph = _compiled_graph(tmp_path, [('Nara', [
        'Nara has a tail. The tail is white. Milo has a tail. His tail is black.'
    ], [
        _assertion('Nara', 'has', 'a tail', qualifiers=[{'kind': 'object_owner', 'value': 'Nara'}]),
        _assertion('The tail', 'color', 'white', qualifiers=[{'kind': 'subject_owner', 'value': 'Nara'}]),
        _assertion('Milo', 'has', 'a tail', qualifiers=[{'kind': 'object_owner', 'value': 'Milo'}]),
        _assertion('His tail', 'color', 'black', qualifiers=[{'kind': 'subject_owner', 'value': 'Milo'}]),
    ])])
    graph.prepare_focused_search(_focused_embed)
    rows = graph.search_graph_focused('What color is Nara?')
    assert {r['fact_fields']['object'] for r in rows} == {'white'}
    assert all(r['fact_fields']['qualifiers'][0]['value'] == 'Nara' for r in rows)


def test_related_identity_memory_is_not_complete_appearance_evidence(tmp_path):
    graph = _compiled_graph(tmp_path, [('Nara', ['Nara is a small antelope. Nara features a long neck. Nara has brown fur.'], [
        _assertion('Nara', 'is a', 'small antelope'),
        _assertion('Nara', 'features', 'a long neck'),
        _assertion('Nara', 'has', 'brown fur'),
    ])])
    plan = graph.relation_index.plan('What does Nara look like?')
    first = [k for k, row in graph._graph_rows.items() if row[3]['relation'] == 'is a']
    assert not graph.relation_index.certificate(plan, first)['sufficient']
    complete = graph.relation_index.certificate(plan, plan['required'])
    assert complete['sufficient'] and complete['required_count'] == 3


def test_unresolved_part_prevents_falsely_certifying_graph_completeness(tmp_path):
    graph = _compiled_graph(tmp_path, [('Nara', ['Nara is a small antelope. The coat is brown.'], [
        _assertion('Nara', 'is a', 'small antelope'), _assertion('The coat', 'color', 'brown')
    ])])
    plan = graph.relation_index.plan('What does Nara look like?')
    certificate = graph.relation_index.certificate(plan, plan['required'])
    assert not certificate['sufficient']
    assert certificate['unbound_source_ids']


def test_relation_directory_deduplicates_same_fact_but_keeps_qualifications(tmp_path):
    graph = _compiled_graph(tmp_path, [
        ('Nara', ['Nara features a long neck.'], [_assertion('Nara', 'features', 'a long neck')]),
        ('Nara [SEP] Description', ['Nara features a long neck. Nara has brown fur.'], [
            _assertion('Nara', 'features', 'a long neck'), _assertion('Nara', 'has', 'brown fur')]),
    ])
    plan = graph.relation_index.plan('What does Nara look like?')
    assert len(plan['terminals']) == 2


def test_event_time_cannot_be_replaced_by_job_relation(tmp_path):
    graph = _compiled_graph(tmp_path, [('Ronan', ['Ronan held the Director post in 2001.'], [
        _assertion('Ronan', 'held position', 'Director'),
        _assertion('Ronan', 'held position', 'Director', qualifiers=[{'kind': 'time', 'value': '2001'}]),
    ])])
    plan = graph.relation_index.plan('When did Ronan hold the Director post?')
    incomplete = [k for k, row in graph._graph_rows.items() if not row[3]['qualifiers']]
    assert not graph.relation_index.certificate(plan, incomplete)['sufficient']
    assert graph.relation_index.certificate(plan, plan['required'])['sufficient']


def test_primary_work_authorship_links_are_source_bound_and_role_specific():
    document = {'id': 'd', 'title': 'Silver Road (musical)',
                'sentences': ['Silver Road is a musical with music and lyrics by Lena and book by Omar.']}
    result = {'facts': [{'subject': 'music and lyrics', 'relation': 'by', 'object': 'Lena', 'qualifiers': []},
                        {'subject': 'book', 'relation': 'by', 'object': 'Omar', 'qualifiers': []},
                        {'subject': 'unrelated tail', 'relation': 'color', 'object': 'white', 'qualifiers': []}]}
    enriched = bind_source_parts(document, result)
    assert all(f['qualifiers'] == [{'kind': 'subject_owner', 'value': 'Silver Road (musical)'}]
               for f in enriched['facts'][:2])
    assert enriched['facts'][2]['qualifiers'] == []


def test_source_import_keeps_complete_section_and_excludes_page_navigation():
    html = '''<div id="mw-content-text"><div class="mw-parser-output"><table><tr><td><p>not source prose</p></td></tr></table>
    <p>Full opening, including background.<sup class="reference">[1]</sup></p>
    <h2>History</h2><p>Early history.</p><h3>Later</h3><p>Later history.</p><h2>References</h2><p>Citation.</p></div></div>'''
    sections = source_sections(html)
    assert sections['Introduction'] == 'Full opening, including background.'
    assert sections['History'] == 'Early history.\n\nLater history.'
    assert 'not source prose' not in json.dumps(sections)


def test_selected_runtime_questions_are_complete_ten_turns_of_screenshot_file():
    from competition_agents.methods.catalog import BASE, CATALOG, DATASETS
    dataset = DATASETS[CATALOG['defaultDataset']]
    original = json.loads((BASE/'data/official_factoid_ranked_topiocqa_003_10turns.json').read_text(encoding='utf-8'))[0]
    runtime = json.loads((BASE/dataset['file']).read_text(encoding='utf-8'))
    assert [q['question'] for q in runtime] == [q['question'] for q in original['tasks']]
    assert len(runtime) == dataset['taskCount'] == 10
    assert [q['turn_id'] for q in runtime] == list(range(1, 11))
    assert all(set(q) == {'id', 'group_id', 'turn_id', 'question'} for q in runtime)


def test_model_no_lookup_is_overridden_for_missing_relation_without_an_extra_model_call(tmp_path):
    from competition_agents.field_collaboration import FieldCollaboration
    graph = _compiled_graph(tmp_path, [('Nara', ['Nara is a small antelope. Nara features a long neck.'], [
        _assertion('Nara', 'is a', 'small antelope'), _assertion('Nara', 'features', 'a long neck')
    ])])
    graph.documents, graph.text_vectors = [], []
    graph.prepare_focused_search(_focused_embed)
    class Client:
        def __init__(self):
            self.calls = []

        def call(self, role, messages):
            payload = json.loads(messages[-1]['content'])
            if role == 'planner':
                answer = {'task': 'What does Nara look like?', 'lookup': False, 'reuse': ['s1']}
            elif role == 'retriever':
                answer = {'facts': [payload['evidence']['facts'][0][:4]]}
            elif role == 'reasoner':
                answer = {'answer': 'Nara is small and has a long neck.', 'support': [payload['facts'][0][0]]}
            else:
                answer = 'Nara is small and has a long neck.'
            self.calls.append({'role': role, 'input_tokens': 1, 'output_tokens': 1, 'elapsed_ms': 1,
                               'request_bytes': 1, 'response_bytes': 1})
            return {'content': answer if isinstance(answer, str) else json.dumps(answer)}
    runtime = FieldCollaboration(graph, Client())
    first = [h for h in graph.search_graph_focused('What is Nara?') if h['fact_fields']['relation'] == 'is a']
    runtime.memory.remember('What is Nara?', 'small antelope', first, 'same')
    result = runtime.run('What does it look like?', history_scope='same')
    assert result['error'] is None
    assert result['planning_control']['model_lookup'] is False
    assert result['planning_control']['lookup'] is True
    assert result['metrics']['knowledge_searches'] == 1
    assert len(result['calls']) == 4
    assert result['evidence_coverage']['sufficient']
    assert result['metrics']['required_fields_restored'] > 0
    delivered = json.loads(result['messages'][-1]['content'])['facts']
    assert any('long neck' in row[3] for row in delivered)
    runtime.memory.close()


def test_latent_high_similarity_does_not_hide_missing_relation(tmp_path):
    from types import SimpleNamespace
    import torch
    from competition_agents.latent_hidden.calibrated_graph_tool import CalibratedGraphTool
    graph = _compiled_graph(tmp_path, [('Nara', ['Nara is a small antelope. Nara features a long neck.'], [
        _assertion('Nara', 'is a', 'small antelope'), _assertion('Nara', 'features', 'a long neck')
    ])])
    class Teacher:
        def encode(self, texts, **kwargs):
            return [[1., 0., 0., 0.] for _ in texts]
    tool = CalibratedGraphTool(graph, SimpleNamespace(torch=torch, device='cpu'), Teacher())
    tool.thresholds = {k: .1 for k in tool.thresholds}
    tool.require_topic_alignment = False
    tool.begin_question('What is Nara?', 'same')
    query = torch.tensor([[1., 0., 0., 0.]])
    tool(query)
    tool.begin_question('What does Nara look like?', 'same')
    result = tool(query)
    assert result['audit']['memory_best_score'] >= .9
    assert not result['audit']['memory_sufficient']
    assert result['audit']['knowledge_searches'] == 1
    assert result['audit']['returned_coverage']['sufficient']
    tool.begin_question('What does Nara look like?', 'same')
    repeated = tool(query)
    assert repeated['audit']['knowledge_searches'] == 0
    assert repeated['audit']['fresh_source_ids'] == []


def test_followup_entity_comes_from_actual_answer_and_its_support_not_gold(tmp_path):
    from competition_agents.field_collaboration import FieldCollaboration
    graph = _compiled_graph(tmp_path, [('River Dance', ['River Dance has music by Lena Morris. Lena Morris is a singer. Omar Vale is a singer.'], [
        _assertion('River Dance', 'music by', 'Lena Morris'),
        _assertion('Lena Morris', 'is a', 'singer'), _assertion('Omar Vale', 'is a', 'singer'),
    ])])
    graph.documents, graph.text_vectors = [], []
    graph.prepare_focused_search(_focused_embed)
    runtime = FieldCollaboration(graph, None)
    hit = next(h for h in graph.search_graph_focused('Who composed River Dance?') if h['fact_fields']['object'] == 'Lena Morris')
    runtime.memory.remember('Who composed River Dance?', 'Lena Morris', [hit], 'one')
    runtime.task = 'Who is the singer?'
    runtime.metrics = {'memory_reference_bindings_used': 0}
    runtime._bind_followup(runtime.memory.history('one'), 'one')
    assert runtime.reference_binding['entities'] == ['lena morris']
    assert runtime.metrics['memory_reference_bindings_used'] == 1
    assert runtime.task == 'Who is the singer?'
    assert runtime.reference_roots == {'entity:lena morris'}
    hits = graph.search_graph_focused(runtime.task, roots=runtime.reference_roots)
    assert {h['fact_fields']['subject'] for h in hits} == {'Lena Morris'}
    # 同一问法下，历史回答若指向没有来源支持的另一个人，不得沿用这个人。
    runtime.memory.episodes[-1]['answer'] = 'Omar Vale'
    runtime.task = 'Who is the singer?'
    runtime._bind_followup(runtime.memory.history('one'), 'one')
    assert runtime.reference_binding is None or 'omar vale' not in runtime.reference_binding['entities']
    assert runtime.task == 'Who is the singer?'
    runtime.memory.close()


def test_bad_generated_references_are_quarantined_only_with_complete_received_sources(tmp_path):
    from competition_agents.field_collaboration import FieldCollaboration
    graph = _compiled_graph(tmp_path, [('Silver Road (musical)', ['Silver Road tells the story of Mira Vale.'], [
        _assertion('Silver Road (musical)', 'tells the story of', 'Mira Vale')
    ])])
    graph.documents, graph.text_vectors = [], []
    graph.prepare_focused_search(_focused_embed)

    class Client:
        def __init__(self):
            self.calls = []

        def call(self, role, messages):
            payload = json.loads(messages[-1]['content'])
            if role == 'planner':
                value = {'task': 'Whose story does Silver Road (musical) tell?', 'lookup': True, 'reuse': []}
            elif role == 'retriever':
                good = [row[:4] for row in payload['evidence']['facts']]
                value = {'facts': [['unknown-source', 'Wrong person', 'owns', 'Other work'], *good]}
            elif role == 'reasoner':
                source, subject, relation, target, *_ = payload['facts'][0]
                value = {'answer': target, 'support': [[source, target, relation, subject]]}
            else:
                assert all(row[1:4] == ['Silver Road (musical)', 'tells the story of', 'Mira Vale'] for row in payload['facts'])
                value = payload['answer']
            self.calls.append({'role': role, 'input_tokens': 1, 'output_tokens': 1, 'elapsed_ms': 1,
                               'request_bytes': 1, 'response_bytes': 1})
            return {'content': value if isinstance(value, str) else json.dumps(value)}

    runtime = FieldCollaboration(graph, Client())
    result = runtime.run('Whose story does Silver Road (musical) tell?')
    assert result['error'] is None and result['prediction'] == 'Mira Vale'
    assert len(result['calls']) == 4
    assert result['metrics']['protocol_reference_recoveries'] == 2
    assert [r['role'] for r in result['protocol_rejections']] == ['retriever', 'reasoner']
    assert all('Wrong person' not in message['content'] for message in result['messages'])
    assert json.loads(result['messages'][-1]['content'])['facts'][0][1:4] == [
        'Silver Road (musical)', 'tells the story of', 'Mira Vale']
    runtime.evidence.clear()
    with pytest.raises(ValueError, match='missing'):
        runtime._reject_reference('reasoner', ['unknown-source'], ValueError('missing'))
    runtime.memory.close()
