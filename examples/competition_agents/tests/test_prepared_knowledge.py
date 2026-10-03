import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from hypergraph_memory_method.prepared_knowledge import (
    SCHEMA, PreparedKnowledge, digest, public_documents, render_fact,
    validate_extraction, write_graph,
)


@pytest.fixture
def library(tmp_path):
    docs = public_documents([{'context': [
        {'title': 'Orion Clock', 'sentences': ['Orion Clock was designed by Mira Vale.']},
        {'title': 'Mira Vale', 'sentences': ['Mira Vale was born in Cedarport.']},
    ]}])
    facts = []
    for doc in docs:
        subject, relation, value = ('Orion Clock', 'designed by', 'Mira Vale') if doc['title'] == 'Orion Clock' else ('Mira Vale', 'born in', 'Cedarport')
        facts += validate_extraction(doc, {'facts': [{'subject': subject, 'relation': relation, 'object': value,
                                                   'polarity': 'positive', 'qualifiers': [], 'sentences': [0]}]})
    (tmp_path / 'text_documents.json').write_text(json.dumps(docs), encoding='utf-8')
    (tmp_path / 'facts.json').write_text(json.dumps(facts), encoding='utf-8')
    np.save(tmp_path / 'text_vectors.npy', np.eye(2, dtype=np.float32))
    write_graph(tmp_path / 'hypergraph.sqlite', docs, facts)
    files = ['text_documents.json', 'text_vectors.npy', 'facts.json', 'hypergraph.sqlite']
    (tmp_path / 'manifest.json').write_text(json.dumps({'schema': SCHEMA, 'status': 'ready', 'corpus_sha256': digest(docs),
        'embedding_model': 'test-model', 'embedding_provider': 'test',
        'files': {f: hashlib.sha256((tmp_path / f).read_bytes()).hexdigest() for f in files}}), encoding='utf-8')
    return tmp_path


def test_import_ignores_evaluation_labels_and_deduplicates():
    doc = {'title': 'A', 'sentences': ['A is a station.']}
    a = public_documents([{'question': 'SECRET', 'answer': 'SECRET', 'context': [doc, doc]}])
    b = public_documents([{'question': 'OTHER', 'supporting_facts': ['OTHER'], 'context': [doc]}])
    assert a == b
    assert len(a) == 1
    assert 'SECRET' not in json.dumps(a)


def test_same_title_different_source_versions_are_not_merged():
    assert len(public_documents([{'context': [{'title': 'A', 'sentences': ['old']}, {'title': 'A', 'sentences': ['new']}]}])) == 2


def test_both_backends_work_before_any_task_and_graph_returns_fields(library):
    db = PreparedKnowledge(library)
    text = db.search_text('clock', lambda q: SimpleNamespace(vector=[1., 0.], provider='test', model='test-model'))
    graph = db.search_graph('Where was the designer of Orion Clock born?', 4)
    assert text and graph
    assert any(h['fact_fields']['object'] == 'Cedarport' for h in graph)
    assert all(' | ' in h['text'] for h in graph)
    assert all(h['retrieval_backend'] == 'prepared_hypergraph' for h in graph)
    assert any('was ' in h['text'] for h in text)


def test_loaded_graph_does_not_parse_facts_again_or_expose_mutable_index(library, monkeypatch):
    from hypergraph_memory_method import prepared_knowledge
    graph = PreparedKnowledge(library)
    expected = graph.search_graph('Where was the designer of Orion Clock born?', 4)
    monkeypatch.setattr(prepared_knowledge.json, 'loads', lambda *args, **kwargs: pytest.fail('query parsed stored facts again'))
    hits = graph.search_graph('Where was the designer of Orion Clock born?', 4)
    assert hits == expected
    hits[0]['fact_fields']['object'] = 'POISON'
    hits[0]['fact_fields']['qualifiers'].append({'kind': 'time', 'value': 'POISON'})
    assert graph.search_graph('Where was the designer of Orion Clock born?', 4) == expected


def test_adapters_ignore_per_task_context_and_never_build(library, monkeypatch):
    from hypergraph_memory_method import context, text_memory
    monkeypatch.setenv('PREPARED_KNOWLEDGE_DIR', str(library))
    monkeypatch.setattr(context, '_build_context_hypergraph', lambda *a: pytest.fail('online construction'))
    monkeypatch.setattr(text_memory, '_context_units', lambda *a: pytest.fail('per-task text access'))
    task = {'context': [{'title': 'POISON', 'sentences': ['fake']}], 'question': 'Orion Clock'}
    assert context.search_context(task)
    assert text_memory.search_semantic_context(task, 'Orion Clock', embed=lambda q: SimpleNamespace(vector=[1., 0.], provider='test', model='test-model'))


def test_missing_or_incomplete_library_does_not_fall_back(tmp_path, monkeypatch):
    from hypergraph_memory_method.context import search_context
    monkeypatch.setenv('PREPARED_KNOWLEDGE_DIR', str(tmp_path))
    with pytest.raises(FileNotFoundError):
        search_context({'context': [], 'question': 'test'})
    (tmp_path / 'manifest.json').write_text(json.dumps({'schema': SCHEMA, 'status': 'incomplete'}))
    with pytest.raises(ValueError, match='禁止在线回退'):
        PreparedKnowledge(tmp_path)


def test_modified_artifact_is_rejected(library):
    (library / 'facts.json').write_text('[]')
    with pytest.raises(ValueError, match='校验失败'):
        PreparedKnowledge(library)


def test_embedding_model_mismatch_rejected(library):
    with pytest.raises(ValueError, match='模型不一致'):
        PreparedKnowledge(library).search_text('clock', lambda q: SimpleNamespace(vector=[1., 0.], provider='test', model='wrong'))


def test_role_independent_protocol_entry_has_no_text_dependency(library, monkeypatch):
    from hypergraph_memory_method.prepared_knowledge import search_prepared_knowledge
    monkeypatch.setenv('PREPARED_KNOWLEDGE_DIR', str(library))
    monkeypatch.setattr(PreparedKnowledge, 'search_text', lambda *a: pytest.fail('protocol read text backend'))
    assert search_prepared_knowledge('Orion Clock', mode='protocol')


def test_missing_sentence_or_ungrounded_entity_is_rejected():
    doc = public_documents([{'context': [{'title': 'A', 'sentences': ['A visits B.', 'A visits C.']}]}])[0]
    fact = {'subject': 'A', 'relation': 'visits', 'object': 'B', 'polarity': 'positive', 'qualifiers': [], 'sentences': [0]}
    with pytest.raises(ValueError, match='未覆盖'):
        validate_extraction(doc, {'facts': [fact]})
    with pytest.raises(ValueError, match='原文锚点'):
        validate_extraction(doc, {'facts': [{**fact, 'object': 'INVENTED', 'sentences': [0, 1]}]})


def test_negation_conditions_and_units_survive_rendering():
    fact = {'subject': 'sensor', 'relation': 'measure', 'object': 'pressure', 'polarity': 'negative',
            'qualifiers': [{'kind': 'condition', 'value': 'during maintenance'}, {'kind': 'unit', 'value': 'kPa'}]}
    view = render_fact(fact)
    assert 'NOT measure' in view and 'during maintenance' in view and 'kPa' in view


def test_cited_sentence_must_ground_object():
    doc = public_documents([{'context': [{'title': 'Chronicle', 'sentences': [
        'Ada visited Rome.', 'Ben visited Paris.']}]}])[0]
    with pytest.raises(ValueError, match='不在引用句'):
        validate_extraction(doc, {'facts': [
            {'subject': 'Ada', 'relation': 'visited', 'object': 'Paris', 'polarity': 'positive', 'qualifiers': [], 'sentences': [0]},
            {'subject': 'Ben', 'relation': 'visited', 'object': 'Paris', 'polarity': 'positive', 'qualifiers': [], 'sentences': [1]},
        ]})


def test_cross_sentence_antecedent_is_allowed_when_both_sources_are_cited():
    doc = public_documents([{'context': [{'title': 'Barton', 'sentences': [
        'Robert Childers Barton was a politician.', 'His father was Charles Barton.']}]}])[0]
    facts = validate_extraction(doc, {'facts': [
        {'subject': 'Charles Barton', 'relation': 'parent of', 'object': 'Robert Childers Barton',
         'polarity': 'positive', 'qualifiers': [], 'sentences': [0, 1]},
    ]})
    assert facts[0]['sentences'] == [0, 1]


def _compiled_graph(tmp_path, rows):
    documents, facts = [], []
    for title, sentences, assertions in rows:
        doc = public_documents([{'context': [{'title': title, 'sentences': sentences}]}])[0]
        documents.append(doc)
        facts.extend(validate_extraction(doc, {'facts': assertions}))
    path = tmp_path / 'graph.sqlite'
    write_graph(path, documents, facts)
    graph = object.__new__(PreparedKnowledge)
    graph.graph_path = path
    from hypergraph_memory_method.source_context import derive_source_aliases, source_alias_index
    graph.documents = documents
    graph.source_aliases = derive_source_aliases(documents)
    graph.graph_aliases = source_alias_index(graph.source_aliases)
    return graph


def _assertion(subject, relation, obj, *, qualifiers=None, sentences=None):
    return {'subject': subject, 'relation': relation, 'object': obj, 'polarity': 'positive',
            'qualifiers': qualifiers or [], 'sentences': [0] if sentences is None else sentences}


def test_bridge_can_enter_object_role_and_preserves_source(tmp_path):
    graph = _compiled_graph(tmp_path, [
        ('Orion Clock', ['Orion Clock was designed by Mira Vale.'], [_assertion('Orion Clock', 'designed by', 'Mira Vale')]),
        ('Cedarport', ['Cedarport is where Mira Vale was born.'], [_assertion('Cedarport', 'where born', 'Mira Vale')]),
        ('Clock style', ['Orion Clock is a tower clock.'], [_assertion('Orion Clock', 'type', 'tower clock')]),
    ])
    hits = graph.search_graph('Where was the designer of Orion Clock born?', 2)
    assert {h['fact_fields']['subject'] for h in hits} == {'Orion Clock', 'Cedarport'}
    assert any(step['to_role'] == 'object' for hit in hits for step in hit['reasoning_path'])
    assert all(graph.sources(h['context_id']) for h in hits)
    # 查询不得留下仍占用文件的连接。
    graph.graph_path.rename(tmp_path / 'closed.sqlite')


def test_entity_mentioned_in_a_field_connects_to_cited_story_context(tmp_path):
    graph = _compiled_graph(tmp_path, [
        ('Astral Tales', ['Astral Tales is a fantasy series.'], [_assertion('Astral Tales', 'is', 'a fantasy series')]),
        ('Echo Archive', ['Echo Archive is a companion to the Astral Tales series.',
                          'Mara tells how raiders enslaved the villages.'], [
            _assertion('Echo Archive', 'is', 'a companion to the Astral Tales series'),
            _assertion('Mara', 'tells', 'how raiders enslaved the villages', sentences=[1])]),
    ])
    hits = graph.search_graph('Which Astral Tales companion describes enslaved villages?', 3)
    assert any(hit['fact_fields']['subject'] == 'Mara' for hit in hits)
    assert any(step.get('via') == 'entity_mention' for hit in hits for step in hit['reasoning_path'])
    assert any(hit['fact_fields']['object'] == 'a companion to the Astral Tales series' for hit in hits)


def test_category_words_do_not_displace_the_named_subject_relation(tmp_path):
    rows = [
        ('Blue Compass', ['Blue Compass is the debut album of South Coastal group ORBIT.'], [
            _assertion('Blue Compass', 'is the debut album of', 'South Coastal group ORBIT')]),
        ('Orbit (group)', ['Orbit is a South Coastal group formed by Aurora Studio.'], [
            _assertion('Orbit', 'is a', 'South Coastal group'), _assertion('Orbit', 'formed by', 'Aurora Studio')]),
    ]
    for i in range(10):
        name = 'Decoy ' + str(i)
        rows.append((name, [name + ' is a South Coastal group formed by Dusk Studio.'], [
            _assertion(name, 'is a', 'South Coastal group'), _assertion(name, 'formed by', 'Dusk Studio')]))
    graph = _compiled_graph(tmp_path, rows)
    hits = graph.search_graph('Who formed the South Coastal group whose debut album is Blue Compass?', 4)
    assert any(hit['fact_fields']['subject'] == 'Orbit' and hit['fact_fields']['object'] == 'Aurora Studio' for hit in hits)
    assert all(step.get('entity') != 'South Coastal group' for hit in hits for step in hit['reasoning_path'])


def test_event_constraints_do_not_mix_different_years(tmp_path):
    graph = _compiled_graph(tmp_path, [
        ('Sales', ['Acme sold devices to Beta in 2021.', 'Acme sold devices to Gamma in 2022.'], [
            _assertion('Acme', 'sold devices to', 'Beta', qualifiers=[{'kind': 'time', 'value': '2021'}]),
            _assertion('Acme', 'sold devices to', 'Gamma', qualifiers=[{'kind': 'time', 'value': '2022'}], sentences=[1]),
        ]),
    ])
    hits = graph.search_graph('Acme sold devices', 4, constraints={'time': '2021'})
    assert len(hits) == 1
    assert hits[0]['fact_fields']['object'] == 'Beta'
    assert hits[0]['fact_fields']['qualifiers'] == [{'kind': 'time', 'value': '2021'}]


def test_owned_entity_query_does_not_get_replaced_by_owner(tmp_path):
    rows = [('Mira', ['Mira has her sisters.', 'Her sisters wore orange.'], [
        _assertion('Mira', 'has', 'her sisters', qualifiers=[{'kind': 'object_owner', 'value': 'Mira'}]),
        _assertion('her sisters', 'wore', 'orange', qualifiers=[{'kind': 'subject_owner', 'value': 'Mira'}], sentences=[1]),
    ])]
    for i in range(16):
        rows.append(('Record ' + str(i), ['Mira visited Place ' + str(i) + '.'], [
            _assertion('Mira', 'visited', 'Place ' + str(i))]))
    graph = _compiled_graph(tmp_path, rows)
    results = graph.search_graph("What did Mira's sisters wear?", 3)
    assert any(h['fact_fields']['object'] == 'orange' for h in results)


def test_verb_inflection_does_not_lose_requested_relation(tmp_path):
    rows = [('Person', ['Mira lived in Cedarport.'], [_assertion('Mira', 'lived in', 'Cedarport')])]
    for i in range(15):
        rows.append(('Work ' + str(i), ['Mira designed Clock ' + str(i) + '.'], [
            _assertion('Mira', 'designed', 'Clock ' + str(i))]))
    graph = _compiled_graph(tmp_path, rows)
    assert graph.search_graph('Where did Mira live?', 1)[0]['fact_fields']['object'] == 'Cedarport'


def test_unresolved_pronouns_and_literal_roles_do_not_merge():
    from hypergraph_memory_method.prepared_knowledge import incidence_key
    first = {'id': 'f1', 'document_id': 'doc'}
    second = {'id': 'f2', 'document_id': 'doc'}
    assert incidence_key('They', 'subject', first) != incidence_key('They', 'subject', second)
    assert incidence_key('May', 'time', first) != incidence_key('May', 'subject', first)
    assert incidence_key('May', 'relation', first) != incidence_key('May', 'subject', first)


def test_exact_entity_anchor_is_not_overrun_by_generic_relation_words(tmp_path):
    rows = [('Mira Vale', ['Mira Vale uses the stage name Zephyr.'], [_assertion('Mira Vale', 'stage name', 'Zephyr')])]
    for i in range(18):
        name = 'Other Person ' + str(i)
        rows.append((name, [name + ' is known by stage name performer and is a consultant.'], [
            _assertion(name, 'is known by stage name and consultant', 'performer')]))
    graph = _compiled_graph(tmp_path, rows)
    hits = graph.search_graph('who is known by stage name Zephyr consultant', 2)
    assert any(hit['fact_fields']['subject'] == 'Mira Vale' for hit in hits)


def test_shared_anchor_does_not_fill_results_with_irrelevant_source_neighbors(tmp_path):
    rows = [
        ('Ocean Life magazine', ['Ocean Life was a companion to New Worlds.'] + [
            'Ocean Life published issue ' + str(i) + '.' for i in range(16)], [
                _assertion('Ocean Life', 'was a companion to', 'New Worlds'), *[
                    _assertion('Ocean Life', 'published', 'issue ' + str(i), sentences=[i + 1])
                    for i in range(16)]]),
        ('Coral Tales', [
            'Coral Tales is an illustrated junior series about ocean life.',
            'Coral Tales has a dolphin narrator.',
        ], [
            _assertion('Coral Tales', 'is', 'an illustrated junior series about ocean life'),
            _assertion('Coral Tales', 'has', 'a dolphin narrator', sentences=[1]),
        ]),
    ]
    graph = _compiled_graph(tmp_path, rows)
    hits = graph.search_graph('ocean life illustrated junior series dolphin narrator companion worlds', 6)
    facts = [hit['fact_fields'] for hit in hits]
    assert any(fact['subject'] == 'Coral Tales' and 'series' in fact['object'] for fact in facts)
    assert any(fact['subject'] == 'Coral Tales' and 'narrator' in fact['object'] for fact in facts)


def test_selected_seed_cannot_lend_residual_score_to_unrelated_neighbors(tmp_path):
    graph = _compiled_graph(tmp_path, [
        ('Mira Vale', ['Mira Vale uses Zephyr and works as a consultant.'] + [
            'Mira Vale received medal ' + str(i) + '.' for i in range(16)], [
                _assertion('Mira Vale', 'uses', 'Zephyr'),
                _assertion('Mira Vale', 'works as', 'consultant'), *[
                    _assertion('Mira Vale', 'received', 'medal ' + str(i), sentences=[i + 1])
                    for i in range(16)]]),
        ('School consultant', ['Casey is a consultant known for efforts to improve performance.'], [
            _assertion('Casey', 'known for', 'efforts to improve performance'),
            _assertion('Casey', 'is', 'consultant')]),
        ('Performance consulting', ['Performance consulting means helping organizations improve performance.'], [
            _assertion('Performance consulting', 'means', 'helping organizations improve performance')]),
    ])
    hits = graph.search_graph('Zephyr consultant known for improving performance', 5)
    assert any(hit['fact_fields']['subject'] == 'Performance consulting' for hit in hits)
    assert all(hit['fact_fields']['relation'] != 'received' for hit in hits)


@pytest.mark.parametrize('relation', ['was not featured on', 'was never issued as', 'has no limit on'])
def test_render_does_not_negate_an_already_negative_predicate(relation):
    fact = {**_assertion('A', relation, 'B'), 'polarity': 'negative'}
    assert 'NOT ' not in render_fact(fact)


def _focused_embed(text):
    """可控关系相似度，用于验证路径与候选裁剪，不依赖模型下载。"""
    text = text.casefold()
    category = (0 if 'born' in text else 1 if 'wear' in text or 'wore' in text
                else 2 if 'lived with' in text or 'alone' in text else 3)
    return SimpleNamespace(vector=np.eye(4)[category].tolist(), provider='test', model='relations')


def test_focused_search_preserves_multihop_bridge_and_isolates_returned_facts(library):
    graph = PreparedKnowledge(library)
    graph.prepare_focused_search(_focused_embed)
    hits = graph.search_graph_focused('Where was the designer of Orion Clock born?')
    assert len(hits) == 2
    assert {h['fact_fields']['object'] for h in hits} == {'Mira Vale', 'Cedarport'}
    terminal = next(h for h in hits if h['fact_fields']['object'] == 'Cedarport')
    bridge = next(h for h in hits if h['fact_fields']['object'] == 'Mira Vale')
    assert bridge['context_id'] in terminal['bridge_source_context_ids']
    hits[0]['fact_fields']['object'] = 'POISON'
    assert all(h['fact_fields']['object'] != 'POISON' for h in graph.search_graph_focused(
        'Where was the designer of Orion Clock born?'))


def test_focused_search_keeps_parallel_companions_without_filling_budget(tmp_path):
    graph = _compiled_graph(tmp_path, [('Mira', [
        'Mira lived with her mother and her sisters.', 'Mira wore blue.', 'Zara lived with her father.'
    ], [
        _assertion('Mira', 'lived with', 'her mother', qualifiers=[{'kind': 'object_owner', 'value': 'Mira'}]),
        _assertion('Mira', 'lived with', 'her sisters', qualifiers=[{'kind': 'object_owner', 'value': 'Mira'}]),
        _assertion('Mira', 'wore', 'blue', sentences=[1]),
        _assertion('Zara', 'lived with', 'her father', qualifiers=[{'kind': 'object_owner', 'value': 'Zara'}], sentences=[2]),
    ])])
    graph._prepare_graph_runtime()
    graph.prepare_focused_search(_focused_embed)
    hits = graph.search_graph_focused('Did Mira live alone?', 8)
    assert {'her mother', 'her sisters'} <= {h['fact_fields']['object'] for h in hits}
    assert all(h['fact_fields']['subject'] != 'Zara' for h in hits)
    assert all(h['retrieval_backend'] == 'focused_hypergraph' for h in hits)


def test_focused_search_prioritizes_owned_group_over_owner(tmp_path):
    graph = _compiled_graph(tmp_path, [('Mira', [
        'Mira has her sisters.', 'Her sisters wore orange.', 'Mira wore blue.'
    ], [
        _assertion('Mira', 'has', 'her sisters', qualifiers=[{'kind': 'object_owner', 'value': 'Mira'}]),
        _assertion('her sisters', 'wore', 'orange', qualifiers=[{'kind': 'subject_owner', 'value': 'Mira'}], sentences=[1]),
        _assertion('Mira', 'wore', 'blue', sentences=[2]),
    ])])
    graph._prepare_graph_runtime()
    graph.prepare_focused_search(_focused_embed)
    hits = graph.search_graph_focused("What did Mira's sisters wear?")
    assert hits[0]['fact_fields']['object'] == 'orange'
    assert all(h['fact_fields']['subject'] == 'Mira' for h in hits if h['fact_fields']['object'] == 'blue')


def test_focused_search_rejects_encoder_change(library):
    graph = PreparedKnowledge(library)
    graph.prepare_focused_search(_focused_embed)
    graph.focused_index.embed = lambda q: SimpleNamespace(vector=[1., 0., 0., 0.], provider='other', model='relations')
    with pytest.raises(ValueError, match='同一编码器'):
        graph.search_graph_focused('Orion Clock')


def test_focused_search_keeps_action_location_despite_embedding_score_gap(tmp_path):
    graph = _compiled_graph(tmp_path, [('Mira', [
        'Mira did not work alone.', 'Mira worked in a workshop.'
    ], [
        {**_assertion('Mira', 'worked', 'alone'), 'polarity': 'negative'},
        _assertion('Mira', 'worked in', 'a workshop', sentences=[1]),
    ])])
    graph._prepare_graph_runtime()
    def misleading_embedding(text):
        return SimpleNamespace(vector=[0., 1.] if 'workshop' in text else [1., 0.], provider='test', model='relations')
    graph.prepare_focused_search(misleading_embedding)
    hits = graph.search_graph_focused('Where did Mira work?')
    assert {h['fact_fields']['object'] for h in hits} == {'alone', 'a workshop'}
    assert {h['fact_fields']['polarity'] for h in hits} == {'negative', 'positive'}
