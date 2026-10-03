from hypergraph_memory_method.discourse_reference import SourceReferenceHistory
from hypergraph_memory_method.relation_coverage import RelationCoverageIndex
from test_prepared_knowledge import _assertion
from test_relation_coverage import _compiled_graph


def test_local_pronouns_never_become_global_entity_anchors(tmp_path):
    graph = _compiled_graph(tmp_path, [
        ('Orion', ['Orion is a film. It received good reviews.'],
         [_assertion('Orion', 'is a', 'film'), _assertion('It', 'received', 'good reviews')]),
        ('Vega', ['Vega is an album. It sold five copies.'],
         [_assertion('Vega', 'is an', 'album'), _assertion('It', 'sold', 'five copies')]),
    ])
    assert not graph.relation_index.anchors('it')
    assert not graph.relation_index.anchors('what response did it receive?')
    assert graph.relation_index.anchors('Orion')


def test_grounded_history_distinguishes_person_and_work_without_gold_answers(tmp_path):
    graph = _compiled_graph(tmp_path, [
        ('Luna (musical)', ['Luna is a musical. Luna has music by Mara. Luna tells the story of Jon.'],
         [_assertion('Luna', 'is a', 'musical'), _assertion('Luna', 'music by', 'Mara'), _assertion('Luna', 'story of', 'Jon')]),
        ('Mara', ['Mara is a singer and songwriter. Mara began her career in 1997.'],
         [_assertion('Mara', 'is a', 'singer and songwriter'), _assertion('Mara', 'began career', '1997')]),
    ])
    graph.documents = [
        {'id': 'a', 'title': 'Luna', 'sentences': ['Luna is a musical.']},
        {'id': 'b', 'title': 'Mara', 'sentences': ['Mara is a singer and songwriter.']},
    ]
    graph.relation_index = RelationCoverageIndex(graph)
    history = SourceReferenceHistory(graph)
    roots = graph.relation_index.anchors('Luna')
    music = [key for key, row in graph._graph_rows.items() if row[3]['relation'] == 'music by']
    history.remember('Who composed Luna?', 'Mara', music, roots)
    person, audit = history.resolve('When did she begin her career?', roots)
    assert person == graph.relation_index.anchors('Mara')
    assert audit['source_ids'] == music
    plot = [key for key, row in graph._graph_rows.items() if row[3]['relation'] == 'story of']
    history.remember('Whose story does the musical tell?', 'Jon', plot, roots)
    work, _ = history.resolve('When was its release?', person)
    assert work == roots
    history.remember('Who composed Luna?', 'Invented Name', music, roots)
    assert not history.turns[-1]['answer_nodes']
    history.reset()
    assert history.resolve('When did she begin?', set()) == (set(), None)


def test_work_type_in_fact_keeps_followup_on_album_instead_of_artist(tmp_path):
    graph = _compiled_graph(tmp_path, [('River', [
        'River is the debut solo studio album by Mara. River was released by North Records.'
    ], [_assertion('River', 'is', 'debut solo studio album'),
        _assertion('River', 'debut album of', 'Mara'),
        _assertion('River', 'was released by', 'North Records')])])
    history = SourceReferenceHistory(graph)
    seen = [k for k, row in graph._graph_rows.items() if row[3]['relation'] != 'was released by']
    history.remember('Which is her debut work?', 'River', seen, {'entity:mara'})
    roots, audit = history.resolve('Who released the album?', {'entity:mara'})
    assert roots == {'entity:river'}
    assert audit['entities'] == ['entity:river']


def test_type_of_object_in_relative_clause_is_not_type_of_person(tmp_path):
    graph = _compiled_graph(tmp_path, [('Mara', [
        'Mara is the first female singer to release four singles from one album.'
    ], [_assertion('Mara', 'is', 'the first female singer to release four singles from one album')])])
    history = SourceReferenceHistory(graph)
    assert 'person' in history.types['entity:mara']
    assert 'album' not in history.types['entity:mara']


def test_relation_values_never_become_illegal_entry_nodes_and_legacy_roots_fall_back(tmp_path):
    from types import SimpleNamespace
    from hypergraph_memory_method.focused_retrieval import FocusedGraphSearch
    from competition_agents.embedding import hash_embedding
    graph = _compiled_graph(tmp_path, [('Mara', ['Mara released River in 1997.'],
        [_assertion('Mara', 'released', 'River'), _assertion('Mara', 'debut year', '1997')])])
    history = SourceReferenceHistory(graph)
    keys = list(graph._graph_rows)
    before = {key: dict(row[3]) for key, row in graph._graph_rows.items()}
    # 重现旧记录里出现空实体、描述和带引号实体的情况。
    graph._graph_nodes[keys[0]]['legacy_value'] = 'entity:'
    history.remember('Which work?', 'River', keys, set())
    roots, _ = history.resolve('When was that?', {'entity:', 'entity:not in graph'})
    assert roots <= graph._graph_named_nodes
    embed = lambda text: SimpleNamespace(vector=hash_embedding(text), provider='test', model='test')
    search = FocusedGraphSearch(graph, embed)
    normal = search.search('Mara released River', roots=graph.relation_index.anchors('Mara'))
    legacy = search.search('Mara released River', roots={'entity:', 'entity:not in graph'})
    assert legacy and normal
    assert before == {key: row[3] for key, row in graph._graph_rows.items()}
