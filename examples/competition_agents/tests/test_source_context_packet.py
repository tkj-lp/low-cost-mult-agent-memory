from hypergraph_memory_method.source_context import derive_source_aliases, evidence_packet


def test_document_identity_and_work_context_survive_projection():
    documents = [{'id': 'person', 'title': 'Mira Vale', 'sentences': ['Mira Jane Vale (born 1970) is an architect.']},
                 {'id': 'work', 'title': 'Orion Clock', 'sentences': ['Orion Clock was designed by Mira Vale.']}]
    aliases = derive_source_aliases(documents)
    assert [(row['alias'], row['name']) for row in aliases] == [('Mira Vale', 'Mira Jane Vale')]
    def hit(doc, s, r, o):
        return {'title': documents[doc]['title'], 'fact_fields': {'document_id': documents[doc]['id'],
            'subject': s, 'relation': r, 'object': o, 'qualifiers': [], 'polarity': 'positive'}}
    packet = evidence_packet({'s1': hit(1, 'Mira Vale', 'designed', 'Orion Clock'),
                              's2': hit(0, 'Mira Jane Vale', 'born in', 'Cedarport')}, aliases)
    assert 'Orion Clock' in packet['sources'].values()
    assert packet['aliases'] == [['Mira Vale', 'Mira Jane Vale', 'd2']]
    assert packet['facts'][1][1:4] == ['Mira Jane Vale', 'born in', 'Cedarport']


def test_similar_but_different_names_are_not_declared_equivalent():
    docs = [{'id': 'a', 'title': 'Mira Vale', 'sentences': ['Mira Wood is an architect.']},
            {'id': 'b', 'title': 'Orion Clock', 'sentences': ['Mira Vale designed Orion Clock.']}]
    assert derive_source_aliases(docs) == []


def test_polarity_and_conditions_survive_materialization():
    hit = {'title': 'Sensor', 'fact_fields': {'document_id': 'd', 'subject': 'Sensor', 'relation': 'measures',
        'object': 'pressure', 'polarity': 'negative', 'qualifiers': [{'kind': 'condition', 'value': 'during maintenance'}]}}
    row = evidence_packet({'s1': hit})['facts'][0]
    assert row[2] == 'NOT measures' and row[4] == 'condition=during maintenance'


def test_source_bound_group_count_is_visible_in_object_not_only_side_qualifiers():
    hit = {'title': 'Office', 'fact_fields': {'document_id': 'd', 'subject': 'Mira',
        'relation': 'works with', 'object': 'her assistants', 'polarity': 'positive',
        'qualifiers': [{'kind': 'object_owner', 'value': 'Mira'}, {'kind': 'count', 'value': '3'}]}}
    row = evidence_packet({'s1': hit})['facts'][0]
    assert row[3] == "Mira's 3 assistants"
    assert row[4] == ''
    assert hit['fact_fields']['object'] == 'her assistants'
