from hypergraph_memory_method.source_context import derive_source_aliases, evidence_packet


def test_mixed_recall_is_grouped_in_source_sentence_order_without_loss():
    def hit(document, sentence, value):
        return {'title': document, 'fact_fields': {'document_id': document, 'sentences': [sentence],
            'subject': document, 'relation': 'records', 'object': value, 'polarity': 'positive',
            'qualifiers': [{'kind': 'condition', 'value': 'original condition'}]}}
    evidence = {'s1': hit('work', 4, 'later event'), 's2': hit('person', 0, 'identity'),
                's3': hit('work', 1, 'first event')}
    packet = evidence_packet(evidence)
    assert [row[0] for row in packet['facts']] == ['s3', 's1', 's2']
    assert {row[0] for row in packet['facts']} == set(evidence)
    assert [row[3] for row in packet['facts']] == ['first event', 'later event', 'identity']
    assert all(row[4] == 'condition=original condition' for row in packet['facts'])
    assert packet['sources'] == {'d1': 'work', 'd2': 'person'}


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
    assert packet['facts'][1][1:4] == ['Mira Vale', 'born in', 'Cedarport']
    assert 'subject_identity=Mira Vale' not in packet['facts'][1][4]
    assert packet['aliases'] == [['Mira Vale', 'Mira Jane Vale', 'd2']]


def test_similar_but_different_names_are_not_declared_equivalent():
    docs = [{'id': 'a', 'title': 'Mira Vale', 'sentences': ['Mira Wood is an architect.']},
            {'id': 'b', 'title': 'Orion Clock', 'sentences': ['Mira Vale designed Orion Clock.']}]
    assert derive_source_aliases(docs) == []


def test_polarity_and_conditions_survive_materialization():
    hit = {'title': 'Sensor', 'fact_fields': {'document_id': 'd', 'subject': 'Sensor', 'relation': 'measures',
        'object': 'pressure', 'polarity': 'negative', 'qualifiers': [{'kind': 'condition', 'value': 'during maintenance'}]}}
    row = evidence_packet({'s1': hit})['facts'][0]
    assert row[2] == 'NOT measures' and row[4] == 'condition=during maintenance'


def test_work_article_surname_is_bound_only_with_explicit_unambiguous_full_name():
    docs = [{'id': 'person', 'title': 'Mira Vale', 'sentences': ['Mira Vale (born 1970) is an architect.']},
            {'id': 'work', 'title': 'Orion Clock', 'sentences': [
                'Orion Clock was designed by Mira Vale. Vale began its construction in 2000.']},
            {'id': 'unknown', 'title': 'Other Clock', 'sentences': ['Vale designed this clock.']}]
    scoped = [r for r in derive_source_aliases(docs) if r['name'] == 'Vale']
    assert [(r['document_id'], r['alias']) for r in scoped] == [('work', 'Mira Vale')]
    docs[1]['sentences'].append('Omar Vale also contributed to this clock.')
    assert not [r for r in derive_source_aliases(docs) if r['name'] == 'Vale']


def test_album_definite_reference_uses_source_identity_not_global_merge():
    docs = [{'id': 'a', 'title': 'Silver Road', 'sentences': [
                'Silver Road is a studio album by Mira Vale. Three singles were released from the album.']},
            {'id': 'b', 'title': 'Gold Road', 'sentences': [
                'Gold Road is a studio album by Omar Vale. Two singles were released from the album.']}]
    aliases = [r for r in derive_source_aliases(docs) if r['name'] == 'the album']
    assert {(r['document_id'], r['alias']) for r in aliases} == {('a', 'Silver Road'), ('b', 'Gold Road')}
    assert all(r['scoped'] for r in aliases)


def test_source_bound_group_count_is_visible_in_object_not_only_side_qualifiers():
    hit = {'title': 'Office', 'fact_fields': {'document_id': 'd', 'subject': 'Mira',
        'relation': 'works with', 'object': 'her assistants', 'polarity': 'positive',
        'qualifiers': [{'kind': 'object_owner', 'value': 'Mira'}, {'kind': 'count', 'value': '3'}]}}
    row = evidence_packet({'s1': hit})['facts'][0]
    assert row[3] == "Mira's 3 assistants"
    assert row[4] == ''
    assert hit['fact_fields']['object'] == 'her assistants'
