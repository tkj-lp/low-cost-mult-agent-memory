import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.build_prepared_knowledge import build
from hypergraph_memory_method.prepared_knowledge import PreparedKnowledge, incidence_key, public_documents
from tools.prepare_coqa_knowledge import prepare


def official_fixture(path):
    story = 'Cotton lived in a barn.\n\nShe lived with her mother.  '
    conversation = {
        'id': 'conversation-a', 'source': 'mctest', 'filename': 'original.txt', 'story': story,
        'questions': [{'turn_id': i, 'input_text': f'Original question {i}?'} for i in range(1, 7)],
        'answers': [{'turn_id': i, 'input_text': f'GOLD_SECRET_{i}', 'span_text': 'SPAN_SECRET'}
                    for i in range(1, 7)],
        'additional_answers': {'annotator': [{'turn_id': 1, 'input_text': 'ALTERNATIVE_SECRET'}]},
    }
    path.write_text(json.dumps({'version': '1.0', 'data': [conversation]}), encoding='utf-8')
    return story


def test_whole_story_preserved_and_answers_cannot_enter_runtime_inputs(tmp_path):
    source, output = tmp_path / 'official.json', tmp_path / 'prepared'
    story = official_fixture(source)
    manifest = prepare(source, output)
    sources = json.loads((output / 'knowledge_sources.json').read_text(encoding='utf-8'))
    docs = public_documents(sources)
    assert docs[0]['sentences'] == [story]
    assert (output / 'source.txt').read_bytes() == story.encode('utf-8')
    questions = json.loads((output / 'questions.json').read_text(encoding='utf-8'))
    assert [q['turn_id'] for q in questions] == [1, 2, 3, 4, 5]
    assert len({q['group_id'] for q in questions}) == 1
    assert questions[1]['question'] == 'Original question 2?'
    for filename in ('questions.json', 'knowledge_sources.json'):
        assert 'SECRET' not in (output / filename).read_text(encoding='utf-8')
    references = json.loads((output / 'references.json').read_text(encoding='utf-8'))
    assert references[0]['reference_answers'] == ['GOLD_SECRET_1', 'ALTERNATIVE_SECRET']
    assert manifest['turns'] == 5


def test_refuses_to_overwrite_selected_conversation(tmp_path):
    source, output = tmp_path / 'official.json', tmp_path / 'prepared'
    official_fixture(source)
    prepare(source, output)
    with pytest.raises(FileExistsError):
        prepare(source, output)


def test_unindexed_build_and_read_never_encode_rank_or_truncate_sources(tmp_path, monkeypatch):
    from tools import build_prepared_knowledge

    samples = [{'context': [
        {'title': 'Cotton', 'sentences': ['Cotton lived in a barn.']},
        {'title': 'Other', 'sentences': ['Other lived in a house.']},
    ]}]
    source = tmp_path / 'sources.json'
    source.write_text(json.dumps(samples), encoding='utf-8')
    docs = public_documents(samples)
    assertions = {
        doc['id']: [{'subject': doc['title'], 'relation': 'lived in',
                     'object': 'barn' if doc['title'] == 'Cotton' else 'house',
                     'polarity': 'positive', 'qualifiers': [], 'sentences': [0]}]
        for doc in docs
    }
    facts_file = tmp_path / 'facts.json'
    facts_file.write_text(json.dumps(assertions), encoding='utf-8')
    monkeypatch.setattr(build_prepared_knowledge, 'prepare_text_index',
                        lambda *a: pytest.fail('unindexed build attempted to encode source'))
    output = tmp_path / 'library'
    build(SimpleNamespace(dataset=str(source), output=str(output), env_file=str(tmp_path / 'absent'),
                          embedding_model='unused', facts_file=str(facts_file), model='test',
                          text_retrieval='full-context'))
    assert not (output / 'text_vectors.npy').exists()
    library = PreparedKnowledge(output)
    no_embed = lambda *a: pytest.fail('full-context read attempted to encode query')
    first = library.search_text('Cotton barn', no_embed, top_k=1)
    second = library.search_text('completely different question', no_embed, top_k=1)
    assert first == second
    assert {hit['text'] for hit in first} == {doc['sentences'][0] for doc in docs}
    assert len(first) == 2
    assert library.search_graph('Where did Cotton live?', 2)
    assert library.manifest['text_index']['mode'] == 'none'


def test_possessive_relations_require_explicit_owner_and_do_not_cross_stories():
    first = {'id': 'f1', 'document_id': 'story-a',
             'qualifiers': [{'kind': 'object_owner', 'value': 'Cotton'}]}
    second = {'id': 'f2', 'document_id': 'story-a',
              'qualifiers': [{'kind': 'subject_owner', 'value': 'Cotton'}]}
    key = incidence_key('her sisters', 'object', first)
    assert key == incidence_key('Her sisters', 'subject', second)
    assert key != incidence_key('Her sisters', 'subject', {**second, 'document_id': 'story-b'})
    assert key != incidence_key('Her sisters', 'subject', {**second, 'qualifiers': []})
    assert key != incidence_key('Her sisters', 'subject', {
        **second, 'qualifiers': [{'kind': 'subject_owner', 'value': 'Other'}]})
