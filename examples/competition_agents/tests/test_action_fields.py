from hypergraph_memory_method.action_fields import extract_actions, render_actions
from hypergraph_memory_method.fields import render_evidence


def test_shared_subject_action_preserves_all_objects():
    body = 'Pine Laboratory measures pressure. Pine Laboratory measures humidity.'
    assert render_actions(body) == 'Pine Laboratory measures: pressure; humidity'
    assert len(render_actions(body)) < len(body)


def test_time_and_actor_are_not_lost():
    body = 'The staff at Pine Laboratory record temperature and rainfall every morning.'
    assert extract_actions(body) == (('The staff at Pine Laboratory', 'record', 'temperature and rainfall', 'every morning'),)


def test_unknown_and_guarded_sentences_remain_verbatim():
    for body in ('Pine Laboratory does not measure pressure.',
                 'Pine Laboratory measures pressure only in winter.',
                 'They measure pressure.', 'Pine Laboratory said "record pressure".',
                 'Pine Laboratory measures pressure when it rains.'):
        assert extract_actions(body) is None
        assert body in render_evidence([body], fields=True)


def test_render_integration_and_text_baseline():
    body = 'Pine Laboratory measures pressure. Pine Laboratory measures humidity.'
    assert 'measures: pressure; humidity' in render_evidence([body], fields=True)
    assert render_evidence([body], fields=False) == '[0] ' + body
