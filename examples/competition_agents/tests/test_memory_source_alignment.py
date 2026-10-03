import json

from competition_agents.field_collaboration import FieldCollaboration
from competition_agents.evidence_memory import SessionEvidenceMemory
from competition_agents.llm.response_content import final_content
from competition_agents.llm.latent_only_chat import LatentOnlyChat
from test_field_collaboration import Client
from test_latent_only import bundle_with_tokenizer
from test_prepared_knowledge import _assertion, _focused_embed
from test_relation_coverage import _compiled_graph


def test_full_source_read_preserves_future_relations_and_prevents_redundant_search(tmp_path):
    graph = _compiled_graph(tmp_path, [('River', ['River is an album. River was released in 2001.'], [
        _assertion('River', 'is', 'album'), _assertion('River', 'was released in', '2001')])])
    graph.prepare_focused_search(_focused_embed)
    graph.text_vectors = []

    class Recheck(Client):
        def call(self, role, messages):
            response = super().call(role, messages)
            if role == 'retriever':
                response['content'] = json.dumps({'missing': ['release date'], 'query': 'River release date'})
            return response

    app = FieldCollaboration(graph, Recheck())
    try:
        first = app.run('What is River?', history_scope='one')
        assert first['error'] is None
        assert {h['fact_fields']['object'] for h in first['memory_written']} == {'album', '2001'}
        second = app.run('When was River released?', history_scope='one')
        assert second['error'] is None
        assert second['metrics']['knowledge_searches'] == 0
        assert len(second['calls']) == 4
        assert second['planning_control']['model_lookup'] is True
        assert second['retrieval_decisions'][0]['stop_reason'] == 'requested_relations_already_received'
        assert '2001' in second['calls'][2]['messages'][-1]['content']
    finally:
        app.memory.close()


def test_answer_containing_later_thinking_marker_is_not_discarded():
    raw = 'The publisher is North Records.\nuser\nOther question\nassistant\n<think></think>Other answer'
    assert final_content(raw) == raw
    assert final_content('<think>working</think>North Records.') == 'North Records.'


def test_bound_recall_retains_entity_paths_without_appending_entire_prior_corpus(tmp_path):
    graph = _compiled_graph(tmp_path, [
        ('River', ['River is an album. River was released in 2001.'], [
            _assertion('River', 'is', 'album'), _assertion('River', 'was released in', '2001')]),
        ('Mountain', ['Mountain is a village. Mountain is in North County.'], [
            _assertion('Mountain', 'is', 'village'), _assertion('Mountain', 'is in', 'North County')]),
    ])
    graph.prepare_focused_search(_focused_embed)
    graph.text_vectors = []
    memory = SessionEvidenceMemory('hypergraph', _focused_embed, graph)
    try:
        all_hits = graph.read_graph_sources()
        memory.remember('Read these sources.', 'Done.', [], 'one', observed_evidence=all_hits)
        _, hits, _ = memory.context('When was it released?', 'one', roots={'entity:river'})
        assert {hit['fact_fields']['object'] for hit in hits} == {'album', '2001'}
        assert len(memory.records) == 4
        assert all(hit['from_memory'] for hit in hits)
    finally:
        memory.close()


def test_latent_decoder_closes_answer_at_next_message_without_storing_fake_question():
    bundle = bundle_with_tokenizer()
    bundle.tokenizer.get_vocab = lambda: {'<|im_start|>': 5}
    class Head(bundle.torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.steps = 0
        def forward(self, value):
            scores = bundle.torch.zeros((*value.shape[:-1], 64))
            scores[..., 3 if self.steps == 0 else 5] = 100
            self.steps += 1
            return scores
    bundle.model.lm_head = Head()
    client = LatentOnlyChat(bundle, feedback_mode='cache_only', latent_steps=0, max_tokens=10)
    client.call('planner', question='Who released River?', scope='one')
    client.call('retriever')
    client.call('reasoner')
    assert client.call('summarizer') == 'final answer'
    last = client.calls[-1]
    assert last['finish_reason'] == 'stop'
    assert last['stop_reason'] == 'next_message_boundary'
    assert last['output_tokens'] == 2
    assert client.seen_ids[-1] == 2
