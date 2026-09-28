import copy
from types import SimpleNamespace

import pytest

from competition_agents.llm.latent_only_chat import LatentOnlyChat
from competition_agents.latent_hidden.graph_state_tool import GraphStateTool
from model_fixture import tiny_bundle


def bundle_with_tokenizer():
    bundle = tiny_bundle()
    class Tokenizer:
        eos_token = '</end>'
        decode_calls = 0

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            text = ''.join('<' + m['role'] + '>' + m['content'] + '</end>' for m in messages)
            return text + ('<assistant>' if add_generation_prompt else '')

        def __call__(self, text, **kwargs):
            ids = [2] if text == self.eos_token else [3 + (sum(text.encode()) + i*7) % 29 for i in range(4)]
            return {'input_ids': bundle.torch.tensor([ids])}

        def decode(self, ids, **kwargs):
            self.decode_calls += 1
            return 'final answer'
    bundle.tokenizer = Tokenizer()
    return bundle


@pytest.mark.parametrize('steps', [0, 2])
def test_only_final_decodes_text_and_every_role_consumes_real_cache(steps):
    bundle = bundle_with_tokenizer()
    weights = {k: v.clone() for k, v in bundle.model.state_dict().items()}
    client = LatentOnlyChat(bundle, latent_steps=steps, max_tokens=2)
    queries = []
    def tool(query):
        assert bundle.torch.is_tensor(query) and query.shape == (1, 32)
        queries.append(query.clone())
        return {'facts': [{'id': 'f0', 'subject': 'Alice', 'relation': 'lived',
                           'object': 'alone', 'polarity': 'negative'}], 'audit': {}}
    client.call('planner', question='Did Alice live alone?', scope='one')
    client.call('retriever', tool=tool)
    client.call('reasoner')
    assert bundle.tokenizer.decode_calls == 0
    assert all(c['output_tokens'] == 0 and c['raw_response_content'] == '' for c in client.calls)
    assert client.call('summarizer') == 'final answer'
    assert bundle.tokenizer.decode_calls == 1
    assert len(queries) == 1 and bundle.torch.isfinite(queries[0]).all()
    assert sum(c['latent_steps'] for c in client.calls) == steps*3
    assert all(c['cached_prefix_positions'] > 0 for c in client.calls[1:])
    assert sum(c['forward_positions'] for c in client.calls) == client.past.get_seq_length()
    assert all(c['layer_positions'] == [c['forward_positions']]*2 for c in client.calls)
    assert all(bundle.torch.equal(v, weights[k]) for k, v in bundle.model.state_dict().items())
    assert all(not layer._forward_pre_hooks for layer in bundle.model.model.layers)
    inputs = [entry for c in client.calls for entry in c['inputs']]
    assert sum(i['category'] == 'external_question' for i in inputs) == 1
    assert sum(i['category'] == 'external_facts' for i in inputs) == 1


def test_external_fact_reuse_does_not_reencode_and_scope_change_resets():
    bundle = bundle_with_tokenizer()
    client = LatentOnlyChat(bundle, latent_steps=0, max_tokens=2)
    fact = {'id': 'f0', 'subject': 'Alice', 'relation': 'color', 'object': 'blue'}
    def tool(_query):
        return {'facts': [copy.deepcopy(fact)], 'audit': {}}
    client.call('planner', question='First question?', scope='one')
    client.call('retriever', tool=tool)
    client.call('planner', question='Second question?', scope='one')
    client.call('retriever', tool=tool)
    assert client.calls[-1]['facts_encoded'] == 0
    assert client.calls[-1]['facts_reused_without_encoding'] == 1
    assert not any(i['category'] == 'external_facts' for i in client.calls[-1]['inputs'])
    fact['object'] = 'green'
    client.call('retriever', tool=tool)
    assert client.calls[-1]['facts_encoded'] == 1
    client.call('planner', question='Another session?', scope='two')
    assert client.calls[-1]['cached_prefix_positions'] == 0
    client.call('retriever', tool=tool)
    assert client.calls[-1]['facts_encoded'] == 1


def test_changed_question_changes_numeric_tool_query_without_decoding():
    bundle = bundle_with_tokenizer()
    queries = []
    def tool(query):
        queries.append(query.clone())
        return {'facts': [], 'audit': {}}
    for question in ['Find the place.', 'Find the owner.']:
        client = LatentOnlyChat(bundle, latent_steps=2)
        client.call('planner', question=question, scope='scope')
        client.call('retriever', tool=tool)
    assert not bundle.torch.allclose(*queries)
    assert bundle.tokenizer.decode_calls == 0


def test_repeated_question_input_to_later_role_is_rejected():
    client = LatentOnlyChat(bundle_with_tokenizer())
    with pytest.raises(ValueError, match='不得重新输入'):
        client.call('retriever', question='duplicate')


def test_graph_tool_consumes_vectors_and_keeps_fact_memory_scoped(monkeypatch):
    bundle = bundle_with_tokenizer()
    torch = bundle.torch
    facts = {'a': {'id': 'a', 'subject': 'Alice', 'relation': 'color', 'object': 'blue',
                    'polarity': 'positive', 'qualifiers': [], 'document_id': 'd'},
             'b': {'id': 'b', 'subject': 'Alice', 'relation': 'lived', 'object': 'alone',
                    'polarity': 'negative', 'qualifiers': [], 'document_id': 'd'}}
    library = SimpleNamespace(
        _graph_rows={k: (0, 'Story', '', v) for k, v in facts.items()},
        _graph_nodes={k: {'subject': 'entity:alice', 'object': 'entity:' + v['object']} for k, v in facts.items()},
        _graph_named_nodes={'entity:alice'}, _graph_owned_names={},
        _graph_neighbors={'entity:alice': [('a',), ('b',)]},
        _graph_links={k: [] for k in facts})
    monkeypatch.setattr(GraphStateTool, '_encode_index', lambda _self: torch.tensor([[1., 0.], [0., 1.]]))
    tool = GraphStateTool(library, bundle, terminals=1)
    tool.begin_question('Alice color?', 'one')
    one = tool(torch.tensor([[1., 0.]]))
    two = tool(torch.tensor([[0., 1.]]))
    assert one['audit']['source_ids'] == ['a']
    assert two['audit']['source_ids'] == ['b']
    assert two['facts'][0]['polarity'] == 'negative'
    assert tool(torch.tensor([[0., 1.]]))['audit']['memory_source_ids'] == ['b']
    tool.begin_question('Alice?', 'two')
    assert tool(torch.tensor([[0., 1.]]))['audit']['fresh_source_ids'] == ['b']
    with pytest.raises(ValueError, match='不接受文字'):
        tool('Alice')
