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


def test_linear_feedback_does_not_select_candidate_tokens_or_change_weights(monkeypatch):
    import competition_agents.llm.latent_only_chat as module
    bundle = bundle_with_tokenizer()
    before = {k: v.clone() for k, v in bundle.model.state_dict().items()}
    monkeypatch.setattr(module, 'align_posterior_hidden', lambda *a, **k: pytest.fail('线性反馈不得调用候选词混合'))
    client = LatentOnlyChat(bundle, feedback_mode='linear', cache_policy='consumer_lifetime')
    client.call('planner', question='Where is the package?', scope='one')
    client.call('retriever', tool=lambda q: {'facts': [{'id': 'source', 'subject': 'Package',
        'relation': 'is in', 'object': 'Depot', 'polarity': 'positive'}], 'audit': {}})
    assert client.calls[-1]['lifetime']['removed_positions'] > 0
    assert client.lifecycle.next_position > client.past.get_seq_length()
    assert client.encoded_facts.keys() == {'source'}
    assert bundle.tokenizer.decode_calls == 0
    assert all(bundle.torch.equal(v, before[k]) for k, v in bundle.model.state_dict().items())
    old_position = client.lifecycle.next_position
    client.call('reasoner')
    assert client.lifecycle.next_position > old_position
    assert sum(b.length for b in client.lifecycle.blocks) == client.past.get_seq_length()


def test_lifetime_preserves_retained_keys_and_values_exactly():
    from transformers.cache_utils import DynamicCache
    from competition_agents.latent_hidden.state_lifecycle import StateLifecycle
    bundle = bundle_with_tokenizer()
    torch = bundle.torch
    cache = DynamicCache()
    for i in range(2):
        key = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4) + 100*i
        cache.update(key, key + 50, i)
    saved = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    state = StateLifecycle()
    state.turn = 1
    state.append(1, 'external_facts', 'planner')
    state.append(1, 'role_instruction', 'planner')
    state.append(1, 'external_question', 'planner')
    result = state.reclaim(cache, 'retriever', torch)
    assert result['removed_positions'] == 1
    assert state.next_position == 3 and cache.get_seq_length() == 2
    for layer, (keys, values) in zip(cache.layers, saved):
        assert torch.equal(layer.keys, keys[:, :, [0, 2], :])
        assert torch.equal(layer.values, values[:, :, [0, 2], :])
    assert result['peak_bytes'] >= result['before_bytes']


def test_compacted_cache_matches_masked_full_cache_at_original_position():
    from competition_agents.latent_hidden.state_lifecycle import StateLifecycle
    bundle = bundle_with_tokenizer()
    torch, model = bundle.torch, bundle.model.model
    with torch.inference_mode():
        past = model(input_ids=torch.tensor([[3, 4, 5]]), use_cache=True).past_key_values
        reference = copy.deepcopy(past)
        state = StateLifecycle()
        state.append(1, 'external_facts', 'planner')
        state.append(1, 'role_instruction', 'planner')
        state.append(1, 'external_question', 'planner')
        state.reclaim(past, 'retriever', torch)
        expected = model(input_ids=torch.tensor([[6]]), past_key_values=reference,
                         attention_mask=torch.tensor([[1, 0, 1, 1]]),
                         cache_position=torch.tensor([3]), position_ids=torch.tensor([[3]]),
                         use_cache=True).last_hidden_state
        actual = model(input_ids=torch.tensor([[6]]), past_key_values=past,
                       attention_mask=torch.ones((1, 3), dtype=torch.long),
                       cache_position=torch.tensor([2]), position_ids=torch.tensor([[state.next_position]]),
                       use_cache=True).last_hidden_state
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_cache_only_continuation_never_maps_hidden_state_to_candidate_input(monkeypatch):
    import competition_agents.llm.latent_only_chat as module
    monkeypatch.setattr(module, 'align_posterior_hidden', lambda *a, **k: pytest.fail('完整缓存续接不执行额外适配'))
    bundle = bundle_with_tokenizer()
    client = LatentOnlyChat(bundle, feedback_mode='cache_only', latent_steps=0, cache_policy='consumer_lifetime')
    client.call('planner', question='Where is the parcel?', scope='one')
    client.call('retriever', tool=lambda q: {'facts': [{'id': 'f1', 'subject': 'Parcel',
        'relation': 'is in', 'object': 'Depot', 'polarity': 'positive'}], 'audit': {}})
    client.call('reasoner')
    assert all(c['latent_steps'] == c['output_tokens'] == 0 for c in client.calls)
    assert all(c['cached_prefix_positions'] > 0 for c in client.calls[1:])
    assert client.calls[-1]['state_after_bytes'] > 0
    assert client.alignment is None and bundle.tokenizer.decode_calls == 0
    assert client.encoded_facts.keys() == {'f1'}


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
