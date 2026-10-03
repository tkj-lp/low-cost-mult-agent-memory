from types import SimpleNamespace

import pytest
import torch

from competition_agents.latent_hidden.calibrated_query import fit_ridge_readout, QueryReadout, INTENTS
from competition_agents.latent_hidden.calibrated_graph_tool import CalibratedGraphTool
from competition_agents.llm.latent_only_chat import LatentOnlyChat
from test_latent_only import bundle_with_tokenizer


def test_readout_learns_known_mapping_and_roundtrips(tmp_path):
    torch.manual_seed(53)
    x = torch.randn(90, 8)
    original = torch.randn(8, 4)
    y = torch.nn.functional.normalize(x @ original, dim=-1)
    readout = fit_ridge_readout(x[:70], y[:70], x[70:80], y[70:80])
    score = (readout(x[80:])*y[80:]).sum(dim=-1).mean()
    assert score > .90
    readout.save(tmp_path/'query.pt')
    restored = QueryReadout.load(tmp_path/'query.pt', 'cpu')
    assert torch.equal(readout(x[80:]), restored(x[80:]))
    with pytest.raises(ValueError, match='维度'):
        restored(torch.ones(1, 10))


@pytest.mark.parametrize('question', ['Find the owner of a red box.', 'CURRENT QUESTION'])
def test_query_features_use_content_and_do_not_generate_intermediate_text(question):
    bundle = bundle_with_tokenizer()
    old = bundle.tokenizer
    class Tokenizer:
        eos_token = old.eos_token
        apply_chat_template = old.apply_chat_template
        decode = old.decode
        def __call__(self, text, **kwargs):
            if not kwargs.get('return_offsets_mapping'):
                return old(text, **kwargs)
            offsets = [(i, min(i+4, len(text))) for i in range(0, len(text), 4)]
            ids = [3+sum(text[a:b].encode())%29 for a, b in offsets]
            return {'input_ids': torch.tensor([ids]), 'offset_mapping': torch.tensor([offsets])}
    bundle.tokenizer = Tokenizer()
    client = LatentOnlyChat(bundle, latent_steps=1, query_mode='features')
    client.call('planner', question=question, scope='one')
    result = []
    def capture(query):
        result.append(query)
        return {'facts': [], 'audit': {}}
    client.call('retriever', tool=capture)
    assert result[0].shape == (1, 96)
    assert old.decode_calls == 0
    assert all(c['output_tokens']==0 for c in client.calls)
    record = next(i for i in client.calls[0]['inputs'] if i['category']=='external_question')
    assert len(record['content_positions']) < record['tokens']
    start = record['text'].index('CURRENT QUESTION:\n') + len('CURRENT QUESTION:\n')
    end = start + len(question)
    assert record['content_positions'] == [i for i in range(record['tokens']) if 4*i < end and 4*(i+1) > start]
    assert all(torch.isclose(part.norm(), torch.tensor(1.)) for part in result[0].split(32, dim=-1))


def fake_tool(tool_class=CalibratedGraphTool, **kwargs):
    facts = {
        'color': {'subject': 'Alice', 'relation': 'was', 'object': 'white', 'polarity': 'positive'},
        'location': {'subject': 'Alice', 'relation': 'lived in', 'object': 'North', 'polarity': 'positive'},
        'companions': {'subject': 'Alice', 'relation': 'lived', 'object': 'alone', 'polarity': 'negative'},
    }
    for k, f in facts.items():
        f.update(id=k, qualifiers=[], document_id='d')
    library = SimpleNamespace(
        _graph_rows={k: (0, 'Story', '', f) for k, f in facts.items()},
        _graph_nodes={k: {'subject': 'entity:alice', 'object': 'entity:'+f['object']} for k, f in facts.items()},
        _graph_named_nodes={'entity:alice'}, _graph_owned_names={},
        _graph_neighbors={'entity:alice': [(k,) for k in facts]}, _graph_links={k: [] for k in facts})
    class Teacher:
        def encode(self, texts, **kwargs):
            result=[]
            for text in texts:
                if 'alone' in text:
                    index=2
                elif any(s in text for s in ('white', 'color')):
                    index=0
                elif any(s in text for s in ('North', 'residence')):
                    index=1
                else:
                    index=3
                result.append([float(i==index) for i in range(4)])
            return result
    tool = tool_class(library, SimpleNamespace(torch=torch, device='cpu'), Teacher(), **kwargs)
    return tool


def test_memory_requires_requested_relation_and_skips_full_graph_only_when_complete(monkeypatch):
    tool = fake_tool()
    tool.begin_question('Alice color?', 'session')
    first = tool(torch.tensor([[1., 0., 0., 0.]]))
    assert first['audit']['knowledge_searches']==1
    assert set(first['audit']['source_ids']) == {'color', 'location', 'companions'}
    tool.begin_question('Where does Alice live?', 'session')
    second = tool(torch.tensor([[0., 1., 0., 0.]]))
    assert second['audit']['memory_sufficient']
    assert second['audit']['knowledge_searches']==0
    assert set(second['audit']['source_ids'])=={'color','location','companions'}
    tool.begin_question('Does Alice live alone?', 'session')
    monkeypatch.setattr(tool, '_paths', lambda: pytest.fail('充分的来源记忆不应执行全图搜索'))
    third=tool(torch.tensor([[0., 0., 1., 0.]]))
    assert third['audit']['memory_sufficient'] and third['audit']['knowledge_searches']==0
    assert third['audit']['fresh_source_ids']==[]
    assert any(f['polarity']=='negative' for f in third['facts'])
    assert next(f['relation'] for f in third['facts'] if f['polarity']=='negative') == 'NOT lived'
    assert third['audit']['experience_applied']==0


def test_numeric_question_reuses_complete_inventory_but_missing_fact_requires_lookup():
    tool=fake_tool()
    query=torch.tensor([[0., 1., 0., 0.]])
    tool.begin_question('Where does Alice live?', 'one')
    tool(query)
    tool.begin_question('Where did Alice live before 2020?', 'one')
    audit = tool(query)['audit']
    assert audit['memory_sufficient'] and audit['knowledge_searches'] == 0
    assert audit['memory_gate_basis'] == 'requested_relation_membership_not_answer_correctness'
    tool.begin_question('Where does Alice live?', 'one')
    tool.records.pop('companions')
    assert tool(query)['audit']['decision_reason']=='incomplete_resident_bundle'
    tool.begin_question('Where does Alice live?', 'two')
    assert not tool.records
    assert tool(query)['audit']['knowledge_searches']==1


def test_memory_paths_are_not_bound_to_previous_query_roots():
    tool = fake_tool()
    query = torch.tensor([[0., 1., 0., 0.]])
    tool.begin_question('Alice?', 'one')
    tool(query)
    for item in tool.resident_paths.values():
        item['roots'] = ('entity:previous unrelated focus',)
    tool.begin_question('Where does Alice live?', 'one')
    result = tool(query)
    assert result['audit']['knowledge_searches'] == 0
    assert set(result['audit']['memory_source_ids']) == set(tool.records)
