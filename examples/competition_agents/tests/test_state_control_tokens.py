from types import SimpleNamespace
import torch

from competition_agents.latent_hidden.posterior_alignment import align_posterior_hidden
from hypergraph_memory_method.source_context import derive_source_aliases


def test_control_tokens_do_not_enter_continuous_message():
    embeddings = torch.nn.Embedding.from_pretrained(torch.eye(4))
    model = SimpleNamespace(get_output_embeddings=lambda: lambda x: x,
                            get_input_embeddings=lambda: embeddings)
    hidden = torch.tensor([[[100., 3., 2., 1.]]])
    vector, audit = align_posterior_hidden(model, hidden, excluded_token_ids=[0])
    assert vector[0, 0, 0] == 0
    assert torch.isfinite(vector).all()
    assert audit['excluded_control_tokens'] == 1


def test_person_surname_is_scoped_and_ambiguous_names_are_not_merged():
    docs = [dict(id='a', title='Anne Example [SEP] Introduction',
                 sentences=['Anne May Example (born 1970) is a writer. Example wrote books.']),
            dict(id='b', title='Anne Example [SEP] Career', sentences=['Example published an album.']),
            dict(id='c', title='Anne Example [SEP] Family', sentences=['Anne Example married John Example. Example moved.'])]
    aliases = derive_source_aliases(docs)
    surnames = [r for r in aliases if r['name'] == 'Example']
    assert {r['document_id'] for r in surnames} == {'a', 'b'}
    assert all(r['scoped'] for r in surnames)
