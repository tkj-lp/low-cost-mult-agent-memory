"""对照屏蔽后的完整缓存，检验裁剪接续的数值一致性及信息保留边界。"""
import copy
from pathlib import Path
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tests"))

from latent_cache_position_probe import PositionProbeChat, select_cache_positions
from competition_agents.llm.latent_only_chat import LatentOnlyChat
from model_fixture import tiny_bundle
from test_latent_only import bundle_with_tokenizer


@pytest.mark.parametrize("keep_positions", [list(range(21, 32)), list(range(6)) + list(range(20, 32))])
def test_original_positions_match_masked_cache_and_reindexed_positions_do_not(keep_positions):
    bundle = tiny_bundle()
    model = bundle.model
    prefix = torch.tensor([[3 + (i * 5) % 29 for i in range(32)]])
    continuation = torch.tensor([[5, 12, 19]])
    keep = torch.tensor(keep_positions)
    with torch.inference_mode():
        past = model(prefix, use_cache=True).past_key_values
        mask = torch.zeros((1, 35), dtype=torch.long)
        mask[0, keep] = 1
        mask[0, 32:] = 1
        reference = model(continuation, past_key_values=copy.deepcopy(past), attention_mask=mask,
                          position_ids=torch.arange(32, 35).unsqueeze(0)).logits
        corrected = copy.deepcopy(past)
        before = sum(layer.keys.numel() + layer.values.numel() for layer in corrected.layers)
        select_cache_positions(corrected, keep)
        after = sum(layer.keys.numel() + layer.values.numel() for layer in corrected.layers)
        assert after < before
        for layer in corrected.layers:
            for tensor in (layer.keys, layer.values):
                assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
        wrong = copy.deepcopy(corrected)
        got = model(continuation, past_key_values=corrected,
                    position_ids=torch.arange(32, 35).unsqueeze(0)).logits
        bad = model(continuation, past_key_values=wrong).logits
    assert torch.allclose(got, reference, atol=2e-6, rtol=2e-5)
    assert float((bad - reference).abs().max()) > 1e-4


def test_invalid_cache_indices_are_rejected_without_mutation():
    bundle = tiny_bundle()
    with torch.inference_mode():
        past = bundle.model(torch.tensor([[3, 7, 11, 5]]), use_cache=True).past_key_values
    saved = past.layers[0].keys.clone()
    for indices in ([], [2, 1], [1, 1], [-1, 2], [0, 4]):
        with pytest.raises(ValueError):
            select_cache_positions(past, torch.tensor(indices, dtype=torch.long))
        assert torch.equal(past.layers[0].keys, saved)


def run_two_questions(client):
    fact = {"id": "f0", "subject": "Alice", "relation": "color", "object": "blue"}
    for question in ("What color?", "Which shade?"):
        client.call("planner", question=question, scope="same")
        client.call("retriever", tool=lambda _: {"facts": [copy.deepcopy(fact)], "audit": {}})
        client.call("reasoner")
        client.call("summarizer")


def test_full_control_matches_current_implementation_and_pruning_keeps_facts():
    bundle = bundle_with_tokenizer()
    weights = {k: v.clone() for k, v in bundle.model.state_dict().items()}
    original = LatentOnlyChat(bundle, latent_steps=2, max_tokens=2)
    full = PositionProbeChat(bundle, variant="full", latent_steps=2, max_tokens=2)
    pruned = PositionProbeChat(bundle, variant="pruned_absolute", latent_steps=2, max_tokens=2)
    for client in (original, full, pruned):
        run_two_questions(client)
        assert client.calls[5]["facts_reused_without_encoding"] == 1
        assert all(c["output_tokens"] == 0 for c in client.calls if c["role"] != "summarizer")
    for left, right in zip(original.past.layers, full.past.layers):
        assert torch.equal(left.keys, right.keys)
        assert torch.equal(left.values, right.values)
    assert [c["inputs"] for c in original.calls] == [c["inputs"] for c in pruned.calls]
    assert "completed_role_instruction" not in pruned.cached_kinds
    assert pruned.cached_kinds.count("common_instruction") == 4
    assert pruned.cached_kinds.count("external_facts") == 4
    assert pruned.cached_kinds.count("latent") == 12
    assert pruned.next_absolute_position == sum(c["forward_positions"] for c in pruned.calls)
    assert pruned.past.get_seq_length() < pruned.next_absolute_position
    assert all(torch.equal(v, weights[k]) for k, v in bundle.model.state_dict().items())
    pruned.call("planner", question="New conversation?", scope="new")
    assert pruned.next_absolute_position == pruned.calls[-1]["forward_positions"]
    assert pruned.calls[-1]["cached_prefix_positions"] == 0
