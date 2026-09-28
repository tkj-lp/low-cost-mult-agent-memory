"""保护来源与约束、消费者期限、跨题回收及已有对照路径一致性。"""
import copy
from pathlib import Path
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tests"))

from latent_cache_lifetime_probe import CacheBlock, LifetimeProbeChat, expiration, reclaimable
from latent_cache_position_probe import PositionProbeChat
from test_latent_cache_position_probe import run_two_questions
from test_latent_only import bundle_with_tokenizer


@pytest.mark.parametrize("kind", ["question", "external_facts", "common_instruction", "answer"])
def test_source_constraints_and_answers_do_not_expire(kind):
    block = CacheBlock(10, 5, kind, "planner", 1, expiration(kind, "planner"))
    for policy in ("full", "immediate", "consumer_lifetime"):
        assert not reclaimable(block, policy=policy, completed_role="summarizer", turn=999)


def test_instruction_survives_until_consumer_and_latent_survives_until_final():
    instruction = CacheBlock(10, 5, "role_instruction", "planner", 1, expiration("role_instruction", "planner"))
    latent = CacheBlock(15, 2, "latent", "planner", 1, expiration("latent", "planner"))
    assert not reclaimable(instruction, policy="consumer_lifetime", completed_role="planner", turn=1)
    assert reclaimable(instruction, policy="consumer_lifetime", completed_role="retriever", turn=1)
    for role in ("planner", "retriever", "reasoner"):
        assert not reclaimable(latent, policy="consumer_lifetime", completed_role=role, turn=1)
    assert reclaimable(latent, policy="consumer_lifetime", completed_role="summarizer", turn=1)
    with pytest.raises(ValueError):
        expiration("unregistered_state", "planner")


@pytest.mark.parametrize("old_variant,new_policy", [("full", "full"), ("pruned_absolute", "immediate")])
def test_controls_preserve_previous_computation_exactly(old_variant, new_policy):
    bundle = bundle_with_tokenizer()
    old = PositionProbeChat(bundle, variant=old_variant, latent_steps=2, max_tokens=2)
    new = LifetimeProbeChat(bundle, policy=new_policy, latent_steps=2, max_tokens=2)
    for client in (old, new):
        run_two_questions(client)
    assert [c["inputs"] for c in old.calls] == [c["inputs"] for c in new.calls]
    for left, right in zip(old.past.layers, new.past.layers):
        assert torch.equal(left.keys, right.keys)
        assert torch.equal(left.values, right.values)
    assert old.next_absolute_position == new.next_absolute_position


def test_live_dependencies_are_kept_then_reclaimed_without_erasing_facts():
    bundle = bundle_with_tokenizer()
    before = {k: v.clone() for k, v in bundle.model.state_dict().items()}
    client = LifetimeProbeChat(bundle, policy="consumer_lifetime", latent_steps=2, max_tokens=2)
    fact = {"id": "f0", "subject": "Alice", "relation": "color", "object": "blue"}
    client.call("planner", question="Keep the full name and the negation.", scope="one")
    assert any(b.kind == "role_instruction" and b.owner == "planner" for b in client.blocks)
    with pytest.raises(ValueError, match="消费者尚未完成"):
        client.call("reasoner")
    client.call("retriever", tool=lambda _: {"facts": [copy.deepcopy(fact)], "audit": {}})
    assert not any(b.kind == "role_instruction" and b.owner == "planner" for b in client.blocks)
    assert sum(b.length for b in client.blocks if b.kind == "latent") == 4
    client.call("reasoner")
    assert sum(b.length for b in client.blocks if b.kind == "latent") == 6
    client.call("summarizer")
    assert not any(b.kind in ("latent", "role_instruction", "boundary") for b in client.blocks)
    assert {b.kind for b in client.blocks} == {"common_instruction", "question", "external_facts", "answer"}
    assert sum(b.length for b in client.blocks) == client.past.get_seq_length()
    client.call("planner", question="What is her color?", scope="one")
    client.call("retriever", tool=lambda _: {"facts": [copy.deepcopy(fact)], "audit": {}})
    assert client.calls[-1]["facts_reused_without_encoding"] == 1
    assert client.calls[-1]["facts_encoded"] == 0
    assert sum(b.length for b in client.blocks if b.kind == "external_facts") == 4
    assert client.next_absolute_position == sum(c["forward_positions"] for c in client.calls)
    assert all(torch.equal(v, before[k]) for k, v in bundle.model.state_dict().items())
    client.call("planner", question="New scope", scope="two")
    assert client.turn == 1
    assert all(b.kind != "external_facts" for b in client.blocks)
