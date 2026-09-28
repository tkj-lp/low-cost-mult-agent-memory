"""独立对照的数值边界、来源隔离和原实现不变性。"""
import copy
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tests"))

from latent_alignment_probe import (
    BASE, LinearAlignment, ResidualAlignment, ProbeChat, calibration_split,
    fit_low_rank, frozen_evidence, unit,
)
from test_latent_only import bundle_with_tokenizer
from competition_agents.llm.latent_only_chat import LatentOnlyChat
from tools.verify_locked_baselines import verify


def test_paper_projection_matches_closed_form_and_mean_embedding_norm():
    bundle = bundle_with_tokenizer()
    model = bundle.model
    aligned = LinearAlignment.from_model(model)
    x = model.get_output_embeddings().weight.detach()
    y = model.get_input_embeddings().weight.detach()
    reference = torch.linalg.solve(x.T @ x + 1e-5 * torch.eye(x.shape[1]), x.T @ y)
    assert torch.allclose(aligned.matrix, reference)
    hidden = torch.randn(3, 1, 32)
    value = aligned(hidden)
    assert torch.isfinite(value).all()
    assert torch.allclose(value.norm(dim=-1), y.norm(dim=-1).mean().expand(3, 1), atol=1e-5)


def test_residual_is_small_and_zero_residual_preserves_base():
    base = LinearAlignment(torch.eye(32), torch.tensor(0.5))
    hidden = torch.randn(4, 1, 32)
    left = torch.randn(32, 3)
    zero = ResidualAlignment(base, {"shared": (left, torch.zeros(3, 32))})
    assert torch.allclose(zero(hidden, "planner"), base(hidden), atol=1e-6)
    residual = ResidualAlignment(base, {"shared": (left, torch.randn(3, 32) * 100)})
    value = residual(hidden, "reasoner")
    assert torch.isfinite(value).all()
    assert torch.allclose(value.norm(dim=-1), torch.full((4, 1), 0.5), atol=1e-6)
    assert ((unit(value) * unit(base(hidden))).sum(-1) >= 0.96).all()


def test_low_rank_fit_generalizes_known_low_rank_map():
    torch.manual_seed(49)
    left = torch.linalg.qr(torch.randn(24, 4)).Q
    features = torch.randn(80, 4) @ left.T
    targets = unit(features) @ left @ torch.randn(4, 24)
    a, b = fit_low_rank(features[:60], targets[:60], rank=4)
    relative_error = ((unit(features[60:]) @ a) @ b - targets[60:]).norm() / targets[60:].norm()
    assert relative_error < 0.01


def test_feedback_keeps_prompts_weights_and_only_final_text_unchanged():
    bundle = bundle_with_tokenizer()
    weights = {k: v.clone() for k, v in bundle.model.state_dict().items()}
    base = LinearAlignment.from_model(bundle.model)
    clients = [LatentOnlyChat(bundle, latent_steps=2, max_tokens=2),
               ProbeChat(bundle, alignment=base, latent_steps=2, max_tokens=2)]
    fact = {"id": "f0", "subject": "Alice", "relation": "color", "object": "blue"}
    for client in clients:
        for question in ("What color?", "Which shade?"):
            client.call("planner", question=question, scope="same")
            client.call("retriever", tool=lambda _: {"facts": [copy.deepcopy(fact)], "audit": {}})
            client.call("reasoner")
            client.call("summarizer")
        assert sum(c["latent_steps"] for c in client.calls) == 12
        assert client.calls[5]["facts_reused_without_encoding"] == 1
        assert all(c["output_tokens"] == 0 for c in client.calls if c["role"] != "summarizer")
    assert [c["inputs"] for c in clients[0].calls] == [c["inputs"] for c in clients[1].calls]
    assert all(torch.equal(v, weights[k]) for k, v in bundle.model.state_dict().items())


def test_real_evidence_replays_three_original_questions_without_gold_answers():
    rows = frozen_evidence(
        BASE.parents[1] / "run_result/Qwen2.5-1.5B-Instruct/validation_v4_ranked153_latent_b71781b6.json",
        BASE / "data/continuous_topiocqa_153_v1/questions.json",
        BASE / "data/prepared_knowledge/topiocqa_153_v2",
    )
    assert len(rows) == 3
    assert rows[0]["facts"][0] == rows[1]["facts"][0]
    assert [len(r["facts"]) for r in rows] == [1, 2, 1]
    assert all("prediction" not in r and "expected_answer" not in r for r in rows)
    split = calibration_split(BASE / "data/prepared_knowledge/topiocqa_119_v2/facts.json",
                              BASE / "data/prepared_knowledge/topiocqa_153_v2")
    train = {f["document_id"] for part, f in split if part == "train"}
    validation = {f["document_id"] for part, f in split if part == "validation"}
    assert not train & validation
    assert len(split) == 24
    verify()
