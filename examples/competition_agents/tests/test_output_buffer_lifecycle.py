"""确认独立缓冲管理不会改变有效缓存、反馈或后续模型输出。"""

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from task_buffer_lifecycle import LastPositionForward
from competition_agents.llm.latentmas_chat import RecordedCore, StateForward
from runtime_fixtures import cache_bundle


@pytest.mark.parametrize('architecture', ['attention', 'hybrid', 'multimodal'])
def test_output_buffer_reclamation_preserves_cache_feedback_and_continuation(architecture):
    bundle = cache_bundle(architecture)
    core = RecordedCore(bundle, 'release_transients')
    matrix = core._latent_realign_matrices[id(core.model)]
    forward = LastPositionForward(bundle)
    core.model = forward
    core._latent_realign_matrices[id(forward)] = matrix
    histories = []
    for compact in (False, True):
        forward.compact = compact
        core.latent_inputs = []
        past = None
        checkpoints = []
        with torch.inference_mode():
            for request in (torch.tensor([[3, 4, 5, 6]]), torch.tensor([[7, 8, 9]])):
                past = core.generate_latent_batch(request, latent_steps=4, past_key_values=past)
                checkpoints.append(copy.deepcopy(past))
            continuation = bundle.model(input_ids=torch.tensor([[10, 11]]),
                                        past_key_values=copy.deepcopy(past)).logits
        histories.append((checkpoints, list(core.latent_inputs), continuation))
    for left, right in zip(histories[0][0], histories[1][0], strict=True):
        assert left.get_seq_length() == right.get_seq_length()
        assert len(left.layers) == len(right.layers)
        for a, b in zip(left.layers, right.layers, strict=True):
            for name in ('keys', 'values', 'conv_states', 'recurrent_states'):
                x, y = getattr(a, name, None), getattr(b, name, None)
                if x is None:
                    assert y is None
                else:
                    assert torch.equal(x, y)
    for left, right in zip(histories[0][1], histories[1][1], strict=True):
        assert torch.equal(left, right)
    assert torch.equal(histories[0][2], histories[1][2])


def test_only_last_output_is_copied_and_every_cache_position_is_retained():
    bundle = cache_bundle()
    request = torch.tensor([[3, 4, 5, 6]])
    with torch.inference_mode():
        reference = StateForward(bundle)(input_ids=request, use_cache=True, return_dict=True)
        managed = LastPositionForward(bundle, compact=True)(input_ids=request,
                                                            use_cache=True, return_dict=True)
    assert managed.hidden_states[-1].shape[1] == 1
    assert torch.equal(reference.hidden_states[-1][:, -1:, :], managed.hidden_states[-1])
    assert managed.hidden_states[-1].untyped_storage().nbytes() == (
        managed.hidden_states[-1].numel() * managed.hidden_states[-1].element_size())
    assert managed.past_key_values.get_seq_length() == request.shape[1]
    for left, right in zip(reference.past_key_values.layers, managed.past_key_values.layers, strict=True):
        assert torch.equal(left.keys, right.keys)
        assert torch.equal(left.values, right.values)
