import torch

from competition_agents.llm.output_repeat_guard import OutputRepeatGuard


def test_only_current_output_is_guarded_not_copying_original_evidence():
    guard = OutputRepeatGuard(4, prompt_length=4)
    scores = torch.zeros((1, 10))
    scores[0, 4] = 100
    output = guard(torch.tensor([[1, 2, 3, 4, 1, 2, 3]]), scores)
    assert int(output.argmax()) == 4
    assert guard.report()['blocked_top1'] == 0


def test_shared_guard_blocks_exact_output_loop_and_leaves_new_tokens_unchanged():
    guard = OutputRepeatGuard(4)
    first = torch.zeros((1, 10))
    guard.apply_output([1, 2, 3, 4], first)
    logits = torch.arange(10, dtype=torch.float32).unsqueeze(0)
    logits[0, 4] = 100
    result = guard.apply_output([1, 2, 3, 4, 1, 2, 3], logits)
    assert torch.isneginf(result[0, 4])
    assert torch.equal(result[0, :4], torch.arange(4, dtype=torch.float32))
    assert int(result.argmax()) == 9
    assert guard.report()['blocked_top1'] == 1


def test_repeat_guard_keeps_required_grammar_continuation_available():
    guard = OutputRepeatGuard(4, preserve_grammar=True)
    guard.apply_output([1, 2, 3, 4], torch.zeros((1, 10)))
    required = torch.full((1, 10), float('-inf'))
    required[0, 4] = 3.
    result = guard.apply_output([1, 2, 3, 4, 1, 2, 3], required)
    assert torch.isfinite(result[0, 4])
    assert guard.report()['grammar_fallbacks'] == 1
    scores = torch.full((1, 10), float('-inf'))
    scores[0, 4], scores[0, 5] = 3., 2.
    result = guard.apply_output([1, 2, 3, 4, 1, 2, 3], scores)
    assert int(result.argmax()) == 5
    assert torch.isneginf(result[0, 4])


def test_protocol_repeat_guard_retains_complete_valid_tool_object():
    import json
    from competition_agents.llm.generation import standard_generate
    from runtime_fixtures import cache_bundle, scripted_logits
    bundle = cache_bundle()
    bundle.output_repeat_ngram_size = 4
    body = {'action': 'deliver', 'query': None,
            'work': {'candidate': 'a' * 64, 'basis': []}}
    hook = scripted_logits(bundle, body)
    try:
        result = standard_generate(bundle, torch.tensor([[3, 4]]), max_tokens=256,
                                   role='reasoner', control=True)
    finally:
        hook.remove()
    assert result['decision_error'] is None
    assert result['action'] == 'deliver'
    assert result['output_repeat_guard']['blocked_top1'] > 0
    assert result['output_repeat_guard']['preserve_grammar']
    assert json.loads(result['content'])['candidate'] != 'a' * 64
    assert result['finish_reason'] == 'stop'
