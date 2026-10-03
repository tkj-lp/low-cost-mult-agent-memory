import sys

import pytest

import run_competition


@pytest.mark.parametrize('backend,options,expected', [
    ('local', [], 32), ('openai-compatible', [], 0),
    ('local', ['--output-repeat-ngram-size', '0'], 0),
])
def test_shared_output_guard_defaults_preserve_remote_interface(tmp_path, monkeypatch, backend, options, expected):
    captured = []

    def run(config, output, event_log):
        captured.append(config)
        return {'status': 'completed', 'model': 'test', 'full_workflow_timing_valid': True,
                'runs': [{'answer_accuracy': 1., 'metrics': {}}]}

    monkeypatch.setattr(run_competition, 'BASE', tmp_path)
    monkeypatch.setattr(run_competition, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['run_competition.py', '--method', 'structured',
                                    '--backend', backend, '--summary-only', *options])
    run_competition.main()
    assert captured[0]['outputRepeatNgramSize'] == expected
    assert captured[0]['latentSteps'] == 40
    assert captured[0]['latentMemoryPolicy'] == 'release_transients'
