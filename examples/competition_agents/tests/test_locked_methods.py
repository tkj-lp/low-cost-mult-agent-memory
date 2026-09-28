import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import runner_server
from competition_agents.methods.catalog import BASE, CATALOG, METHODS, create_runtime
from competition_agents.methods.runner import run
from competition_agents.methods.text import SimilarityTextMemory


def test_frontend_and_backend_share_exactly_five_methods_and_two_retrieval_modes():
    assert list(METHODS) == ['plain', 'structured', 'latent', 'autoform', 'llmlingua']
    assert {r['value'] for r in CATALOG['retrievalModes']} == {'semantic', 'hypergraph'}
    for method, spec in METHODS.items():
        config = runner_server.normalize_config({'collaborationMode': method})
        command = runner_server.build_command(config)
        assert command[1].endswith('run_competition.py')
        assert command[command.index('--method')+1] == method
        assert config['retrievalMode'] == spec['retrieval']
        assert config['memoryMode'] == spec['memory']
    for retrieval in ['keyword', 'tag', 'fusion']:
        with pytest.raises(ValueError):
            runner_server.normalize_config({'retrievalMode': retrieval})
    with pytest.raises(ValueError):
        runner_server.normalize_config({'collaborationMode': 'protocol-vector'})
    with pytest.raises(ValueError):
        runner_server.normalize_config({'collaborationMode': 'latent', 'backend': 'openai-compatible'})
    with pytest.raises(ValueError):
        runner_server.normalize_config({'collaborationMode': 'latent', 'localModel': 'other'})


def test_text_memory_selects_whole_units_and_does_not_claim_missing_sources_present():
    def encode(text):
        return [float('red' in text), float('north' in text), 0.01]
    memory = SimilarityTextMemory(encode, limit=1)
    original = 'red\nEntire original paragraph.\nDo not split this text.'
    memory.append('one', original, corpus_version='v1', contains_sources=True)
    memory.append('one', 'north is where she went', corpus_version='v1', contains_sources=False)
    text, version, _ = memory.read_for_question('one', 'red?')
    assert text == original and version == 'v1'
    text, version, _ = memory.read_for_question('one', 'north?')
    assert text == 'north is where she went' and version is None
    assert memory.read_for_question('other', 'red?') == ('', None, 0)


@pytest.mark.parametrize('module', ['text', 'protocol', 'autoform', 'llmlingua'])
def test_text_based_entries_do_not_import_latent_runtime(module):
    code = f"import competition_agents.methods.{module}; import sys; assert 'competition_agents.llm.latent_only_chat' not in sys.modules; assert 'competition_agents.llm.experimental_state_chat' not in sys.modules"
    result = subprocess.run([sys.executable, '-c', code], cwd=BASE, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


class Client:
    model = 'test-client'

    def __init__(self, protocol=False):
        self.calls, self.protocol = [], protocol

    def call(self, role, messages, **kwargs):
        if self.protocol:
            payload = json.loads(messages[-1]['content'])
            if role == 'planner':
                content = json.dumps({'task': payload['question'], 'lookup': True, 'reuse': []})
            elif role == 'retriever':
                evidence = payload.get('evidence', payload.get('facts', {}))
                # 格式流转测试只要求引用确实收到的事实；不据此声称模型答题准确率。
                rows = evidence.get('facts', []) if isinstance(evidence, dict) else evidence
                if not rows:
                    rows = next(v['facts'] for v in payload.values() if isinstance(v, dict) and 'facts' in v)
                row = rows[0]
                content = json.dumps({'facts': [row[:4]]})
            elif role == 'reasoner':
                content = json.dumps({'answer': 'White.', 'support': [r[0] for r in payload['facts']]})
            else:
                content = 'White.'
        else:
            content = 'White.' if role == 'summarizer' else 'Inspect the original source about Cotton.'
        self.calls.append({'role': role, 'messages': messages, 'response': {'content': content},
                           'input_tokens': 1, 'output_tokens': 1, 'elapsed_ms': 1.,
                           'request_bytes': len(json.dumps(messages).encode()), 'response_bytes': len(content.encode()),
                           'finish_reason': 'stop'})
        return {'content': content}


def resources():
    return {'encode': lambda text: [1., .1],
            'embed': lambda text: SimpleNamespace(vector=[1., .1], provider='test', model='test')}


@pytest.mark.parametrize('method', ['plain', 'structured', 'autoform', 'llmlingua'])
def test_complete_runtime_emits_frontend_results_for_selected_method(method, tmp_path, monkeypatch):
    if method == 'llmlingua':
        from competition_agents.methods import llmlingua
        monkeypatch.setattr(llmlingua, '_get_official_compressor', lambda: object())
        monkeypatch.setattr(llmlingua, 'compress_message', lambda text, **kwargs: SimpleNamespace(payload=text, provider='llmlingua2'))
    config = {**runner_server.normalize_config({'collaborationMode': method}), 'taskCount': 1}
    events = tmp_path/'events.jsonl'
    result = run(config, tmp_path/'result', events, client=Client(method == 'structured'), resources=resources())
    row = result['runs'][0]['task_results'][0]
    assert row['error'] is None
    assert result['experiment_mode'] == METHODS[method]['experimentMode']
    assert row['model_calls'] == (7 if method == 'autoform' else 4)
    assert len(row['messages']) == 3
    assert row['actual_message_bytes'] == sum(m['wire_bytes'] for m in row['messages'])
    log = [json.loads(line) for line in events.read_text(encoding='utf-8').splitlines()]
    assert log[0]['event_type'] == 'run_start' and log[-1]['event_type'] == 'run_done'
    assert len([e for e in log if e['event_type'] == 'agent_message']) == 3
    assert result['runs'][0]['events'] == log
    assert len([e for e in log if e['event_type'] == 'task_queue']) == 1
    assert not any(e.get('to_agent') == 'Experience' for e in log)


def test_frontend_metrics_preserve_unmeasured_state_payload_and_real_text_bytes():
    for value in (None, 1024):
        payload = {'experiment_mode': 'hidden_api' if value is None else 'baseline_text'}
        task = {'task_index': 1, 'metrics': {'actual_message_bytes': value}}
        record = runner_server.method_record_from_run(payload, {
            'metrics': {'actual_message_bytes': value, 'text_chars': 100, 'state_peak_bytes': 999},
            'task_results': [task]}, 'test')
        assert record['actualMessageBytes'] == value
        assert record['communicationCost'] == value
        assert record['statePeakBytes'] == 999
        assert record['series']['actualMessageBytes'] == [value]
