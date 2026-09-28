import hashlib
import json
from pathlib import Path

import pytest

import runner_server
from competition_agents.methods.catalog import BASE, CATALOG, DATASETS
from competition_agents.methods.runner import load_tasks, run
from hypergraph_memory_method.prepared_knowledge import public_documents
from tools.prepare_ranked_continuous import prepare


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def test_ten_turns_match_branch_and_official_source_without_injected_history():
    dataset = DATASETS[CATALOG['defaultDataset']]
    rows = read(BASE / dataset['file'])
    branch = read(BASE / 'data/official_factoid_ranked_topiocqa_003_10turns.json')[0]
    official = {row['Turn_no']: row for row in map(json.loads, (BASE / 'data/official_sources/topiocqa_valid.jsonl').read_text(encoding='utf-8').splitlines()) if row['Conversation_no'] == 153}
    references = read(BASE / dataset['references'])
    assert len(rows) == len(references) == 10
    assert [row['turn_id'] for row in rows] == list(range(1, 11))
    assert [row['question'] for row in rows] == [row['question'] for row in branch['tasks']]
    for row, reference in zip(rows, references):
        assert set(row) == {'id', 'group_id', 'turn_id', 'question'}
        assert row['question'] == official[row['turn_id']]['Question']
        assert reference['id'] == row['id']
        assert reference['reference_answers'][0] == official[row['turn_id']]['Answer']
    assert len(load_tasks(dataset, 10)) == 10
    assert len(load_tasks(dataset, 1)) == 1


def test_knowledge_still_contains_exactly_the_same_complete_source_sections():
    dataset = DATASETS[CATALOG['defaultDataset']]
    root = (BASE / dataset['file']).parent
    source = root / 'knowledge_sources.json'
    old = BASE / 'data/continuous_topiocqa_153_v1/knowledge_sources.json'
    assert source.read_bytes() == old.read_bytes()
    knowledge = BASE / dataset['knowledge']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == read(knowledge / 'manifest.json')['dataset_sha256']
    assert public_documents(read(source)) == read(knowledge / 'text_documents.json')
    branch = read(BASE / 'data/official_factoid_ranked_topiocqa_003_10turns.json')[0]
    assert {row['title'] for row in read(source)[0]['context']} == {row['title'] for row in branch['context']}
    # 原三题及历史结果的输入仍存在，没有原地改成十题。
    assert len(read(old.with_name('questions.json'))) == 3


def test_reusing_snapshots_does_not_download_or_rebuild_knowledge(tmp_path, monkeypatch):
    from tools import prepare_ranked_continuous

    monkeypatch.setattr(prepare_ranked_continuous.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('禁止重新联网下载来源'))
    output = tmp_path / 'ten'
    result = prepare(BASE / 'data/official_factoid_ranked_topiocqa_003_10turns.json',
                     BASE / 'data/official_sources/topiocqa_valid.jsonl', output,
                     reuse_sources=BASE / 'data/continuous_topiocqa_153_v1')
    assert result['selected_turns'] == 10
    assert len(list((output / 'source_snapshots').glob('*.html'))) == 4


@pytest.mark.parametrize('dataset', list(DATASETS))
def test_frontend_api_and_command_always_use_ten_tasks(dataset):
    for old_count in (None, 1, 3, 20):
        config = runner_server.normalize_config({'dataset': dataset, 'taskCount': old_count})
        assert config['taskCount'] == 10
        command = runner_server.build_command(config)
        assert command[command.index('--limit') + 1] == '10'
        assert len(load_tasks(DATASETS[dataset], 10)) == 10


def test_truncated_dataset_is_rejected_before_model_calls(tmp_path):
    dataset = DATASETS[CATALOG['defaultDataset']]
    path = tmp_path / 'questions.json'
    path.write_text(json.dumps(read(BASE / dataset['file'])[:3]), encoding='utf-8')
    with pytest.raises(ValueError, match='实际题数'):
        load_tasks({**dataset, 'file': str(path)}, 10)


@pytest.mark.parametrize('method', ['plain', 'structured'])
def test_ten_turn_queue_and_results_flow_through_existing_roles(method, tmp_path):
    from test_locked_methods import Client, resources

    config = runner_server.normalize_config({'collaborationMode': method})
    events = tmp_path / 'events.jsonl'
    result = run(config, tmp_path / 'result', events, client=Client(method == 'structured'), resources=resources())
    rows = result['runs'][0]['task_results']
    assert len(rows) == result['task_count'] == 10
    assert [row['turn_id'] for row in rows] == list(range(1, 11))
    log = [json.loads(line) for line in events.read_text(encoding='utf-8').splitlines()]
    queue = next(event['tasks'] for event in log if event['event_type'] == 'task_queue')
    assert [row['task_id'] for row in queue] == [row['task_id'] for row in rows]
    assert log[-1]['event_type'] == 'run_done'
