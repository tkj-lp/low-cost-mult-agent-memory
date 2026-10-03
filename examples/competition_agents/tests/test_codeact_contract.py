"""列表任务不能泄漏标准答案，也不能使整组评分中断。"""
import json

from competition_agents.methods.catalog import DATASETS, BASE
from competition_agents.methods.runner import load_tasks
from competition_agents.methods.codeact import evaluate_execution
from competition_agents.restricted_python_codeact import run_restricted_python


def test_all_ten_task_contracts_and_reference_execution():
    dataset = next(d for d in DATASETS.values() if d['datasetType'] == 'codeact')
    tasks = load_tasks(dataset, 10)
    original = json.loads((BASE/dataset['file']).read_text(encoding='utf-8'))
    for row, task in zip(original, tasks):
        public = task['public_task']
        assert 'expected' not in public and 'reference_code' not in public
        if isinstance(row['expected'], list):
            assert public['output_fields'] == []
            assert public['output_schema'] == {'type': 'array'}
        actual = run_restricted_python(row['reference_code'], inputs=row['inputs'],
                                       allowed_imports=row['allowed_imports'], timeout_ms=5000)
        assert evaluate_execution(actual, row['expected'])['correct'], (row['id'], actual)


def test_list_scores_partial_wrong_order_and_failed_execution():
    def score(value, expected, status='ok'):
        return evaluate_execution({'result': value, 'sandbox_status': status}, expected)
    assert score(['a'], ['a', 'b'])['score'] == .5
    assert score(['b', 'a'], ['a', 'b'])['score'] == 0
    assert not score(['a', 'b', 'c'], ['a', 'b'])['correct']
    assert score([], [])['correct']
    assert not score(['a'], ['a'], 'error')['correct']
    assert score({'a': 3}, ['a'])['score'] == 0
    assert score(3.00000001, 3)['correct']
