import json
from types import SimpleNamespace

import pytest

import runner_server
from competition_agents.methods.catalog import DATASETS
from competition_agents.methods.runner import load_tasks
from hypergraph_memory_method.source_context import derive_source_aliases, source_alias_index
from hypergraph_memory_method.prepared_knowledge import incidence_key
from tools.prepare_official_continuous import prepare


def test_summary_counts_success_failure_and_waiting_separately():
    events = [{'event_type': 'run_start', 'task_count': 3},
              {'event_type': 'task_done', 'task_id': 'one'},
              {'event_type': 'task_done', 'task_id': 'one'},
              {'event_type': 'task_failed', 'task_id': 'two'},
              {'event_type': 'run_done', 'status': 'failed'}]
    summary = runner_server.event_summary(events)
    assert summary['completedTasks'] == 1
    assert summary['failedTasks'] == 1
    assert summary['taskCount'] == 3


def test_old_failed_completion_event_is_not_counted_as_success():
    summary = runner_server.event_summary([{'event_type': 'task_done', 'task_id': 'one',
                                          'return_result': {'error': 'invalid source'}}])
    assert summary['completedTasks'] == 0 and summary['failedTasks'] == 1


def test_current_events_never_include_another_run(tmp_path, monkeypatch):
    file = tmp_path/'events.jsonl'
    file.write_text('\n'.join(json.dumps({'run_id': value}) for value in ['old', 'current']), encoding='utf-8')
    monkeypatch.setattr(runner_server, 'EVENT_LOG_FILE', file)
    monkeypatch.setattr(runner_server.STATE, 'run_id', 'current')
    assert runner_server.read_events() == [{'run_id': 'current'}]
    monkeypatch.setattr(runner_server.STATE, 'run_id', '')
    assert runner_server.read_events() == []


def test_handoff_chart_series_preserves_actual_values_and_unmeasured_values():
    rows = [{'task_index': 2, 'metrics': {'handoff_payload_bytes': 1234}},
            {'task_index': 1, 'metrics': {'handoff_payload_bytes': 456}},
            {'task_index': 3, 'metrics': {}}]
    assert runner_server.per_task_series(rows, 'handoff_payload_bytes') == [456., 1234., None]


@pytest.mark.parametrize('stopped,expected', [(True, 'stopped'), (False, 'failed')])
def test_terminal_process_cannot_be_called_completed_on_partial_result(monkeypatch, stopped, expected):
    monkeypatch.setattr(runner_server, 'refresh_process_state', lambda: None)
    monkeypatch.setattr(runner_server, 'read_events', lambda: [
        {'event_type': 'run_start', 'task_count': 3}, {'event_type': 'task_done', 'task_id': 'one'},
        {'event_type': 'run_done'}])
    monkeypatch.setattr(runner_server.STATE, 'process', SimpleNamespace(poll=lambda: 1))
    monkeypatch.setattr(runner_server.STATE, 'stopped', stopped)
    monkeypatch.setattr(runner_server.STATE, 'last_error', '')
    assert runner_server.status_payload()['status'] == expected


def test_official_questions_and_references_are_separated(tmp_path):
    from competition_agents.methods.catalog import BASE
    report = prepare(BASE/'data/official_topiocqa_001_10turns.json',
                     BASE/'data/official_sources/topiocqa_valid.jsonl', tmp_path)
    assert report['conversation'] == 119 and report['selected_turns'] == 3
    questions = json.loads((tmp_path/'questions.json').read_text(encoding='utf-8'))
    assert [row['question'] for row in questions] == ['what is an oribi?', 'where is it found?', 'how does it look like?']
    assert all(set(row) == {'id', 'group_id', 'turn_id', 'question'} for row in questions)
    sources = json.loads((tmp_path/'knowledge_sources.json').read_text(encoding='utf-8'))[0]['context']
    assert len(sources) == 7 and all(row.get('source') != 'official_annotated_rationale' for row in sources)


def test_codeact_public_input_excludes_reference_code_and_values():
    tasks = load_tasks(DATASETS['codeact'], 3)
    assert len(tasks) == 3
    for row in tasks:
        assert set(row['public_task']) == {'task', 'inputs', 'allowed_imports', 'output_fields'}
        assert 'expected' not in row['public_task'] and 'reference_code' not in row['public_task']


def test_source_scoped_descriptions_do_not_merge_different_animals():
    docs = [{'id': 'one', 'title': 'Oryx [SEP] Description', 'sentences': ['The oryx is a large antelope. This antelope has long horns.']},
            {'id': 'two', 'title': 'Oribi [SEP] Description', 'sentences': ['The oribi is a small antelope. This antelope has a short tail.']}]
    aliases = source_alias_index(derive_source_aliases(docs))
    def key(doc, value):
        return incidence_key(value, 'subject', {'id': doc, 'document_id': doc, 'qualifiers': []}, aliases)
    assert key('one', 'The oryx') == key('one', 'This antelope') == 'entity:oryx'
    assert key('two', 'This antelope') == 'entity:oribi'
    assert key('other', 'This antelope').startswith('local:')


@pytest.mark.parametrize('structured', [False, True])
def test_codeact_executes_model_code_and_exposes_four_real_calls(structured):
    from competition_agents.methods.codeact import CodeActCollaboration, evaluate_execution
    from competition_agents.methods.catalog import BASE
    from test_locked_methods import resources
    class Client:
        calls = []
        def call(self, role, messages):
            outputs = {'planner': '{"steps":["add inputs"],"query":"sum"}' if structured else 'Add the supplied values.',
                       'retriever': '[]' if structured else 'Python sum is available.',
                       'reasoner': '{"code":"result = {\\"sum\\": sum(inputs[\\"values\\"])}"}' if structured else 'result = {"sum": sum(inputs["values"])}',
                       'summarizer': 'The result is 13.'}
            text = outputs[role]
            self.calls.append({'role': role, 'messages': messages, 'elapsed_ms': 1,
                               'input_tokens': 1, 'output_tokens': 1, 'finish_reason': 'stop'})
            return {'content': text}
    client = Client()
    client.calls = []
    runtime = CodeActCollaboration(client, {**resources(), 'knowledge': BASE/DATASETS['codeact']['knowledge']}, structured)
    result = runtime.run_task({'task': 'Sum the values', 'inputs': {'values': [6, 7]},
                               'allowed_imports': [], 'output_fields': ['sum'],
                               'reference_code': 'raise RuntimeError("REFERENCE MUST NOT RUN")', 'expected': {'sum': 'PRIVATE_EXPECTED_SENTINEL'}},
                              history_scope='test')
    runtime.memory.close()
    assert result['error'] is None
    assert result['execution_result']['result'] == {'sum': 13}
    assert evaluate_execution(result['execution_result'], {'sum': 13})['correct']
    assert len(client.calls) == 4 and len(result['messages']) == 4
    assert 'REFERENCE MUST NOT RUN' not in json.dumps(client.calls)
    assert 'PRIVATE_EXPECTED_SENTINEL' not in json.dumps(client.calls)
    assert result['metrics']['communication_bytes'] == sum(m['wire_bytes'] for m in result['messages'])
