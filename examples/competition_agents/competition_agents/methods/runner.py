"""运行固定方法并将真实调用、消息与检索记录转换为前端事件。"""
import json
from pathlib import Path
import time
import os
from uuid import uuid4

from competition_agents.answer_evaluation import answer_coverage
from competition_agents.runtime_events import configure_event_log, emit_runtime_event
from .catalog import CATALOG, DATASETS, METHODS, BASE, create_runtime, validate

AGENTS = {'planner': 'Planner', 'retriever': 'Retriever', 'reasoner': 'Reasoner', 'summarizer': 'Summarizer', 'autoform': 'AutoForm'}


def memory_rows(value, scope):
    if isinstance(value, list):
        return [{**row, 'memory_id': row.get('context_id', row.get('memory_id', str(i))),
                 'summary': row.get('text', row.get('summary', '')), 'memory_layer': 'evidence'}
                for i, row in enumerate(value)]
    if not isinstance(value, dict):
        return []
    if value.get('evidence'):
        return memory_rows(list(value['evidence'].values()), scope)
    if value.get('text') and value.get('committed', True):
        return [{'memory_id': scope, 'summary': value['text'], 'memory_layer': 'evidence',
                 'retrieval': value.get('retrieval', {})}]
    return [{'memory_id': key, 'summary': '来源事实：' + key, 'memory_layer': 'evidence'}
            for key in value.get('source_ids', [])]


def load_tasks(dataset, count):
    rows = json.loads((BASE/dataset['file']).read_text(encoding='utf-8'))
    if len(rows) != dataset['taskCount']:
        raise ValueError('数据文件实际题数与配置不一致，禁止截短运行后宣称完成。')
    if type(count) is not int or not 1 <= count <= len(rows):
        raise ValueError('请求题数超出数据文件范围。')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('数据文件包含重复任务编号。')
    if dataset['datasetType'] == 'codeact':
        return [{'id': row['id'], 'question': row['task'], 'scope': 'codeact',
                 'answer': row['expected'], 'public_task': {**{k: row[k] for k in ('task', 'inputs', 'allowed_imports')},
                                                          'output_fields': list(row['expected']) if isinstance(row['expected'], dict) else [],
                                                          'output_schema': {'type': 'object' if isinstance(row['expected'], dict) else 'array'}}}
                for row in rows[:count]]
    if [row['turn_id'] for row in rows] != list(range(1, len(rows) + 1)) or len({row['group_id'] for row in rows}) != 1:
        raise ValueError('连续数据必须是同一会话从第一轮开始的完整顺序。')
    if any(set(row) != {'id', 'group_id', 'turn_id', 'question'} for row in rows):
        raise ValueError('公开问题文件不得包含答案或人工提供的历史上下文。')
    questions = rows[:count]
    references = {row['id']: row['reference_answers'] for row in json.loads((BASE/dataset['references']).read_text(encoding='utf-8'))}
    return [{**row, 'answer': references[row['id']], 'scope': row.get('group_id', row['id'])} for row in questions]


class ObservedClient:
    def __init__(self, client, context):
        self.client, self.context = client, context
        self.calls = client.calls

    def __getattr__(self, name):
        return getattr(self.client, name)

    def call(self, role, messages, **kwargs):
        emit_runtime_event({**self.context, 'event_type': 'agent_start', 'to_agent': AGENTS.get(role, role)})
        result = self.client.call(role, messages, **kwargs)
        call = self.calls[-1]
        emit_runtime_event({**self.context, 'event_type': 'agent_output', 'from_agent': AGENTS.get(role, role),
                            'return_result': result, 'text_chars': len(result.get('content') or ''),
                            'elapsed_ms': call['elapsed_ms']})
        return result


def frontend_task(task, result, index, method):
    metrics = result['metrics']
    calls = result['calls']
    if 'public_task' in task:
        from .codeact import evaluate_execution
        coverage = evaluate_execution(result.get('execution_result', {}), task['answer'])
    else:
        coverage = answer_coverage(result['prediction'], task['answer'])
    messages = [{**m, 'sender': AGENTS.get(m['from'], m['from']), 'receiver': AGENTS.get(m['to'], m['to']),
                 'payload': m['content'], 'char_cost': len(m['content']), 'actual_message_bytes': m['wire_bytes']}
                for m in result.get('messages', [])]
    metrics['text_chars'] = sum(len(m['content']) for m in messages if m.get('kind') != 'external_tool')
    metrics['actual_message_bytes'] = metrics.get('communication_bytes')
    metrics.setdefault('handoff_payload_bytes', metrics.get('communication_bytes'))
    metrics.setdefault('handoff_measurement', 'serialized_message')
    hits = memory_rows(result.get('memory_read'), task['id'] + ':read')
    writes = memory_rows(result.get('memory_written'), task['id'] + ':write')
    if result.get('experience_written'):
        writes.append(result['experience_written'])
    return {**result, 'task_id': task['id'], 'task_index': index, 'turn_id': task.get('turn_id'),
            'status': ('failed' if not result['prediction'] else 'completed_with_issues'
                       if result['error'] or result.get('workflow_issues') or not result.get('workflow_complete', True) else 'completed'),
            'final_answer': result['prediction'], 'expected_answer': task['answer'], 'answer_coverage': coverage,
            'answer_correct': coverage['correct'],
            'answer_accuracy': coverage['score'],
            'elapsed_ms': metrics['elapsed_ms'], 'elapsed_seconds': metrics['elapsed_ms']/1000,
            'model_calls': len(calls), 'text_chars': metrics['text_chars'],
            'actual_message_bytes': metrics.get('communication_bytes'), 'messages': messages,
            'memory_hits': hits, 'memory_hit_ids': [row['memory_id'] for row in hits],
            'memory_hit_count': len(hits), 'memory_writes': writes,
            'memory_retrieval_mode': method['retrieval'], 'memory_organization': method['memory'],
            'agent_outputs': [{'agent': AGENTS.get(c['role'], c['role']),
                               'output': c.get('raw_response_content') or c.get('response', {}).get('content', ''),
                               'elapsed_ms': c['elapsed_ms']} for c in calls]}


def run(config, output_dir, event_log, *, client=None, resources=None):
    config = {**config,
              'latentSteps': config.get('latentSteps', CATALOG.get('defaultLatentSteps', 40)),
              'latentMemoryPolicy': config.get('latentMemoryPolicy', CATALOG.get('defaultLatentMemoryPolicy', 'release_transients'))}
    method = validate(config)
    dataset = DATASETS[config['dataset']]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    context = {'run_id': config.get('runId') or os.environ.get('COMPETITION_RUN_ID') or uuid4().hex,
               'experiment_mode': method['experimentMode'], 'dataset_file': dataset['file'],
               'dataset_type': dataset['datasetType'], 'task_count': config['taskCount'],
               'memory_retrieval_mode': method['retrieval'], 'memory_organization': method['memory']}
    configure_event_log(str(event_log), context)
    tasks = load_tasks(dataset, config['taskCount'])
    emit_runtime_event({**context, 'event_type': 'task_queue',
                        'tasks': [{'task_index': i, 'task_id': t['id'], 'task_title': t['question']}
                                  for i, t in enumerate(tasks, 1)]})
    started = time.perf_counter()
    if client is None:
        if config['backend'] == 'local':
            from competition_agents.llm.native_local_chat import NativeLocalChat
            client = NativeLocalChat.from_pretrained(config['localModel'], max_tokens=config.get('maxTokens', 1024), seed=42, threads=4,
                                                    output_repeat_ngram_size=config.get('outputRepeatNgramSize', CATALOG.get('defaultOutputRepeatNgramSize', 0)))
        else:
            from competition_agents.llm.audited_chat import AuditedChat
            client = AuditedChat(config['remoteModel'], max_tokens=config.get('maxTokens', 1024), seed=42)
    if resources is None:
        from types import SimpleNamespace
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', local_files_only=True, device='cpu')
        encode = lambda text: encoder.encode(text, normalize_embeddings=True).tolist()
        resources = {'encoder': encoder, 'encode': encode,
                     'embed': lambda text: SimpleNamespace(vector=encode(text), provider='local', model='sentence-transformers/all-MiniLM-L6-v2')}
    resources = {**resources, 'knowledge': BASE/dataset['knowledge'],
                  'task_kind': dataset['datasetType'],
                  'memory_enabled': config.get('memoryEnabled', True),
                  'max_searches': config.get('maxSearches', 2),
                  'latent_steps': config['latentSteps'],
                  'latent_memory_policy': config['latentMemoryPolicy'],
                 'experience_db': (output/'experience.sqlite' if config.get('isolatedMemory') else
                                   os.environ.get('HYPERGRAPH_MEMORY_DB') or output/'experience.sqlite')}
    task_context = dict(context)
    def latent_event(event):
        if event.get('event') == 'start':
            emit_runtime_event({**task_context, 'event_type': 'agent_start',
                                'to_agent': AGENTS.get(event['role'], event['role'])})
            return
        emit_runtime_event({**task_context, 'event_type': 'agent_output',
                            'from_agent': AGENTS.get(event['role'], event['role']),
                            'elapsed_ms': event['seconds']*1000,
                            'return_result': {'content': event.get('answer') or '',
                                              'lookup_query': event.get('lookup_query'),
                                              'representation': 'final_text' if event.get('answer') else 'hidden_state'}})
        following = {'planner': 'Retriever', 'retriever': 'Reasoner', 'reasoner': 'Summarizer'}
        if event['role'] in following:
            emit_runtime_event({**task_context, 'event_type': 'state_transfer',
                                'from_agent': AGENTS[event['role']], 'to_agent': following[event['role']],
                                'action_type': 'reuse_computation_state', 'actual_message_bytes': None,
                                'handoff_payload_bytes': (event.get('handoff') or {}).get('payload_bytes'),
                                'text_chars': 0, 'return_result': {
                                    **(event.get('handoff') or {}),
                                    'representation': 'shared_computation_state',
                                    'summary': '续接同进程计算状态；跨机器传输量未测量。'}})
    resources['latent_event_sink'] = latent_event
    runtime = create_runtime(config, ObservedClient(client, task_context), resources)
    if config.get('warmup') and config['backend'] == 'local':
        from competition_agents.llm.native_local_chat import NativeLocalChat
        warm = NativeLocalChat(client.bundle, max_tokens=1, output_repeat_ngram_size=client.output_repeat_ngram_size)
        warm.call('summarizer', [{'role': 'user', 'content': 'Say ready.'}])
        del warm
    setup_ms = (time.perf_counter()-started)*1000
    rows = []
    try:
        for index, task in enumerate(tasks, 1):
            task_context.update(task_id=task['id'], task_index=index, question=task['question'], task_title=task['question'])
            emit_runtime_event({**task_context, 'event_type': 'task_start', 'to_agent': 'Planner'})
            raw = (runtime.run_task(task['public_task'], history_scope=task['scope']) if 'public_task' in task
                   else runtime.run(task['question'], history_scope=task['scope']))
            row = frontend_task(task, raw, index, method)
            from .experiment_metrics import task_metrics
            task_metrics(row, method['value'])
            rows.append(row)
            audit = row.get('retrieval_audit', {})
            memory_read = row.get('memory_read')
            query = audit.get('query_kind') or (memory_read.get('query') if isinstance(memory_read, dict) else None) or task['question']
            emit_runtime_event({**task_context, 'event_type': 'memory_query',
                                'from_agent': 'Retriever',
                                'to_agent': 'memory', 'recorded_after_task': True,
                                'elapsed_ms': row['metrics'].get('memory_search_ms', 0),
                                'input_params': {'query': query},
                                'return_result': {'hit_count': len(row['memory_hits']), 'memory_layer': 'evidence'}})
            for event_type, records in (('memory_hit', row['memory_hits']), ('memory_write', row['memory_writes'])):
                for record in records:
                    emit_runtime_event({**task_context, 'event_type': event_type,
                                        'from_agent': 'memory' if event_type == 'memory_hit' else 'Summarizer',
                                        'to_agent': 'Retriever' if event_type == 'memory_hit' else 'memory',
                                        'memory_refs': [record['memory_id']], 'return_result': record,
                                        'recorded_after_task': True})
            for message in row['messages']:
                emit_runtime_event({**task_context, 'event_type': 'agent_message',
                                    'from_agent': message['sender'], 'to_agent': message['receiver'],
                                    'payload': message['content'], 'text_chars': message['char_cost'],
                                    'recorded_after_task': True,
                                    'actual_message_bytes': message['actual_message_bytes']})
            if row.get('execution_result'):
                emit_runtime_event({**task_context, 'event_type': 'execution_result', 'from_agent': 'Executor',
                                    'return_result': row['execution_result'], 'recorded_after_task': True})
            emit_runtime_event({**task_context, 'event_type': 'task_failed' if not row.get('workflow_complete', True) else 'task_done',
                                'status': row['status'], 'from_agent': 'Summarizer',
                                'elapsed_ms': row['elapsed_ms'], 'text_chars': row['text_chars'],
                                'return_result': {'answer': row['prediction'], 'final_answer': row['prediction'], 'error': row['error']}})
            (output/'tasks.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        runtime.close()
    metrics = {key: sum(r['metrics'].get(key, 0) or 0 for r in rows) for key in (
        'input_tokens', 'output_tokens', 'elapsed_ms', 'model_service_ms', 'knowledge_search_ms',
        'knowledge_searches', 'memory_search_ms', 'memory_write_ms', 'experience_reads', 'experience_applied', 'experience_writes')}
    for key in ('input_tokens', 'output_tokens'):
        values = [row['metrics'].get(key) for row in rows]
        metrics[key] = sum(values) if all(value is not None for value in values) else None
    metrics['token_usage_complete'] = all(row['metrics'].get('token_usage_complete', True) for row in rows)
    metrics.update(actual_message_bytes=(sum(r['actual_message_bytes'] for r in rows)
                                        if all(r['actual_message_bytes'] is not None for r in rows) else None),
                   text_chars=sum(r['text_chars'] for r in rows), task_count=len(rows),
                   elapsed_seconds=metrics['elapsed_ms']/1000, model_calls=sum(r['model_calls'] for r in rows))
    metrics['state_peak_bytes'] = max((r['metrics'].get('state_peak_bytes', 0) for r in rows), default=0)
    metrics['state_reuse_count'] = sum(r['metrics'].get('state_reuse_count', 0) for r in rows)
    metrics['message_count'] = sum(len(r['messages']) for r in rows)
    metrics['handoff_payload_bytes'] = sum(r['metrics'].get('handoff_payload_bytes') or 0 for r in rows)
    metrics['handoff_measurement'] = 'resident_state' if method['value'] == 'latent' else 'serialized_message'
    metrics['execution_ms'] = sum(r['metrics'].get('execution_ms', 0) for r in rows)
    metrics['failed_tasks'] = sum(not r.get('workflow_complete', True) for r in rows)
    metrics['completed_tasks'] = len(rows) - metrics['failed_tasks']
    metrics['tasks_with_issues'] = sum(r['status'] == 'completed_with_issues' for r in rows)
    metrics['memory_hit_rate'] = sum(bool(r['memory_hits']) for r in rows)/len(rows)
    from .experiment_metrics import aggregate
    metrics.update(aggregate(rows))
    from hashlib import sha256
    result = {**context, 'schema_version': 3, 'baseline_version': CATALOG['version'],
              'backend': config['backend'],
              'dataset_sha256': sha256((BASE/dataset['file']).read_bytes()).hexdigest(),
              'knowledge_corpus_sha256': json.loads((BASE/dataset['knowledge']/'manifest.json').read_text(encoding='utf-8'))['corpus_sha256'],
              'evaluation_metric': 'execution_value_coverage' if dataset['datasetType'] == 'codeact' else 'reference_coverage',
              'status': ('failed' if metrics['failed_tasks'] else 'completed_with_issues'
                         if metrics['tasks_with_issues'] else 'completed'),
              'model': client.model, 'setup_ms_excluded': setup_ms, 'run_config': config,
              'full_workflow_timing_valid': all(r.get('workflow_complete', True) for r in rows),
              'role_prompt_hashes': rows[0].get('role_prompt_hashes', {}) if rows else {},
              'extractive_fallback_enabled': False,
              'runs': [{**context, 'mode': method['value'], 'metrics': metrics, 'task_results': rows,
                        'answer_accuracy': sum(r['answer_accuracy'] for r in rows)/len(rows),
                        'memory': list(getattr(runtime.memory, 'records', {}).values())}]}
    emit_runtime_event({**context, 'event_type': 'run_done', 'elapsed_ms': metrics['elapsed_ms'],
                        'status': result['status'],
                        'return_result': {'completed': metrics['completed_tasks'], 'failed': metrics['failed_tasks'], 'tasks': len(rows)}})
    result['runs'][0]['events'] = [json.loads(line) for line in Path(event_log).read_text(encoding='utf-8').splitlines() if line.strip()]
    (output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result
