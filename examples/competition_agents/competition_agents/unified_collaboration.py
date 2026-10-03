"""唯一主流程：角色决定记忆是否够用，传输后端不改变角色工作。"""

from __future__ import annotations

import copy
import hashlib
import time
from uuid import uuid4

from .collaboration_memory import CollaborationMemory
from .collaboration_transport import CollaborationTransport, encode
from .collaboration_fields import can_reuse_work
from .llm.response_content import normalize_lookup_query
from .role_prompts import ROLE_ORDER, role_prompt, prompt_digest


class UnifiedCollaboration:
    def __init__(self, client, resources, *, method, max_searches=2):
        if method not in ('plain', 'structured', 'latent', 'autoform', 'llmlingua'):
            raise ValueError('未知协作方式。')
        if max_searches < 1:
            raise ValueError('补查预算必须为正。')
        self.method, self.client, self.max_searches = method, client, max_searches
        self.task_kind = resources.get('task_kind', 'continuous')
        self.memory = CollaborationMemory(resources, graph=method in ('structured', 'latent'),
                                          enabled=resources.get('memory_enabled', True))
        self.transport = CollaborationTransport(method, client, self.memory.knowledge)
        self.resident = None
        if method == 'latent':
            from .llm.latentmas_chat import LatentMASChat
            self.resident = LatentMASChat(client,
                event_sink=resources.get('latent_event_sink'),
                latent_steps=resources.get('latent_steps', 40),
                memory_policy=resources.get('latent_memory_policy', 'release_transients'))

    def _call(self, role, snapshot, scope, turn_key):
        prompt = role_prompt(self.task_kind, role)
        started = time.perf_counter()
        call_log = self.resident.calls if self.resident is not None else self.client.calls
        call_count = len(call_log)
        response = {'content': '', 'action': None, 'lookup_query': None}
        allow_reuse = can_reuse_work(role, snapshot, self.task_kind)
        try:
            if self.resident is not None:
                state_scope = scope if self.memory.enabled else scope + ':' + turn_key
                response = self.resident.call_role(role, prompt, copy.deepcopy(snapshot), scope=state_scope,
                    turn_key=turn_key, packet=self.transport.packet(snapshot),
                    identifiers=self.transport.identifiers,
                    allow_reuse=allow_reuse,
                    task_kind=self.task_kind,
                    external_program=self.task_kind == 'codeact' and role == 'reasoner')
                record = self.resident.calls[-1]
            else:
                response = self.client.call(role, [
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': self.transport.render(snapshot, receiver=role)
                     + '\n\nCurrent task to complete:\n' + encode(snapshot['task'])}],
                    role_control={'allow_reuse': allow_reuse, 'task_kind': self.task_kind}
                                 if role != 'summarizer' else False,
                    shared_roles=True)
                record = self.client.calls[-1]
                record['role_prompt_sha256'] = prompt_digest(self.task_kind, role)
            content = response.get('content')
            if content is None:
                content = ''
            if not isinstance(content, str):
                raise TypeError('模型接口返回的工作内容不是文本。')
            response['content'] = content
            response['lookup_query'] = (normalize_lookup_query(response.get('lookup_query'))
                                        if response.get('action') == 'lookup' else None)
            self.outputs[role] = content
            snapshot['work'].append({'role': role, 'content': content,
                                     'action': response.get('action'),
                                     'lookup_query': response['lookup_query'],
                                     'status': record.get('finish_reason', 'unknown')})
            for message in record.get('state_issues', []):
                snapshot['diagnostics'].append({'role': role, 'kind': 'state_recovery', 'message': message})
            if record.get('context_budget_reduced'):
                snapshot['diagnostics'].append({'role': role, 'kind': 'context_budget_reduced',
                    'message': '完整输入已保留；受模型上下文上限限制，本次最多生成'
                               + str(record['effective_output_budget']) + '个词元。'})
            if record.get('finish_reason') == 'length':
                snapshot['diagnostics'].append({'role': role, 'kind': 'output_truncated',
                    'message': '输出达到统一预算；已取得的来源和实际输出继续交接。'})
            if role != 'summarizer' and response.get('action') not in ('deliver', 'lookup', 'reuse'):
                snapshot['diagnostics'].append({'role': role, 'kind': 'decision_unavailable',
                    'message': response.get('decision_error') or '工具决定未取得；不能据此声明证据充分。'})
            self.stages.append({'role': role, 'status': 'returned',
                                'action': response.get('action'), 'finish_reason': record.get('finish_reason'),
                                'elapsed_ms': (time.perf_counter() - started) * 1000})
        except Exception as exc:
            issue = {'role': role, 'kind': 'backend_error', 'message': str(exc)[:500]}
            if len(call_log) == call_count:
                # 未返回用量的失败调用仍计次数和耗时，不能从统计中消失。
                call_log.append({'role': role, 'model': self.client.model, 'input_tokens': None,
                    'output_tokens': None, 'elapsed_ms': (time.perf_counter()-started)*1000,
                    'finish_reason': 'error', 'error': issue['message'],
                    'role_prompt': prompt, 'role_prompt_sha256': prompt_digest(self.task_kind, role),
                    'response': {'content': '', 'action': None}})
            snapshot['diagnostics'].append(issue)
            self.stages.append({**issue, 'status': 'error', 'elapsed_ms': (time.perf_counter()-started)*1000})
        return response

    def _handoff(self, sender, receiver, snapshot):
        try:
            self.transport.handoff(sender, receiver, snapshot)
        except Exception as exc:
            # 压缩插件失败不能丢掉原任务、证据或阻断后续角色。
            snapshot['diagnostics'].append({'role': sender, 'kind': 'transport_error', 'message': str(exc)[:500]})

    def _merge_evidence(self, snapshot, hits, requester):
        merged = []
        for original in hits:
            hit = copy.deepcopy(original)
            key = hit['context_id']
            previous = snapshot['evidence'].get(key)
            identity = lambda value: value.get('fact_fields') or {
                'title': value.get('title', ''), 'text': value.get('text', '')}
            if previous is not None and identity(previous) != identity(hit):
                # 原编号留在审计中；不能用另一版本静默覆盖已经交接的事实。
                hit['original_context_id'] = key
                key += ':version:' + hashlib.sha256(encode(identity(hit)).encode('utf-8')).hexdigest()[:16]
                hit['context_id'] = key
                snapshot['diagnostics'].append({'role': requester, 'kind': 'source_version_conflict',
                    'message': '同一来源编号出现不同内容，保留双方并记录原编号。'})
            snapshot['evidence'].setdefault(key, hit)
            merged.append(hit)
        return merged

    def _recall_memory(self, snapshot, scope, query, requester, *, query_source):
        """按实际查询读取已存记忆，新增内容交回角色，不替角色判断充分性。"""
        if not self.memory.enabled:
            return False
        query_key = normalize_lookup_query(query).casefold()
        if query_key in self.memory_queries_checked:
            return False
        self.memory_queries_checked.add(query_key)
        self.metrics['memory_searches'] += 1
        if requester != 'initial':
            self.metrics['memory_requeries'] += 1
        before = set(snapshot['evidence'])
        new_units = []
        recalled = {'history': [], 'evidence': [], 'text': '', 'retrieval': {}}
        started = time.perf_counter()
        try:
            recalled = self.memory.recall(query, scope, self.task_kind)
        except Exception as exc:
            snapshot['diagnostics'].append({'role': 'memory', 'kind': 'memory_error', 'message': str(exc)[:500]})
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.metrics['memory_search_ms'] += elapsed_ms
        snapshot['diagnostics'].extend(recalled.get('diagnostics', []))
        history_keys = {encode(row) for row in snapshot['history']}
        for row in recalled['history']:
            if encode(row) not in history_keys:
                snapshot['history'].append(copy.deepcopy(row))
                history_keys.add(encode(row))
        if recalled.get('experience') is not None:
            snapshot['experience'] = recalled['experience']
        merged = self._merge_evidence(snapshot, recalled['evidence'], 'memory')
        for hit in merged:
            # 本轮刚查库并写入的事实不能算作跨轮旧证据复用。
            if hit.get('original_context_id', hit['context_id']) in self.turn_start_memory_ids:
                self.memory_evidence.setdefault(hit['context_id'], copy.deepcopy(hit))
        units = recalled.get('text_units', [])
        if not units and recalled.get('text'):
            units = [{'id': hashlib.sha256(recalled['text'].encode('utf-8')).hexdigest(),
                      'text': recalled['text']}]
        for unit in units:
            if unit['id'] not in self.memory_text_units:
                self.memory_text_units[unit['id']] = copy.deepcopy(unit)
                new_units.append(unit['id'])
        snapshot['remembered_text'] = '\n\n'.join(unit['text'] for unit in self.memory_text_units.values())
        new_ids = [key for key in snapshot['evidence'] if key not in before]
        self.memory_reads.append({'query': query, 'query_source': query_source, 'requester': requester,
            'elapsed_ms': elapsed_ms, 'retrieval': copy.deepcopy(recalled.get('retrieval', {})),
            'source_ids': [hit['context_id'] for hit in merged], 'new_source_ids': new_ids,
            'text_unit_ids': [unit['id'] for unit in units], 'new_text_unit_ids': new_units,
            'diagnostics': copy.deepcopy(recalled.get('diagnostics', [])),
            'sufficiency_decided_by': 'retriever_and_reasoner'})
        if requester != 'initial':
            self.metrics['memory_requery_new_evidence'] += len(new_ids)
        # 非空返回并不自动完成任务；只有新增信息才触发重新交给角色检查。
        # 没有新增或角色再次请求同一查询时，继续查库，避免重复读取无限循环。
        return bool(new_ids or new_units)

    def _lookup(self, snapshot, scope, requester, *, query=None):
        query = normalize_lookup_query(query)
        query_source = 'role_tool_argument'
        if query is None:
            # 接口异常不停止四角色流程；保底查询必须显式记录，不能冒充规划结果。
            query = snapshot['task']['question']
            query_source = 'original_question_fallback'
            snapshot['diagnostics'].append({'role': requester, 'kind': 'lookup_query_unavailable',
                'message': '未取得角色的工具查询参数，使用原题保底；此次检索未使用规划后的查询。'})
        if self._recall_memory(snapshot, scope, query, requester, query_source=query_source):
            return True
        # 记忆重查不消费查库预算；即使查库预算用完，也先尝试尚未读过的记忆查询。
        if len(self.searches) >= self.max_searches:
            snapshot['diagnostics'].append({'role': requester, 'kind': 'lookup_budget_exhausted',
                'message': '仍有补查请求，预算已用完；没有把已有证据判为充分。'})
            return False
        request = {'query': query, 'expanded': bool(self.searches)}
        audit = {'query': query, 'query_source': query_source,
                 'original_question': snapshot['task']['question'], 'requester': requester,
                 'tool_request_bytes': len(encode(request).encode('utf-8'))}
        self._external(requester, 'knowledge', request)
        started = time.perf_counter()
        try:
            hits = self.memory.search(query, scope, expanded=request['expanded'])
            hits = self._merge_evidence(snapshot, hits, requester)
            self.searches.append({**audit,
                                  'source_ids': [hit['context_id'] for hit in hits],
                                  'result': copy.deepcopy(hits), 'tool_result_bytes': len(encode(hits).encode())})
            self._external('knowledge', requester, hits)
        except Exception as exc:
            failure = {'error': str(exc)[:500]}
            self.searches.append({**audit, 'source_ids': [], **failure,
                                  'tool_result_bytes': len(encode(failure).encode('utf-8'))})
            self._external('knowledge', requester, failure)
            snapshot['diagnostics'].append({'role': requester, 'kind': 'knowledge_error', 'message': str(exc)[:500]})
            return False
        finally:
            self.metrics['knowledge_search_ms'] += (time.perf_counter()-started)*1000
        write_started = time.perf_counter()
        try:
            self.metrics['memory_facts_added'] += self.memory.observe(query, hits, scope)
        except Exception as exc:
            # 记忆写入失败不撤回本次查库结果，也不重复计作查库失败。
            snapshot['diagnostics'].append({'role': 'memory', 'kind': 'memory_write_error',
                                            'message': str(exc)[:500]})
        finally:
            self.metrics['memory_write_ms'] += (time.perf_counter()-write_started)*1000
        return True

    def _retrieve(self, snapshot, scope, turn_key):
        while True:
            result = self._call('retriever', snapshot, scope, turn_key)
            if result.get('action') in ('deliver', 'reuse'):
                return
            if result.get('action') is None:
                # 接口失败不等于证据充分，也不能伪造一次模型查库决定。
                return
            if not self._lookup(snapshot, scope, 'retriever', query=result.get('lookup_query')):
                return

    def _reason(self, snapshot, scope, turn_key):
        while True:
            result = self._call('reasoner', snapshot, scope, turn_key)
            if result.get('action') in ('deliver', 'reuse'):
                return result
            if result.get('action') is None:
                # 工作结果仍完整交付；不能因控制接口异常把结果删除。
                return result
            self._handoff('reasoner', 'retriever', snapshot)
            if not self._lookup(snapshot, scope, 'reasoner', query=result.get('lookup_query')):
                return result
            self._retrieve(snapshot, scope, turn_key)
            self._handoff('retriever', 'reasoner', snapshot)

    def _execute(self, response, snapshot, scope, turn_key):
        from .methods.codeact import generated_code
        from .restricted_python_codeact import run_restricted_python
        code = response['content']
        try:
            try:
                code = generated_code(code)
            except (ValueError, SyntaxError):
                # 所有方式共用一次程序格式修正；不读取参考程序、答案或评分。
                snapshot['diagnostics'].append({'role': 'reasoner', 'kind': 'program_format',
                    'message': '请生成可执行程序并给 result 赋值，不能用答案代替程序。'})
                repaired = self._reason(snapshot, scope, turn_key)
                code = generated_code(repaired['content'])
            execute_started = time.perf_counter()
            try:
                execution = run_restricted_python(code, inputs=snapshot['task']['inputs'],
                    allowed_imports=snapshot['task']['allowed_imports'], timeout_ms=5000)
            finally:
                self.metrics['execution_ms'] += (time.perf_counter()-execute_started)*1000
            execution['generated_code'] = code
        except Exception as exc:
            execution = {'sandbox_status': 'error', 'stderr': str(exc)[:1000], 'generated_code': code}
        self._external('reasoner', 'executor', {'code': code, 'inputs': snapshot['task']['inputs'],
            'allowed_imports': snapshot['task']['allowed_imports'], 'timeout_ms': 5000})
        self._external('executor', 'summarizer', execution)
        snapshot['execution'] = execution

    def _external(self, sender, receiver, payload):
        content = encode(payload)
        envelope = {'from': sender, 'to': receiver, 'content': content}
        self.transport.messages.append({**envelope, 'content_bytes': len(content.encode()),
                                        'wire_bytes': len(encode(envelope).encode()), 'kind': 'external_tool'})

    def run(self, question, *, history_scope='single'):
        if self.task_kind != 'continuous':
            raise ValueError('代码任务需要明确的程序输入。')
        return self._run({'question': question}, history_scope)

    def run_task(self, task, *, history_scope):
        if self.task_kind != 'codeact':
            raise ValueError('当前入口不是代码任务。')
        public = {key: copy.deepcopy(task[key]) for key in (
            'task', 'inputs', 'allowed_imports', 'output_fields', 'output_schema') if key in task}
        public['question'] = public.pop('task')
        return self._run(public, history_scope)

    def _run(self, task, scope):
        started, begin = time.perf_counter(), len(self.client.calls)
        resident_begin = len(self.resident.calls) if self.resident is not None else 0
        self.outputs, self.stages, self.searches = {}, [], []
        self.memory_reads, self.memory_queries_checked = [], set()
        self.memory_evidence, self.memory_text_units = {}, {}
        self.turn_start_memory_ids = set(self.memory.records)
        self.transport.messages = []
        self.metrics = {'memory_search_ms': 0., 'memory_write_ms': 0., 'knowledge_search_ms': 0.,
                        'memory_facts_added': 0, 'memory_searches': 0, 'memory_requeries': 0,
                        'memory_requery_new_evidence': 0,
                        'execution_ms': 0.}
        snapshot = {'task': copy.deepcopy(task), 'history': [], 'evidence': {}, 'work': [],
                    'execution': None, 'diagnostics': [], 'experience': None, 'remembered_text': ''}
        self._recall_memory(snapshot, scope, task['question'], 'initial', query_source='original_question')
        turn_key = uuid4().hex
        self._call('planner', snapshot, scope, turn_key)
        self._handoff('planner', 'retriever', snapshot)
        self._retrieve(snapshot, scope, turn_key)
        self._handoff('retriever', 'reasoner', snapshot)
        reasoned = self._reason(snapshot, scope, turn_key)
        if self.task_kind == 'codeact':
            self._execute(reasoned, snapshot, scope, turn_key)
        self._handoff('reasoner', 'summarizer', snapshot)
        final = self._call('summarizer', snapshot, scope, turn_key)
        prediction = final['content'].strip()
        if not prediction:
            snapshot['diagnostics'].append({'role': 'summarizer', 'kind': 'empty_final_answer',
                                            'message': '最终角色没有返回可交付的回答。'})
        backend_errors = [s for s in self.stages if s['status'] == 'error']
        execution = snapshot['execution']
        error = '; '.join(s['role'] + ': ' + s['message'] for s in backend_errors) or None
        if execution and execution.get('sandbox_status') != 'ok':
            error = (error + '; ' if error else '') + '代码执行失败：' + (
                execution.get('stderr') or execution.get('blocked_reason') or execution.get('sandbox_status', ''))
        if not prediction:
            error = (error + '; ' if error else '') + '最终回答为空。'
        # 回答错误、达到输出预算、执行器返回真实错误都不同于提前终止四角色流程。
        returned_roles = {stage['role'] for stage in self.stages if stage['status'] == 'returned'}
        completed = bool(prediction) and set(ROLE_ORDER) <= returned_roles
        write_started = time.perf_counter()
        committed = {'written': [], 'experience': None, 'experience_writes': 0}
        memory_committed = not self.memory.enabled
        lifecycle = {'released': False, 'removed_positions': 0, 'milliseconds': 0.}
        try:
            committed = self.memory.commit(task['question'], prediction, snapshot, scope,
                                           self.task_kind, completed=completed)
            snapshot['diagnostics'].extend(committed.get('diagnostics', []))
            memory_committed = not any(d.get('kind') in ('evidence_write_error', 'memory_write_error')
                                       for d in committed.get('diagnostics', []))
        except Exception as exc:
            snapshot['diagnostics'].append({'role': 'memory', 'kind': 'memory_write_error', 'message': str(exc)[:500]})
        self.metrics['memory_write_ms'] += (time.perf_counter()-write_started)*1000
        if self.resident is not None:
            # 回答、事实、来源写回完成后，当前任务才没有活跃消费者。
            lifecycle = self.resident.finish_task(persisted=memory_committed)
        calls = self.resident.calls[resident_begin:] if self.resident is not None else self.client.calls[begin:]
        memory_ids = list(self.memory_evidence)
        def measured_total(key):
            values = [call.get(key) for call in calls]
            return sum(values) if all(value is not None for value in values) else None
        agent_messages = [message for message in self.transport.messages if message.get('kind') != 'external_tool']
        tool_messages = [message for message in self.transport.messages if message.get('kind') == 'external_tool']
        self.metrics.update(elapsed_ms=(time.perf_counter()-started)*1000,
            input_tokens=measured_total('input_tokens'), output_tokens=measured_total('output_tokens'),
            token_usage_complete=all(call.get('input_tokens') is not None and call.get('output_tokens') is not None for call in calls),
            model_service_ms=sum(c['elapsed_ms'] for c in calls), knowledge_searches=len(self.searches),
            knowledge_request_bytes=sum(search['tool_request_bytes'] for search in self.searches),
            knowledge_result_bytes=sum(search['tool_result_bytes'] for search in self.searches),
            lookup_query_fallbacks=sum(search['query_source'] != 'role_tool_argument' for search in self.searches),
            memory_evidence_delivered=len(memory_ids),
            memory_text_units_delivered=len(self.memory_text_units),
            memory_only_tasks=int(bool(memory_ids) and not self.searches and completed),
            communication_bytes=None if self.resident is not None else sum(m['wire_bytes'] for m in agent_messages),
            external_tool_message_bytes=sum(m['wire_bytes'] for m in tool_messages),
            message_content_bytes=sum(m['content_bytes'] for m in self.transport.messages),
            experience_reads=int(bool(snapshot.get('experience'))), experience_applied=0,
            experience_writes=committed.get('experience_writes', 0))
        if self.resident is not None:
            self.metrics.update(state_peak_bytes=max(self.resident.peak,
                max((c.get('state_peak_bytes', 0) for c in calls), default=0)),
                handoff_payload_bytes=sum(c.get('handoff', {}).get('payload_bytes', 0) for c in calls),
                handoff_measurement='resident_state', state_reuse_count=sum(c.get('reused_input_tokens', 0) > 0 for c in calls),
                cache_recovery_positions=sum(c.get('cache_recovery_positions', 0) for c in calls),
                forward_positions=sum(c.get('forward_positions', 0) for c in calls),
                facts_encoded=sum(c.get('facts_encoded', 0) for c in calls),
                facts_reused_without_encoding=sum(c.get('facts_reused_without_encoding', 0) for c in calls),
                active_hidden_handoff_bytes=sum(c.get('handoff', {}).get('active_hidden_bytes', 0) for c in calls),
                cache_reclaimed_positions=lifecycle['removed_positions'],
                cache_reclamation_ms=lifecycle['milliseconds'])
        return {'question': task['question'], 'prediction': prediction, 'error': error,
                'history_scope': scope, 'implementation': 'shared-role-typed-decisions-task-lifecycle-v4',
                'cache_lifecycle': lifecycle,
                'workflow_complete': completed, 'workflow_issues': snapshot['diagnostics'], 'stages': self.stages,
                'unreturned_roles': [role for role in ROLE_ORDER if role not in returned_roles],
                'role_prompt_hashes': {r: next((c['role_prompt_sha256'] for c in calls if c['role'] == r),
                                               prompt_digest(self.task_kind, r)) for r in ROLE_ORDER},
                'calls': calls, 'messages': self.transport.messages, 'outputs': self.outputs, 'searches': self.searches,
                'wire_identifier_map': copy.deepcopy(self.transport.identifiers),
                'execution_result': execution or {}, 'source_evidence': snapshot['evidence'],
                'memory_read': {'history': snapshot['history'], 'evidence': self.memory_evidence,
                    'text': snapshot['remembered_text'], 'text_units': list(self.memory_text_units.values()),
                    'query': task['question'], 'reads': self.memory_reads,
                    'retrieval': self.memory_reads[0]['retrieval'] if self.memory_reads else {}},
                'memory_written': committed['written'], 'experience_written': committed['experience'],
                'retrieval_audit': {'memory_source_ids': memory_ids, 'source_ids': list(snapshot['evidence']),
                    'knowledge_searches': len(self.searches), 'decision_source': 'retriever_and_reasoner',
                    'query_source': 'role_tool_argument_with_explicit_fallback',
                    'query_fallbacks': self.metrics['lookup_query_fallbacks']},
                'metrics': self.metrics}

    def close(self):
        try:
            self.memory.close()
        finally:
            if self.resident is not None:
                self.resident.close()


def create_collaboration(client, resources, *, method):
    return UnifiedCollaboration(client, resources, method=method,
                                max_searches=resources.get('max_searches', 2))


def default_resources(knowledge):
    """兼容显式对象入口；不采用哈希向量冒充相似度模型。"""
    from .embedding import embed_text
    embed = lambda text: embed_text(text, provider='local')
    return {'knowledge': knowledge, 'embed': embed,
            'encode': lambda text: embed(text).vector, 'task_kind': 'continuous'}
