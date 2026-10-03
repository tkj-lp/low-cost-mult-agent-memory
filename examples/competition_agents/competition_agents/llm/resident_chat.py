"""驻留完整缓存的协作后端；工具动作和最终答案使用标准生成接口。"""

from __future__ import annotations

import hashlib
import copy
import json
import time

from .experimental_state_chat import cache_bytes
from .generation import standard_generate
from .forward_timing import ForwardTiming
from .native_local_chat import state_message, text_decoder
from .response_content import final_content
from ..collaboration_transport import projected_history
from ..role_prompts import shared_instruction_parts, split_role_prompt
from .context_budget import ContextCapacityError, context_limit
from ..latent_hidden.state_lifecycle import TaskStateLifecycle


class ResidentChat:
    """同任务跨角色保留完整状态，任务提交后回收工作状态；不压缩有效缓存。"""

    def __init__(self, native, *, event_sink=None):
        self.bundle, self.model = native.bundle, native.model
        self.max_tokens = native.max_tokens
        self.event_sink = event_sink
        self.calls = []
        self.reset()

    def reset(self):
        self.lifecycle = TaskStateLifecycle()
        self.past = self.last_hidden = None
        self.scope = self.turn_key = None
        self.ids, self.encoded_facts, self.encoded_sources = [], {}, {}
        self.selected_ids = None
        self.encoded_history, self.encoded_aliases = set(), set()
        self.experience_signature, self.execution_signature = None, None
        self.diagnostic_count = 0
        self.peak = 0
        self.identifiers = {}
        self.shared_instruction_signature = None
        self.shared_instruction_signatures = set()

    def finish_task(self, *, persisted):
        """只有任务信息已写入记忆或显式关闭记忆，才允许释放任务计算状态。"""
        started = time.perf_counter()
        before = self._state_size()['payload_bytes']
        if not persisted:
            return {'released': False, 'removed_positions': 0, 'milliseconds': 0.,
                    'reason': 'memory_commit_not_confirmed'}
        positions = len(self.ids)
        self.past = self.last_hidden = None
        self.past, self.ids = self.lifecycle.restore_prefix()
        self.turn_key, self.selected_ids = None, None
        self.encoded_facts, self.encoded_sources = {}, {}
        self.encoded_history, self.encoded_aliases = set(), set()
        self.experience_signature = self.execution_signature = None
        self.diagnostic_count = 0
        self.identifiers = {}
        if str(self.bundle.device).startswith('cuda'):
            self.bundle.torch.cuda.synchronize(self.bundle.device)
        # 当前题的峰值已随角色记录保存，下一题从回收后的实际驻留量重新计。
        self.peak = self._state_size()['payload_bytes']
        return {'released': True, 'removed_positions': positions - len(self.ids),
                'retained_prefix_positions': len(self.ids), 'before_bytes': before,
                'after_bytes': self._state_size()['payload_bytes'],
                'milliseconds': (time.perf_counter() - started) * 1000,
                'policy': 'full_task_state_until_memory_commit_then_restore_shared_prefix'}

    @staticmethod
    def _signature(value):
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()

    def acknowledge_history(self, history):
        # 本轮原题和最终回答已在缓存中；登记记忆的同一记录，避免下轮重输。
        self.encoded_history.update(self._signature(row) for row in history)
        self.peak = max(self.peak, self._state_size()['payload_bytes'])

    def _state_size(self):
        hidden_bytes = int(self.last_hidden.numel() * self.last_hidden.element_size()) if self.last_hidden is not None else 0
        metadata = {'scope': self.scope, 'turn_key': self.turn_key, 'next_position': len(self.ids),
                    'fact_signatures': self.encoded_facts, 'source_titles': self.encoded_sources,
                    'history_signatures': sorted(self.encoded_history),
                    'alias_signatures': sorted(self.encoded_aliases), 'selected_ids': self.selected_ids,
                    'experience_signature': self.experience_signature,
                    'execution_signature': self.execution_signature, 'diagnostic_count': self.diagnostic_count,
                    'identifier_table': self.identifiers}
        metadata['shared_instruction_signature'] = self.shared_instruction_signature
        metadata['shared_instruction_signatures'] = sorted(self.shared_instruction_signatures)
        metadata_bytes = len(json.dumps(metadata, ensure_ascii=False).encode('utf-8'))
        # 官方生成接收整段逻辑词元历史，其整数张量载荷也属于可续接状态。
        token_history_bytes = len(self.ids) * 8  # 生成输入为六十四位整数。
        prefix_bytes = cache_bytes(self.lifecycle.prefix_cache) + len(self.lifecycle.prefix_ids) * 8
        return {'cache_bytes': cache_bytes(self.past), 'active_hidden_bytes': hidden_bytes,
                'retained_prefix_bytes': prefix_bytes,
                'token_history_bytes': token_history_bytes, 'metadata_bytes': metadata_bytes,
                'payload_bytes': cache_bytes(self.past) + hidden_bytes + token_history_bytes + metadata_bytes + prefix_bytes}

    def _ensure_state(self):
        if not self.ids or (self.past is not None and self.past.get_seq_length() == len(self.ids)):
            return
        # 异常后的可变缓存不可直接续接；保留原始词元重建，实际重复计算单列。
        original = self.ids[:]
        self.past, self.last_hidden, self.ids = None, None, []
        try:
            values = self.bundle.torch.tensor([original], device=self.bundle.device, dtype=self.bundle.torch.long)
            self._forward(ids=values)
        except Exception:
            self.past, self.last_hidden, self.ids = None, None, original
            raise
        self.metrics['cache_recovery_positions'] += len(original)

    def close(self):
        # 会话结束还要释放公共前缀；任务内的工作状态由 finish_task 管理。
        self.reset()

    def _forward(self, *, ids=None, embeds=None):
        torch = self.bundle.torch
        values = ids if ids is not None else embeds
        count = int(values.shape[1])
        old = self.past.get_seq_length() if self.past is not None else 0
        limit = context_limit(self.bundle)
        if limit is not None and old + count > limit:
            raise ContextCapacityError(f'完整状态加新增输入为{old + count}个词元，超过模型上限{limit}；未执行超限前向。')
        options = {'past_key_values': self.past, 'use_cache': True, 'return_dict': True,
                   'attention_mask': torch.ones((1, old + count), device=values.device, dtype=torch.long),
                   'cache_position': torch.arange(old, old + count, device=values.device)}
        options['input_ids' if ids is not None else 'inputs_embeds'] = values
        timer = ForwardTiming(torch, values.device)
        timer.before()
        with torch.inference_mode():
            output = text_decoder(self.bundle)(**options)
        timer.after()
        self.metrics['appended_forward_ms'] += sum(timer.milliseconds())
        self.past = output.past_key_values
        # 本路径不执行末层反馈；无用的末层切片会拖住整段输出张量的存储。
        # 完整各层缓存保持原样，仅不保留未被下游读取的输出引用。
        self.last_hidden = None
        # 连续位置也占据真实缓存位置；占位词元只用于生成接口定位后缀，不重新送入模型。
        self.ids.extend(ids[0].tolist() if ids is not None else
                        [self.bundle.tokenizer.pad_token_id or 0] * count)
        self.metrics['forward_positions'] += count
        self.peak = max(self.peak, self._state_size()['payload_bytes'])

    def _tokens(self, text):
        return self.bundle.tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids'].to(self.bundle.device)

    def _append(self, text, category):
        started = time.perf_counter()
        ids = self._tokens(text)
        if not ids.shape[1]:
            return
        self.inputs.append({'category': category, 'text': text, 'tokens': int(ids.shape[1])})
        self.metrics['input_tokens'] += int(ids.shape[1])
        self._forward(ids=ids)
        self.metrics['input_processing_ms'] += (time.perf_counter() - started) * 1000

    def _render(self, role, content, *, open_assistant=False):
        # 公共规则仅在起始处使用系统角色；后续职责作为本次请求，不能插入新的系统消息。
        messages = [{'role': role, 'content': content}]
        text = state_message(self.bundle, messages, open_assistant=open_assistant, continuation=bool(self.ids))
        bos = self.bundle.tokenizer.bos_token
        if self.past is not None and bos and text.startswith(bos):
            text = text[len(bos):]
        return text

    def _data(self, content, category):
        self._append(self._render('user', json.dumps(content, ensure_ascii=False, separators=(',', ':'))), category)

    def _evidence(self, packet):
        fresh_sources = {key: value for key, value in packet['sources'].items()
                         if self.encoded_sources.get(key) != value}
        fresh, signatures = [], {}
        for row in packet['facts']:
            signature = hashlib.sha256(json.dumps(row, ensure_ascii=False).encode()).hexdigest()
            if self.encoded_facts.get(row[0]) != signature:
                fresh.append(row)
                signatures[row[0]] = signature
            else:
                self.metrics['facts_reused_without_encoding'] += 1
        aliases = [row for row in packet.get('aliases', []) if self._signature(row) not in self.encoded_aliases]
        if fresh or fresh_sources or aliases:
            self._data({'new_original_evidence': {
                'columns': packet['columns'], 'sources': fresh_sources,
                'facts': fresh, 'aliases': aliases}}, 'external_facts')
            # 只有实际前向完成才能登记已编码，失败不能使下一角色跳过未输入事实。
            self.encoded_facts.update(signatures)
            self.encoded_sources.update(fresh_sources)
            self.encoded_aliases.update(self._signature(row) for row in aliases)
            self.metrics['facts_encoded'] += len(fresh)
        selected = tuple(row[0] for row in packet['facts'])
        if selected != self.selected_ids:
            # 缓存命中仍需传达本题选中了什么；不得把“已编码”误当作“本轮已交接”。
            self._data({'evidence_selected_for_current_task': list(selected)}, 'evidence_selection')
            self.selected_ids = selected

    def call_role(self, role, prompt, snapshot, *, scope, turn_key, packet, external_program=False,
                  identifiers=None, allow_reuse=True, task_kind='continuous'):
        try:
            return self._call_role(role, prompt, snapshot, scope=scope, turn_key=turn_key,
                                   packet=packet, external_program=external_program, identifiers=identifiers,
                                   allow_reuse=allow_reuse, task_kind=task_kind)
        except Exception:
            # 生成函数可能已原位改变缓存；保留已提交词元，下一角色重新核对位置。
            self.past, self.last_hidden = None, None
            raise

    def _call_role(self, role, prompt, snapshot, *, scope, turn_key, packet, external_program=False,
                   identifiers=None, allow_reuse=True, task_kind='continuous'):
        started = time.perf_counter()
        if scope != self.scope:
            self.reset()
            self.scope = scope
        if identifiers is not None:
            self.identifiers = copy.deepcopy(identifiers)
        self.metrics = {key: 0 for key in ('input_tokens', 'forward_positions', 'latent_steps',
                        'alignment_ms', 'facts_encoded', 'facts_reused_without_encoding', 'cache_recovery_positions',
                        'input_processing_ms', 'appended_forward_ms')}
        self.inputs = []
        before = self._state_size()['payload_bytes']
        reused_positions = len(self.ids) if self.past is not None else 0
        self._ensure_state()
        if self.event_sink:
            self.event_sink({'role': role, 'event': 'start'})
        common, current = split_role_prompt(prompt)
        common_signature = self._signature(common) if common else None
        fragments = shared_instruction_parts(common)
        shared_instruction_reused = bool(fragments) and all(
            self._signature(part) in self.shared_instruction_signatures for part in fragments)
        for part in fragments:
            signature = self._signature(part)
            if signature not in self.shared_instruction_signatures:
                self._append(self._render('system', part), 'shared_instruction')
                self.shared_instruction_signatures.add(signature)
        self.shared_instruction_signature = common_signature
        self.lifecycle.capture_prefix(self.past, self.ids)
        if turn_key != self.turn_key:
            history = [row for row in snapshot['history'] if self._signature(row) not in self.encoded_history]
            self._data({'current_task': snapshot['task'],
                        'new_conversation': projected_history(history, self.identifiers)}, 'external_question')
            self.encoded_history.update(self._signature(row) for row in history)
            self.turn_key, self.selected_ids = turn_key, None
            self.diagnostic_count = 0
        self._evidence(packet)
        experience = (snapshot.get('experience') or {}).get('relation_paths')
        experience_signature = self._signature(experience)
        if experience_signature != self.experience_signature:
            if experience is not None or self.experience_signature is not None:
                self._data({'retrieval_experience_for_current_task': experience}, 'external_experience')
            self.experience_signature = experience_signature
        if snapshot.get('execution') is not None and self._signature(snapshot['execution']) != self.execution_signature:
            self._data({'actual_execution': snapshot['execution']}, 'external_tool_result')
            self.execution_signature = self._signature(snapshot['execution'])
        new_diagnostics = snapshot.get('diagnostics', [])[self.diagnostic_count:]
        if new_diagnostics:
            self._data({'workflow_status': new_diagnostics}, 'workflow_status')
            self.diagnostic_count = len(snapshot['diagnostics'])
        # 当前问题与当前职责必须是本次请求的末尾，诊断和旧角色工作只是数据。
        request = 'Current task to complete:\n' + json.dumps(snapshot['task'], ensure_ascii=False) + '\n\n' + current
        closed = self._render('user', request)
        opened = self._render('user', request, open_assistant=True)
        if not opened.startswith(closed) or not opened[len(closed):]:
            raise ValueError('官方模板不能安全分离生成前缀。')
        self._append(closed, 'role_instruction')
        prefix = self._tokens(opened[len(closed):])
        self.inputs.append({'category': 'generation_prefix', 'text': opened[len(closed):], 'tokens': int(prefix.shape[1])})
        self.metrics['input_tokens'] += int(prefix.shape[1])
        logical_input = self.bundle.torch.tensor([self.ids + prefix[0].tolist()], device=self.bundle.device)
        control = role != 'summarizer'
        # 每个角色实际使用原生生成完成工作。中间词元只留在计算状态中，
        # 只在外部工具边界解码查询或程序；角色工作不经文本重新输入下游。
        # 不用零步占位或未知映射代替推理。
        result = standard_generate(self.bundle, logical_input, max_tokens=self.max_tokens,
            past=self.past, control=control, prompt_tokens=int(prefix.shape[1]),
            decode_content=role == 'summarizer' or external_program, allow_reuse=allow_reuse,
            role=role, task_kind=task_kind, reserve_tokens=len(self._ending_ids()))
        self.ids.extend(prefix[0].tolist())
        self.ids.extend(result['output_token_ids'])
        self.past = result['past_key_values']
        finalize_started = time.perf_counter()
        state_issues = []
        try:
            if self.past is None:
                # 已经生成的工作不能丢弃；从完整词元重建并记录额外计算。
                state_issues.append('官方生成没有返回缓存，已使用保留的原始词元重建。')
                self.metrics['forward_positions'] += int(prefix.shape[1]) + max(len(result['output_token_ids']) - 1, 0)
                self._ensure_state()
            else:
                self.metrics['forward_positions'] += self.past.get_seq_length() - result['cached_input_tokens']
            # 官方生成通常尚未前向处理最后一个输出词元；补齐缓存，不改词元或答案。
            missing = len(self.ids) - self.past.get_seq_length()
            if missing < 0:
                raise ValueError('官方缓存位置超过逻辑会话长度。')
            if missing:
                pending_ids = self.ids[-missing:]
                pending = self.bundle.torch.tensor([pending_ids], device=self.bundle.device)
                self.ids = self.ids[:-missing]
                try:
                    self._forward(ids=pending)
                except Exception:
                    self.ids.extend(pending_ids)
                    raise
            # 从官方模板取得完整的消息关闭片段，不把模型的结束词元写成下一角色内容。
            ending_ids = self._ending_ids()
            # 结束词元可能已由模型生成，模板仍可能要求换行等后缀。
            # 只补齐官方模板尚缺的部分，最终角色截断时也关闭该消息。
            overlap = max((size for size in range(min(len(ending_ids), len(self.ids)) + 1)
                           if size == 0 or self.ids[-size:] == ending_ids[:size]), default=0)
            remaining = ending_ids[overlap:]
            if remaining:
                self.inputs.append({'category': 'boundary', 'tokens': len(remaining)})
                self.metrics['input_tokens'] += len(remaining)
                values = self.bundle.torch.tensor([remaining], device=self.bundle.device)
                self._forward(ids=values)
        except Exception as exc:
            self.past, self.last_hidden = None, None
            state_issues.append('生成内容已保留，缓存接续需重建：' + str(exc)[:300])
        state = self._state_size()
        state_finalize_ms = (time.perf_counter() - finalize_started) * 1000
        self.peak = max(self.peak, state['payload_bytes'])
        content = result['content'] if control else final_content(
            result['content'], thinking_prefix='<think>' in opened[-64:])
        finish = result['finish_reason']
        record = {'role': role, 'model': self.model, **self.metrics,
                  'role_prompt': prompt, 'role_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                  'shared_roles': True, 'role_instruction_message_role': 'user',
                  'inputs': self.inputs, 'raw_response_content': result['content'],
                  'raw_output_token_ids': result['output_token_ids'],
                  'response': {'content': content, 'action': result['action'],
                               'lookup_query': result['lookup_query'], 'decision_error': result['decision_error']},
                  'output_tokens': result['output_tokens'], 'finish_reason': finish,
                  'effective_eos_token_ids': result['effective_eos_token_ids'],
                  'control_stop_reason': result['control_stop_reason'],
                  'generation_call_ms': result['generation_call_ms'],
                  'generation_prefill_forward_ms': result['generation_prefill_forward_ms'],
                  'generation_decode_forward_ms': result['generation_decode_forward_ms'],
                  'generation_forward_calls': result['generation_forward_calls'],
                  'forward_timing_kind': result['forward_timing_kind'],
                  'state_finalize_ms': state_finalize_ms,
                  'shared_instruction_reused': shared_instruction_reused,
                  'shared_instruction_sha256': common_signature,
                  'elapsed_ms': (time.perf_counter() - started) * 1000,
                  'state_before_bytes': before,
                  'state_after_bytes': state['payload_bytes'], 'state_peak_bytes': self.peak,
                  'cache_after_bytes': state['cache_bytes'], 'state_issues': state_issues,
                  'reused_input_tokens': reused_positions if not self.metrics['cache_recovery_positions'] else 0,
                  'generation_backend': 'native_standard_generate',
                  'intermediate_token_generation': control and not external_program,
                  'tool_query_bytes': len((result['lookup_query'] or '').encode('utf-8')),
                  'output_repeat_guard': result['output_repeat_guard'],
                  'grammar_setup_ms': result['grammar_setup_ms'],
                  'context_limit': result['context_limit'],
                  'effective_output_budget': result['effective_output_budget'],
                  'context_budget_reduced': result['context_budget_reduced'],
                  'external_program_generation': external_program}
        record['cached_input_tokens'] = result['cached_input_tokens']
        if role != 'summarizer':
            record['handoff'] = {**state,
                                 'cache_transport': 'same_process_same_device_resident',
                                 'network_transfer_bytes': None}
        self.calls.append(record)
        self.metrics = None
        if self.event_sink:
            self.event_sink({'role': role, 'event': 'done', 'seconds': record['elapsed_ms'] / 1000,
                             'output_tokens': result['output_tokens'], 'answer': content,
                             'lookup_query': result['lookup_query'],
                             'handoff': record.get('handoff')})
        return {'content': content, 'action': result['action'],
                'lookup_query': result['lookup_query'], 'finish_reason': finish,
                'decision_error': result['decision_error']}

    def _ending_ids(self):
        marker = self.bundle.tokenizer.apply_chat_template([
            {'role': 'user', 'content': '__TASK_BOUNDARY__'},
            {'role': 'assistant', 'content': '__ROLE_END__'}],
            tokenize=False, add_generation_prompt=False,
            **getattr(self.bundle, 'template_kwargs', {}))
        if '__ROLE_END__' not in marker:
            raise ValueError('官方模板未保留角色结束定位标记。')
        return self._tokens(marker.split('__ROLE_END__', 1)[1])[0].tolist()
