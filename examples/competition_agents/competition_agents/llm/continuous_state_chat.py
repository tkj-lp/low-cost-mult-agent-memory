"""逐层缓存接续；中间角色只计算新增指令，工具参数与最终答案正常生成。"""
from __future__ import annotations

import copy
import hashlib
import json
import time

from .context_budget import context_limit
from .generation import standard_generate
from .resident_chat import ResidentChat
from .response_content import final_content
from ..collaboration_transport import projected_history
from ..role_prompts import role_duty_prompt, split_role_prompt


TOOL_INTERFACE = (
    'Use the runtime decision interface. Return one JSON object with exactly two fields, '
    'in this order: action, query. The action is deliver, lookup, or reuse. '
    'Deliver hands off your new work using available evidence; query is null. '
    'Lookup requests knowledge because a needed fact or relation is missing; '
    'query is a concise search string naming the resolved entities, missing relation '
    'and necessary conditions. Reuse adopts preceding work unchanged after checking '
    'it against the evidence, when you have no new work to add; query is null. '
    'Resolve references using the plan, conversation and available evidence; '
    'do not merely repeat an unresolved question. '
    'Do not output your plan, answer candidate, reasoning or evidence list. '
    'Your internal work remains in the shared computation state.')


class ContinuousStateChat(ResidentChat):
    """同题各层缓存原样接续，不把末层重新送入输入层，跨题仅保留公共前缀。"""

    def __init__(self, native, *, event_sink=None):
        super().__init__(native, event_sink=event_sink)

    def _close_message(self):
        ending = self._ending_ids()
        overlap = max((n for n in range(min(len(ending), len(self.ids)) + 1)
                       if n == 0 or self.ids[-n:] == ending[:n]), default=0)
        if ending[overlap:]:
            self.metrics['input_tokens'] += len(ending[overlap:])
            self.inputs.append({'category': 'boundary', 'tokens': len(ending[overlap:])})
            self._forward(ids=self.bundle.torch.tensor([ending[overlap:]], device=self.bundle.device))

    def _request(self, content):
        closed = self._render('user', content)
        opened = self._render('user', content, open_assistant=True)
        if not opened.startswith(closed) or not opened[len(closed):]:
            raise ValueError('官方模板不能安全分离生成前缀。')
        self._append(closed, 'role_instruction')
        return self._tokens(opened[len(closed):]), opened[len(closed):]

    def _internal_role(self, current):
        """仅缓存对照；正式隐状态后端覆写此处执行官方隐式推理。"""
        self._append(self._render('user', current), 'role_instruction')

    def _generate(self, prefix, prefix_text, *, tool_only, role, task_kind, allow_reuse=False):
        self.inputs.append({'category': 'generation_prefix', 'text': prefix_text, 'tokens': int(prefix.shape[1])})
        self.metrics['input_tokens'] += int(prefix.shape[1])
        logical = self.bundle.torch.tensor([self.ids + prefix[0].tolist()], device=self.bundle.device)
        result = standard_generate(self.bundle, logical,
            max_tokens=min(self.max_tokens, 256) if tool_only else self.max_tokens,
            past=self.past, control=tool_only, tool_only=tool_only, allow_reuse=allow_reuse,
            prompt_tokens=int(prefix.shape[1]), role=role, task_kind=task_kind,
            reserve_tokens=len(self._ending_ids()))
        result['output_kind'] = 'tool_parameters' if tool_only else ('final_answer' if role == 'summarizer' else 'external_program')
        finalize_started = time.perf_counter()
        self.ids.extend(prefix[0].tolist())
        self.ids.extend(result['output_token_ids'])
        self.past = result['past_key_values']
        if self.past is None:
            self._ensure_state()
        else:
            self.metrics['forward_positions'] += self.past.get_seq_length() - result['cached_input_tokens']
            missing = len(self.ids) - self.past.get_seq_length()
            if missing < 0:
                raise ValueError('生成缓存超过实际逻辑位置。')
            if missing:
                pending = self.ids[-missing:]
                self.ids = self.ids[:-missing]
                try:
                    self._forward(ids=self.bundle.torch.tensor([pending], device=self.bundle.device))
                except Exception:
                    self.ids.extend(pending)
                    raise
        self._close_message()
        self.metrics['state_finalize_ms'] += (time.perf_counter() - finalize_started) * 1000
        return result

    def _call_role(self, role, prompt, snapshot, *, scope, turn_key, packet, external_program=False,
                   identifiers=None, allow_reuse=True, task_kind='continuous'):
        started = time.perf_counter()
        if scope != self.scope:
            self.reset()
            self.scope = scope
        if identifiers is not None:
            self.identifiers = copy.deepcopy(identifiers)
        self.metrics = {key: 0 for key in ('input_tokens', 'forward_positions', 'latent_steps',
            'latent_forward_ms', 'alignment_ms', 'role_prefill_positions', 'role_prefill_ms', 'facts_encoded', 'facts_reused_without_encoding',
            'cache_recovery_positions', 'input_processing_ms', 'appended_forward_ms', 'state_finalize_ms')}
        self.inputs = []
        before = self._state_size()['payload_bytes']
        reused = len(self.ids) if self.past is not None else 0
        self._ensure_state()
        if self.event_sink:
            self.event_sink({'role': role, 'event': 'start'})
        duty = role_duty_prompt(task_kind, role)
        common, current = split_role_prompt(duty)
        # 其他两种方式也在每次角色调用末尾指明同一个当前任务。
        # 原始题目不改写，防止长会话中的历史问题被误认为当前问题。
        current += '\n\nCurrent task to complete:\n' + json.dumps(snapshot['task'], ensure_ascii=False, separators=(',', ':'))
        signature = self._signature(common)
        shared_reused = signature in self.shared_instruction_signatures
        if not shared_reused:
            self._append(self._render('system', common), 'shared_instruction')
            self.shared_instruction_signatures.add(signature)
        self.shared_instruction_signature = signature
        self.lifecycle.capture_prefix(self.past, self.ids)
        if turn_key != self.turn_key:
            history = [r for r in snapshot['history'] if self._signature(r) not in self.encoded_history]
            self._data({'current_task': snapshot['task'],
                        'new_conversation': projected_history(history, self.identifiers)}, 'external_question')
            self.encoded_history.update(self._signature(r) for r in history)
            self.turn_key, self.selected_ids = turn_key, None
            self.diagnostic_count = 0
        self._evidence(packet)
        experience = (snapshot.get('experience') or {}).get('relation_paths')
        exp_signature = self._signature(experience)
        if exp_signature != self.experience_signature:
            if experience is not None or self.experience_signature is not None:
                self._data({'retrieval_experience_for_current_task': experience}, 'external_experience')
            self.experience_signature = exp_signature
        if snapshot.get('execution') is not None and self._signature(snapshot['execution']) != self.execution_signature:
            self._data({'actual_execution': snapshot['execution']}, 'external_tool_result')
            self.execution_signature = self._signature(snapshot['execution'])
        diagnostics = snapshot.get('diagnostics', [])[self.diagnostic_count:]
        if diagnostics:
            self._data({'workflow_status': diagnostics}, 'workflow_status')
            self.diagnostic_count = len(snapshot['diagnostics'])
        # 当前题在本题状态中只编码一次，后续角色的职责始终处于新增证据之后。
        role_started = time.perf_counter()
        role_positions = self.metrics['forward_positions']
        generated, content, action, query, decision_error = [], '', 'deliver', None, None
        if role == 'summarizer':
            prefix, prefix_text = self._request(current)
            self.metrics['role_prefill_positions'] = self.metrics['forward_positions'] - role_positions
            self.metrics['role_prefill_ms'] = (time.perf_counter() - role_started) * 1000
            result = self._generate(prefix, prefix_text, tool_only=False, role=role, task_kind=task_kind)
            generated.append(result)
            content = final_content(result['content'], thinking_prefix='<think>' in prefix_text)
            action = None
        elif role == 'planner':
            self._internal_role(current)
            self.metrics['role_prefill_positions'] = self.metrics['forward_positions'] - role_positions
            self.metrics['role_prefill_ms'] = (time.perf_counter() - role_started) * 1000
        else:
            self._internal_role(current)
            # 内部工作完成后，只有外部检索所需的参数经过文字接口。
            tool_prefix, tool_text = self._request(TOOL_INTERFACE +
                '\nAvailable original evidence rows: ' + str(len(packet['facts'])) + '.')
            self.metrics['role_prefill_positions'] = self.metrics['forward_positions'] - role_positions
            self.metrics['role_prefill_ms'] = (time.perf_counter() - role_started) * 1000
            result = self._generate(tool_prefix, tool_text, tool_only=True, role=role,
                                    task_kind=task_kind, allow_reuse=allow_reuse)
            generated.append(result)
            action, query, decision_error = result['action'], result['lookup_query'], result['decision_error']
            if external_program and action == 'deliver':
                code_prefix, code_text = self._request(current)
                result = self._generate(code_prefix, code_text, tool_only=False, role=role, task_kind=task_kind)
                generated.append(result)
                content = final_content(result['content'], thinking_prefix='<think>' in code_text)
        state = self._state_size()
        self.peak = max(self.peak, state['payload_bytes'])
        total = lambda key: sum(g.get(key) or 0 for g in generated)
        tool_outputs = sum(g['output_tokens'] for g in generated if g['output_kind'] == 'tool_parameters')
        finish = 'length' if any(g['finish_reason'] == 'length' for g in generated) else 'stop'
        record = {'role': role, 'model': self.model, **self.metrics,
            'role_prompt': duty, 'role_prompt_sha256': hashlib.sha256(duty.encode()).hexdigest(),
            'declared_text_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'role_duty_sha256': hashlib.sha256(duty.encode()).hexdigest(),
            'tool_interface_prompt': TOOL_INTERFACE if role in ('retriever', 'reasoner') else None,
            'shared_roles': True, 'role_instruction_message_role': 'user', 'inputs': self.inputs,
            'raw_response_content': content, 'raw_output_token_ids': [i for g in generated for i in g['output_token_ids']],
            'generation_records': generated,
            'response': {'content': content, 'action': action, 'lookup_query': query, 'decision_error': decision_error},
            'output_tokens': total('output_tokens'), 'tool_output_tokens': tool_outputs,
            'intermediate_work_output_tokens': 0, 'finish_reason': finish,
            'effective_eos_token_ids': generated[-1]['effective_eos_token_ids'] if generated else [],
            'control_stop_reason': generated[-1]['control_stop_reason'] if generated else 'role_input_processed',
            'generation_call_ms': total('generation_call_ms'),
            'generation_prefill_forward_ms': total('generation_prefill_forward_ms'),
            'generation_decode_forward_ms': total('generation_decode_forward_ms'),
            'generation_forward_calls': total('generation_forward_calls'),
            'external_generation_calls': len(generated),
            'forward_timing_kind': 'cuda_events' if str(self.bundle.device).startswith('cuda') else 'cpu_wall',
            'shared_instruction_reused': shared_reused,
            'shared_instruction_sha256': signature, 'elapsed_ms': (time.perf_counter() - started) * 1000,
            'state_before_bytes': before, 'state_after_bytes': state['payload_bytes'], 'state_peak_bytes': self.peak,
            'cache_after_bytes': state['cache_bytes'], 'state_issues': [],
            'reused_input_tokens': reused if not self.metrics['cache_recovery_positions'] else 0,
            'generation_backend': 'per_layer_cache_reuse_native_external_generation',
            'intermediate_token_generation': False, 'tool_query_bytes': len((query or '').encode()),
            'output_repeat_guard': generated[-1]['output_repeat_guard'] if generated else None,
            'grammar_setup_ms': total('grammar_setup_ms'), 'context_limit': context_limit(self.bundle),
            'effective_output_budget': generated[-1]['effective_output_budget'] if generated else 0,
            'context_budget_reduced': any(g['context_budget_reduced'] for g in generated),
            'external_program_generation': external_program, 'cached_input_tokens': reused,
            'feedback_mode': getattr(self, 'feedback_mode', 'disabled_no_last_layer_reinjection'),
            'state_mechanism': getattr(self, 'state_mechanism', 'reuse_each_layer_cache_only_process_new_role_and_external_input'),
            'upstream_commit': getattr(self, 'upstream_commit', None),
            'latent_step_budget': getattr(self, 'latent_step_budget', 0),
            'latent_memory_policy': getattr(self, 'memory_policy', None)}
        # 生成接口返回的缓存对象不能进入结果文件，也不能留下额外缓存引用。
        for g in generated:
            g.pop('past_key_values', None)
        if role != 'summarizer':
            record['handoff'] = {**state, 'cache_transport': 'same_process_same_device_resident',
                                 'network_transfer_bytes': None}
        self.calls.append(record)
        self.metrics = None
        if self.event_sink:
            self.event_sink({'role': role, 'event': 'done', 'seconds': record['elapsed_ms'] / 1000,
                             'output_tokens': record['output_tokens'], 'answer': content,
                             'lookup_query': query, 'handoff': record.get('handoff')})
        return {'content': content, 'action': action, 'lookup_query': query,
                'finish_reason': finish, 'decision_error': decision_error}
