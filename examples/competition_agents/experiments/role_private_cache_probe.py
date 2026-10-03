"""三题限定原型：按消费期限回收操作信息，完整事实与末端交付状态保留。"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import weakref

BASE = Path(os.environ.get('COMPETITION_PROJECT_ROOT') or
            Path(__file__).resolve().parents[1]).resolve()
sys.path.insert(0, str(BASE))

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer

from competition_agents.llm.experimental_state_chat import cache_bytes
from competition_agents.llm.native_local_chat import state_message, text_decoder
from competition_agents.llm.continuous_state_chat import TOOL_INTERFACE
from competition_agents.role_prompts import ROLE_ORDER, role_duty_prompt, split_role_prompt


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def history_metadata_spans(content):
    """定位历史记录的控制字段；不以关键词匹配问答正文。"""
    payload = json.dumps(content, ensure_ascii=False, separators=(',', ':'))
    if not isinstance(content.get('new_conversation'), list):
        return payload, []
    prefix = '"new_conversation":'
    # 由结构化对象的顺序计算位置，正文中的同名字串不参与定位。
    cursor = 1
    for key, value in content.items():
        key_text = json.dumps(key, ensure_ascii=False)
        value_text = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        if key == 'new_conversation':
            cursor += len(prefix) + 1
            break
        cursor += len(key_text) + 1 + len(value_text) + 1
    spans = []
    for record in content['new_conversation']:
        rendered = json.dumps(record, ensure_ascii=False, separators=(',', ':'))
        if isinstance(record, dict):
            field = cursor + 1
            for key, value in record.items():
                key_text = json.dumps(key, ensure_ascii=False)
                value_text = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
                end = field + len(key_text) + 1 + len(value_text)
                if key == 'evidence_ids' and isinstance(value, list) and all(isinstance(v, str) for v in value):
                    # 整个列表均有对应事实时，列表的重复键名和括号也可以回收。
                    # 若有缺失编号，这个范围保持有效，只分别回收已对应的编号。
                    spans.append((field, end, 'duplicate_history_source_map', value))
                    # 每个编号分别登记：未在当前事实表出现的编号不允许回收。
                    item = field + len(key_text) + 2
                    for identifier in value:
                        literal = json.dumps(identifier, ensure_ascii=False)
                        spans.append((item, item + len(literal), 'duplicate_history_source_id', [identifier]))
                        item += len(literal) + 1
                elif (key == 'verified' and value is False) or (key == 'completed' and value is True):
                    # 共用规则已将历史答案视为未验证；完成标记已由记忆读取处理。
                    spans.append((field, end, 'consumed_history_status', []))
                field = end + 1
        cursor += len(rendered) + 1
    return payload, spans


class RolePrivateCache(DynamicCache):
    """逻辑位置不回退；物理键值只移除明确登记的旧操作提示位置。"""

    def __init__(self, config):
        super().__init__(config=config)
        self.absolute_length = 0
        self.retained_positions = []
        self.private_positions = {}
        self.pending_positions = []
        self.protected_positions = set()

    def get_seq_length(self, layer_idx=0):
        return self.absolute_length

    def begin(self, count):
        self.pending_positions = list(range(self.absolute_length, self.absolute_length + count))

    def commit(self):
        self.retained_positions.extend(self.pending_positions)
        self.absolute_length += len(self.pending_positions)
        self.pending_positions = []
        for layer in self.layers:
            if type(layer) is DynamicLayer and layer.get_seq_length() != len(self.retained_positions):
                raise ValueError('物理缓存位置表与实际键值长度不一致。')

    def causal_mask(self, count, device, dtype):
        queries = torch.arange(self.absolute_length, self.absolute_length + count, device=device)
        keys = torch.tensor(self.retained_positions + self.pending_positions, device=device)
        allowed = keys[None, :] <= queries[:, None]
        mask = torch.zeros((count, len(keys)), device=device, dtype=dtype)
        mask.masked_fill_(~allowed, torch.finfo(dtype).min)
        return mask[None, None, :, :]

    def retire(self, completed_role, *, available_facts=(), completed_internal_calls=0,
               completed_tool_calls=0):
        stage = ROLE_ORDER.index(completed_role)
        available_facts = set(available_facts)
        removed = [position for position in self.retained_positions
                   if position in self.private_positions and
                   self.private_positions[position]['expires_after'] <= stage and
                   self.private_positions[position].get('after_internal_call', 0) <= completed_internal_calls and
                   self.private_positions[position].get('after_tool_call', 0) <= completed_tool_calls and
                   set(self.private_positions[position].get('requires_fact_ids', ())) <= available_facts]
        before = peak = cache_bytes(self)
        if not removed:
            return {'removed_positions': 0, 'before_bytes': before,
                    'after_bytes': before, 'compaction_peak_bytes': peak}
        if any(type(layer) not in (DynamicLayer, LinearAttentionLayer) for layer in self.layers):
            raise ValueError('未知缓存布局，未执行任何删除。')
        if set(removed) & self.protected_positions:
            raise ValueError('待回收位置与原始证据或问答正文重合，未执行删除。')
        removed_set = set(removed)
        keep = [i for i, position in enumerate(self.retained_positions) if position not in removed_set]
        kinds = {}
        for position in removed:
            kind = self.private_positions[position]['kind']
            kinds[kind] = kinds.get(kind, 0) + 1
        checked = 0
        for layer in self.layers:
            if type(layer) is not DynamicLayer:
                continue
            indices = torch.tensor(keep, device=layer.keys.device, dtype=torch.long)
            # 先替换并释放旧键，再复制值，避免新键、新值与整层旧缓存同时存在。
            # 剩余字节与绝对位置不变；每次分配的临时空间仍计入峰值。
            for name in ('keys', 'values'):
                existing = getattr(layer, name)
                replacement = existing.index_select(-2, indices)
                peak = max(peak, cache_bytes(self) + replacement.numel() * replacement.element_size())
                setattr(layer, name, replacement)
                del existing, replacement
            checked += 1
        self.retained_positions = [self.retained_positions[i] for i in keep]
        for position in removed:
            del self.private_positions[position]
        return {'removed_positions': len(removed), 'removed_by_kind': kinds,
                'before_bytes': before, 'after_bytes': cache_bytes(self),
                'compaction_peak_bytes': peak, 'absolute_length': self.absolute_length,
                'retained_attention_positions': len(keep),
                'compacted_attention_layers': checked,
                'recurrent_states_unchanged': True}


class PrivateInstructionManager:
    def __init__(self, bundle, *, policy='instructions', delivery_positions=8):
        if policy not in ('instructions', 'consumed_work', 'consumed_controls'):
            raise ValueError('未知角色内部缓存回收方案。')
        if not 1 <= delivery_positions <= 40:
            raise ValueError('必须保留至少一个真实末端交付位置。')
        self.bundle = bundle
        self.policy = policy
        self.delivery_positions = delivery_positions
        self.original_task = None
        self.decoder = text_decoder(bundle)
        self.handles = []
        self.current_role = None
        self.forward_records = []
        self.cache_ref = None
        self.active_cache = None
        self.task_peak = 0
        self.role_peak = 0
        self.internal_calls = 0
        self.tool_calls = 0
        self.reclamations = []
        self.attached_chat = None
        self.original_methods = {}

    def __enter__(self):
        self.handles = [self.decoder.register_forward_pre_hook(self.before, with_kwargs=True),
                        self.decoder.register_forward_hook(self.after, with_kwargs=True)]
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.active_cache = self.cache_ref = None
        if self.attached_chat is not None:
            for name, method in self.original_methods.items():
                setattr(self.attached_chat, name, method)
        self.attached_chat = None
        self.original_methods = {}

    def sample(self, past):
        value = cache_bytes(past)
        self.task_peak = max(self.task_peak, value)
        self.role_peak = max(self.role_peak, value)
        return value

    def before(self, module, args, kwargs):
        past = kwargs.get('past_key_values')
        if past is None:
            past = RolePrivateCache(module.config)
        elif not isinstance(past, RolePrivateCache):
            # 官方生成可能预先建立空缓存；只替换空对象，不替换任何已有状态。
            if past.get_seq_length() != 0:
                raise ValueError('不能用另一缓存对象替换已有计算状态。')
            past = RolePrivateCache(module.config)
        values = kwargs.get('inputs_embeds')
        if values is None:
            values = kwargs.get('input_ids')
        if values is None and args:
            values = args[0]
        if values is None or values.shape[0] != 1:
            raise ValueError('本原型只支持无填充的单会话输入。')
        count = int(values.shape[1])
        past.begin(count)
        self.sample(past)
        if len(past.retained_positions) != past.absolute_length:
            # 不能把裁剪后的物理长度当旋转位置，不能把后缀提前到旧空洞中。
            mask = kwargs.get('attention_mask')
            if mask is not None and mask.ndim == 2 and not bool(torch.all(mask == 1)):
                raise ValueError('本原型不改写带填充或额外屏蔽规则的输入。')
            kwargs['attention_mask'] = past.causal_mask(
                count, values.device, module.get_input_embeddings().weight.dtype)
        kwargs['past_key_values'] = past
        self.active_cache = past
        self.cache_ref = weakref.ref(past)
        self.forward_records.append({'role': self.current_role,
            'logical_prefix_positions': past.absolute_length,
            'retained_prefix_positions': len(past.retained_positions),
            'new_forward_positions': count, 'kind': 'embeds' if kwargs.get('inputs_embeds') is not None else 'ids'})
        return args, kwargs

    def after(self, module, args, kwargs, output):
        past = output.past_key_values
        past.commit()
        self.sample(past)
        self.active_cache = None

    def mark(self, past, text, start, fragment, owner, kind, expiry, **guards):
        if not fragment or text.count(fragment) != 1:
            raise ValueError('角色私有提示无法唯一定位，未登记删除。')
        left = text.index(fragment)
        right = left + len(fragment)
        self.mark_span(past, text, start, left, right, owner, kind, expiry, **guards)

    def mark_span(self, past, text, start, left, right, owner, kind, expiry, **guards):
        if not 0 <= left < right <= len(text):
            raise ValueError('登记范围越过原文边界，未登记删除。')
        offsets = self.bundle.tokenizer(text, add_special_tokens=False,
                                       return_offsets_mapping=True)['offset_mapping']
        selected = [start + i for i, (a, b) in enumerate(offsets)
                    if b > a and a >= left and b <= right]
        # 与正文相邻、跨越边界的词元不回收，当前任务和消息边界全部保留。
        for position in selected:
            past.private_positions[position] = {'owner': owner, 'kind': kind,
                                                'expires_after': expiry, **guards}
            past.protected_positions.discard(position)

    def collect(self, chat, phase):
        result = chat.past.retire(self.current_role,
            available_facts=chat.encoded_facts, completed_internal_calls=self.internal_calls,
            completed_tool_calls=self.tool_calls)
        self.task_peak = max(self.task_peak, result['compaction_peak_bytes'])
        self.role_peak = max(self.role_peak, result['compaction_peak_bytes'])
        if result['removed_positions']:
            self.reclamations.append({'role': self.current_role, 'phase': phase,
                'completed_internal_calls': self.internal_calls,
                'completed_tool_calls': self.tool_calls,
                'available_fact_ids': sorted(chat.encoded_facts),
                'protected_data_positions': len(chat.past.protected_positions),
                'all_protected_positions_retained': chat.past.protected_positions <= set(chat.past.retained_positions),
                **result})
        return result

    def attach(self, chat):
        original_internal, original_append, original_data = chat._internal_role, chat._append, chat._data
        original_request, original_generate, original_evidence = chat._request, chat._generate, chat._evidence
        self.attached_chat = chat
        self.original_methods = {name: getattr(chat, name) for name in
            ('_internal_role', '_append', '_data', '_request', '_generate', '_evidence')}
        expanded = self.policy == 'consumed_controls'

        def data(content, category):
            if category == 'external_question':
                self.original_task = content['current_task']
                self.internal_calls = 0
                self.tool_calls = 0
            start = len(chat.ids)
            text = chat._render('user', json.dumps(content, ensure_ascii=False, separators=(',', ':')))
            original_data(content, category)
            if not expanded:
                return
            if category in ('external_question', 'external_facts', 'external_tool_result'):
                chat.past.protected_positions.update(range(start, len(chat.ids)))
            if category == 'external_question':
                payload, spans = history_metadata_spans(content)
                offset = text.index(payload)
                for left, right, kind, required in spans:
                    self.mark_span(chat.past, text, start, offset + left, offset + right,
                        self.current_role, kind, 0, requires_fact_ids=required, after_internal_call=1)
            elif category == 'external_experience' and set(content) == {'retrieval_experience_for_current_task'}:
                self.mark_span(chat.past, text, start, 0, len(text), self.current_role,
                               'consumed_retrieval_routes', 1, after_tool_call=1)
            elif category == 'evidence_selection':
                selected = content.get('evidence_selected_for_current_task')
                if isinstance(selected, list) and set(selected) == set(chat.encoded_facts):
                    self.mark_span(chat.past, text, start, 0, len(text), self.current_role,
                                   'duplicate_fact_selection', 0, requires_fact_ids=selected)
                    self.collect(chat, 'fact_selection_registered')

        def evidence(packet):
            original_evidence(packet)
            if expanded:
                self.collect(chat, 'original_fact_table_encoded')

        def internal(current):
            start = len(chat.ids)
            text = state_message(self.bundle, [{'role': 'user', 'content': current}],
                open_assistant=True, continuation=bool(chat.ids), template_kwargs={})
            original_internal(current)
            self.internal_calls += 1
            _, private = split_role_prompt(role_duty_prompt('continuous', self.current_role))
            stage = ROLE_ORDER.index(self.current_role)
            expiry = stage if self.policy != 'instructions' else min(stage + 1, 3)
            guard = {'after_tool_call': self.tool_calls + 1} if expanded and stage in (1, 2) else {}
            self.mark(chat.past, text, start, private, self.current_role, 'old_role_duty', expiry, **guard)
            if self.policy != 'instructions':
                marker = '\n\nCurrent task to complete:\n'
                task_copy = current[current.index(marker) + len(marker):]
                if self.original_task is None or json.loads(task_copy) != self.original_task:
                    raise ValueError('没有逐字段相同的原题副本，不能删除该输入。')
                self.mark(chat.past, text, start, marker + task_copy, self.current_role,
                          'duplicate_current_task', stage, **guard)
                size = len(self.bundle.tokenizer(text, add_special_tokens=False)['input_ids'])
                # 此处是明确的近似试验，不把早期内部步骤宣称为已证明无用。
                # 不重写留下的表示；保留每个角色末端八个位置和全部原始证据。
                for position in range(start + size, start + size + chat.latent_step_budget - self.delivery_positions):
                    chat.past.private_positions[position] = {
                        'owner': self.current_role, 'kind': 'consumed_internal_steps',
                        'expires_after': 0 if expanded else min(stage + 1, 3),
                        **({'after_internal_call': self.internal_calls + 1} if expanded else {})}
            if expanded:
                self.collect(chat, 'role_internal_work_completed')

        def append(text, category):
            start = len(chat.ids)
            original_append(text, category)
            if expanded and category == 'shared_instruction':
                chat.past.protected_positions.update(range(start, len(chat.ids)))
            if category == 'role_instruction' and TOOL_INTERFACE in text:
                # 只回收通用动作语法；检索查询、行动结果和证据数量仍保留。
                if expanded:
                    self.mark_span(chat.past, text, start, 0, len(text), self.current_role,
                                   'consumed_tool_syntax', ROLE_ORDER.index(self.current_role))
                else:
                    self.mark(chat.past, text, start, TOOL_INTERFACE, self.current_role,
                              'consumed_tool_syntax', ROLE_ORDER.index(self.current_role))

        def request(content):
            result = original_request(content)
            if expanded and self.current_role == 'summarizer':
                # 最终角色已读完职责、原题与上游工作；此时回收，而不是等答案生成后。
                self.internal_calls += 1
                self.collect(chat, 'final_request_prefill_completed')
            return result

        def generate(*args, **kwargs):
            start = len(chat.ids)
            result = original_generate(*args, **kwargs)
            if expanded and kwargs.get('tool_only'):
                if result.get('decision_error') is None:
                    self.tool_calls += 1
                if result.get('action') in ('deliver', 'reuse') and result.get('lookup_query') is None:
                    # 外部控制已取得动作；没有查询内容或候选答案的控制回复不进入下一角色。
                    for position in range(start, len(chat.ids)):
                        chat.past.private_positions[position] = {'owner': self.current_role,
                            'kind': 'consumed_tool_control', 'expires_after': ROLE_ORDER.index(self.current_role)}
                self.collect(chat, 'tool_decision_generated')
            return result

        chat._internal_role, chat._append, chat._data = internal, append, data
        chat._request, chat._generate, chat._evidence = request, generate, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference-root', required=True)
    parser.add_argument('--policy', choices=('instructions', 'consumed_work', 'consumed_controls'), default='instructions')
    # 从接口上禁止扩大到完整十轮。
    parser.add_argument('--task-count', type=int, choices=(3,), default=3)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    from types import SimpleNamespace
    from sentence_transformers import SentenceTransformer
    from competition_agents.answer_evaluation import answer_coverage
    from competition_agents.llm.native_local_chat import NativeLocalChat
    from competition_agents.unified_collaboration import UnifiedCollaboration
    from task_buffer_lifecycle import cache_fingerprints, json_digest

    print(json.dumps({'event': 'model_loading', 'model': args.model}), flush=True)
    native = NativeLocalChat.from_pretrained(args.model, max_tokens=1024,
        threads=4, output_repeat_ngram_size=32)
    bundle = native.bundle
    if not bundle.tokenizer.is_fast:
        raise ValueError('分词器没有可靠的原文位置映射，不能试验删除。')
    warm = NativeLocalChat(bundle, max_tokens=1, output_repeat_ngram_size=32)
    warm.call('warmup', [{'role': 'user', 'content': 'Reply ready.'}])
    versions = {name: parameter._version for name, parameter in bundle.model.named_parameters()}
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                                  local_files_only=True, device='cpu')
    encode = lambda text: encoder.encode(text, normalize_embeddings=True).tolist()
    resources = {'knowledge': BASE / 'data/prepared_knowledge/topiocqa_153_v4',
        'encode': encode, 'embed': lambda text: SimpleNamespace(vector=encode(text),
            provider='local', model='sentence-transformers/all-MiniLM-L6-v2'),
        'task_kind': 'continuous', 'latent_steps': 40, 'latent_memory_policy': 'release_transients',
        'memory_enabled': True, 'experience_db': ':memory:'}
    print(json.dumps({'event': 'alignment_preparing'}), flush=True)
    runtime = UnifiedCollaboration(native, resources, method='latent', max_searches=2)
    chat = runtime.resident
    questions = json.loads((BASE / 'data/continuous_topiocqa_153_10turns_v1/questions.json').read_text(encoding='utf-8'))[:3]
    references = {r['id']: r['reference_answers'] for r in json.loads(
        (BASE / 'data/continuous_topiocqa_153_10turns_v1/references.json').read_text(encoding='utf-8'))}
    old = json.loads((Path(args.reference_root) / 'current_release_transients_tasks.json').read_text(encoding='utf-8'))[:3]
    tasks, boundaries = [], []
    try:
        with PrivateInstructionManager(bundle, policy=args.policy, delivery_positions=8) as manager:
            manager.attach(chat)

            def event(event):
                if event['event'] == 'start':
                    manager.current_role = event['role']
                    manager.role_peak = cache_bytes(chat.past)
                    return
                call = chat.calls[-1]
                before = cache_fingerprints(chat.past)
                mark_table = dict(chat.past.private_positions)
                if event['role'] == 'summarizer':
                    # 最终输出后已有整题回收，不能再复制键值去做没有消费者的压缩。
                    size = cache_bytes(chat.past)
                    result = {'removed_positions': 0, 'before_bytes': size,
                        'after_bytes': size, 'compaction_peak_bytes': size,
                        'reason': 'final_output_completed_no_further_compaction'}
                else:
                    result = manager.collect(chat, 'role_returned')
                manager.task_peak = max(manager.task_peak, result['compaction_peak_bytes'])
                manager.role_peak = max(manager.role_peak, result['compaction_peak_bytes'])
                # 逐个剩余关系位置的字节原值由保留索引的组件检查保证；全量逐层来源保留在审计中。
                after = cache_fingerprints(chat.past)
                boundaries.append({'role': event['role'], 'cache_before': before,
                    'cache_after': after, 'retirement': result,
                    'private_position_table_before': mark_table,
                    'inputs_sha256': json_digest(call['inputs']),
                    'output_ids': call['raw_output_token_ids'],
                    'role_cache_peak_bytes_including_compaction': manager.role_peak})
                save(output / 'live_boundaries.json', {'task_id': task['id'], 'boundaries': boundaries})
                print(json.dumps({'event': 'role_done', 'task_id': task['id'],
                    'role': event['role'], 'retired': result['removed_positions'],
                    'cache_mib_before': result['before_bytes'] / 1048576,
                    'cache_mib_after': result['after_bytes'] / 1048576}), flush=True)

            chat.event_sink = event
            for task, original in zip(questions, old, strict=True):
                boundaries = []
                manager.task_peak = 0
                manager.forward_records = []
                manager.reclamations = []
                gc.collect()
                started = time.perf_counter()
                row = runtime.run(task['question'], history_scope=task['group_id'])
                coverage = answer_coverage(row['prediction'], references[task['id']])
                row.update(task_id=task['id'], boundary_audit=boundaries,
                    new_forward_audit=manager.forward_records,
                    consumption_retirement_audit=manager.reclamations,
                    answer_coverage=coverage,
                    cache_peak_bytes_including_compaction=manager.task_peak,
                    cache_boundary_peak_bytes=max((b['retirement']['before_bytes'] for b in boundaries), default=0),
                    original_cache_boundary_peak_bytes=original['cache_peak_bytes'],
                    original_answer=original['prediction'], original_answer_coverage=original['answer_coverage'],
                    same_final_answer=row['prediction'] == original['prediction'],
                    same_queries_and_facts=[(s['query'],s['result']) for s in row['searches']] ==
                        [(s['query'],s['result']) for s in original['searches']],
                    seconds_including_audit=time.perf_counter() - started)
                tasks.append(row)
                save(output / 'tasks.json', tasks)
                print(json.dumps({'event': 'task_done', 'model': args.model,
                    'task_id': task['id'], 'workflow_complete': row['workflow_complete'],
                    'cache_peak_mib': manager.task_peak / 1048576,
                    'original_cache_peak_mib': original['cache_peak_bytes'] / 1048576,
                    'same_answer': row['same_final_answer'], 'coverage': coverage,
                    'answer': row['prediction']}, ensure_ascii=False), flush=True)
    finally:
        runtime.close()
    unchanged = versions == {name: parameter._version for name, parameter in bundle.model.named_parameters()}
    save(output / 'summary.json', {'model': args.model, 'task_count': len(tasks),
        'policy': args.policy, 'delivery_positions_per_internal_call': 40 if args.policy == 'instructions' else 8,
        'scope': '仅隐状态，真实连续问答前三题；角色提示、已消费内部步骤与重复控制信息的独立回收试验',
        'compaction': '逐个张量替换，计入复制临时空间，保留绝对位置和所有递归状态',
        'internal_step_retirement_is_approximate': args.policy != 'instructions',
        'weights_unchanged': unchanged, 'formal_baseline_changed': False,
        'all_workflows_complete': all(r['workflow_complete'] for r in tasks),
        'tasks': [{k: r[k] for k in ('task_id','prediction','answer_coverage',
            'original_answer','original_answer_coverage','same_final_answer','same_queries_and_facts',
            'cache_peak_bytes_including_compaction','cache_boundary_peak_bytes',
            'original_cache_boundary_peak_bytes','seconds_including_audit')} for r in tasks]})
    print(json.dumps({'event': 'model_done', 'model': args.model, 'task_count': len(tasks),
                      'weights_unchanged': unchanged}), flush=True)


if __name__ == '__main__':
    main()
