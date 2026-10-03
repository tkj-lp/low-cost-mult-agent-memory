"""本机题内缓冲管理原型：不裁剪在用缓存、事实或角色输入。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import threading
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

BASE = Path(os.environ.get('COMPETITION_PROJECT_ROOT') or
            Path(__file__).resolve().parents[1]).resolve()
sys.path.insert(0, str(BASE))

from competition_agents.llm.experimental_state_chat import cache_bytes
from competition_agents.llm.latentmas_chat import StateForward


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    # numpy 的视图直接交给散列，不额外复制完整缓存到字节字符串。
    import torch
    raw = value.view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw).cast('B')).hexdigest()


def json_digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def cache_fingerprints(past):
    return {
        'positions': past.get_seq_length(),
        'bytes': cache_bytes(past),
        'layer_count': len(past.layers),
        'tensors': [
            {'layer': i, 'name': name, 'shape': list(value.shape),
             'dtype': str(value.dtype), 'sha256': tensor_digest(value)}
            for i, layer in enumerate(past.layers)
            for name in ('keys', 'values', 'conv_states', 'recurrent_states')
            if (value := getattr(layer, name, None)) is not None
        ],
    }


class LastPositionForward(StateForward):
    """仅供实验；完整各层缓存不变，末位输出可复制到独立的小缓冲。"""

    def __init__(self, bundle, *, compact=False):
        super().__init__(bundle)
        self.compact = compact
        self.records = []
        self.original_output_ref = None

    def __call__(self, **kwargs):
        result = super().__call__(**kwargs)
        original = result.hidden_states[-1]
        self.original_output_ref = weakref.ref(original)
        before = original.untyped_storage().nbytes()
        states = []
        for value in result.hidden_states:
            # 上游的这条入口只读最后一个位置。复制数值，不聚合或改写表示。
            # 不动 past_key_values，不删除任何缓存位置，也不改位置编号。
            if self.compact and (value.shape[1] > 1 or
                                value.untyped_storage().nbytes() >
                                value.numel() * value.element_size()):
                value = value[:, -1:, :].detach().clone()
            states.append(value)
        after = states[-1].untyped_storage().nbytes()
        self.records.append({
            'input_positions': original.shape[1],
            'original_output_storage_bytes': before,
            'retained_output_storage_bytes': after,
            'released_output_storage_bytes': before - after,
            'last_hidden_sha256': tensor_digest(states[-1][:, -1, :]),
            'cache_positions': result.past_key_values.get_seq_length(),
            'cache_bytes': cache_bytes(result.past_key_values),
        })
        return SimpleNamespace(past_key_values=result.past_key_values,
                               hidden_states=tuple(states))


class ProcessMemory:
    """采样当前进程的真实内存；它不是显存，也不是理论张量字节数。"""

    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.before = self.peak = self.process.memory_info().rss
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.done.wait(0.02):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __exit__(self, *args):
        self.after = self.process.memory_info().rss
        self.peak = max(self.peak, self.after)
        self.done.set()
        self.thread.join()

    def report(self):
        return {'rss_before_bytes': self.before, 'rss_peak_bytes': self.peak,
                'rss_after_bytes': self.after, 'sample_interval_ms': 20,
                'scope': '同进程完整答题，包含审计开销；处理器内存，不是显存'}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def component_check(core, forward, bundle, output, *, latent_steps=40):
    """完整真实知识输入后再接一段指令，逐层比较缓存及下一位置分数。"""
    import copy
    from competition_agents.llm.native_local_chat import state_message

    docs_path = BASE / 'data/prepared_knowledge/topiocqa_153_v4/text_documents.json'
    documents = json.loads(docs_path.read_text(encoding='utf-8'))
    # 不读取题目答案；来源全文两组完全相同，没有预先挑选答案事实。
    prompt = state_message(bundle, [{'role': 'user', 'content': json.dumps(
        {'source_documents': documents}, ensure_ascii=False)}], open_assistant=True)
    suffix_text = state_message(bundle, [{'role': 'user', 'content':
        'Keep every entity relation and qualification in the supplied sources available.'}],
        open_assistant=True, continuation=True)
    ids = bundle.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')['input_ids'].to(bundle.device)
    suffix = bundle.tokenizer(suffix_text, add_special_tokens=False, return_tensors='pt')['input_ids'].to(bundle.device)
    references, rows = [], []
    for compact in (False, True):
        forward.compact = compact
        forward.records.clear()
        core.latent_inputs = []
        core.pre_aligned = None
        cuda_memory = {}
        cuda = str(bundle.device).startswith('cuda')
        if cuda:
            bundle.torch.cuda.synchronize(bundle.device)
            bundle.torch.cuda.empty_cache()
            cuda_memory['cuda_allocated_before_bytes'] = bundle.torch.cuda.memory_allocated(bundle.device)
            bundle.torch.cuda.reset_peak_memory_stats(bundle.device)
        started = time.perf_counter()
        with bundle.torch.inference_mode():
            past = core.generate_latent_batch(ids, latent_steps=latent_steps)
            checkpoints = [cache_fingerprints(past)]
            past = core.generate_latent_batch(suffix, latent_steps=latent_steps, past_key_values=past)
            checkpoints.append(cache_fingerprints(past))
            aligned = [tensor_digest(value) for value in core.latent_inputs]
            if cuda:
                bundle.torch.cuda.synchronize(bundle.device)
                cuda_memory.update(
                    cuda_peak_allocated_bytes=bundle.torch.cuda.max_memory_allocated(bundle.device),
                    cuda_allocated_after_bytes=bundle.torch.cuda.memory_allocated(bundle.device),
                    cuda_memory_scope='两段完整隐式推理，含核验；不含下面复制缓存后的输出分数检查')
            query = bundle.tokenizer('Final answer:', add_special_tokens=False,
                                     return_tensors='pt')['input_ids'].to(bundle.device)
            scores = bundle.model(input_ids=query, past_key_values=copy.deepcopy(past),
                                  use_cache=True).logits[:, -1, :].detach().clone()
        if references:
            scores_identical = bool(bundle.torch.equal(scores.cpu(), references[0]))
            cache_identical = checkpoints == rows[0]['checkpoints']
            feedback_identical = aligned == rows[0]['feedback_sha256']
        else:
            references.append(scores.cpu())
            scores_identical = cache_identical = feedback_identical = True
        row = {'mode': 'compact_last_position' if compact else 'current_release_transients',
               'input_positions': ids.shape[1], 'continuation_positions': suffix.shape[1],
               'latent_steps_per_request': latent_steps, 'checkpoints': checkpoints,
               'feedback_sha256': aligned, 'buffers': list(forward.records),
               'cache_identical': cache_identical, 'feedback_identical': feedback_identical,
               'next_token_scores_identical': scores_identical,
               'elapsed_seconds': time.perf_counter() - started}
        row.update(cuda_memory)
        rows.append(row)
        save_json(output / 'component_check.json', rows)
        print(json.dumps({'event': 'component_done', 'mode': row['mode'],
                          'cache_identical': cache_identical, 'feedback_identical': feedback_identical,
                          'next_token_scores_identical': scores_identical}, ensure_ascii=False), flush=True)
        del past, scores
        core.latent_inputs = []
        core.pre_aligned = None
        gc.collect()
    assert all(rows[-1][key] for key in ('cache_identical', 'feedback_identical',
                                        'next_token_scores_identical'))
    return rows


def compare_tasks(baseline, managed):
    comparisons = []
    for left, right in zip(baseline, managed, strict=True):
        a, b = left['boundary_audit'], right['boundary_audit']
        checks = {
            'same_question': left['question'] == right['question'],
            'same_role_order': [c['role'] for c in a] == [c['role'] for c in b],
            'same_role_inputs': [c['inputs_sha256'] for c in a] == [c['inputs_sha256'] for c in b],
            'all_layer_cache_identical': [c['cache'] for c in a] == [c['cache'] for c in b],
            'same_feedback_vectors': [c['aligned_sha256'] for c in a] == [c['aligned_sha256'] for c in b],
            'same_last_hidden_vectors': [c['last_hidden_sha256'] for c in a] == [c['last_hidden_sha256'] for c in b],
            'same_generated_token_ids': [c['output_ids'] for c in a] == [c['output_ids'] for c in b],
            'same_retrieval_queries_and_facts': [(s['query'], s['result']) for s in left['searches']]
                                              == [(s['query'], s['result']) for s in right['searches']],
            'same_source_evidence': left['source_evidence'] == right['source_evidence'],
            'same_memory_read': left['memory_read']['evidence'] == right['memory_read']['evidence'],
            'same_final_answer': left['prediction'] == right['prediction'],
        }
        comparisons.append({'task_id': left['task_id'], **checks,
                            'all_checks_passed': all(checks.values())})
    return comparisons


def instrument_core(core, bundle):
    alignment = core._latent_realign_matrices[id(core.model)]
    forward = LastPositionForward(bundle)
    core.model = forward
    core._latent_realign_matrices[id(forward)] = alignment
    alignment_original = core._apply_latent_realignment

    def observe_alignment(hidden, model):
        aligned = alignment_original(hidden, model)
        forward.records[-1].update(aligned_sha256=tensor_digest(aligned),
            original_output_alive_at_feedback=forward.original_output_ref() is not None)
        return aligned

    core._apply_latent_realignment = observe_alignment
    return forward


def fixed_evidence_check(native, core, forward, output, args):
    """缓存专项：完整保留实际召回，排除工具参数失败；不冒充端到端成绩。"""
    from competition_agents.answer_evaluation import answer_coverage
    from competition_agents.collaboration_transport import CollaborationTransport
    from competition_agents.llm.generation import standard_generate
    from competition_agents.llm.native_local_chat import state_message
    from competition_agents.llm.response_content import final_content
    from competition_agents.role_prompts import role_duty_prompt, split_role_prompt
    from hypergraph_memory_method.prepared_knowledge import PreparedKnowledge

    bundle = native.bundle
    source = BASE / 'outputs/report_full_20261002/server-results/raw/qwen9/qa/latent/result.json'
    archived = json.loads(source.read_text(encoding='utf-8'))['runs'][0]['task_results'][:args.task_count]
    questions = json.loads((BASE / 'data/continuous_topiocqa_153_10turns_v1/questions.json').read_text(encoding='utf-8'))[:args.task_count]
    reference_rows = json.loads((BASE / 'data/continuous_topiocqa_153_10turns_v1/references.json').read_text(encoding='utf-8'))
    references = {row['id']: row['reference_answers'] for row in reference_rows}
    knowledge = PreparedKnowledge(BASE / 'data/prepared_knowledge/topiocqa_153_v4')
    component = component_check(core, forward, bundle, output, latent_steps=40)
    runs = {}
    for compact in (False, True):
        mode = 'compact_last_position' if compact else 'current_release_transients'
        forward.compact = compact
        forward.records.clear()
        history = []
        rows = []
        runs[mode] = rows
        for task, archived_task in zip(questions, archived, strict=True):
            assert task['question'] == archived_task['question']
            # 使用该题实际召回的全部事实。参考答案、候选和旧模型回答不进入输入。
            evidence = archived_task['source_evidence']
            if not all(key in knowledge._graph_rows and
                       value['fact_fields'] == knowledge._graph_rows[key][3]
                       for key, value in evidence.items()):
                raise ValueError('归档召回与现有超图事实不一致，禁止混用知识库版本。')
            transport = CollaborationTransport('latent', None, knowledge)
            snapshot = {'task': {'question': task['question']}, 'history': [], 'evidence': evidence}
            packet = transport.packet(snapshot)
            common, _ = split_role_prompt(role_duty_prompt('continuous', 'planner'))
            initial = state_message(bundle, [{'role': 'system', 'content': common},
                {'role': 'user', 'content': json.dumps({'current_task': snapshot['task'],
                    'conversation': history, 'original_evidence': packet}, ensure_ascii=False,
                    separators=(',', ':'))}], open_assistant=False)
            past = None
            logical_ids = []
            checkpoints = []
            start_buffer = len(forward.records)
            started = time.perf_counter()

            def append(text, role):
                nonlocal past
                ids = bundle.tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids'].to(bundle.device)
                # 本段单独验证跨角色缓存及最终生成；不注入连续向量。
                # 40步连续向量的安全性由上面的独立组件对照验证。
                past = core.generate_latent_batch(ids, latent_steps=0, past_key_values=past)
                logical_ids.extend(ids[0].tolist())
                checkpoints.append({'role': role, 'input_sha256': tensor_digest(ids),
                                    'cache': cache_fingerprints(past)})
                print(json.dumps({'event': 'fixed_evidence_stage', 'mode': mode,
                    'task_id': task['id'], 'role': role, 'cache_positions': past.get_seq_length()},
                    ensure_ascii=False), flush=True)

            gc.collect()
            with ProcessMemory() as measurement, bundle.torch.inference_mode():
                append(initial, 'complete_evidence_input')
                for role in ('planner', 'retriever', 'reasoner'):
                    _, duty = split_role_prompt(role_duty_prompt('continuous', role))
                    append(state_message(bundle, [{'role': 'user', 'content': duty}],
                                         open_assistant=False, continuation=True), role)
                _, duty = split_role_prompt(role_duty_prompt('continuous', 'summarizer'))
                suffix_text = state_message(bundle, [{'role': 'user', 'content': duty +
                    '\n\nCurrent task to complete:\n' + json.dumps(snapshot['task'])}],
                    open_assistant=True, continuation=True)
                suffix = bundle.tokenizer(suffix_text, add_special_tokens=False,
                    return_tensors='pt')['input_ids'].to(bundle.device)
                all_ids = bundle.torch.tensor([logical_ids + suffix[0].tolist()], device=bundle.device)
                result = standard_generate(bundle, all_ids, max_tokens=args.max_tokens,
                                           past=past, role='summarizer')
                checkpoints.append({'role': 'summarizer', 'input_sha256': tensor_digest(suffix),
                                    'cache': cache_fingerprints(result['past_key_values'])})
                answer = final_content(result['content']).strip()
            buffers = forward.records[start_buffer:]
            row = {'task_id': task['id'], 'question': task['question'], 'prediction': answer,
                'answer_coverage': answer_coverage(answer, references[task['id']]),
                'fact_count': len(packet['facts']), 'evidence_sha256': json_digest(packet),
                'evidence': packet, 'source_evidence': evidence,
                'history_received': list(history), 'checkpoints': checkpoints,
                'generated_ids': result['output_token_ids'], 'finish_reason': result['finish_reason'],
                'buffers': list(buffers), 'process_memory': measurement.report(),
                'elapsed_seconds': time.perf_counter() - started}
            rows.append(row)
            save_json(output / (mode + '_fixed_evidence_tasks.json'), rows)
            print(json.dumps({'event': 'fixed_evidence_task_done', 'mode': mode,
                'task_id': task['id'], 'answer': answer, 'fact_count': row['fact_count'],
                'coverage': row['answer_coverage']['score'], 'seconds': row['elapsed_seconds']},
                ensure_ascii=False), flush=True)
            history.append({'question': task['question'], 'answer': answer, 'verified': False})
            del past, result
    comparisons = []
    for a, b in zip(runs['current_release_transients'], runs['compact_last_position'], strict=True):
        checks = {'same_full_evidence': a['evidence_sha256'] == b['evidence_sha256'],
            'same_history': a['history_received'] == b['history_received'],
            'all_layer_cache_and_inputs_identical': a['checkpoints'] == b['checkpoints'],
            'same_generated_token_ids': a['generated_ids'] == b['generated_ids'],
            'same_final_answer': a['prediction'] == b['prediction']}
        comparisons.append({'task_id': a['task_id'], **checks, 'all_checks_passed': all(checks.values())})
    summary = {'model': args.model, 'device': bundle.device,
        'scope': '缓存专项，使用既有完整真实召回；不包含在线检索或四角色隐式推理成绩',
        'source_archive': str(source), 'source_archive_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'dataset': 'TopiOCQA 官方第153组前三轮',
        'component_latent_steps_per_request': 40, 'answer_trial_latent_steps': 0,
        'max_output_tokens': args.max_tokens, 'seed': 42, 'threads': args.threads,
        'component_check': component, 'comparisons': comparisons,
        'all_checks_passed': all(c['all_checks_passed'] for c in comparisons),
        'runs': {mode: [{key: value for key, value in row.items() if key not in (
            'source_evidence', 'evidence', 'checkpoints', 'buffers', 'generated_ids')}
            | {'cache_peak_bytes': max(c['cache']['bytes'] for c in row['checkpoints']),
               'largest_retained_output_storage_bytes': max(b['retained_output_storage_bytes'] for b in row['buffers']),
               'largest_original_output_storage_bytes': max(b['original_output_storage_bytes'] for b in row['buffers'])}
            for row in rows] for mode, rows in runs.items()}}
    save_json(output / 'summary.json', summary)
    print(json.dumps({'event': 'experiment_done', 'comparisons': comparisons,
                      'output': str(output)}, ensure_ascii=False), flush=True)
    assert summary['all_checks_passed'], '缓存专项不一致，保留结果，禁止接入正式基线。'
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--scope', choices=('fixed_evidence', 'end_to_end'), default='fixed_evidence')
    parser.add_argument('--task-count', type=int, choices=range(1, 11), default=3)
    parser.add_argument('--component-only', action='store_true')
    parser.add_argument('--skip-component', action='store_true')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    from competition_agents.answer_evaluation import answer_coverage
    from competition_agents.collaboration_memory import CollaborationMemory
    from competition_agents.collaboration_transport import CollaborationTransport
    from competition_agents.llm.native_local_chat import NativeLocalChat
    from competition_agents.unified_collaboration import UnifiedCollaboration

    print(json.dumps({'event': 'loading_model', 'model': args.model}), flush=True)
    native = NativeLocalChat.from_pretrained(args.model, max_tokens=args.max_tokens,
        threads=args.threads, output_repeat_ngram_size=32)
    bundle = native.bundle
    if str(bundle.device).startswith('cuda'):
        warm = NativeLocalChat(bundle, max_tokens=1, output_repeat_ngram_size=32)
        warm.call('warmup', [{'role': 'user', 'content': 'Reply ready.'}])
    versions = {name: value._version for name, value in bundle.model.named_parameters()}
    if args.component_only:
        from competition_agents.llm.latentmas_chat import RecordedCore
        core = RecordedCore(bundle, 'release_transients')
        forward = instrument_core(core, bundle)
        component = component_check(core, forward, bundle, output, latent_steps=40)
        save_json(output / 'summary.json', {'model': args.model, 'device': bundle.device,
            'scope': '完整40步隐式推理的缓存与显存专项', 'component_check': component,
            'all_checks_passed': all(component[-1][key] for key in (
                'cache_identical', 'feedback_identical', 'next_token_scores_identical')),
            'weights_unchanged': versions == {
                name: value._version for name, value in bundle.model.named_parameters()}})
        return
    if args.scope == 'fixed_evidence':
        from competition_agents.llm.latentmas_chat import RecordedCore
        print(json.dumps({'event': 'preparing_alignment', 'scope': args.scope}), flush=True)
        core = RecordedCore(bundle, 'release_transients')
        forward = instrument_core(core, bundle)
        summary = fixed_evidence_check(native, core, forward, output, args)
        summary['weights_unchanged'] = versions == {
            name: value._version for name, value in bundle.model.named_parameters()}
        save_json(output / 'summary.json', summary)
        assert summary['weights_unchanged'], '模型参数发生变化。'
        return
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                                  local_files_only=True, device='cpu')
    encode = lambda text: encoder.encode(text, normalize_embeddings=True).tolist()
    resources = {'knowledge': BASE / 'data/prepared_knowledge/topiocqa_153_v4',
                 'encode': encode, 'embed': lambda text: SimpleNamespace(
                     vector=encode(text), provider='local', model='sentence-transformers/all-MiniLM-L6-v2'),
                 'task_kind': 'continuous', 'latent_steps': 40,
                 'latent_memory_policy': 'release_transients', 'memory_enabled': True,
                 'experience_db': ':memory:'}
    print(json.dumps({'event': 'preparing_alignment_and_graph'}), flush=True)
    runtime = UnifiedCollaboration(native, resources, method='latent', max_searches=2)
    chat, core = runtime.resident, runtime.resident.core
    forward = instrument_core(core, bundle)
    component = [] if args.skip_component else component_check(core, forward, bundle, output)
    questions = json.loads((BASE / 'data/continuous_topiocqa_153_10turns_v1/questions.json').read_text(encoding='utf-8'))[:args.task_count]
    # 标准答案只在 runtime.run 已返回后用于评分。
    reference_rows = json.loads((BASE / 'data/continuous_topiocqa_153_10turns_v1/references.json').read_text(encoding='utf-8'))
    references = {r['id']: r['reference_answers'] for r in reference_rows}
    runs = {}
    current_boundaries = []
    buffer_offset = 0

    def event_sink(event):
        nonlocal buffer_offset
        print(json.dumps({'event': event['event'], 'mode': mode, 'task_id': task['id'],
                          'role': event['role'], 'seconds': event.get('seconds')}, ensure_ascii=False), flush=True)
        if event['event'] != 'done':
            return
        call = chat.calls[-1]
        buffers = forward.records[buffer_offset:]
        current_boundaries.append({
            'role': event['role'], 'cache': cache_fingerprints(chat.past),
            'inputs_sha256': json_digest(call['inputs']),
            'output_ids': call['raw_output_token_ids'],
            'last_hidden_sha256': [r['last_hidden_sha256'] for r in buffers],
            'aligned_sha256': [r['aligned_sha256'] for r in buffers if 'aligned_sha256' in r],
            'buffers': list(buffers),
        })
        buffer_offset = len(forward.records)
        save_json(output / 'live_boundaries.json', {'mode': mode, 'task_id': task['id'],
                                                  'boundaries': current_boundaries})

    chat.event_sink = event_sink
    try:
        for compact in (False, True):
            mode = 'compact_last_position' if compact else 'current_release_transients'
            if compact:
                runtime.memory.close()
                runtime.memory = CollaborationMemory(resources, graph=True, enabled=True)
                runtime.transport = CollaborationTransport('latent', native, runtime.memory.knowledge)
            chat.reset()
            chat.calls.clear()
            native.calls.clear()
            forward.records.clear()
            forward.compact = compact
            buffer_offset = 0
            core.latent_inputs = []
            core.pre_aligned = None
            rows = []
            runs[mode] = rows
            for task in questions:
                current_boundaries = []
                gc.collect()
                with ProcessMemory() as measurement:
                    row = runtime.run(task['question'], history_scope=task['group_id'])
                row.update(task_id=task['id'], boundary_audit=current_boundaries,
                    process_memory=measurement.report(),
                    cuda_peak_allocated_bytes=max((c.get('cuda_peak_allocated_bytes', 0)
                        for c in row['calls']), default=0),
                    cache_peak_bytes=max((c.get('cache_after_bytes', 0)
                        for c in row['calls']), default=0),
                    answer_coverage=answer_coverage(row['prediction'], references[task['id']]))
                rows.append(row)
                save_json(output / (mode + '_tasks.json'), rows)
                print(json.dumps({'event': 'task_done', 'mode': mode, 'task_id': task['id'],
                    'answer': row['prediction'], 'workflow_complete': row['workflow_complete'],
                    'answer_coverage': row['answer_coverage'],
                    'seconds': row['metrics']['elapsed_ms'] / 1000,
                    'knowledge_searches': row['metrics']['knowledge_searches'],
                    'memory_evidence_delivered': row['metrics']['memory_evidence_delivered']},
                    ensure_ascii=False), flush=True)
    finally:
        runtime.close()
    comparisons = compare_tasks(runs['current_release_transients'], runs['compact_last_position'])
    unchanged = versions == {name: value._version for name, value in bundle.model.named_parameters()}
    summary = {'model': args.model, 'device': bundle.device,
        'scope': '完整顺序连续问答的题内输出缓冲原型；真实本地检索与记忆，含一致性核验开销',
        'task_count': len(questions),
        'dataset': 'TopiOCQA 官方第153组', 'knowledge': str(resources['knowledge']),
        'latent_steps_per_internal_role': 40, 'max_output_tokens': args.max_tokens,
        'seed': 42, 'threads': args.threads, 'weights_unchanged': unchanged,
        'all_active_cache_positions_preserved': True, 'component_check': component,
        'comparisons': comparisons, 'all_checks_passed': unchanged and all(r['all_checks_passed'] for r in comparisons),
        'runs': {mode: [{'task_id': r['task_id'], 'prediction': r['prediction'],
            'answer_coverage': r['answer_coverage'], 'metrics': r['metrics'],
            'process_memory': r['process_memory'],
            'cuda_peak_allocated_bytes': r['cuda_peak_allocated_bytes'],
            'cache_peak_bytes': r['cache_peak_bytes'],
            'largest_retained_output_storage_bytes': max((b['retained_output_storage_bytes']
                for c in r['boundary_audit'] for b in c['buffers']), default=0),
            'largest_original_output_storage_bytes': max((b['original_output_storage_bytes']
                for c in r['boundary_audit'] for b in c['buffers']), default=0)} for r in rows]
            for mode, rows in runs.items()}}
    save_json(output / 'summary.json', summary)
    print(json.dumps({'event': 'experiment_done', 'comparisons': comparisons,
                      'weights_unchanged': unchanged, 'output': str(output)}, ensure_ascii=False), flush=True)
    assert summary['all_checks_passed'], '原型与基线不一致，结果已保存，禁止接入正式基线。'


if __name__ == '__main__':
    main()
