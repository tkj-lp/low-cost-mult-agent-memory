"""单卡串行实验队列：保留每次运行和失败；不依成绩选择题目或删除结果。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from competition_agents.methods.catalog import BASE, CATALOG, DATASETS

MODELS = ['Qwen/Qwen3-4B-Instruct-2507', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 'microsoft/Phi-4-mini-instruct']
METHODS = ['plain', 'structured', 'latent', 'autoform', 'llmlingua']


def dump(path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['pilot', 'full'], required=True)
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--max-searches', type=int, default=CATALOG.get('defaultMaxSearches', 2))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.max_tokens < 1 or args.max_searches < 1:
        parser.error('生成和检索预算必须为正整数。')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    state_path = args.output/'queue.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if args.resume and state_path.exists() else {'jobs': []}
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    env.pop('HYPERGRAPH_MEMORY_DB', None)
    env.setdefault('LLMLINGUA2_DEVICE_MAP', 'cuda')
    env['LLMLINGUA2_ENABLE_OFFICIAL'] = env['LLMLINGUA2_STRICT'] = '1'
    plan = {'phase': args.phase, 'models': args.models, 'methods': args.methods, 'datasets': list(DATASETS),
            'limit': 1 if args.phase == 'pilot' else 10, 'created': datetime.now(timezone.utc).isoformat(),
            'baseline_version': CATALOG['version'],
            'generation_budget': {m: args.max_tokens for m in args.models},
            'max_searches': args.max_searches,
            'base_weights_trainable': False, 'single_gpu_serial': True}
    plan_path = args.output/'plan.json'
    if plan_path.exists():
        saved = json.loads(plan_path.read_text(encoding='utf-8'))
        if any(saved.get(key) != plan[key] for key in (
            'baseline_version', 'phase', 'models', 'methods', 'datasets', 'limit', 'generation_budget', 'max_searches')):
            raise ValueError('不能把不同版本或参数的运行拼在同一结果目录。')
    else:
        dump(plan_path, plan)
    for model_index, model in enumerate(args.models):
        for dataset_index, (dataset_id, dataset) in enumerate(DATASETS.items()):
            slug = model.split('/')[-1] + '--' + dataset_id
            # 顺序轮转避免总让同一种方式排在第一；同一张卡不并行跑组。
            offset = (model_index + dataset_index) % len(args.methods)
            methods = args.methods[offset:] + args.methods[:offset]
            jobs = []
            for method in methods:
                key = slug + '--' + method
                command = [sys.executable, 'run_competition.py', '--method', method, '--dataset', dataset_id,
                           '--local-model', model, '--limit', str(plan['limit']),
                           '--max-tokens', str(plan['generation_budget'][model]),
                           '--max-searches', str(plan['max_searches']), '--warmup', '--summary-only', '--isolated-memory',
                           '--output-dir', str(args.output/key), '--event-log-file', str(args.output/(key+'.events.jsonl'))]
                jobs.append((key, command))
            for key, command in jobs:
                if any(j['key'] == key and 'exit_code' in j for j in state['jobs']):
                    continue
                if any(j['key'] == key for j in state['jobs']):
                    raise ValueError('上次进程中断的任务需要人工核查，不能覆盖：' + key)
                record = {'key': key, 'command': command, 'started': datetime.now(timezone.utc).isoformat()}
                state['jobs'].append(record)
                dump(state_path, state)
                print(json.dumps({'start': key}), flush=True)
                stamp = time.perf_counter()
                try:
                    with (args.output/(key+'.log')).open('w', encoding='utf-8') as log:
                        proc = subprocess.run(command, cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=10800)
                    record['exit_code'] = proc.returncode
                except subprocess.TimeoutExpired:
                    record['exit_code'] = -1
                    record['error'] = '三小时任务上限；记录失败，不与完整运行时间比较。'
                record['process_seconds_including_setup'] = time.perf_counter()-stamp
                result_file = args.output/key/'result.json'
                if result_file.exists():
                    result = json.loads(result_file.read_text(encoding='utf-8'))
                    run = result['runs'][0]
                    record['result'] = {'coverage': run['answer_accuracy'], 'metrics': run['metrics'],
                                        'full_workflow_timing_valid': result['full_workflow_timing_valid']}
                    record['result_sha256'] = hashlib.sha256(result_file.read_bytes()).hexdigest()
                dump(state_path, state)
                print(json.dumps({'finished': key, 'exit_code': record['exit_code'], 'seconds': record['process_seconds_including_setup']}), flush=True)
    state['finished'] = datetime.now(timezone.utc).isoformat()
    dump(state_path, state)


if __name__ == '__main__':
    main()
