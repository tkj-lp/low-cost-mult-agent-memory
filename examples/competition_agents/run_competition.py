"""比赛唯一运行入口；每次进程只加载并执行一种固定协作方案。"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from uuid import uuid4

from competition_agents.methods.catalog import BASE, CATALOG, DATASETS, METHODS
from competition_agents.methods.runner import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--dataset', choices=DATASETS, default=CATALOG['defaultDataset'])
    parser.add_argument('--backend', choices=['local', 'openai-compatible'], default='local')
    parser.add_argument('--local-model', default=CATALOG['defaultLocalModel'])
    parser.add_argument('--remote-model', default=CATALOG['defaultRemoteModel'])
    parser.add_argument('--limit', type=int, help='默认运行所选数据集固定十题；调试可显式截取前缀。')
    parser.add_argument('--event-log-file', default=str(BASE/'frontend/public/runtime/run_events.jsonl'))
    parser.add_argument('--output-dir')
    parser.add_argument('--calibration', help='仅兼容旧命令；当前共用流程不加载该离线映射。')
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--max-searches', type=int, default=CATALOG.get('defaultMaxSearches', 2),
                        help='所有协作方式共用的知识库补查次数上限。')
    parser.add_argument('--disable-memory', action='store_true', help='关闭跨题记忆，四角色主流程仍正常执行。')
    parser.add_argument('--latent-steps', type=int, default=CATALOG.get('defaultLatentSteps', 40),
                        help='各中间角色统一隐式推理步数；不随题目或答案调整。')
    parser.add_argument('--latent-memory-policy', choices=['full', 'release_transients'],
                        default=CATALOG.get('defaultLatentMemoryPolicy', 'release_transients'),
                        help='完整缓存接续；可释放无用临时输出，任务提交后统一回收。')
    parser.add_argument('--output-repeat-ngram-size', type=int, default=None,
                        help='仅阻止本次输出重复长片段；0沿用原始生成接口。各对照组须一致。')
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--isolated-memory', action='store_true')
    args = parser.parse_args()
    if args.output_repeat_ngram_size is None:
        args.output_repeat_ngram_size = CATALOG.get('defaultOutputRepeatNgramSize', 0) if args.backend == 'local' else 0
    if args.output_repeat_ngram_size != 0 and args.output_repeat_ngram_size < 2:
        parser.error('输出重复保护长度应为0或至少2。')
    if args.output_repeat_ngram_size and args.backend != 'local':
        parser.error('输出重复保护当前仅支持本地后端。')
    env_file = BASE/'.env.local'
    if env_file.exists():
        for line in env_file.read_text(encoding='utf-8').splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                name, value = line.split('=', 1)
                os.environ.setdefault(name.strip(), value.strip().strip('\"').strip("'"))
    method = METHODS[args.method]
    config = {'collaborationMode': args.method, 'retrievalMode': method['retrieval'], 'memoryMode': method['memory'],
              'backend': args.backend, 'localModel': args.local_model, 'remoteModel': args.remote_model,
              'dataset': args.dataset, 'calibration': args.calibration, 'maxTokens': args.max_tokens, 'warmup': args.warmup,
              'maxSearches': args.max_searches, 'memoryEnabled': not args.disable_memory,
              'latentSteps': args.latent_steps, 'latentMemoryPolicy': args.latent_memory_policy,
              'isolatedMemory': args.isolated_memory,
              'outputRepeatNgramSize': args.output_repeat_ngram_size,
              'taskCount': args.limit if args.limit is not None else DATASETS[args.dataset].get('runTaskCount', CATALOG['defaultTaskCount'])}
    output = args.output_dir or BASE/'runtime/runs'/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+args.method+'-'+uuid4().hex[:6])
    result = run(config, output, args.event_log_file)
    print('=== run_result_json ===')
    printed = ({'status': result['status'], 'model': result['model'], 'output': str(output),
                'coverage': result['runs'][0]['answer_accuracy'], 'metrics': result['runs'][0]['metrics']}
               if args.summary_only else result)
    print(json.dumps(printed, ensure_ascii=False), flush=True)
    if not result['full_workflow_timing_valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
