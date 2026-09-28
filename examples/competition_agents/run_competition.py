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
    args = parser.parse_args()
    env_file = BASE/'.env.local'
    if env_file.exists():
        for line in env_file.read_text(encoding='utf-8').splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                name, value = line.split('=', 1)
                os.environ.setdefault(name.strip(), value.strip().strip('\"').strip("'"))
    method = METHODS[args.method]
    config = {'collaborationMode': args.method, 'retrievalMode': method['retrieval'], 'memoryMode': method['memory'],
              'backend': args.backend, 'localModel': args.local_model, 'remoteModel': args.remote_model,
              'dataset': args.dataset, 'taskCount': args.limit if args.limit is not None else DATASETS[args.dataset].get('runTaskCount', CATALOG['defaultTaskCount'])}
    output = args.output_dir or BASE/'runtime/runs'/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+args.method+'-'+uuid4().hex[:6])
    result = run(config, output, args.event_log_file)
    print('=== run_result_json ===')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not result['full_workflow_timing_valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
