"""三模型各三题的独立缓存回收原型队列，不支持十轮运行。"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    plan = json.loads((ROOT / 'plan.json').read_text(encoding='utf-8'))
    if plan['task_count'] != 3 or len(plan['models']) != 3:
        raise ValueError('只允许三个固定模型，每个模型前三题。')
    project = Path(plan['project'])
    os.environ.update(COMPETITION_PROJECT_ROOT=str(project),
        COMPETITION_MODEL_MANIFEST='/root/autodl-tmp/setup-logs/models.json',
        HF_HOME='/root/autodl-tmp/hf-cache', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    os.environ.pop('HYPERGRAPH_MEMORY_DB', None)
    mismatches = [name for name, digest in plan['frozen_sources'].items()
        if hashlib.sha256((project / name).read_bytes()).hexdigest() != digest]
    save(ROOT / 'source_verification.json', {'checked_files': len(plan['frozen_sources']),
                                           'mismatches': mismatches})
    if mismatches:
        raise RuntimeError('正式源码有变化，未启动：' + str(mismatches))
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                   '--format=csv,noheader'], text=True).strip()
    save(ROOT / 'environment.json', {'python': sys.version, 'gpu': gpu,
                                    'project': str(project), 'task_count': 3})
    queue_path = ROOT / 'queue.json'
    if queue_path.exists():
        raise RuntimeError('已有队列记录，不覆盖或重复启动。')
    state = {'status': 'running', 'started': stamp(), 'task_count_per_model': 3,
             'total_new_task_runs_allowed': 9, 'jobs': []}
    save(queue_path, state)
    for item in plan['models']:
        job = {**item, 'status': 'running', 'started': stamp()}
        state['jobs'].append(job)
        save(queue_path, state)
        command = [sys.executable, '-u', str(ROOT / 'role_private_cache_probe.py'),
            '--model', item['model'], '--task-count', '3', '--output', str(ROOT / item['key']),
            '--policy', plan.get('policy_key', 'instructions'),
            '--reference-root', str(Path(plan['reference_root']) / item['key'])]
        print(json.dumps({'event': 'model_start', **item}), flush=True)
        with (ROOT / (item['key'] + '.log')).open('w', encoding='utf-8') as logfile:
            process = subprocess.run(command, cwd=project, stdout=logfile, stderr=subprocess.STDOUT)
        job.update(status='finished' if process.returncode == 0 else 'error',
                   exit_code=process.returncode, finished=stamp())
        summary_path = ROOT / item['key'] / 'summary.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            job.update(task_count=summary['task_count'],
                       all_workflows_complete=summary['all_workflows_complete'])
        save(queue_path, state)
        print(json.dumps({'event': 'model_done', **job}), flush=True)
    state.update(status='finished' if all(j['exit_code'] == 0 for j in state['jobs']) else 'finished_with_errors',
                 finished=stamp())
    save(queue_path, state)
    differences = [name for name, digest in plan['frozen_sources'].items()
        if hashlib.sha256((project / name).read_bytes()).hexdigest() != digest]
    save(ROOT / 'postrun_source_verification.json', {'checked_files': len(plan['frozen_sources']),
                                                  'mismatches': differences})


if __name__ == '__main__':
    main()
