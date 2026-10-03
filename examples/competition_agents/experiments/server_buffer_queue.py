"""串行补测三模型题内缓冲管理；冻结主程序，保留每题原始结果。"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    plan = json.loads((ROOT / 'plan.json').read_text(encoding='utf-8'))
    project = Path(plan['project'])
    os.environ.update(COMPETITION_PROJECT_ROOT=str(project),
        COMPETITION_MODEL_MANIFEST='/root/autodl-tmp/setup-logs/models.json',
        HF_HOME='/root/autodl-tmp/hf-cache', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    os.environ.pop('HYPERGRAPH_MEMORY_DB', None)
    state_path = ROOT / 'queue.json'
    state = {'status': 'preparing', 'started': stamp(), 'jobs': []}
    save(state_path, state)
    differences = [name for name, digest in plan['frozen_sources'].items()
        if not (project / name).is_file() or
        hashlib.sha256((project / name).read_bytes()).hexdigest() != digest]
    save(ROOT / 'source_verification.json', {'checked_files': len(plan['frozen_sources']),
                                            'mismatches': differences})
    if differences:
        raise RuntimeError('冻结源码不一致，未启动实验：' + str(differences))
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                   '--format=csv,noheader'], text=True).strip()
    save(ROOT / 'environment.json', {'python': sys.version, 'gpu': gpu,
        'project': str(project), 'runner_sha256': hashlib.sha256(
            (ROOT / 'task_buffer_lifecycle.py').read_bytes()).hexdigest()})
    state['status'] = 'running'
    save(state_path, state)
    for item in plan['models']:
        output = ROOT / item['key']
        if (output / 'summary.json').exists():
            summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
            if summary.get('all_checks_passed') and summary.get('task_count') == plan['task_count']:
                state['jobs'].append({**item, 'status': 'already_finished'})
                save(state_path, state)
                continue
        if output.exists():
            # 不覆盖未完成数据；只有显式另建目录才能重新跑该组。
            raise RuntimeError('结果目录已存在且没有完整核验结果：' + str(output))
        job = {**item, 'status': 'running', 'started': stamp()}
        state['jobs'].append(job)
        save(state_path, state)
        command = [sys.executable, '-u', str(ROOT / 'task_buffer_lifecycle.py'),
            '--scope', 'end_to_end', '--task-count', str(plan['task_count']),
            '--max-tokens', str(plan['max_tokens']), '--model', item['model'],
            '--output', str(output)]
        print(json.dumps({'event': 'model_start', **item, 'command': command}), flush=True)
        with (ROOT / (item['key'] + '.log')).open('w', encoding='utf-8') as logfile:
            process = subprocess.run(command, cwd=project, stdout=logfile, stderr=subprocess.STDOUT)
        job.update(finished=stamp(), exit_code=process.returncode,
                   status='finished' if process.returncode == 0 else 'error')
        if (output / 'summary.json').exists():
            summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
            job.update(all_checks_passed=summary['all_checks_passed'], task_count=summary['task_count'])
        save(state_path, state)
        print(json.dumps({'event': 'model_done', **job}), flush=True)
        if process.returncode:
            state.update(status='error', finished=stamp(), failed_model=item['key'])
            save(state_path, state)
            raise RuntimeError('该组未通过，已保留原始记录：' + item['key'])
    state.update(status='finished', finished=stamp())
    save(state_path, state)
    print(json.dumps({'event': 'queue_done', 'finished': state['finished']}), flush=True)


if __name__ == '__main__':
    main()
