"""服务器运行前检查：只核验数据、依赖和设备，不调用答题模型。"""
import argparse
import importlib.metadata
import json
import platform

from competition_agents.methods.catalog import BASE, CATALOG
from tools.verify_locked_baselines import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--files-only', action='store_true')
    args = parser.parse_args()
    locked = verify()
    report = {'version': CATALOG['version'], 'verified_files': len(locked['files']),
              'python': platform.python_version(), 'datasets': [], 'errors': []}
    for dataset in CATALOG['datasets']:
        rows = json.loads((BASE / dataset['file']).read_text(encoding='utf-8'))
        if len(rows) != dataset['taskCount']:
            report['errors'].append('数据数量不一致：' + dataset['value'])
        report['datasets'].append({'id': dataset['value'], 'available': len(rows), 'run_count': dataset['runTaskCount']})
    if not args.files_only:
        packages = {}
        for name in ('torch', 'transformers', 'sentence-transformers', 'numpy', 'snowballstemmer', 'llmlingua', 'pytest', 'beautifulsoup4'):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
                report['errors'].append('缺少依赖：' + name)
        report['packages'] = packages
        if packages['torch']:
            import torch
            report['gpu_available'] = torch.cuda.is_available()
            report['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            report['cuda_build'] = torch.version.cuda
            if args.require_gpu and not report['gpu_available']:
                report['errors'].append('当前实例没有可用显卡，请启用显卡后再跑最终实验。')
        try:
            from huggingface_hub import snapshot_download
            report['model_cache'] = {}
            for name in (CATALOG['defaultLocalModel'], 'sentence-transformers/all-MiniLM-L6-v2',
                         'microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank'):
                try:
                    report['model_cache'][name] = snapshot_download(name, local_files_only=True)
                except Exception:
                    report['model_cache'][name] = None
                    report['errors'].append('未找到完整模型快照：' + name)
        except ImportError:
            report['errors'].append('缺少模型缓存检查依赖。')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return bool(report['errors'])


if __name__ == '__main__':
    raise SystemExit(main())
