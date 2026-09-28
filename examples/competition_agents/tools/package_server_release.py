"""打包整个仓库的代码和实验材料，排除密钥、依赖缓存及运行临时文件。"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

from competition_agents.methods.catalog import BASE, CATALOG
from tools.verify_locked_baselines import verify

EXCLUDED_DIRS = {'.git', 'node_modules', '__pycache__', '.pytest_cache', '.ruff_cache',
                 '.codex_cleanup_backups', '.codex_doc_review', '.doc_work', '.venv', '.venvs',
                 'outputs', 'run_result', 'runtime', 'extraction_cache', 'artifacts', '.ssh'}


def excluded_file(path):
    name = path.name.lower()
    return ((name.startswith('.env') and name != '.env.example')
            or path.suffix.lower() in {'.pyc', '.pyo', '.log', '.key', '.pem', '.p12', '.pfx'}
            or name in {'id_rsa', 'id_ed25519', 'credentials', '.netrc'})


def build(output, release_name):
    verify()
    if not release_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in release_name):
        raise ValueError('发布目录名称仅接受字母、数字、横线及下划线。')
    root = BASE.parents[1]
    files = []
    for directory, folders, names in os.walk(root):
        folders[:] = sorted(name for name in folders if name not in EXCLUDED_DIRS and not (Path(directory) / name).is_symlink())
        for name in sorted(names):
            path = Path(directory) / name
            if not excluded_file(path) and not path.is_symlink() and path.resolve() != output.resolve():
                files.append(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('不能覆盖已发布压缩包。')
    manifest = {'release': release_name, 'version': CATALOG['version'],
                'excluded_directories': sorted(EXCLUDED_DIRS),
                'excluded_secrets': '环境密钥文件、私钥和凭据；仅包含环境配置占位示例。',
                'model_weights': '不包含基础模型权重；保留项目已有校准文件。', 'files': {}}
    with tarfile.open(output, 'w:gz') as archive:
        for path in files:
            data = path.read_bytes()
            relative = path.relative_to(root).as_posix()
            manifest['files'][relative] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(release_name + '/' + relative)
            info.size = len(data)
            info.mode = 0o755 if path.suffix == '.sh' else 0o644
            archive.addfile(info, io.BytesIO(data))
        data = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
        info = tarfile.TarInfo(release_name + '/SERVER_RELEASE.json')
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    report = {'archive': str(output), 'release': release_name, 'files': len(files), 'bytes': output.stat().st_size,
              'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(output.suffix + '.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve(), args.name), ensure_ascii=False))
