"""核验固定方案源码、提示词、知识库和校准文件；发布新版本时显式重建清单。"""
import argparse
import hashlib
import json

from competition_agents.methods.catalog import BASE, CATALOG

LOCK = BASE/'config/baselines.lock.json'


def source_files():
    paths = [BASE/'run_competition.py', BASE/'runner_server.py', BASE/'config/methods.json']
    for folder in ('competition_agents', 'hypergraph_memory_method', 'baselines', 'tools'):
        paths.extend(p for p in (BASE/folder).rglob('*.py') if 'tests' not in p.parts)
    for folder in ('data/continuous_coqa_v1', 'data/continuous_topiocqa_119_v1', 'data/codeact_tool_sources', 'data/prepared_knowledge'):
        paths.extend(p for p in (BASE/folder).rglob('*') if p.is_file() and 'extraction_cache' not in p.parts)
    paths.extend([BASE/'data/codeact_tasks.json', BASE/'data/official_topiocqa_001_10turns.json'])
    for dataset in CATALOG['datasets']:
        paths.append(BASE/dataset['file'])
        if dataset.get('references'):
            paths.append(BASE/dataset['references'])
            parent = (BASE/dataset['file']).parent
            paths.extend(p for p in parent.iterdir() if p.is_file())
    paths.append(BASE/'data/official_factoid_ranked_topiocqa_003_10turns.json')
    paths.extend(p for p in (BASE/'frontend/src').rglob('*') if p.is_file())
    return sorted(set(paths))


def verify():
    saved = json.loads(LOCK.read_text(encoding='utf-8'))
    changed = [name for name, digest in saved['files'].items()
               if not (BASE/name).is_file() or hashlib.sha256((BASE/name).read_bytes()).hexdigest() != digest]
    actual = {p.relative_to(BASE).as_posix() for p in source_files()}
    changed.extend(sorted(actual-set(saved['files'])))
    if changed:
        raise ValueError('固定方案发生未登记变化：' + ', '.join(changed))
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='确认发布当前版本后重建校验清单。')
    args = parser.parse_args()
    if args.write:
        payload = {'version': CATALOG['version'], 'methods': CATALOG['methods'],
                   'files': {p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in source_files()}}
        LOCK.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    saved = verify()
    print('固定方案核验通过，文件数：' + str(len(saved['files'])))


if __name__ == '__main__':
    main()
