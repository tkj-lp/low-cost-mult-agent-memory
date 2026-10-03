"""保留同源文档和抽取事实，仅在新目录重建实体关联索引。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from hypergraph_memory_method.prepared_knowledge import write_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    shutil.copytree(args.source, args.output)
    started = time.perf_counter()
    docs = json.loads((args.output/'text_documents.json').read_text(encoding='utf-8'))
    facts = json.loads((args.output/'facts.json').read_text(encoding='utf-8'))
    temporary = args.output/'reindexed.sqlite'
    write_graph(temporary, docs, facts)
    temporary.replace(args.output/'hypergraph.sqlite')
    path = args.output/'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest['files']['hypergraph.sqlite'] = hashlib.sha256((args.output/'hypergraph.sqlite').read_bytes()).hexdigest()
    manifest['reindex'] = {'source': args.source.as_posix(), 'seconds': time.perf_counter()-started,
                           'policy': 'same documents and facts; scoped person surname identities'}
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
