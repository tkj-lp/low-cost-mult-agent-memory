"""仅重建来源事实的超图关联；文本及抽取事实保持逐字不变。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from hypergraph_memory_method.prepared_knowledge import write_graph


def reindex(source, output, calibration=None, calibration_output=None):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    for name in ('text_documents.json', 'facts.json'):
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != manifest['files'][name]:
            raise ValueError('旧版本知识内容校验失败。')
        shutil.copy2(source/name, output/name)
    documents = json.loads((output/'text_documents.json').read_text(encoding='utf-8'))
    facts = json.loads((output/'facts.json').read_text(encoding='utf-8'))
    write_graph(output/'hypergraph.sqlite', documents, facts)
    manifest['files']['hypergraph.sqlite'] = hashlib.sha256((output/'hypergraph.sqlite').read_bytes()).hexdigest()
    manifest.update(reindexed_from=str(source), graph_organization='source-scoped-title-alias-v2')
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    if calibration:
        previous = json.loads((Path(calibration)/'manifest.json').read_text(encoding='utf-8'))
        if any(previous['config']['knowledge_files'][name] != manifest['files'][name]
               for name in ('facts.json', 'text_documents.json')):
            raise ValueError('事实或文本变化必须重新拟合，不能迁移已有校准。')
        # 校准样本只取事实内容和原始主体；只改图关联不改变隐藏特征或监督向量。
        dest = Path(calibration_output)
        dest.mkdir(parents=True, exist_ok=False)
        for name in ('readout.pt', 'validation_vectors.pt'):
            shutil.copy2(Path(calibration)/name, dest/name)
        previous['config']['knowledge_files'] = manifest['files']
        previous['graph_reindex_note'] = '仅图关联变化；事实、原文与校准样本不变，未重新拟合权重。'
        (dest/'manifest.json').write_text(json.dumps(previous, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--calibration')
    parser.add_argument('--calibration-output')
    args = parser.parse_args()
    reindex(args.source, args.output, args.calibration, args.calibration_output)
