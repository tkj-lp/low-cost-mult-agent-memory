"""校验已发布知识库和真实问题检索；不调用模型，不更改知识内容。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import statistics
import time

from hypergraph_memory_method.prepared_knowledge import PreparedKnowledge, validate_extraction


def verify(root, dataset=None):
    library = PreparedKnowledge(root)
    facts = json.loads((library.root / 'facts.json').read_text(encoding='utf-8'))
    groups = defaultdict(list)
    for fact in facts:
        groups[fact['document_id']].append(fact)
    for document in library.documents:
        checked = validate_extraction(document, {'facts': groups[document['id']]})
        if {f['id'] for f in checked} != {f['id'] for f in groups[document['id']]}:
            raise ValueError('事实摘要或来源字段不一致。')
    with closing(sqlite3.connect(library.graph_path.as_uri() + '?mode=ro', uri=True)) as db:
        integrity = db.execute('PRAGMA integrity_check').fetchone()[0]
        if integrity != 'ok':
            raise ValueError('数据库完整性检查失败：' + integrity)
        counts = {table: db.execute('SELECT count(*) FROM ' + table).fetchone()[0]
                  for table in ('documents', 'facts', 'incidence', 'fact_sources', 'constraints')}
        counts['nodes'] = db.execute('SELECT count(DISTINCT entity) FROM incidence').fetchone()[0]
        counts['local_reference_nodes'] = db.execute("SELECT count(DISTINCT entity) FROM incidence WHERE entity LIKE 'local:%'").fetchone()[0]
        counts['covered_sentences'] = db.execute('SELECT count(*) FROM (SELECT DISTINCT document_id,sentence_index FROM fact_sources)').fetchone()[0]
        for table in ('incidence', 'fact_sources', 'constraints'):
            if db.execute('SELECT count(*) FROM ' + table + ' x LEFT JOIN facts f ON f.id=x.fact_id WHERE f.id IS NULL').fetchone()[0]:
                raise ValueError('存在悬空超边引用：' + table)
        if {r[0] for r in db.execute('SELECT id FROM facts')} != {f['id'] for f in facts}:
            raise ValueError('事实文件与数据库超边不一致。')
        if {r[0] for r in db.execute('SELECT id FROM search')} != {f['id'] for f in facts}:
            raise ValueError('倒排索引覆盖不完整。')
    if counts['covered_sentences'] != sum(len(d['sentences']) for d in library.documents):
        raise ValueError('原文句子引用覆盖不完整。')
    tasks = []
    if dataset:
        for sample in json.loads(Path(dataset).read_text(encoding='utf-8')):
            started = time.perf_counter()
            hits = library.search_graph(sample['question'], 12)
            elapsed = (time.perf_counter() - started) * 1000
            # 金标仅在检索之后用于评价，不传给检索器或构建器。
            gold = {row[0] for row in sample.get('supporting_facts', [])}
            retrieved = {hit['title'] for hit in hits}
            tasks.append({'id': sample['id'], 'question': sample['question'], 'milliseconds': elapsed,
                          'facts': len(hits), 'retrieved_titles': sorted(retrieved),
                          'gold_support_titles': sorted(gold),
                          'support_title_recall': len(gold & retrieved) / len(gold) if gold else None,
                          'fact_ids': [hit['context_id'] for hit in hits]})
    report = {'status': 'passed', 'schema': library.manifest['schema'], 'corpus_sha256': library.manifest['corpus_sha256'],
              'counts': counts, 'database_integrity': integrity, 'artifact_checksums': 'passed',
              'source_validation': 'structure, document anchors, cited object/qualifier anchors and sentence-reference coverage',
              'semantic_verification': 'not a proof of semantic correctness or lossless extraction',
              'query_scope': 'one direct graph search per question, top 12 facts; no planner or answer model',
              'queries': tasks}
    if tasks:
        scored = [t['support_title_recall'] for t in tasks if t['support_title_recall'] is not None]
        report['query_summary'] = {'count': len(tasks), 'nonempty': sum(bool(t['facts']) for t in tasks),
            'median_milliseconds': statistics.median(t['milliseconds'] for t in tasks),
            'mean_support_title_recall': statistics.mean(scored) if scored else None}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--knowledge-dir', required=True)
    parser.add_argument('--dataset-file')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = verify(args.knowledge_dir, args.dataset_file)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'queries'}, ensure_ascii=True))
