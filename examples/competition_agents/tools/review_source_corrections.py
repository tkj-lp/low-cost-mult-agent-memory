"""记录依据原文逐项核对的抽取修正；不导入任务问答。"""
import json
import os
from copy import deepcopy
from pathlib import Path

from tools.build_prepared_knowledge import PROMPT, read_env
from hypergraph_memory_method.prepared_knowledge import digest, validate_extraction


def run(root):
    read_env(Path('.env.local'))
    docs = json.loads((root / 'text_documents.json').read_text(encoding='utf-8'))
    target = root / 'source_review_corrections.json'
    corrections = json.loads(target.read_text(encoding='utf-8')) if target.exists() else {}
    audit = []
    failures = []
    for doc in docs:
        fp = digest({'document': doc, 'prompt': PROMPT, 'model': 'qwen3.7-flash', 'endpoint': os.environ['LLM_BASE_URL'], 'enable_thinking': False})
        path = root / 'extraction_cache' / (fp + '.json')
        if not path.exists():
            path = root / 'extraction_cache' / (fp + '.attempt-2.json')
        original = json.loads(path.read_text(encoding='utf-8'))['result']
        result = deepcopy(corrections.get(doc['id'], original))
        before = deepcopy(result)
        source = ' '.join([doc['title'], *doc['sentences']])
        title = doc['title']
        for row in result['facts']:
            obj = row.get('object', '')
            rel = row['relation']
            # 以下均为已对照来源检查的否定方向；负面事件不等于事件未发生。
            if row['polarity'] == 'negative' and (rel.startswith('lost ') or rel in ('fell', 'fate during', 'had to be dismantled', 'outcome', 'refused to include') or 'unlawful' in rel or 'unlawful' in row['subject']):
                row['polarity'] = 'positive'
            # 去掉模型自行补出的外层引号，仅在去掉后精确存在于原文时执行。
            for owner, key in [(row, 'subject'), (row, 'object'), *[(q, 'value') for q in row.get('qualifiers', [])]]:
                value = owner.get(key, '')
                if value and value not in source:
                    for candidate in (value.strip('"'), value.removesuffix('"')):
                        if candidate and candidate in source:
                            owner[key] = candidate
                            break
            if title == 'Shirley Temple' and 'Czechoslovakia' in obj:
                row.update(relation='appointed United States ambassador to', object='Czechoslovakia')
            if title == 'Cho Kyuhyun' and obj.startswith('member of') and 'sub-groups' in obj:
                row.update(relation='member of Super Junior subgroup', object='Super Junior-K.R.Y.' if 'K.R.Y.' in obj else 'Super Junior-M')
            if title == 'Andre Norton Award' and row['subject'] == 'graphic novels':
                row.update(relation='eligible for', object='the prize')
            if title == 'Goštautai' and row['subject'] == 'masculine form':
                row.update(subject='Goštautai', relation='masculine form', object='Goštautas')
            if title == 'Get Christie Love!' and obj == 'from the adaptation':
                row.update(object='')
            if title == 'Get Christie Love!' and obj == 'in place of "Christie Opara"':
                row.update(object='"Christie Opara"', relation='replaced')
            if title == 'America East Conference' and row['subject'] == 'members of America East Conference':
                row.update(subject='America East Conference', relation='members located mainly in')
            if title == 'Janet Waldo' and obj == 'radio actress':
                row.update(object='radio and voice actress')
            if title == 'The Hour of Bewilderbeast' and row['subject'] == "several of the album's eighteen tracks":
                row.update(subject="the album's eighteen tracks", qualifiers=[{'kind': 'scope', 'value': 'several of which'}])
            if title == '2014 Bahrain Grand Prix' and row['subject'] == "Hamilton's first in Bahrain":
                row.update(subject='Hamilton', relation='Bahrain victory ordinal', object='first')
            if title == 'The Wonderful Thing About Tiggers':
                if row['subject'] == 'that instance':
                    row.update(subject='Tigger', relation='first sang song at', object='the same instance', qualifiers=[])
                if obj == '"The Many Adventures of Winnie the Pooh" ride':
                    row.update(object='The Many Adventures of Winnie the Pooh ride')
            if title == 'Kansas City Scout':
                if row['subject'] == 'covering all the major highways in the Kansas City metro':
                    row.update(subject='cover all the major highways in the Kansas City metro', polarity='positive')
                for q in row.get('qualifiers', []):
                    if q['value'] == 'Since it was created':
                        q['value'] = "Since the project's beginning"
            if title == 'Steven Adler' and obj == 'second season of the reality TV show "Celebrity Rehab with Dr. Drew"':
                row.update(object='Celebrity Rehab with Dr. Drew', qualifiers=[{'kind': 'season', 'value': 'second'}])
            if title == 'Case Gym':
                if obj in ("Boston University Terriers men's basketball teams", "Boston University Terriers women's basketball teams"):
                    row.update(object="Boston University Terriers men's and women's basketball teams")
                if obj == 'Walter Brown Arena' and rel == 'location relative to':
                    row['relation'] = 'above'
                if obj == 'Walter Brown Arena' and rel == 'played in':
                    row['qualifiers'] = [{'kind': 'frequency', 'value': 'on occasion'}]
        if title == 'Amanda Lepore':
            def fact(relation, obj, indices, qualifiers=None):
                return {'subject': 'Lepore', 'relation': relation, 'object': obj, 'polarity': 'positive', 'qualifiers': qualifiers or [], 'sentences': indices}
            result['facts'].extend([
                fact('participated in', 'Artists and Prostitutes 1985–2005', [3], [{'kind': 'location', 'value': 'New York City'}, {'kind': 'installation', 'value': 'voyeuristic life-sized set'}]),
                fact('released', 'several singles', [4]),
                {'subject': 'several singles', 'relation': 'written by and/or recorded with', 'object': 'Cazwell', 'polarity': 'positive', 'qualifiers': [{'kind': 'scope', 'value': 'many'}], 'sentences': [4]},
                fact('released debut studio album', 'I.. . Amanda Lepore', [5, 6, 7], [{'kind': 'time', 'value': '2011'}, {'kind': 'label', 'value': 'Peace Bisquit'}]),
            ])
        try:
            validate_extraction(doc, result)
        except ValueError as exc:
            failures.append({'title': title, 'document_id': doc['id'], 'error': str(exc)})
            # 已完成部分的修正保留为候选，不能发布。
            (root / 'extraction_cache' / (doc['id'] + '.manual-candidate.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            continue
        if result != original:
            corrections[doc['id']] = result
        if result != before:
            audit.append({'document_id': doc['id'], 'title': title, 'before_sha256': digest(before), 'after_sha256': digest(result), 'basis': 'source text only; no question or answer labels'})
    target.write_text(json.dumps(corrections, ensure_ascii=False, indent=2), encoding='utf-8')
    (root / 'manual_review_audit.json').write_text(json.dumps({'changes': audit, 'remaining': failures}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'corrected_documents': len(corrections), 'remaining': failures}, ensure_ascii=True))


if __name__ == '__main__':
    import sys
    run(Path(sys.argv[1]))
