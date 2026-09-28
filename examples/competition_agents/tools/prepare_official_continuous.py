"""核对原分支会话与官方题目；来源、问题和评测答案分开落盘。"""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(branch_file, official_file, output, limit=3):
    group = json.loads(branch_file.read_text(encoding='utf-8'))[0]
    conversation = group['original_conversation_no']
    official = {r['Turn_no']: r for r in (
        json.loads(line) for line in official_file.read_text(encoding='utf-8').splitlines()
    ) if r['Conversation_no'] == conversation}
    for task in group['tasks']:
        if task['question'] != official[task['original_turn_no']]['Question']:
            raise ValueError('分支问题与官方原始会话不一致。')
    # 不导入官方对话中的参考回答，也不将答案解释拼成知识来源。
    sources = [r for r in group['context'] if r.get('source') != 'official_annotated_rationale']
    if not sources or any(set(r) - {'title', 'sentences', 'source'} for r in sources):
        raise ValueError('来源结构需要人工检查。')
    questions, references = [], []
    for task in group['tasks'][:limit]:
        turn = task['original_turn_no']
        row = official[turn]
        identity = f'topiocqa-{conversation}-{turn}'
        questions.append({'id': identity, 'group_id': f'topiocqa-{conversation}',
                          'turn_id': turn, 'question': row['Question']})
        answers = list(dict.fromkeys([row['Answer']] + [a['Answer'] for a in row['Additional_answers']]))
        references.append({'id': identity, 'reference_answers': answers})
    manifest = {
        'dataset': 'TopiOCQA', 'conversation': conversation, 'official_turns': len(official),
        'repository_turns': len(group['tasks']), 'selected_turns': len(questions),
        'selection': '原分支第一组官方会话，按原顺序取前几轮，不根据模型成绩筛选。',
        'official_url': 'https://huggingface.co/datasets/McGill-NLP/TopiOCQA/blob/543f3fc424da32c8b024fb5ae182946860be5517/data/topiocqa_valid.jsonl',
        'branch': 'personal/codex/competition-console-update-20260706',
        'commit': '43301ee774083ffadd43eb0f4cf80b1c550d5fc6',
        'branch_file': branch_file.name, 'official_file': official_file.name,
        'branch_sha256': hashlib.sha256(branch_file.read_bytes()).hexdigest(),
        'official_sha256': hashlib.sha256(official_file.read_bytes()).hexdigest(),
        'source_documents': len(sources),
        'excluded_annotation_documents': len(group['context']) - len(sources),
        'source_policy': '保留分支内整段来源；排除答案解释，不扩写、不补全原始资料中缺失的数值。',
        'question_verification': '全部分支问题逐字匹配官方原始会话；来源段落取自分支。',
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [('questions.json', questions), ('references.json', references),
                        ('knowledge_sources.json', [{'context': sources}]), ('provenance.json', manifest)]:
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch-file', default='data/official_topiocqa_001_10turns.json')
    parser.add_argument('--official-file', default='data/official_sources/topiocqa_valid.jsonl')
    parser.add_argument('--output', default='data/continuous_topiocqa_119_v1')
    parser.add_argument('--limit', type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(prepare(Path(args.branch_file), Path(args.official_file), Path(args.output), args.limit), ensure_ascii=False))
