"""将官方连续问答拆为完整原文、在线问题和独立评分文件，不按答案截取来源。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_URL = 'https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json'


def prepare(source_file, output):
    source_file, output = Path(source_file), Path(output)
    raw = source_file.read_bytes()
    dataset = json.loads(raw)
    # 在任何模型实验前固定官方文件第一段会话的前五轮，不按运行成绩挑题。
    conversation = dataset['data'][0]
    story = conversation['story']
    turns = conversation['questions'][:5]
    if not isinstance(story, str) or not story.strip() or len(turns) != 5:
        raise ValueError('第一段会话必须有完整故事和至少五轮问题。')
    if [q['turn_id'] for q in turns] != [1, 2, 3, 4, 5]:
        raise ValueError('会话轮次不连续。')
    output.mkdir(parents=True, exist_ok=False)
    scope = 'coqa-' + conversation['id']
    knowledge = [{'context': [{'title': conversation['filename'], 'sentences': [story]}]}]
    questions = [{'id': scope + '-turn-' + str(q['turn_id']), 'group_id': scope,
                  'turn_id': q['turn_id'], 'question': q['input_text']} for q in turns]
    answers = {a['turn_id']: a for a in conversation['answers']}
    additional = conversation.get('additional_answers', {})
    references = []
    for q in questions:
        turn = q['turn_id']
        answer = answers[turn]
        variants = [answer['input_text']]
        for annotator in additional.values():
            variants.extend(a['input_text'] for a in annotator if a['turn_id'] == turn)
        references.append({'id': q['id'], 'reference_answers': list(dict.fromkeys(variants)),
                           'official_answer': answer})
    files = {'knowledge_sources.json': knowledge, 'questions.json': questions, 'references.json': references}
    for name, value in files.items():
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    # 额外保存逐字符相同的原文，便于直接核对；不添加解释或答案标记。
    (output / 'source.txt').write_bytes(story.encode('utf-8'))
    manifest = {
        'dataset': 'CoQA', 'version': dataset.get('version'), 'official_url': SOURCE_URL,
        'official_file_sha256': hashlib.sha256(raw).hexdigest(),
        'conversation_id': conversation['id'], 'source_collection': conversation['source'],
        'selection': 'first conversation in official dev file; first five consecutive turns',
        'turns': 5, 'documents': 1, 'source_characters': len(story),
        'source_sha256': hashlib.sha256(story.encode('utf-8')).hexdigest(),
        'source_policy': 'whole official story, verbatim; no split, expansion, answer spans or rewritten questions',
        'text_retrieval': 'full-context; no chunk index, vector index or preselection',
        'extractor_model': 'qwen3.7-flash-2026-07-15',
        'online_input': 'questions.json only; preceding answers must come from the current run',
        'offline_input': 'knowledge_sources.json only',
        'evaluation_only': 'references.json',
        'files': {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                  for name in [*files, 'source.txt']},
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-file', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_file, args.output), ensure_ascii=True))
