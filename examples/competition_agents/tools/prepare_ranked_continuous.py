"""从指定分支的连续窗口取题，下载完整来源章节；答案与建库输入隔离。"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.parse
import urllib.request

from bs4 import BeautifulSoup


def source_sections(html):
    soup = BeautifulSoup(html, 'html.parser')
    body = soup.select_one('#mw-content-text .mw-parser-output')
    if body is None:
        raise ValueError('原文章节容器不存在。')
    for node in body.select('sup.reference, .mw-editsection, style, script, table, .navbox, .hatnote'):
        node.decompose()
    sections, active = defaultdict(list), [(1, 'Introduction')]
    for node in body.find_all(['h2', 'h3', 'h4', 'p']):
        value = re.sub(r'\s+', ' ', node.get_text('', strip=False)).strip()
        if not value:
            continue
        if node.name != 'p':
            level = int(node.name[1])
            active = [(depth, name) for depth, name in active if depth < level and name != 'Introduction']
            active.append((level, value))
        else:
            for _, name in active:
                sections[name].append(value)
    return {name: '\n\n'.join(parts) for name, parts in sections.items()}


def prepare(branch_file, official_file, output, *, limit=None, reuse_sources=None):
    group = json.loads(branch_file.read_text(encoding='utf-8'))[0]
    if group['source_dataset'] != 'TopiOCQA':
        raise ValueError('此来源工具仅支持已核验的 TopiOCQA 原始格式。')
    official = {row['Turn_no']: row for row in map(json.loads, official_file.read_text(encoding='utf-8').splitlines())
                if row['Conversation_no'] == group['original_conversation_no']}
    count = len(group['tasks']) if limit is None else limit
    if type(count) is not int or not 1 <= count <= len(group['tasks']):
        raise ValueError('题目数量必须在原始会话范围内。')
    selected = group['tasks'][:count]
    if [task['original_turn_no'] for task in selected] != list(range(1, count + 1)):
        raise ValueError('本次选择必须从完整会话第一轮开始，禁止偷偷补入标准历史答案。')
    for task in group['tasks']:
        if task['question'] != official[task['original_turn_no']]['Question']:
            raise ValueError('分支问题与官方题目不一致。')
    titles = sorted({item['title'] for item in group['context']})
    pages, sources, audit = {}, [], []
    reused = None
    if reuse_sources is not None:
        reuse_sources = Path(reuse_sources)
        reused = json.loads((reuse_sources / 'provenance.json').read_text(encoding='utf-8'))
        for key, path in [('branch_sha256', branch_file), ('official_sha256', official_file)]:
            if reused[key] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('复用来源与当前原始数据不一致。')
        sources = json.loads((reuse_sources / 'knowledge_sources.json').read_text(encoding='utf-8'))[0]['context']
        audit = reused['sources']
        if sorted(row['title'] for row in sources) != titles or sorted(row['title'] for row in audit) != titles:
            raise ValueError('来源没有覆盖完整会话涉及的全部章节。')
        by_title = {row['title']: row for row in sources}
        for row in audit:
            body = by_title[row['title']]['sentences']
            if len(body) != 1 or hashlib.sha256(body[0].encode()).hexdigest() != row['text_sha256']:
                raise ValueError('原文章节校验失败：' + row['title'])
            raw = reuse_sources / 'source_snapshots' / (row['html_sha256'] + '.html')
            if hashlib.sha256(raw.read_bytes()).hexdigest() != row['html_sha256']:
                raise ValueError('原网页快照校验失败：' + row['title'])
    output.mkdir(parents=True, exist_ok=False)
    raw_dir = output / 'source_snapshots'
    raw_dir.mkdir()
    if reused:
        for digest in {row['html_sha256'] for row in audit}:
            shutil.copyfile(reuse_sources / 'source_snapshots' / (digest + '.html'), raw_dir / (digest + '.html'))
    for title in ([] if reused else titles):
        article, section = map(str.strip, title.split('[SEP]', 1))
        url = 'https://en.wikipedia.org/wiki/' + urllib.parse.quote(article.replace(' ', '_'))
        if article not in pages:
            request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (research source snapshot)'})
            with urllib.request.urlopen(request, timeout=45) as response:
                raw = response.read()
            checksum = hashlib.sha256(raw).hexdigest()
            (raw_dir / (checksum + '.html')).write_bytes(raw)
            pages[article] = (source_sections(raw), checksum)
        sections, checksum = pages[article]
        if not sections.get(section):
            raise ValueError('原文章节未找到：' + title)
        body = sections[section]
        sources.append({'title': title, 'sentences': [body]})
        audit.append({'title': title, 'url': url, 'html_sha256': checksum,
                      'text_sha256': hashlib.sha256(body.encode()).hexdigest(), 'characters': len(body)})
    questions, references = [], []
    for task in selected:
        turn = task['original_turn_no']
        row = official[turn]
        identity = f"topiocqa-{group['original_conversation_no']}-{turn}"
        questions.append({'id': identity, 'group_id': f"topiocqa-{group['original_conversation_no']}",
                          'turn_id': turn, 'question': task['question']})
        references.append({'id': identity, 'reference_answers': list(dict.fromkeys(
            [row['Answer']] + [answer['Answer'] for answer in row['Additional_answers']]))})
    manifest = {'dataset': 'TopiOCQA', 'conversation': group['original_conversation_no'],
                'branch_file': branch_file.name, 'branch_sha256': hashlib.sha256(branch_file.read_bytes()).hexdigest(),
                'branch': 'personal/codex/competition-console-update-20260706',
                'selected_turns': len(selected), 'original_turns': [t['original_turn_no'] for t in selected],
                'selection': f'用户指定分支中的原始会话，从第一轮按原顺序取{count}题；不补入标准历史答案。',
                'knowledge_policy': '完整会话文件涉及的原文完整章节；不导入答案标注，不按测试题数裁剪章节正文。',
                'source_version': '维基百科公开页面快照；不是原始评测年代的冻结知识库。',
                'fetched_at': datetime.now(timezone.utc).isoformat(), 'sources': audit,
                'official_sha256': hashlib.sha256(official_file.read_bytes()).hexdigest()}
    if reused:
        manifest.update(fetched_at=reused['fetched_at'], reused_sources=reuse_sources.name,
                        prepared_at=datetime.now(timezone.utc).isoformat())
    for name, value in [('questions.json', questions), ('references.json', references),
                        ('knowledge_sources.json', [{'context': sources}]), ('provenance.json', manifest)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch-file', required=True)
    parser.add_argument('--official-file', default='data/official_sources/topiocqa_valid.jsonl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, help='默认导出完整会话；仅调试时截取从第一轮开始的前缀。')
    parser.add_argument('--reuse-sources', help='复用并校验已有完整章节及网页快照，避免重新下载导致来源漂移。')
    args = parser.parse_args()
    print(json.dumps(prepare(Path(args.branch_file), Path(args.official_file), Path(args.output),
                             limit=args.limit, reuse_sources=args.reuse_sources), ensure_ascii=False))
