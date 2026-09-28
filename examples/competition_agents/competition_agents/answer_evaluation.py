"""按标准答案覆盖度评分；额外内容不扣分，明确矛盾由来源审查单独否决。"""
from collections import Counter
import re
import unicodedata


def answer_tokens(value):
    normalized = unicodedata.normalize('NFKC', str(value)).casefold()
    normalized = re.sub(r'\b(?:a|an|the)\b', ' ', normalized)
    return re.findall(r'[a-z0-9]+|[\u3400-\u9fff]', normalized)


def answer_coverage(prediction, references, *, contradiction=False):
    """词元召回不等于事实验证；调用者须保存来源审查状态。"""
    if isinstance(references, str):
        references = [references]
    references = [r for r in references if r and answer_tokens(r)]
    if not references:
        return {'coverage': None, 'score': None, 'correct': None, 'status': 'unscored', 'contradiction': contradiction}
    predicted = Counter(answer_tokens(prediction))
    candidates = []
    for reference in references:
        gold = Counter(answer_tokens(reference))
        overlap = sum((predicted & gold).values())
        candidates.append((overlap / sum(gold.values()), reference))
    coverage, matched = max(candidates, key=lambda item: item[0])
    score = 0.0 if contradiction else coverage
    return {'coverage': coverage, 'score': score, 'correct': score == 1.0,
            'status': 'contradicted' if contradiction else 'full' if score == 1 else 'partial' if score else 'missing',
            'matched_reference': matched, 'contradiction': contradiction}
