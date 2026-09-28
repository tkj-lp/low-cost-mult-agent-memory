"""可追溯的实体、动作、对象抽取；未知句法保持原文。"""

from __future__ import annotations

import re
from functools import lru_cache

from .evidence_spans import sentence_spans


@lru_cache(maxsize=4096)
def extract_actions(body: str) -> tuple[tuple[str, str, str, str], ...] | None:
    """每项为主语、原形态动作、对象、时间；仅接受完整匹配的独立断言。"""
    if re.search(r'["“”`;\n\r]|\b(?:not|never|if|unless|except|only|may|might|must|however|but|said|reported)\b', body, re.I):
        return None
    rows = []
    for span in sentence_spans(body):
        sentence = span['text'].strip().removesuffix('.')
        match = re.fullmatch(
            r'(?P<subject>[A-Z][\w ,\-]*?) '
            r'(?P<action>record(?:s|ed)?|measure(?:s|d)?|monitor(?:s|ed)?|collect(?:s|ed)?|'
            r'design(?:s|ed)?|founded|founds|opened|opens|built|builds) '
            r'(?P<object>[^.!?]+?)(?: (?P<time>(?:every|each) [^.!?]+))?', sentence)
        if not match or re.search(r'\b(?:he|she|it|they|its|their|who|which|that|is|was|are|were|has|have|had|will|would|can|could|should|to)\b', match['subject'], re.I):
            return None
        # 从句、并列动作和语态不得误作简单对象；退回原文而非猜测。
        if re.search(r'\b(?:because|while|although|when|where|who|which|that)\b', match['object'], re.I):
            return None
        rows.append((match['subject'], match['action'], match['object'], match['time'] or ''))
    return tuple(rows) or None


def render_actions(body: str, source: str = '') -> str | None:
    rows = extract_actions(body)
    if rows is None:
        return None
    # 同主语、同动作、同时间共享字段，所有对象仍然保留。
    groups: dict[tuple[str, str, str], list[str]] = {}
    for subject, action, value, time in rows:
        groups.setdefault((subject, action, time), []).append(value)
    lines = []
    for (subject, action, time), values in groups.items():
        actor = '' if subject == source else ('staff' if subject == 'The staff at ' + source else subject)
        relation = ' '.join(part for part in (actor, action, time) if part)
        lines.append(relation + ': ' + '; '.join(dict.fromkeys(values)))
    return '\n'.join(lines)
