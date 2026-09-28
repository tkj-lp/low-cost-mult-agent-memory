"""从已召回的来源选取原文片段，不读取答案标签或生成新事实。"""

from __future__ import annotations

import re

from .text import question_content_tokens


def _terms(text):
    return {
        term[:-1]
        if len(term) > 4
        and term.endswith("s")
        and not term.endswith(("ss", "us", "is"))
        else term
        for term in question_content_tokens(text)
    }


def sentence_spans(source: str) -> list[dict]:
    """返回可直接回读的完整句位置；代码作为不可拆分单元。"""
    if "```" in source:
        return [{"start": 0, "end": len(source), "text": source}]
    boundaries = [0]
    for match in re.finditer(r"(?<=[.!?。！？])\s+(?=[A-Z\u3400-\u9fff])", source):
        # 不把小数、常见缩写后的文字误当作独立事实。
        prefix = source[: match.start()]
        if re.search(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|vs|etc|[A-Z])\.$", prefix):
            continue
        boundaries.append(match.end())
    boundaries.append(len(source))
    spans = []
    for start, end in zip(boundaries, boundaries[1:]):
        while end > start and source[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append({"start": start, "end": end, "text": source[start:end]})
    return spans


def select_support_spans(source: str, question: str, *, topic: str = "") -> list[dict]:
    """保留最高相关完整句及依赖邻句，不截断限定条件或并列答案。"""
    full = [{"start": 0, "end": len(source), "text": source}]
    if not question.strip() or "```" in source:
        return full
    spans = sentence_spans(source)
    if len(spans) < 2:
        return full
    query = _terms(question) - _terms(topic)
    tokens = [_terms(span["text"]) for span in spans]
    # 同一主体的重复出现不应压过真正被问到的属性。
    frequencies = {term: sum(term in row for row in tokens) for term in query}
    scores = [sum(1.0 / frequencies[t] for t in query & row) for row in tokens]
    best = max(scores, default=0.0)
    if best <= 0 or max(scores) - min(scores) < 1e-9:
        return full
    chosen = {i for i, score in enumerate(scores) if abs(score - best) < 1e-9}
    dependent = re.compile(
        r"^(?:he|she|it|its|they|their|these|those|this|that|also|however|"
        r"but|except|instead|respectively)\b|^(?:其|它|他|她|这|但|此外|其中)",
        re.I,
    )
    # 保留代词的前文、否定例外和列表延续，不对引用范围做硬截断。
    changed = True
    while changed:
        before = set(chosen)
        for i in list(chosen):
            if i and dependent.search(spans[i]["text"]):
                chosen.add(i - 1)
            if i + 1 < len(spans) and dependent.search(spans[i + 1]["text"]):
                chosen.add(i + 1)
        changed = before != chosen
    return [spans[i] for i in sorted(chosen)]


def support_text(
    source: str, question: str, *, topic: str = ""
) -> tuple[str, list[dict]]:
    spans = select_support_spans(source, question, topic=topic)
    return " ".join(span["text"] for span in spans), spans
