"""可追溯的抽取式事实字段；不把模型答案或评测标注变为事实。"""

from __future__ import annotations

import re
from functools import lru_cache

from .entities import extract_canonical_entities
from .evidence_spans import support_text
from .text import stable_id
from .action_fields import render_actions


@lru_cache(maxsize=4096)
def _source_fact(text):
    source, separator, body = text.partition(": ")
    if not separator or len(source) > 100 or "\n" in source:
        return "", text
    return source, body


def source_envelope(original: str, selected: str) -> tuple[str, list[int] | None]:
    """选句不能丢掉原始来源标题；补回的标题也是可定位的原文片段。"""
    source, _ = _source_fact(original)
    if source and not selected.startswith(source + ": "):
        return source + ": " + selected, [0, len(source) + 1]
    return selected, None


@lru_cache(maxsize=4096)
def _subject_fields(source: str, fact: str) -> str:
    """只将显式重复主语移至共享字段，不改写谓词、限定词或指代。"""
    if not source or any(mark in fact for mark in ('"', '“', '”', '\n', '\r')):
        return fact
    subject = re.escape(source)
    # 限于句首；引语、条件从句及宾语中的同名实体保持不动。
    fact = re.sub(r"(^|(?<=[.!?]) )The staff at " + subject + r" ",
                  lambda m: m[1] + "staff: ", fact)
    return re.sub(r"(^|(?<=[.!?]) )" + subject + r" ",
                  lambda m: m[1], fact)


def _relation_field(fact: str) -> str:
    """简单完整断言拆分关系和值；复杂句不猜测句法边界。"""
    if re.search(r'[;\n\r"“”]|\b(?:not|never|unless|if|except|may|might|must|only)\b', fact, re.I):
        return fact
    match = re.fullmatch(
        r"(?P<relation>(?:staff: )?(?:record|records|measure|measures|monitor|monitors|collect|collects)) "
        r"(?P<value>[^.!?]+?)(?P<time> every [^.!?]+)?[.]?", fact)
    if match:
        return (match["relation"].replace("staff: ", "staff ")
                + (match["time"] or "") + ": " + match["value"])
    match = re.fullmatch(
        r"(?P<relation>(?:is|was|are|were) (?:born in|designed by|located in|in)) "
        r"(?P<value>[^.!?]+)[.]?", fact)
    if match:
        return match["relation"] + ": " + match["value"]
    return fact


def fact_fields(text: str, *, question: str = "", source_id: str = "") -> dict:
    # 保留完整关系句，不能用删停用词的方法丢掉否定、时间或并列成分。
    selected, spans = support_text(text, question) if question else (text, [])
    selected, title_span = source_envelope(text, selected)
    if title_span is not None:
        spans = [{"start": title_span[0], "end": title_span[1]}, *spans]
    source, body = _source_fact(selected)
    return {
        "source_id": source_id or "fact-" + stable_id(text, length=16),
        "source": source,
        "fact": body,
        "text": selected,
        "entities": extract_canonical_entities(selected),
        "source_spans": [[span["start"], span["end"]] for span in spans],
    }


def recalled_evidence(hits: list[dict]) -> list[str]:
    """仅恢复历史检索事实；旧答案、推理结论、经验和金标均不能混入。"""
    result = []
    for hit in hits:
        if hit.get("memory_type", "evidence") != "evidence":
            continue
        fields = hit.get("evidence_fields", [])
        if fields:
            values = [item["text"] for item in fields
                      if isinstance(item, dict) and isinstance(item.get("text"), str)]
        else:
            values = [step.get("text", "")
                      for step in (hit.get("evidence_chain") or {}).get("steps", [])
                      if step.get("kind") == "evidence"
                      and (step.get("source") or {}).get("source_type") == "agent_retrieval"]
            if not values:
                values = (hit.get("trace") or {}).get("evidence", [])
        result.extend(value for value in values if isinstance(value, str) and value.strip())
    return list(dict.fromkeys(result))


def render_evidence(texts: list[str], *, fields: bool = False, question: str = "") -> str:
    """模型数据面不含哈希、检索分数、图诊断或存储引用。"""
    if fields:
        records = {}
        for text in dict.fromkeys(texts):
            selected = fact_fields(text, question=question)["text"] if question else text
            source, fact = _source_fact(selected)
            actions = render_actions(fact, source)
            fact = actions if actions is not None else (_relation_field(_subject_fields(source, fact)) if source else fact)
            key = source or "facts"
            values = records.setdefault(key, [])
            if fact not in values:
                values.append(fact)
        return "\n".join(
            (key + ":\n" if key != "facts" else "")
            + "\n".join("  " + value for value in values)
            for key, values in records.items()
        ) or "(none)"
    return "\n".join(f"[{index}] {text}" for index, text in enumerate(texts)) or "(none)"


def render_evidence_records(records: list[dict]) -> str:
    """共享列名，保留选择角色使用的原始编号及重复证据的独立编号。"""
    lines = ["id | evidence"]
    for record in records:
        value = render_evidence([record["text"]], fields=True)
        # 多行和分隔符按字符串转义，不能伪造下一条证据的编号。
        value = value.replace("\\", "\\\\").replace("\n", " ").replace("\r", " ").replace("|", "\\|")
        lines.append(f'{record["id"]} | {value}')
    return "\n".join(lines)


def render_protocol_records(records: list[dict], columns: tuple[str, ...]) -> str:
    """历史和策略共享字段名；保留值、顺序、空值及记录边界。"""
    if not records:
        return "[]"

    def cell(value):
        return (str(value).replace("\\", "\\\\").replace("\r", "\\r")
                .replace("\n", "\\n").replace("|", "\\|"))

    return "\n".join([
        " | ".join(columns),
        *(" | ".join(cell(row.get(column, "")) for column in columns) for row in records),
    ])
