from __future__ import annotations

from typing import Any


NODE_WEIGHTS = {
    "tag": 3.0,
    "keyword": 1.0,
}


def score_record(record: dict[str, Any], matched_nodes: set[str]) -> float:
    """按命中的节点类型给记忆打分，避免常见词命中后返回所有记忆。"""

    score = 0.0
    for node in matched_nodes:
        kind, _, _ = node.partition(":")
        score += NODE_WEIGHTS.get(kind, 0.5)
    # 同分时让信息更丰富的总结略微靠前。
    score += min(len(record.get("summary", "")) / 1000.0, 0.25)
    return round(score, 4)
