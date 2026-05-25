from __future__ import annotations

import math


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两个 latent 向量的余弦相似度，用于接收方消费 hidden state。"""

    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
