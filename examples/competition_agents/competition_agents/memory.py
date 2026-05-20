from __future__ import annotations

from datetime import datetime, timezone

from competition_agents.state import simple_embedding


def add_memory(
    memory: list[dict],
    *,
    source_agent: str,
    task_topic: str,
    summary: str,
    tags: list[str],
) -> list[dict]:
    """向当前内存存储追加一条可复用记忆。"""

    # 元数据字段保持稳定，方便网页和超图模块对接。
    record = {
        "memory_id": f"mem-{len(memory) + 1:04d}",
        "source_agent": source_agent,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_topic": task_topic,
        "summary": summary,
        "tags": tags,
        "embedding": simple_embedding(task_topic + " " + summary),
    }
    return [*memory, record]


def search_memory(memory: list[dict], query: str, *, top_k: int) -> list[dict]:
    """用简单的关键词和标签分数检索记忆。"""

    if not memory:
        return []

    # 当前本地打分逻辑故意保持简单，后续会替换成超图检索。
    query_tokens = set(query.lower().split())

    def score(record: dict) -> tuple[int, int]:
        haystack = " ".join(
            [
                record.get("task_topic", ""),
                record.get("summary", ""),
                " ".join(record.get("tags", [])),
            ]
        ).lower()
        overlap = len(query_tokens.intersection(haystack.split()))
        tag_bonus = sum(1 for tag in record.get("tags", []) if tag.lower() in query.lower())
        return overlap + tag_bonus, len(record.get("summary", ""))

    # 只返回正分结果，避免无关任务被计为命中。
    ranked = sorted(memory, key=score, reverse=True)
    return [record for record in ranked[:top_k] if score(record)[0] > 0]
