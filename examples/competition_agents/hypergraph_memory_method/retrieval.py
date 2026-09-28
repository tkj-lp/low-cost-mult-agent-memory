from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Literal

from competition_agents.embedding import embed_text

from .index import _cosine_similarity, compatible_embedding, get_cached_hypergraph_index
from .memory_model import normalize_memory_record, validate_memory_ids
from .text import tokenize

RetrievalMode = Literal["keyword", "tag", "semantic", "hypergraph", "fusion"]
RETRIEVAL_MODE_VALUES: tuple[RetrievalMode, ...] = (
    "keyword",
    "tag",
    "semantic",
    "hypergraph",
    "fusion",
)
TEXT_RETRIEVAL_MODES = {"keyword", "tag", "semantic"}
HYPERGRAPH_RETRIEVAL_MODES = {"hypergraph", "fusion"}


def _text_document(record: dict[str, Any]) -> str:
    values = [
        record.get("task_topic", ""),
        record.get("summary", ""),
        record.get("group_id", ""),
        record.get("turn_id", ""),
        " ".join(str(value) for value in record.get("artifact_types", []) or []),
        " ".join(str(value) for value in record.get("tags", []) or []),
        record.get("problem_pattern", ""),
        record.get("solution_pattern", ""),
        record.get("failure_pattern", ""),
        " ".join(str(value) for value in record.get("trigger_conditions", []) or []),
        " ".join(str(value) for value in record.get("supporting_evidence", []) or []),
        " ".join(str(value) for value in record.get("evidence", []) or []),
    ]
    return "\n".join(str(value).strip() for value in values if str(value).strip())


def _prepared(memory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_record in memory:
        record = normalize_memory_record(raw_record)
        for key in (
            "hypergraph",
            "evidence_chain",
            "evidence_chain_id",
            "evidence_chain_valid",
            "evidence_chain_completeness",
        ):
            record.pop(key, None)
        record["memory_organization"] = "text"
        record["search_document"] = _text_document(record)
        records.append(record)
    return records


def _compact_text_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_id": record.get("memory_id"),
        "global_memory_id": record.get("global_memory_id"),
        "memory_type": record.get("memory_type", "evidence"),
        "memory_organization": "text",
        "task_topic": record.get("task_topic", ""),
        "summary": record.get("summary", ""),
        "tags": record.get("tags", []),
        "supporting_evidence": record.get("supporting_evidence", []),
    }


def search_keyword_memory(
    memory: list[dict[str, Any]],
    query: str,
    *,
    top_k: int,
    _records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """使用带逆文档频率的关键词基线检索文本记忆。"""

    if top_k <= 0:
        return []
    records = _records if _records is not None else _prepared(memory)
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return []
    document_tokens = {
        record["memory_id"]: set(tokenize(str(record.get("search_document") or "")))
        for record in records
    }
    frequencies = Counter(
        token
        for tokens in document_tokens.values()
        for token in tokens
        if token in query_tokens
    )
    scored: list[tuple[float, dict[str, Any]]] = []
    for record in records:
        tokens = document_tokens[record["memory_id"]]
        overlap = query_tokens & tokens
        if not overlap:
            continue
        score = sum(
            math.log((len(records) + 1) / (frequencies[token] + 0.5)) + 1.0
            for token in overlap
        )
        scored.append(
            (
                score,
                {
                    **record,
                    "retrieval_mode": "keyword",
                    "memory_organization": "text",
                    "retrieval_score": round(score, 4),
                    "matched_keywords": sorted(overlap),
                    "reusable_memory": _compact_text_projection(record),
                },
            )
        )
    return [
        item
        for _, item in sorted(
            scored, key=lambda pair: (-pair[0], pair[1]["memory_id"])
        )[:top_k]
    ]


def search_tag_memory(
    memory: list[dict[str, Any]],
    query: str,
    *,
    top_k: int,
    _records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """使用精确标签交集检索文本记忆。"""

    if top_k <= 0:
        return []
    query_tags = set(tokenize(query))
    scored: list[tuple[float, dict[str, Any]]] = []
    records = _records if _records is not None else _prepared(memory)
    for record in records:
        tags = {
            str(tag).strip().lower().replace("_", "-") for tag in record.get("tags", [])
        }
        overlap = query_tags & tags
        if not overlap:
            continue
        precision = len(overlap) / max(1, len(tags))
        coverage = len(overlap) / max(1, len(query_tags))
        score = len(overlap) * 3.0 + precision + coverage
        scored.append(
            (
                score,
                {
                    **record,
                    "retrieval_mode": "tag",
                    "memory_organization": "text",
                    "retrieval_score": round(score, 4),
                    "matched_tags": sorted(overlap),
                    "reusable_memory": _compact_text_projection(record),
                },
            )
        )
    return [
        item
        for _, item in sorted(
            scored, key=lambda pair: (-pair[0], pair[1]["memory_id"])
        )[:top_k]
    ]


def search_semantic_memory(
    memory: list[dict[str, Any]],
    query: str,
    *,
    top_k: int,
    _records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """在文本记忆上执行独立向量召回。"""

    if top_k <= 0 or not str(query).strip():
        return []
    query_embedding = embed_text(query)
    scored: list[tuple[float, dict[str, Any]]] = []
    records = _records if _records is not None else _prepared(memory)
    if query_embedding.provider == "hash":
        # A whole-text digest does not preserve meaning. Report a lexical
        # fallback instead of ranking unrelated texts by random hash distance.
        return [
            {
                **hit,
                "retrieval_mode": "semantic",
                "semantic_provider": "hash",
                "semantic_fallback": True,
            }
            for hit in search_keyword_memory(
                memory, query, top_k=top_k, _records=records
            )
        ]
    for record in records:
        values = record.get("embedding")
        if not compatible_embedding(query_embedding, record):
            record = dict(record)
            document_embedding = embed_text(str(record.get("search_document") or ""))
            values = document_embedding.vector
            record["embedding"] = values
            record["embedding_provider"] = document_embedding.provider
            record["embedding_model"] = document_embedding.model
        if not compatible_embedding(query_embedding, record):
            continue
        try:
            score = _cosine_similarity(
                query_embedding.vector, [float(value) for value in values]
            )
        except (TypeError, ValueError):
            continue
        scored.append(
            (
                score,
                {
                    **record,
                    "retrieval_mode": "semantic",
                    "memory_organization": "text",
                    "retrieval_score": round(score, 4),
                    "embedding_score": round(score, 4),
                    "semantic_provider": query_embedding.provider,
                    "semantic_fallback": query_embedding.provider == "hash",
                    "reusable_memory": _compact_text_projection(record),
                },
            )
        )
    return [
        item
        for _, item in sorted(
            scored, key=lambda pair: (-pair[0], pair[1]["memory_id"])
        )[:top_k]
    ]


def search_memory_by_mode(
    memory: list[dict[str, Any]],
    query: str,
    *,
    top_k: int,
    mode: RetrievalMode = "hypergraph",
    memory_types: set[str] | None = None,
    type_limits: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """以稳定输入输出切换五种记忆检索模式。"""

    if top_k <= 0 or not str(query).strip():
        return []
    allowed_types = set(memory_types or ()) | set(type_limits or ())
    include_failed = allowed_types == {"failed_experience"}
    memory = [
        record
        for record in memory
        if (
            not allowed_types
            or str(record.get("memory_type") or "evidence") in allowed_types
        )
        and (
            include_failed
            or (
                record.get("use_as_positive", True)
                and str(record.get("quality_status") or "")
                not in {"failed", "rejected"}
                and str(record.get("review_status") or "")
                not in {"rejected", "archived", "superseded"}
                and not record.get("pruned", False)
                and not (
                    record.get("memory_type") == "experience"
                    and record.get("review_status") in {"candidate", "pending"}
                )
            )
        )
    ]
    validate_memory_ids(memory)
    if not memory:
        return []
    candidate_k = len(memory) if type_limits else top_k
    if mode == "keyword":
        return _limit_by_memory_type(
            search_keyword_memory(memory, query, top_k=candidate_k),
            top_k=top_k,
            memory_types=memory_types,
            type_limits=type_limits,
        )
    if mode == "tag":
        return _limit_by_memory_type(
            search_tag_memory(memory, query, top_k=candidate_k),
            top_k=top_k,
            memory_types=memory_types,
            type_limits=type_limits,
        )
    if mode == "semantic":
        return _limit_by_memory_type(
            search_semantic_memory(memory, query, top_k=candidate_k),
            top_k=top_k,
            memory_types=memory_types,
            type_limits=type_limits,
        )
    if mode == "hypergraph":
        hits = [
            {**hit, "memory_organization": "hypergraph"}
            for hit in get_cached_hypergraph_index(memory).search(
                query, top_k=candidate_k
            )
        ]
        return _limit_by_memory_type(
            hits,
            top_k=top_k,
            memory_types=memory_types,
            type_limits=type_limits,
        )
    if mode != "fusion":
        raise ValueError(f"Unsupported retrieval mode: {mode}")

    # 全局超图索引已经集成实体、关键词、标签、语义补召回和关系扩展。
    # 不再在外层重复规范化全部记忆、执行三次全量文本检索。
    # 单类型限额不需要取回整个库；多类型仍保留完整候选以满足各类配额。
    kinds = {str(record.get("memory_type") or "evidence") for record in memory}
    candidate_k = len(memory) if type_limits and len(kinds) > 1 else top_k
    indexed_hits = get_cached_hypergraph_index(memory).search(query, top_k=candidate_k)
    hits = [
        {
            **hit,
            "retrieval_mode": "fusion",
            "memory_organization": "hypergraph",
            "fusion_implementation": "indexed_multichannel",
            "fusion_score": hit["retrieval_score"],
            "fusion_channels": hit.get("recall_channels", []),
        }
        for hit in indexed_hits
    ]
    return _limit_by_memory_type(
        hits,
        top_k=top_k,
        memory_types=memory_types,
        type_limits=type_limits,
    )


def _limit_by_memory_type(
    hits: list[dict[str, Any]],
    *,
    top_k: int,
    memory_types: set[str] | None,
    type_limits: dict[str, int] | None,
) -> list[dict[str, Any]]:
    allowed = set(memory_types or [])
    limits = {
        str(key): max(0, int(value)) for key, value in (type_limits or {}).items()
    }
    if limits:
        allowed |= set(limits)
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for hit in hits:
        memory_type = str(hit.get("memory_type", "evidence") or "evidence")
        if allowed and memory_type not in allowed:
            continue
        if limits and counts[memory_type] >= limits.get(memory_type, 0):
            continue
        selected.append(hit)
        counts[memory_type] += 1
        if len(selected) >= top_k:
            break
    return selected


def compare_retrieval_modes(
    memory: list[dict[str, Any]], query: str, *, top_k: int
) -> dict[str, list[dict[str, Any]]]:
    """使用相同记忆与查询生成可复现的模式对比结果。"""

    return {
        mode: search_memory_by_mode(memory, query, top_k=top_k, mode=mode)
        for mode in ("keyword", "tag", "semantic", "hypergraph", "fusion")
    }
