from __future__ import annotations

import json
import math
import hashlib
import pickle
import threading
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any

from competition_agents.embedding import embed_text

from .builder import build_hypergraph_view
from .entities import extract_canonical_entities
from .evolution import prepare_searchable_memory, retention_profile
from .gating import evaluate_candidate_gate
from .memory_model import (
    compact_reuse_projection,
    compact_trace,
    normalize_memory_record,
    validate_memory_ids,
)
from .query_profile import build_query_profile, profile_nodes
from .schema import (
    NODE_CLAIM,
    NODE_ENTITY,
    NODE_EVIDENCE_SOURCE,
    NODE_FAILURE_PATTERN,
    NODE_KEYWORD,
    NODE_MEMORY_REF,
    NODE_PROBLEM_PATTERN,
    NODE_SOLUTION_PATTERN,
    NODE_SUPPORTING_FACT,
    NODE_TAG,
    NODE_TRIGGER,
    STRUCTURED_NODE_KINDS,
)
from .scoring import node_kind, score_components
from .text import node_id, normalize_text, stable_id, tokenize

EXPANSION_NODE_KINDS = {
    NODE_ENTITY,
    NODE_EVIDENCE_SOURCE,
    NODE_MEMORY_REF,
    NODE_SUPPORTING_FACT,
    NODE_PROBLEM_PATTERN,
    NODE_SOLUTION_PATTERN,
    NODE_FAILURE_PATTERN,
    NODE_TRIGGER,
    NODE_CLAIM,
}

_INDEX_CACHE_LIMIT = 8
_INDEX_CACHE: "OrderedDict[str, HypergraphIndex]" = OrderedDict()
_INDEX_CACHE_LOCK = threading.Lock()


def _memory_index_signature(memory: list[dict[str, Any]]) -> str:
    """Fingerprint graph-relevant fields while ignoring mutable usage counters."""

    # Include vectors, provenance and review state: changing any of them must
    # invalidate cached rankings even when the caller omits a version bump.
    ignored = {
        "hypergraph",
        "retrieval_count",
        "reuse_count",
        "reuse_success_count",
        "reuse_failure_count",
        "last_retrieved_at",
        "last_reused_at",
        "confidence",
    }
    rows = []
    for record in memory:
        row = {key: value for key, value in record.items() if key not in ignored}
        # Fingerprint numeric vectors without formatting every float as text.
        # Recompute from contents so in-place edits still invalidate the index.
        if isinstance(row.get("embedding"), list):
            row["embedding"] = hashlib.sha256(
                pickle.dumps(row["embedding"], protocol=5)
            ).hexdigest()
        if isinstance(row.get("trace"), dict):
            row["trace"] = compact_trace(row["trace"])
        rows.append(row)
    serialized = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str)
    return stable_id(serialized, length=32)


def clear_hypergraph_index_cache() -> None:
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE.clear()


def get_cached_hypergraph_index(memory: list[dict[str, Any]]) -> "HypergraphIndex":
    """Reuse the immutable index for repeated queries over one memory snapshot."""

    signature = _memory_index_signature(memory)
    with _INDEX_CACHE_LOCK:
        cached = _INDEX_CACHE.get(signature)
        if cached is not None:
            _INDEX_CACHE.move_to_end(signature)
            cached.refresh_runtime_fields(memory)
            return cached

    built = HypergraphIndex.from_memory_records(memory)
    with _INDEX_CACHE_LOCK:
        cached = _INDEX_CACHE.get(signature)
        if cached is not None:
            _INDEX_CACHE.move_to_end(signature)
            cached.refresh_runtime_fields(memory)
            return cached
        _INDEX_CACHE[signature] = built
        while len(_INDEX_CACHE) > _INDEX_CACHE_LIMIT:
            _INDEX_CACHE.popitem(last=False)
    return built


def query_nodes(query: str, profile: dict[str, Any] | None = None) -> set[str]:
    """Return graph nodes activated by the current query."""

    nodes = profile_nodes(profile or build_query_profile(query))
    for token in tokenize(query):
        nodes.add(node_id(NODE_KEYWORD, token))
        nodes.add(node_id(NODE_TAG, token))
    for entity in extract_canonical_entities(query):
        nodes.add(node_id(NODE_ENTITY, entity))
    return nodes


def _all_hyperedges(hypergraph: dict[str, Any]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen: set[str] = set()
    event = hypergraph.get("event_hyperedge")
    if isinstance(event, dict):
        edges.append(event)
        seen.add(str(event.get("hyperedge_id")))
    for edge in hypergraph.get("hyperedges", []):
        if not isinstance(edge, dict):
            continue
        edge_id = str(edge.get("hyperedge_id"))
        if edge_id in seen:
            continue
        edges.append(edge)
        seen.add(edge_id)
    return edges


def _node_sort_key(node: str) -> tuple[int, str]:
    kind = node_kind(node)
    if kind in STRUCTURED_NODE_KINDS:
        return (0, node)
    if kind == NODE_TAG:
        return (1, node)
    return (2, node)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dimensions = len(left)
    if dimensions == 0 or dimensions != len(right):
        return 0.0
    if not all(math.isfinite(value) for value in [*left, *right]):
        return 0.0
    left_values = left[:dimensions]
    right_values = right[:dimensions]
    left_norm = math.hypot(*left_values)
    right_norm = math.hypot(*right_values)
    if not left_norm or not right_norm or not math.isfinite(left_norm + right_norm):
        return 0.0
    cosine = sum(
        (a / left_norm) * (b / right_norm) for a, b in zip(left_values, right_values)
    )
    # Map cosine from [-1, 1] to [0, 1] so it can be fused with positive graph scores.
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def _embedding_score(query_embedding: list[float], record: dict[str, Any]) -> float:
    record_embedding = record.get("embedding")
    if not isinstance(record_embedding, list):
        return 0.0
    try:
        vector = [float(value) for value in record_embedding]
    except (TypeError, ValueError):
        return 0.0
    return _cosine_similarity(query_embedding, vector)


def compatible_embedding(result: Any, record: dict[str, Any]) -> bool:
    """Only compare vectors produced by the same declared embedding space."""

    return (
        result.provider != "hash"
        and record.get("embedding_provider") == result.provider
        and record.get("embedding_model") == result.model
        and isinstance(record.get("embedding"), list)
        and len(record["embedding"]) == len(result.vector)
    )


def _continuous_reuse_score(
    profile: dict[str, Any], record: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    """Boost records that satisfy the current continuous-task reuse contract."""

    group_ids = {
        normalize_text(str(value))
        for value in profile.get("group_ids", [])
        if str(value).strip()
    }
    artifacts = {
        normalize_text(str(value))
        for value in profile.get("artifacts", [])
        if str(value).strip()
    }
    record_group = normalize_text(str(record.get("group_id") or ""))
    record_turn = normalize_text(str(record.get("turn_id") or ""))
    record_artifacts = {
        normalize_text(str(value))
        for value in record.get("artifact_types", []) or []
        if str(value).strip()
    }

    score = 0.0
    reasons: list[str] = []
    if group_ids and record_group in group_ids:
        score += 4.0
        reasons.append("same_group")
    artifact_overlap = artifacts & record_artifacts
    if artifact_overlap:
        score += min(len(artifact_overlap) * 4.0, 10.0)
        reasons.append("artifact_match")
    return round(score, 4), {
        "record_group": record_group,
        "record_turn": record_turn,
        "record_artifacts": sorted(record_artifacts),
        "matched_reuse_reasons": reasons,
    }


class HypergraphIndex:
    """In-memory hypergraph index for reusable competition-agent memory."""

    def __init__(
        self,
        *,
        memory_by_id: dict[str, dict[str, Any]],
        node_to_memory_ids: dict[str, set[str]],
        node_to_hyperedge_ids: dict[str, set[str]],
        hyperedge_to_memory_id: dict[str, str],
        memory_to_nodes: dict[str, set[str]],
    ) -> None:
        self.memory_by_id = memory_by_id
        self.node_to_memory_ids = node_to_memory_ids
        self.node_to_hyperedge_ids = node_to_hyperedge_ids
        self.hyperedge_to_memory_id = hyperedge_to_memory_id
        self.memory_to_nodes = memory_to_nodes
        self.token_to_memory_ids: dict[str, set[str]] = defaultdict(set)
        self.tag_to_memory_ids: dict[str, set[str]] = defaultdict(set)
        for memory_id, record in memory_by_id.items():
            for token in set(tokenize(str(record.get("search_document") or ""))):
                self.token_to_memory_ids[token].add(memory_id)
            for tag in record.get("tags", []):
                self.tag_to_memory_ids[str(tag).strip().lower().replace("_", "-")].add(
                    memory_id
                )

    @classmethod
    def from_memory_records(cls, memory: list[dict[str, Any]]) -> "HypergraphIndex":
        """Build a global inverted index from memory records."""

        memory_by_id: dict[str, dict[str, Any]] = {}
        node_to_memory_ids: dict[str, set[str]] = defaultdict(set)
        node_to_hyperedge_ids: dict[str, set[str]] = defaultdict(set)
        hyperedge_to_memory_id: dict[str, str] = {}
        memory_to_nodes: dict[str, set[str]] = defaultdict(set)

        validate_memory_ids(memory)
        for raw_record in prepare_searchable_memory(memory):
            record = normalize_memory_record(raw_record)
            memory_id = record["memory_id"]
            hypergraph = build_hypergraph_view(record, persist=False, _normalized=True)
            record["hypergraph"] = hypergraph
            memory_by_id[memory_id] = record

            for hyperedge in _all_hyperedges(hypergraph):
                hyperedge_id = str(hyperedge["hyperedge_id"])
                hyperedge_to_memory_id[hyperedge_id] = memory_id
                for node in hyperedge.get("nodes", []):
                    node_to_memory_ids[str(node)].add(memory_id)
                    node_to_hyperedge_ids[str(node)].add(hyperedge_id)
                    memory_to_nodes[memory_id].add(str(node))

        return cls(
            memory_by_id=memory_by_id,
            node_to_memory_ids=dict(node_to_memory_ids),
            node_to_hyperedge_ids=dict(node_to_hyperedge_ids),
            hyperedge_to_memory_id=hyperedge_to_memory_id,
            memory_to_nodes=dict(memory_to_nodes),
        )

    def search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        """执行多路召回、跨记忆关系扩展和证据链约束排序。"""

        if top_k <= 0 or not str(query).strip():
            return []

        profile = build_query_profile(query)
        query_embedding_result = embed_text(query)
        query_embedding = query_embedding_result.vector
        query_token_set = set(tokenize(query))
        matched_by_memory: dict[str, dict[str, Any]] = {}

        def candidate(memory_id: str) -> dict[str, Any]:
            return matched_by_memory.setdefault(
                memory_id,
                {
                    "matched_nodes": set(),
                    "matched_hyperedges": set(),
                    "reasoning_path": [],
                    "recall_channels": set(),
                    "lexical_overlap": 0,
                    "tag_overlap": 0,
                    "semantic_score": 0.0,
                    "graph_expansion_score": 0.0,
                },
            )

        for node in query_nodes(query, profile):
            for memory_id in self.node_to_memory_ids.get(node, set()):
                entry = candidate(memory_id)
                entry["recall_channels"].add("hypergraph_node")
                entry["matched_nodes"].add(node)

                for hyperedge_id in self.node_to_hyperedge_ids.get(node, set()):
                    if self.hyperedge_to_memory_id.get(hyperedge_id) != memory_id:
                        continue
                    entry["matched_hyperedges"].add(hyperedge_id)
                    entry["reasoning_path"].append(
                        {
                            "query_node": node,
                            "hyperedge_id": hyperedge_id,
                            "memory_id": memory_id,
                        }
                    )

        lexical_counts: dict[str, int] = defaultdict(int)
        tag_counts: dict[str, int] = defaultdict(int)
        for token in query_token_set:
            for memory_id in self.token_to_memory_ids.get(token, ()):
                lexical_counts[memory_id] += 1
            for memory_id in self.tag_to_memory_ids.get(token, ()):
                tag_counts[memory_id] += 1
        for memory_id in sorted(lexical_counts.keys() | tag_counts.keys()):
            lexical_overlap = lexical_counts.get(memory_id, 0)
            tag_overlap = tag_counts.get(memory_id, 0)
            if lexical_overlap:
                entry = candidate(memory_id)
                entry["recall_channels"].add("keyword")
                entry["lexical_overlap"] = lexical_overlap
                entry["reasoning_path"].append(
                    {
                        "query_node": "keyword:lexical-overlap",
                        "hyperedge_id": f"hyperedge:{memory_id}:keyword-recall",
                        "memory_id": memory_id,
                    }
                )
            if tag_overlap:
                entry = candidate(memory_id)
                entry["recall_channels"].add("tag")
                entry["tag_overlap"] = tag_overlap
                entry["reasoning_path"].append(
                    {
                        "query_node": "tag:exact-overlap",
                        "hyperedge_id": f"hyperedge:{memory_id}:tag-recall",
                        "memory_id": memory_id,
                    }
                )

        # 哈希向量只用于接口回退，不具备语义召回资格；真实向量可独立产生候选。
        if query_embedding_result.provider != "hash":
            semantic_candidates = [
                (_embedding_score(query_embedding, record), memory_id)
                for memory_id, record in self.memory_by_id.items()
                if compatible_embedding(query_embedding_result, record)
            ]
            semantic_limit = max(top_k * 4, 8)
            for semantic_score, memory_id in sorted(semantic_candidates, reverse=True)[
                :semantic_limit
            ]:
                if semantic_score < 0.62:
                    continue
                entry = candidate(memory_id)
                entry["recall_channels"].add("semantic")
                entry["semantic_score"] = semantic_score
                entry["reasoning_path"].append(
                    {
                        "query_node": "semantic:embedding",
                        "hyperedge_id": f"hyperedge:{memory_id}:semantic-recall",
                        "memory_id": memory_id,
                    }
                )

        # 从直接召回的记忆沿实体、来源、规则和记忆引用扩展到相关历史记忆。
        seed_ids = list(matched_by_memory)
        max_expansion_degree = max(
            3, min(20, int(math.sqrt(max(1, len(self.memory_by_id)))) + 1)
        )
        for seed_id in seed_ids:
            for shared_node in sorted(self.memory_to_nodes.get(seed_id, set())):
                if node_kind(shared_node) not in EXPANSION_NODE_KINDS:
                    continue
                neighbors = self.node_to_memory_ids.get(shared_node, set())
                if len(neighbors) > max_expansion_degree:
                    continue
                inverse_frequency_weight = 1.0 / max(
                    1.0, math.log2(len(neighbors) + 1.0)
                )
                for neighbor_id in neighbors:
                    if neighbor_id == seed_id:
                        continue
                    entry = candidate(neighbor_id)
                    entry["recall_channels"].add("hypergraph_expansion")
                    entry["graph_expansion_score"] += inverse_frequency_weight
                    entry["matched_nodes"].add(shared_node)
                    neighbor_edges = [
                        edge_id
                        for edge_id in self.node_to_hyperedge_ids.get(
                            shared_node, set()
                        )
                        if self.hyperedge_to_memory_id.get(edge_id) == neighbor_id
                    ]
                    entry["matched_hyperedges"].update(neighbor_edges)
                    entry["reasoning_path"].append(
                        {
                            "query_node": f"memory:{seed_id}",
                            "hyperedge_id": (
                                neighbor_edges[0]
                                if neighbor_edges
                                else f"hyperedge:{neighbor_id}:graph-expansion"
                            ),
                            "memory_id": neighbor_id,
                            "seed_memory_id": seed_id,
                            "via_node": shared_node,
                        }
                    )

        ranked_hits = []
        for memory_id, entry in matched_by_memory.items():
            record = self.memory_by_id[memory_id]
            matched_nodes = entry["matched_nodes"]
            gate = evaluate_candidate_gate(query, profile, record)
            if not gate.accepted:
                continue
            embedding_score = max(
                float(entry.get("semantic_score", 0.0) or 0.0),
                _embedding_score(query_embedding, record)
                if compatible_embedding(query_embedding_result, record)
                else 0.0,
            )
            scores = score_components(
                record, matched_nodes, embedding_score=embedding_score
            )
            lexical_component = min(float(entry["lexical_overlap"]), 8.0) * 1.5
            tag_component = min(float(entry["tag_overlap"]), 4.0) * 3.5
            expansion_component = min(float(entry["graph_expansion_score"]), 6.0) * 1.25
            chain_query_component = (
                float(record.get("evidence_chain_completeness", 0.0) or 0.0) * 4.0
                if profile.get("evidence_chain_requested")
                else 0.0
            )
            continuous_reuse_score, continuous_reuse_debug = _continuous_reuse_score(
                profile, record
            )
            scores["retrieval_score"] = round(
                scores["retrieval_score"]
                + gate.score
                + lexical_component
                + tag_component
                + expansion_component
                + chain_query_component
                + continuous_reuse_score,
                4,
            )
            if scores["retrieval_score"] <= 0:
                continue

            reasoning_path = sorted(
                entry["reasoning_path"],
                key=lambda path: (
                    _node_sort_key(path["query_node"]),
                    path["hyperedge_id"],
                    path["memory_id"],
                ),
            )
            ranked_hits.append(
                {
                    **record,
                    **scores,
                    **gate.as_dict(),
                    "retrieval_mode": "hypergraph_multichannel",
                    "recall_channels": sorted(entry["recall_channels"]),
                    "lexical_overlap_score": round(lexical_component, 4),
                    "tag_overlap_score": round(tag_component, 4),
                    "graph_expansion_score": round(expansion_component, 4),
                    "evidence_chain_query_score": round(chain_query_component, 4),
                    "continuous_reuse_score": continuous_reuse_score,
                    "continuous_reuse_debug": continuous_reuse_debug,
                    "matched_nodes": sorted(matched_nodes, key=_node_sort_key),
                    "matched_hyperedges": sorted(entry["matched_hyperedges"]),
                    "reasoning_path": reasoning_path,
                }
            )

        selected = sorted(
            ranked_hits,
            key=lambda hit: (
                -hit["retrieval_score"],
                -hit["gap_coverage_score"],
                -hit["structural_score"],
                -hit["evidence_chain_quality_score"],
                -hit["continuous_reuse_score"],
                -hit["experience_score"],
                -hit["lifecycle_score"],
                -hit["retention_score"],
                -hit["reuse_score"],
                -hit["strategy_score"],
                -hit["evidence_score"],
                -hit["tag_score"],
                -hit["candidate_gate_score"],
                hit["keyword_score"],
                hit["memory_id"],
            ),
        )[:top_k]
        for hit in selected:
            hit["reusable_memory"] = compact_reuse_projection(
                self.memory_by_id[hit["memory_id"]], _normalized=True
            )
        # Consumers may annotate hits. Never let them mutate the cached index.
        return deepcopy(selected)

    def refresh_runtime_fields(self, memory: list[dict[str, Any]]) -> None:
        """Refresh counters that affect ranking without rebuilding graph topology."""

        count_fields = (
            "retrieval_count",
            "reuse_count",
            "reuse_success_count",
            "reuse_failure_count",
        )
        raw_by_id = {str(record.get("memory_id") or ""): record for record in memory}
        for memory_id, cached in self.memory_by_id.items():
            sources = [
                raw_by_id[value]
                for value in {memory_id, *cached.get("merged_memory_ids", [])}
                if value in raw_by_id
            ]
            if not sources:
                continue
            for field in count_fields:
                cached[field] = sum(
                    int(source.get(field, 0) or 0) for source in sources
                )
            for field in ("last_retrieved_at", "last_reused_at"):
                values = [str(source[field]) for source in sources if source.get(field)]
                if values:
                    cached[field] = max(values)
                else:
                    cached.pop(field, None)
            cached["confidence"] = max(
                float(source.get("confidence", 0.5) or 0.0) for source in sources
            )
            cached.update(retention_profile(cached))
