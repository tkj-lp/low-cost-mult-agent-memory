from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from competition_agents.embedding import embed_text

from .builder import build_hypergraph_view
from .evolution import prepare_searchable_memory
from .gating import evaluate_candidate_gate
from .memory_model import compact_reuse_projection, normalize_memory_record
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
from .text import node_id, tokenize

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


def query_nodes(query: str, profile: dict[str, Any] | None = None) -> set[str]:
    """Return graph nodes activated by the current query."""

    nodes = profile_nodes(profile or build_query_profile(query))
    for token in tokenize(query):
        nodes.add(node_id(NODE_KEYWORD, token))
        nodes.add(node_id(NODE_TAG, token))
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
    dimensions = min(len(left), len(right))
    if dimensions == 0:
        return 0.0
    left_values = left[:dimensions]
    right_values = right[:dimensions]
    dot = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(a * a for a in left_values))
    right_norm = math.sqrt(sum(b * b for b in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    # Map cosine from [-1, 1] to [0, 1] so it can be fused with positive graph scores.
    return max(0.0, min(1.0, (dot / (left_norm * right_norm) + 1.0) / 2.0))


def _embedding_score(query_embedding: list[float], record: dict[str, Any]) -> float:
    record_embedding = record.get("embedding")
    if not isinstance(record_embedding, list):
        return 0.0
    try:
        vector = [float(value) for value in record_embedding]
    except (TypeError, ValueError):
        return 0.0
    return _cosine_similarity(query_embedding, vector)


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

    @classmethod
    def from_memory_records(cls, memory: list[dict[str, Any]]) -> "HypergraphIndex":
        """Build a global inverted index from memory records."""

        memory_by_id: dict[str, dict[str, Any]] = {}
        node_to_memory_ids: dict[str, set[str]] = defaultdict(set)
        node_to_hyperedge_ids: dict[str, set[str]] = defaultdict(set)
        hyperedge_to_memory_id: dict[str, str] = {}
        memory_to_nodes: dict[str, set[str]] = defaultdict(set)

        for raw_record in prepare_searchable_memory(memory):
            record = normalize_memory_record(raw_record)
            memory_id = record["memory_id"]
            hypergraph = build_hypergraph_view(record, persist=False)
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

        semantic_candidates: list[tuple[float, str]] = []
        for memory_id, record in self.memory_by_id.items():
            document_tokens = set(tokenize(str(record.get("search_document") or "")))
            lexical_overlap = len(query_token_set & document_tokens)
            tags = {
                str(tag).strip().lower().replace("_", "-")
                for tag in record.get("tags", [])
            }
            tag_overlap = len(query_token_set & tags)
            semantic_score = _embedding_score(query_embedding, record)
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
            semantic_candidates.append((semantic_score, memory_id))

        # 哈希向量只用于接口回退，不具备语义召回资格；真实向量可独立产生候选。
        if query_embedding_result.provider != "hash":
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
                _embedding_score(query_embedding, record),
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
            scores["retrieval_score"] = round(
                scores["retrieval_score"]
                + gate.score
                + lexical_component
                + tag_component
                + expansion_component
                + chain_query_component,
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
                    "matched_nodes": sorted(matched_nodes, key=_node_sort_key),
                    "matched_hyperedges": sorted(entry["matched_hyperedges"]),
                    "reasoning_path": reasoning_path,
                    "reusable_memory": compact_reuse_projection(record),
                }
            )

        return sorted(
            ranked_hits,
            key=lambda hit: (
                -hit["retrieval_score"],
                -hit["gap_coverage_score"],
                -hit["structural_score"],
                -hit["evidence_chain_quality_score"],
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
