from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

from competition_agents.embedding import embed_text

from .builder import ensure_hypergraph_view
from .gating import evaluate_candidate_gate
from .evolution import prepare_searchable_memory
from .query_profile import build_query_profile, profile_nodes
from .schema import NODE_KEYWORD, NODE_TAG, STRUCTURED_NODE_KINDS
from .scoring import node_kind, score_components
from .text import node_id, tokenize


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
    ) -> None:
        self.memory_by_id = memory_by_id
        self.node_to_memory_ids = node_to_memory_ids
        self.node_to_hyperedge_ids = node_to_hyperedge_ids
        self.hyperedge_to_memory_id = hyperedge_to_memory_id

    @classmethod
    def from_memory_records(cls, memory: list[dict[str, Any]]) -> "HypergraphIndex":
        """Build a global inverted index from memory records."""

        memory_by_id: dict[str, dict[str, Any]] = {}
        node_to_memory_ids: dict[str, set[str]] = defaultdict(set)
        node_to_hyperedge_ids: dict[str, set[str]] = defaultdict(set)
        hyperedge_to_memory_id: dict[str, str] = {}

        for record in prepare_searchable_memory(memory):
            memory_id = record["memory_id"]
            hypergraph = ensure_hypergraph_view(record)
            memory_by_id[memory_id] = record

            for hyperedge in _all_hyperedges(hypergraph):
                hyperedge_id = str(hyperedge["hyperedge_id"])
                hyperedge_to_memory_id[hyperedge_id] = memory_id
                for node in hyperedge.get("nodes", []):
                    node_to_memory_ids[str(node)].add(memory_id)
                    node_to_hyperedge_ids[str(node)].add(hyperedge_id)

        return cls(
            memory_by_id=memory_by_id,
            node_to_memory_ids=dict(node_to_memory_ids),
            node_to_hyperedge_ids=dict(node_to_hyperedge_ids),
            hyperedge_to_memory_id=hyperedge_to_memory_id,
        )

    def search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        """Activate query nodes, aggregate paths to memory, and return top hits."""

        profile = build_query_profile(query)
        query_embedding = embed_text(query).vector
        matched_by_memory: dict[str, dict[str, Any]] = {}
        for node in query_nodes(query, profile):
            for memory_id in self.node_to_memory_ids.get(node, set()):
                entry = matched_by_memory.setdefault(
                    memory_id,
                    {
                        "matched_nodes": set(),
                        "matched_hyperedges": set(),
                        "reasoning_path": [],
                    },
                )
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

        ranked_hits = []
        for memory_id, entry in matched_by_memory.items():
            record = self.memory_by_id[memory_id]
            matched_nodes = entry["matched_nodes"]
            gate = evaluate_candidate_gate(query, profile, record)
            if not gate.accepted:
                continue
            embedding_score = _embedding_score(query_embedding, record)
            scores = score_components(record, matched_nodes, embedding_score=embedding_score)
            scores["retrieval_score"] = round(scores["retrieval_score"] + gate.score, 4)
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
                    "matched_nodes": sorted(matched_nodes, key=_node_sort_key),
                    "matched_hyperedges": sorted(entry["matched_hyperedges"]),
                    "reasoning_path": reasoning_path,
                }
            )

        return sorted(
            ranked_hits,
            key=lambda hit: (
                -hit["gap_coverage_score"],
                -hit["structural_score"],
                -hit["experience_score"],
                -hit["lifecycle_score"],
                -hit["retention_score"],
                -hit["reuse_score"],
                -hit["strategy_score"],
                -hit["evidence_score"],
                -hit["tag_score"],
                -hit["candidate_gate_score"],
                -hit["retrieval_score"],
                hit["keyword_score"],
                hit["memory_id"],
            ),
        )[:top_k]
