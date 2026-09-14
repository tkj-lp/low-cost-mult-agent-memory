from __future__ import annotations

import importlib
import json
import math
import time
from collections import OrderedDict, defaultdict
from typing import Any, Callable

from .contracts import MemoryEncoder, RetrievalRequest
from .entities import canonical_entity, extract_canonical_entities
from .memory_model import compact_reuse_projection
from .service_store import ServiceStore, json_text
from .text import indexed_tokens, node_id


def checked_vector(values: Any) -> list[float]:
    if isinstance(values, (str, bytes, dict)):
        raise ValueError("embedding must be a sequence of numbers")
    vector = [float(value) for value in values]
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding must be a non-empty finite vector")
    if not any(vector):
        raise ValueError("embedding must have non-zero norm")
    scale = max(abs(value) for value in vector)
    scaled = [value / scale for value in vector]
    norm = math.sqrt(sum(value * value for value in scaled))
    return [value / norm for value in scaled]


def personalized_rank(
    seeds: dict[str, float],
    edges: dict[tuple[str, str], float],
    *,
    iterations: int = 40,
) -> dict[str, float]:
    """Weighted random walk on the bounded projection of hyperedge incidences."""

    if not seeds:
        return {}
    total = sum(seeds.values())
    restart = {key: value / total for key, value in seeds.items()}
    neighbors: dict[str, dict[str, float]] = defaultdict(dict)
    for (left, right), weight in edges.items():
        if left == right or weight <= 0:
            continue
        neighbors[left][right] = weight
        neighbors[right][left] = weight
    nodes = set(restart) | set(neighbors)
    rank = {key: restart.get(key, 0.0) for key in nodes}
    for _ in range(iterations):
        dangling = sum(rank[key] for key in nodes if not neighbors.get(key))
        next_rank = {
            key: (0.15 + 0.85 * dangling) * restart.get(key, 0.0) for key in nodes
        }
        for key, links in neighbors.items():
            denominator = sum(links.values())
            for target, weight in links.items():
                next_rank[target] += 0.85 * rank[key] * weight / denominator
        difference = sum(abs(next_rank[key] - rank[key]) for key in nodes)
        rank = next_rank
        if difference < 1e-9:
            break
    return rank


class IndexedRetriever:
    def __init__(
        self,
        store: ServiceStore,
        namespace: str,
        encoder: MemoryEncoder | None,
        *,
        vector_backend: str = "exact",
        reranker: Callable[[str, list[dict[str, Any]]], list[float]] | None = None,
    ) -> None:
        if vector_backend not in {"exact", "hnsw"}:
            raise ValueError("vector_backend must be exact or hnsw")
        self.store, self.namespace, self.encoder = store, namespace, encoder
        self.vector_backend, self.reranker = vector_backend, reranker
        self._queries: OrderedDict[tuple[str, str], list[float]] = OrderedDict()
        self._vectors: tuple[Any, ...] | None = None
        self._hnsw = None
        if vector_backend == "hnsw":
            try:
                self._hnsw = importlib.import_module("hnswlib")
            except ImportError as exc:
                raise RuntimeError(
                    "hnsw backend requires hnswlib; no silent fallback"
                ) from exc

    def _semantic(
        self, request: RetrievalRequest, where: str, params: list[Any]
    ) -> tuple[list[str], dict[str, Any]]:
        if self.encoder is None or not self.encoder.model_id:
            raise RuntimeError(
                "semantic/fusion retrieval requires a configured memory encoder"
            )
        import numpy as np

        key = (self.encoder.model_id, request.query)
        query_cached = key in self._queries
        if not query_cached:
            encoded = self.encoder.encode([request.query])
            if len(encoded) != 1:
                raise ValueError("encoder returned an unexpected batch size")
            self._queries[key] = checked_vector(encoded[0])
        self._queries.move_to_end(key)
        while len(self._queries) > 128:
            self._queries.popitem(last=False)
        query = np.asarray(self._queries[key], dtype=np.float32)
        cache_key = (self.store.revision, self.encoder.model_id, where, tuple(params))
        cache_hit = self._vectors is not None and self._vectors[0] == cache_key
        if not cache_hit:
            rows = self.store.read_rows(
                "SELECT v.global_memory_id, v.dimensions, v.vector_json FROM memory_vectors v "
                "JOIN memory_scopes s ON v.global_memory_id=s.global_memory_id "
                f"WHERE {where} AND v.model_id=? ORDER BY v.global_memory_id",
                [*params, self.encoder.model_id],
            )
            if rows and {int(row[1]) for row in rows} != {len(query)}:
                raise ValueError(
                    "stored/query embedding dimensions differ; rebuild with the same encoder"
                )
            ids = [str(row[0]) for row in rows]
            eligible_count = int(
                self.store.read_rows(
                    f"SELECT count(*) FROM memory_scopes s WHERE {where}",
                    params,
                )[0][0]
            )
            if len(ids) != eligible_count:
                raise RuntimeError(
                    "some memories lack vectors for this encoder; call reindex_embeddings first"
                )
            matrix = np.asarray(
                [checked_vector(json.loads(row[2])) for row in rows], dtype=np.float32
            )
            ann = None
            if ids:
                matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
                if self._hnsw is not None:
                    ann = self._hnsw.Index(space="cosine", dim=len(query))
                    ann.init_index(
                        max_elements=len(ids), ef_construction=100, M=16, random_seed=17
                    )
                    ann.add_items(matrix, np.arange(len(ids)), num_threads=1)
                    ann.set_ef(max(64, request.candidate_limit))
            self._vectors = (cache_key, ids, matrix, ann)
        _, ids, matrix, ann = self._vectors
        diagnostics = {
            "vector_cache_hit": cache_hit,
            "query_embedding_cache_hit": query_cached,
            "vector_count": len(ids),
            "vector_backend": self.vector_backend,
        }
        if not ids:
            return [], diagnostics
        query /= np.linalg.norm(query)
        count = min(request.candidate_limit, len(ids))
        if ann is None:
            similarities = matrix @ query
            positions = np.argsort(-similarities, kind="stable")[:count]
        else:
            ann.set_ef(max(64, request.candidate_limit))
            labels, _ = ann.knn_query(query, k=count, num_threads=1)
            positions = labels[0]
            subset = matrix[positions] @ query
            return [
                ids[int(i)] for i, score in zip(positions, subset) if float(score) > 0.2
            ], diagnostics
        return [
            ids[int(i)] for i in positions if float(similarities[int(i)]) > 0.2
        ], diagnostics

    def _expand(
        self,
        seeds: dict[str, float],
        request: RetrievalRequest,
        where: str,
        params: list[Any],
    ) -> tuple[dict[tuple[str, str], float], list[dict[str, Any]], set[str]]:
        edges: dict[tuple[str, str], float] = {}
        paths: list[dict[str, Any]] = []
        visited = set(seeds)
        frontier = sorted(seeds)
        seen_nodes: set[str] = set()

        def connect(
            left: str, right: str, relation: str, via: str, weight: float
        ) -> None:
            if left == right:
                return
            if right not in visited and len(visited) >= request.candidate_limit:
                return
            visited.add(right)
            pair = tuple(sorted((left, right)))
            edges[pair] = max(edges.get(pair, 0.0), weight)
            item = {"from": left, "to": right, "relation": relation, "via": via}
            if item not in paths:
                paths.append(item)

        for _ in range(request.graph_hops):
            if not frontier:
                break
            before = set(visited)
            placeholders = ",".join("?" for _ in frontier)
            rows = self.store.read_rows(
                "SELECT i.global_memory_id, i.index_key, i.index_type FROM memory_index i "
                "JOIN memory_scopes s ON s.global_memory_id=i.index_key "
                f"WHERE {where} AND i.global_memory_id IN ({placeholders}) "
                "AND i.index_type IN ('source_reference','knowledge_reference') "
                "ORDER BY i.global_memory_id,i.index_key LIMIT ?",
                [*params, *frontier, request.candidate_limit * 4],
            )
            for row in rows:
                connect(str(row[0]), str(row[1]), str(row[2]), str(row[1]), 3.0)
            rows = self.store.read_rows(
                "SELECT i.index_key, i.global_memory_id, i.index_type FROM memory_index i "
                "JOIN memory_scopes s ON s.global_memory_id=i.global_memory_id "
                f"WHERE {where} AND i.index_key IN ({placeholders}) "
                "AND i.index_type IN ('source_reference','knowledge_reference') "
                "ORDER BY i.index_key,i.global_memory_id LIMIT ?",
                [*params, *frontier, request.candidate_limit * 4],
            )
            for row in rows:
                connect(str(row[0]), str(row[1]), str(row[2]), str(row[0]), 3.0)
            memberships = self.store.read_rows(
                "SELECT global_memory_id,index_key FROM memory_index WHERE "
                f"global_memory_id IN ({placeholders}) AND index_type='hypergraph_node' "
                "AND index_key LIKE 'entity:%' ORDER BY index_key,global_memory_id",
                frontier,
            )
            owners: dict[str, list[str]] = defaultdict(list)
            for row in memberships:
                owners[str(row[1])].append(str(row[0]))
            for node in sorted(owners)[: request.candidate_limit * 4]:
                if node in seen_nodes:
                    continue
                seen_nodes.add(node)
                neighbors = self.store.read_rows(
                    "SELECT i.global_memory_id FROM memory_index i JOIN memory_scopes s "
                    "ON i.global_memory_id=s.global_memory_id "
                    f"WHERE {where} AND i.index_type='hypergraph_node' AND i.index_key=? "
                    "ORDER BY i.global_memory_id LIMIT ?",
                    [*params, node, request.max_node_degree + 1],
                )
                if len(neighbors) > request.max_node_degree:
                    continue
                for left in owners[node]:
                    for neighbor in neighbors:
                        connect(
                            left,
                            str(neighbor[0]),
                            "shared_entity",
                            node,
                            1.0 / max(1, len(neighbors) - 1),
                        )
            frontier = sorted(visited - before)
        return edges, paths, visited

    def search(self, request: RetrievalRequest) -> dict[str, Any]:
        started = time.perf_counter()
        where, params = self.store.eligible_sql(
            self.namespace, request, all_layers=True
        )
        rankings: dict[str, list[str]] = {}
        diagnostics: dict[str, Any] = {}
        tokens = list(dict.fromkeys(indexed_tokens(request.query)))[:32]
        if tokens and request.mode != "semantic":
            expression = " OR ".join('"' + token + '"' for token in tokens)
            rows = self.store.read_rows(
                "SELECT memory_search.global_memory_id, bm25(memory_search) AS rank "
                "FROM memory_search JOIN memory_scopes s "
                "ON memory_search.global_memory_id=s.global_memory_id "
                f"WHERE memory_search MATCH ? AND {where} ORDER BY rank,1 LIMIT ?",
                [expression, *params, request.candidate_limit],
            )
            rankings["keyword"] = [str(row[0]) for row in rows]
        if request.mode == "tag":
            rows = (
                self.store.read_rows(
                    "SELECT i.global_memory_id, SUM(i.weight) score FROM memory_index i "
                    "JOIN memory_scopes s ON i.global_memory_id=s.global_memory_id "
                    f"WHERE {where} AND i.index_type='tag' AND i.index_key IN ("
                    + ",".join("?" for _ in tokens)
                    + ") GROUP BY i.global_memory_id ORDER BY score DESC,1 LIMIT ?",
                    [*params, *tokens, request.candidate_limit],
                )
                if tokens
                else []
            )
            rankings = {"tag": [str(row[0]) for row in rows]}
        if request.mode in {"semantic", "fusion"}:
            rankings["semantic"], diagnostics = self._semantic(request, where, params)
        entities = extract_canonical_entities(request.query, explicit=request.entities)
        entity_nodes = [
            node_id("entity", canonical_entity(value)) for value in entities
        ]
        if entity_nodes and request.mode in {"hypergraph", "fusion"}:
            rows = self.store.read_rows(
                "SELECT i.global_memory_id, COUNT(*) score FROM memory_index i "
                "JOIN memory_scopes s ON i.global_memory_id=s.global_memory_id "
                f"WHERE {where} AND i.index_type='hypergraph_node' AND i.index_key IN ("
                + ",".join("?" for _ in entity_nodes)
                + ") GROUP BY i.global_memory_id ORDER BY score DESC,1 LIMIT ?",
                [*params, *entity_nodes, request.candidate_limit],
            )
            rankings["entity"] = [str(row[0]) for row in rows]
        if request.seed_memory_ids:
            ids = list(request.seed_memory_ids)
            allowed = self.store.read_rows(
                f"SELECT s.global_memory_id FROM memory_scopes s WHERE {where} AND s.global_memory_id IN ("
                + ",".join("?" for _ in ids)
                + ")",
                [*params, *ids],
            )
            rankings["explicit"] = sorted(str(row[0]) for row in allowed)
        scores: dict[str, float] = defaultdict(float)
        channels: dict[str, dict[str, int]] = defaultdict(dict)
        for channel, ids in rankings.items():
            for rank, memory_id in enumerate(ids, 1):
                scores[memory_id] += 1.0 / (60 + rank)
                channels[memory_id][channel] = rank
        ordered = sorted(scores, key=lambda key: (-scores[key], key))
        seed_limit = min(request.candidate_limit, max(8, request.top_k * 2))
        seeds = {key: scores[key] for key in ordered[:seed_limit]}
        recall_elapsed = time.perf_counter() - started
        paths: list[dict[str, Any]] = []
        candidates = set(ordered[: request.candidate_limit])
        graph_scores: dict[str, float] = {}
        if request.mode in {"hypergraph", "fusion"}:
            edges, paths, candidates = self._expand(seeds, request, where, params)
            graph_scores = personalized_rank(seeds, edges)
            for rank, key in enumerate(
                sorted(graph_scores, key=lambda key: (-graph_scores[key], key)), 1
            ):
                scores[key] += 1.0 / (60 + rank)
                channels[key]["graph"] = rank
        graph_elapsed = time.perf_counter() - started - recall_elapsed
        records = self.store.fetch_ids(sorted(candidates))
        feedback_scores = {}
        for key, record in records.items():
            success = int(record.get("verified_success_count", 0))
            failure = int(record.get("verified_failure_count", 0))
            utility = (success + 1.0) / (success + failure + 2.0)
            feedback_scores[key] = utility
            scores[key] *= 1.0 + 0.1 * (utility - 0.5)
        ordered = sorted(
            (
                key
                for key, record in records.items()
                if record["memory_layer"] in request.layers
            ),
            key=lambda key: (-scores[key], key),
        )
        reranked = False
        if self.reranker is not None and len(ordered) > 1:
            first, second = (scores[key] for key in ordered[:2])
            if first - second <= first * 0.1:
                subset = ordered[:12]
                values = self.reranker(
                    request.query,
                    [compact_reuse_projection(records[key]) for key in subset],
                )
                if len(values) != len(subset) or not all(
                    math.isfinite(value) for value in values
                ):
                    raise ValueError(
                        "reranker must return one finite score per candidate"
                    )
                ordered[: len(subset)] = [
                    key
                    for _, key in sorted(
                        zip(values, subset), key=lambda item: (-item[0], item[1])
                    )
                ]
                reranked = True
        hits: list[dict[str, Any]] = []
        omitted: list[str] = []
        per_layer: dict[str, int] = defaultdict(int)
        for key in ordered:
            record = records[key]
            layer = record["memory_layer"]
            if per_layer[layer] >= request.layer_limits.get(layer, request.top_k):
                continue
            memory = compact_reuse_projection(
                record, max_steps=request.max_evidence_steps
            )
            memory.update(
                {
                    field: record.get(field)
                    for field in (
                        "source_agent",
                        "created_at",
                        "task_id",
                        "review_status",
                    )
                }
            )
            hit = {
                "memory_id": key,
                "memory_layer": layer,
                "score": round(scores[key], 8),
                "verified_utility": round(feedback_scores[key], 6),
                "channel_ranks": channels[key],
                "graph_score": round(graph_scores.get(key, 0.0), 8),
                "relation_paths": [
                    path for path in paths if key in (path["from"], path["to"])
                ][:8],
                "memory": memory,
            }
            if len(json_text([*hits, hit]).encode("utf-8")) > request.max_payload_bytes:
                omitted.append(key)
                continue
            hits.append(hit)
            per_layer[layer] += 1
            if len(hits) >= request.top_k:
                break
        diagnostics.update(
            {
                "candidate_count": len(candidates),
                "records_loaded": len(records),
                "seed_count": len(seeds),
                "relation_count": len(paths),
                "reranked": reranked,
                "recall_ms": round(recall_elapsed * 1000, 3),
                "graph_ms": round(graph_elapsed * 1000, 3),
                "total_ms": round((time.perf_counter() - started) * 1000, 3),
                "payload_bytes": len(json_text(hits).encode("utf-8")),
                "budget_omitted_ids": omitted,
            }
        )
        return {"hits": hits, "diagnostics": diagnostics, "batch_id": request.batch_id}
