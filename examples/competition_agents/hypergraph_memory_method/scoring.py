from __future__ import annotations

from typing import Any

from .schema import (
    NODE_ACTION,
    NODE_AGENT,
    NODE_ARCHIVE_SCOPE,
    NODE_ARTIFACT,
    NODE_CAPABILITY,
    NODE_DATASET,
    NODE_ENTITY,
    NODE_EVIDENCE,
    NODE_EXECUTION,
    NODE_EXPECTED_REUSE,
    NODE_GAP,
    NODE_EXPERIENCE,
    NODE_GROUP,
    NODE_KEYWORD,
    NODE_MEMORY_REF,
    NODE_MESSAGE,
    NODE_METRIC,
    NODE_OUTCOME,
    NODE_LIFECYCLE,
    NODE_REUSE_BENEFIT,
    NODE_STATE,
    NODE_STATE_REF,
    NODE_STRATEGY,
    NODE_SUPPORTING_FACT,
    NODE_TAG,
    NODE_TOOL,
    NODE_TURN,
)


NODE_WEIGHTS = {
    NODE_EXPECTED_REUSE: 8.0,
    NODE_MEMORY_REF: 8.0,
    NODE_EXPERIENCE: 7.0,
    NODE_REUSE_BENEFIT: 6.5,
    NODE_STRATEGY: 6.0,
    NODE_ARTIFACT: 6.0,
    NODE_GAP: 5.5,
    NODE_STATE: 5.0,
    NODE_STATE_REF: 4.5,
    NODE_SUPPORTING_FACT: 4.5,
    NODE_EVIDENCE: 4.0,
    NODE_ENTITY: 3.5,
    NODE_TAG: 3.0,
    NODE_AGENT: 2.0,
    NODE_ACTION: 2.0,
    NODE_CAPABILITY: 2.0,
    NODE_OUTCOME: 2.0,
    NODE_EXECUTION: 2.0,
    NODE_TOOL: 2.0,
    NODE_METRIC: 1.5,
    NODE_GROUP: 1.5,
    NODE_TURN: 1.5,
    NODE_ARCHIVE_SCOPE: 1.5,
    NODE_LIFECYCLE: 1.5,
    NODE_DATASET: 1.0,
    NODE_MESSAGE: 0.5,
    NODE_KEYWORD: 1.0,
}

EMBEDDING_SCORE_WEIGHT = 8.0


def node_kind(node: str) -> str:
    return node.partition(":")[0]


def score_components(
    record: dict[str, Any],
    matched_nodes: set[str],
    *,
    embedding_score: float | None = None,
) -> dict[str, float]:
    """Score one memory hit with explicit sub-scores for explanation."""

    base_score = 0.0
    gap_score = 0.0
    reuse_score = 0.0
    evidence_score = 0.0
    state_ref_score = 0.0
    strategy_score = 0.0
    experience_score = 0.0

    for node in matched_nodes:
        kind = node_kind(node)
        weight = NODE_WEIGHTS.get(kind, 0.25)
        base_score += weight
        if kind == NODE_GAP:
            gap_score += weight
        elif kind in {NODE_EXPECTED_REUSE, NODE_MEMORY_REF}:
            reuse_score += weight
        elif kind in {NODE_ARTIFACT, NODE_SUPPORTING_FACT, NODE_EVIDENCE, NODE_ENTITY}:
            evidence_score += weight
        elif kind in {NODE_STATE, NODE_STATE_REF}:
            state_ref_score += weight
        elif kind == NODE_STRATEGY:
            strategy_score += weight
        elif kind in {NODE_EXPERIENCE, NODE_REUSE_BENEFIT}:
            experience_score += weight

    structural_score = gap_score + reuse_score + evidence_score + state_ref_score + strategy_score + experience_score
    embedding_component = max(0.0, float(embedding_score or 0.0)) * EMBEDDING_SCORE_WEIGHT
    summary_bonus = min(len(record.get("summary", "")) / 1000.0, 0.25)
    hypergraph_score = base_score + summary_bonus
    retrieval_score = hypergraph_score + min(structural_score, 20.0) * 0.25 + embedding_component

    return {
        "retrieval_score": round(retrieval_score, 4),
        "hypergraph_score": round(hypergraph_score, 4),
        "embedding_score": round(float(embedding_score or 0.0), 4),
        "gap_coverage_score": round(gap_score, 4),
        "reuse_score": round(reuse_score, 4),
        "evidence_score": round(evidence_score, 4),
        "state_ref_score": round(state_ref_score, 4),
        "strategy_score": round(strategy_score, 4),
        "experience_score": round(experience_score, 4),
    }


def score_record(record: dict[str, Any], matched_nodes: set[str]) -> float:
    """Backward-compatible scalar score helper."""

    return score_components(record, matched_nodes)["retrieval_score"]
