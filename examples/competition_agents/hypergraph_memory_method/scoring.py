from __future__ import annotations

from typing import Any

from .distill import lifecycle_stage, reuse_success_rate
from .schema import (
    NODE_ACTION,
    NODE_AGENT,
    NODE_ARCHIVE_SCOPE,
    NODE_ARTIFACT,
    NODE_CAPABILITY,
    NODE_CLAIM,
    NODE_DATASET,
    NODE_ENTITY,
    NODE_EVIDENCE,
    NODE_EVIDENCE_CHAIN,
    NODE_EVIDENCE_SOURCE,
    NODE_EXECUTION,
    NODE_EXPECTED_REUSE,
    NODE_EXPERIENCE,
    NODE_EXPERIENCE_RULE,
    NODE_EXPERIENCE_SIGNATURE,
    NODE_FAILURE_PATTERN,
    NODE_GAP,
    NODE_GROUP,
    NODE_KEYWORD,
    NODE_LIFECYCLE,
    NODE_MEMORY_LAYER,
    NODE_MEMORY_REF,
    NODE_MEMORY_TYPE,
    NODE_MESSAGE,
    NODE_METRIC,
    NODE_OUTCOME,
    NODE_PROBLEM_PATTERN,
    NODE_RELATION,
    NODE_REUSE_BENEFIT,
    NODE_SOLUTION_PATTERN,
    NODE_STATE,
    NODE_STATE_REF,
    NODE_STRATEGY,
    NODE_SUPPORTING_FACT,
    NODE_TAG,
    NODE_TOOL,
    NODE_TRIGGER,
    NODE_TURN,
    NODE_VERSION,
)

NODE_WEIGHTS = {
    NODE_EXPECTED_REUSE: 8.0,
    NODE_MEMORY_REF: 8.0,
    NODE_EXPERIENCE_RULE: 8.0,
    NODE_EXPERIENCE_SIGNATURE: 7.5,
    NODE_PROBLEM_PATTERN: 7.5,
    NODE_SOLUTION_PATTERN: 7.5,
    NODE_TRIGGER: 7.0,
    NODE_FAILURE_PATTERN: 6.5,
    NODE_EXPERIENCE: 7.0,
    NODE_REUSE_BENEFIT: 6.5,
    NODE_STRATEGY: 6.0,
    NODE_MEMORY_LAYER: 5.5,
    NODE_MEMORY_TYPE: 5.0,
    NODE_ARTIFACT: 6.0,
    NODE_GAP: 5.5,
    NODE_STATE: 5.0,
    NODE_STATE_REF: 4.5,
    NODE_SUPPORTING_FACT: 4.5,
    NODE_EVIDENCE: 4.0,
    NODE_EVIDENCE_CHAIN: 8.5,
    NODE_CLAIM: 7.0,
    NODE_EVIDENCE_SOURCE: 6.0,
    NODE_RELATION: 5.5,
    NODE_VERSION: 0.5,
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

LIFECYCLE_SCORES = {
    "generalized": 5.0,
    "validated": 3.0,
    "promoted-candidate": 1.0,
    "episodic": 0.0,
    "needs-validation": -1.5,
    "deprecated": -10.0,
}


def node_kind(node: str) -> str:
    return node.partition(":")[0]


def score_components(
    record: dict[str, Any],
    matched_nodes: set[str],
    *,
    embedding_score: float | None = None,
) -> dict[str, Any]:
    """Score one memory hit with explicit sub-scores for explanation."""

    base_score = 0.0
    gap_score = 0.0
    reuse_score = 0.0
    evidence_score = 0.0
    state_ref_score = 0.0
    strategy_score = 0.0
    experience_score = 0.0
    evidence_chain_score = 0.0
    tag_score = 0.0
    keyword_score = 0.0

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
        if kind in {
            NODE_EVIDENCE_CHAIN,
            NODE_CLAIM,
            NODE_EVIDENCE_SOURCE,
            NODE_RELATION,
        }:
            evidence_chain_score += weight
        elif kind in {NODE_STATE, NODE_STATE_REF}:
            state_ref_score += weight
        elif kind == NODE_STRATEGY:
            strategy_score += weight
        elif kind in {
            NODE_EXPERIENCE,
            NODE_EXPERIENCE_RULE,
            NODE_EXPERIENCE_SIGNATURE,
            NODE_PROBLEM_PATTERN,
            NODE_SOLUTION_PATTERN,
            NODE_FAILURE_PATTERN,
            NODE_TRIGGER,
            NODE_REUSE_BENEFIT,
            NODE_MEMORY_LAYER,
            NODE_MEMORY_TYPE,
        }:
            experience_score += weight
        elif kind == NODE_TAG:
            tag_score += weight
        elif kind == NODE_KEYWORD:
            keyword_score += weight

    structural_score = (
        gap_score
        + reuse_score
        + evidence_score
        + evidence_chain_score
        + state_ref_score
        + strategy_score
        + experience_score
    )
    embedding_component = (
        max(0.0, float(embedding_score or 0.0)) * EMBEDDING_SCORE_WEIGHT
    )
    summary_bonus = min(len(record.get("summary", "")) / 1000.0, 0.25)
    hypergraph_score = base_score + summary_bonus
    structure_priority_bonus = min(structural_score, 36.0) * 0.6
    keyword_penalty = (
        keyword_score * 0.35 if structural_score == 0 else keyword_score * 0.15
    )
    reuse_count = float(record.get("reuse_count", 0) or 0)
    reuse_success_count = float(record.get("reuse_success_count", 0) or 0)
    success_rate = reuse_success_rate(record)
    current_lifecycle = lifecycle_stage(record)
    lifecycle_score = LIFECYCLE_SCORES.get(current_lifecycle, 0.0)
    reuse_success_bonus = min(reuse_success_count, 5.0) * 0.8
    reuse_count_bonus = min(reuse_count, 8.0) * 0.2
    success_rate_bonus = success_rate * 3.0 if reuse_count > 0 else 0.0
    retention_value = record.get("retention_score", 0.5)
    retention_score = max(
        0.0, min(1.0, float(0.5 if retention_value is None else retention_value))
    )
    retention_component = (retention_score - 0.5) * 4.0
    chain_completeness = max(
        0.0,
        min(1.0, float(record.get("evidence_chain_completeness", 0.0) or 0.0)),
    )
    chain_quality_component = chain_completeness * 4.0
    if record.get("evidence_chain_valid") is False and record.get("evidence_chain_id"):
        chain_quality_component -= 1.0
    retrieval_score = (
        hypergraph_score
        + structure_priority_bonus
        + embedding_component
        + reuse_success_bonus
        + reuse_count_bonus
        + success_rate_bonus
        + lifecycle_score
        + retention_component
        + chain_quality_component
        - keyword_penalty
    )

    return {
        "retrieval_score": round(retrieval_score, 4),
        "hypergraph_score": round(hypergraph_score, 4),
        "embedding_score": round(float(embedding_score or 0.0), 4),
        "structural_score": round(structural_score, 4),
        "tag_score": round(tag_score, 4),
        "keyword_score": round(keyword_score, 4),
        "lifecycle_score": round(lifecycle_score, 4),
        "retention_score": round(retention_score, 4),
        "retention_component": round(retention_component, 4),
        "evidence_chain_completeness": round(chain_completeness, 4),
        "evidence_chain_quality_score": round(chain_quality_component, 4),
        "experience_lifecycle_stage": current_lifecycle,
        "experience_success_rate": round(success_rate, 4),
        "experience_feedback": {
            "lifecycle_stage": current_lifecycle,
            "reuse_count": int(reuse_count),
            "reuse_success_count": int(reuse_success_count),
            "success_rate": round(success_rate, 4),
            "lifecycle_score": round(lifecycle_score, 4),
        },
        "gap_coverage_score": round(gap_score, 4),
        "reuse_score": round(reuse_score, 4),
        "evidence_score": round(evidence_score, 4),
        "state_ref_score": round(state_ref_score, 4),
        "strategy_score": round(strategy_score, 4),
        "experience_score": round(experience_score, 4),
        "evidence_chain_score": round(evidence_chain_score, 4),
    }


def score_record(record: dict[str, Any], matched_nodes: set[str]) -> float:
    """Backward-compatible scalar score helper."""

    return score_components(record, matched_nodes)["retrieval_score"]
