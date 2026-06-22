from __future__ import annotations

import re
from typing import Any

from .schema import (
    NODE_ARTIFACT,
    NODE_ARCHIVE_SCOPE,
    NODE_EXPERIENCE,
    NODE_EXPERIENCE_RULE,
    NODE_EXPECTED_REUSE,
    NODE_FAILURE_PATTERN,
    NODE_GAP,
    NODE_INTENT,
    NODE_KEYWORD,
    NODE_MEMORY_LAYER,
    NODE_MEMORY_REF,
    NODE_PROBLEM_PATTERN,
    NODE_REUSE_BENEFIT,
    NODE_SOLUTION_PATTERN,
    NODE_STRATEGY,
    NODE_STATE,
    NODE_STATE_REF,
    NODE_TAG,
    NODE_TRIGGER,
)
from .text import limited_tokens, node_id, normalize_text
from .gating import infer_trigger_families


ARTIFACT_HINTS = {
    "answer": {"answer", "result", "final"},
    "supporting-facts": {"supporting", "facts", "fact"},
    "evidence": {"evidence", "chain"},
    "plan": {"plan", "planning"},
    "state": {"state", "latent", "ref", "reference"},
    "memory": {"memory", "reuse", "previous", "prior", "history"},
    "metrics": {"metrics", "cost", "performance", "elapsed"},
    "execution": {"execution", "codeact", "sandbox", "tool"},
    "strategy": {"strategy", "plan", "handoff", "coordination"},
    "experience": {"experience", "episodic", "insight", "distill", "rule"},
    "trajectory": {"trajectory", "trace", "message", "handoff", "agent"},
    "reuse-benefit": {"benefit", "reuse", "saving", "cost"},
}


def _artifact_from_expected_reuse(value: str) -> str:
    _, _, artifact = value.partition(".")
    return normalize_text(artifact or value)


def build_query_profile(query: str, task_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a lightweight query profile used by graph retrieval."""

    normalized_query = normalize_text(query)
    tokens = set(limited_tokens(query, limit=32))
    task_meta = task_meta or {}

    expected_reuse = {
        normalize_text(match)
        for match in re.findall(r"turn-\d+\.[a-zA-Z_][a-zA-Z0-9_]*", normalized_query)
    }
    if isinstance(task_meta, dict):
        for value in task_meta.get("expected_reuse", []):
            if value:
                expected_reuse.add(normalize_text(str(value)))

    artifacts: set[str] = set()
    for artifact, hints in ARTIFACT_HINTS.items():
        if tokens & hints or any(hint in normalized_query for hint in hints):
            artifacts.add(artifact)
    artifacts.update(_artifact_from_expected_reuse(value) for value in expected_reuse)
    if "previous" in normalized_query or "prior" in normalized_query or "reuse" in normalized_query:
        artifacts.add("memory")
    if "evidence chain" in normalized_query:
        artifacts.update({"evidence", "supporting-facts"})

    gaps: set[str] = set()
    if {"memory", "reuse", "previous", "prior", "history"} & tokens:
        gaps.update({"missing-memory", "partial-context", "memory-grounded-reasoning"})
    if {"reason", "answer", "question"} & tokens:
        gaps.add("needs-reasoning")
    if {"evidence", "facts", "fact"} & tokens:
        gaps.add("external-supporting-facts")
    if {"codeact", "sandbox", "tool", "execution"} & tokens:
        gaps.add("tool-execution")
    if {"experience", "distill", "insight"} & tokens:
        gaps.add("experience-distillation")
    if {"trace", "trajectory", "handoff", "message"} & tokens:
        gaps.add("agent-trajectory")

    intent = "reuse" if {"reuse", "previous", "prior", "history"} & tokens else "lookup"
    if {"summarize", "summary"} & tokens:
        intent = "summarize"
    elif {"compare", "cost", "performance"} & tokens:
        intent = "compare"
    elif {"design", "plan"} & tokens:
        intent = "design"

    state_refs = set(re.findall(r"state-\d+", normalized_query))
    memory_refs = set(re.findall(r"mem-\d+", normalized_query))
    strategies = set()
    if {"handoff", "coordination"} & tokens:
        strategies.add("gap-aware-coordination")
    if {"state", "latent", "ref"} & tokens:
        strategies.add("state-ref-passing")
    if {"memory", "reuse", "previous", "prior"} & tokens:
        strategies.add("memory-ref-passing")
        strategies.add("expected-reuse-planning")

    experiences = set()
    if {"evidence", "facts", "chain"} & tokens:
        experiences.add("supporting-fact-chain")
        experiences.add("evidence-state-captured")
    if {"answer", "result"} & tokens:
        experiences.add("answer-state-captured")
    if {"memory", "reuse", "previous", "prior"} & tokens:
        experiences.add("memory-grounded-reasoning")

    memory_layers = set()
    if {"evidence", "facts", "fact", "answer"} & tokens:
        memory_layers.add("evidence")
    if {"experience", "rule", "strategy", "failure", "trigger"} & tokens:
        memory_layers.add("experience")
    if {"trace", "trajectory", "message", "handoff", "agent"} & tokens:
        memory_layers.add("trajectory")
    if {"outcome", "metric", "metrics", "benefit", "saving", "cost"} & tokens:
        memory_layers.add("outcome")

    rule_nodes = {
        "problem_patterns": set(),
        "solution_patterns": set(),
        "failure_patterns": set(),
        "triggers": set(),
    }
    trigger_families = infer_trigger_families(query)
    if "temporal" in trigger_families:
        rule_nodes["problem_patterns"].add("multiple-temporal-candidates-with-constraints")
        rule_nodes["solution_patterns"].add("choose-date-matching-all-constraints")
        rule_nodes["triggers"].add("temporal-query")
    if "attribute" in trigger_families:
        rule_nodes["problem_patterns"].add("attribute-selection-among-related-facts")
        rule_nodes["solution_patterns"].add("pick-fact-matching-requested-attribute")
        rule_nodes["triggers"].add("attribute-lookup-query")
    if "location" in trigger_families:
        rule_nodes["triggers"].add("multi-location-query")
    if "entity-list" in trigger_families:
        rule_nodes["triggers"].add("entity-list-query")
    if "reference" in trigger_families:
        rule_nodes["triggers"].add("reference-resolution-query")
    if {"failure", "avoid", "wrong", "mistake"} & tokens:
        rule_nodes["failure_patterns"].add("avoid-historical-error")
    for token in tokens:
        if token.endswith("_query") or token.endswith("-query") or token in {"temporal", "attribute", "reference"}:
            rule_nodes["triggers"].add(token.replace("_", "-"))

    reuse_benefits = set()
    if {"cost", "communication", "chars"} & tokens:
        reuse_benefits.add("communication-cost-accounted")
    if {"latent", "state"} & tokens:
        reuse_benefits.add("latent-transfer-accounted")

    return {
        "intent": intent,
        "expected_reuse": sorted(expected_reuse),
        "question": task_meta.get("question", query) if isinstance(task_meta, dict) else query,
        "gaps": sorted(gaps),
        "artifacts": sorted(artifacts),
        "keywords": sorted(tokens),
        "state_refs": sorted(state_refs),
        "memory_refs": sorted(memory_refs),
        "strategies": sorted(strategies),
        "experiences": sorted(experiences),
        "reuse_benefits": sorted(reuse_benefits),
        "memory_layers": sorted(memory_layers),
        "trigger_families": sorted(trigger_families),
        **{key: sorted(value) for key, value in rule_nodes.items()},
    }


def profile_nodes(profile: dict[str, Any]) -> set[str]:
    """Convert a query profile into candidate graph nodes."""

    nodes: set[str] = set()
    intent = profile.get("intent")
    if intent:
        nodes.add(node_id(NODE_INTENT, str(intent)))
    for value in profile.get("expected_reuse", []):
        nodes.add(node_id(NODE_EXPECTED_REUSE, str(value)))
    for value in profile.get("gaps", []):
        nodes.add(node_id(NODE_GAP, str(value)))
    for value in profile.get("artifacts", []):
        nodes.add(node_id(NODE_ARTIFACT, str(value)))
        if value in {"plan", "evidence", "answer"}:
            nodes.add(node_id(NODE_STATE, str(value)))
    for value in profile.get("state_refs", []):
        nodes.add(node_id(NODE_STATE_REF, str(value)))
    for value in profile.get("memory_refs", []):
        nodes.add(node_id(NODE_MEMORY_REF, str(value)))
    for value in profile.get("strategies", []):
        nodes.add(node_id(NODE_STRATEGY, str(value)))
    for value in profile.get("experiences", []):
        nodes.add(node_id(NODE_EXPERIENCE, str(value)))
    for value in profile.get("memory_layers", []):
        nodes.add(node_id(NODE_MEMORY_LAYER, str(value)))
    for value in profile.get("problem_patterns", []):
        nodes.add(node_id(NODE_PROBLEM_PATTERN, str(value)))
    for value in profile.get("solution_patterns", []):
        nodes.add(node_id(NODE_SOLUTION_PATTERN, str(value)))
    for value in profile.get("failure_patterns", []):
        nodes.add(node_id(NODE_FAILURE_PATTERN, str(value)))
    for value in profile.get("triggers", []):
        nodes.add(node_id(NODE_TRIGGER, str(value)))
    if profile.get("memory_layers") or profile.get("experiences"):
        nodes.add(node_id(NODE_EXPERIENCE_RULE, "rule-card"))
    for value in profile.get("reuse_benefits", []):
        nodes.add(node_id(NODE_REUSE_BENEFIT, str(value)))
    for value in profile.get("archive_scopes", []):
        nodes.add(node_id(NODE_ARCHIVE_SCOPE, str(value)))
    for token in profile.get("keywords", []):
        nodes.add(node_id(NODE_KEYWORD, str(token)))
        nodes.add(node_id(NODE_TAG, str(token)))
    return nodes
