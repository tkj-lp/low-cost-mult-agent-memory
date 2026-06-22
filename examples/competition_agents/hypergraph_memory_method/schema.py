from __future__ import annotations

from typing import Final


NODE_MEMORY: Final = "memory"
NODE_MEMORY_TYPE: Final = "memory_type"
NODE_MEMORY_LAYER: Final = "memory_layer"
NODE_TASK: Final = "task"
NODE_GROUP: Final = "group"
NODE_TURN: Final = "turn"
NODE_DATASET: Final = "dataset"
NODE_AGENT: Final = "agent"
NODE_ACTION: Final = "action"
NODE_MESSAGE: Final = "message"
NODE_TAG: Final = "tag"
NODE_KEYWORD: Final = "keyword"
NODE_EXPECTED_REUSE: Final = "expected_reuse"
NODE_ARTIFACT: Final = "artifact"
NODE_STATE: Final = "state"
NODE_STATE_REF: Final = "state_ref"
NODE_MEMORY_REF: Final = "memory_ref"
NODE_GAP: Final = "gap"
NODE_SUPPORTING_FACT: Final = "supporting_fact"
NODE_EVIDENCE: Final = "evidence"
NODE_ENTITY: Final = "entity"
NODE_METRIC: Final = "metric"
NODE_OUTCOME: Final = "outcome"
NODE_EXECUTION: Final = "execution"
NODE_TOOL: Final = "tool"
NODE_CAPABILITY: Final = "capability"
NODE_INTENT: Final = "intent"
NODE_STRATEGY: Final = "strategy"
NODE_EXPERIENCE: Final = "experience"
NODE_EXPERIENCE_RULE: Final = "experience_rule"
NODE_EXPERIENCE_SIGNATURE: Final = "experience_signature"
NODE_PROBLEM_PATTERN: Final = "problem_pattern"
NODE_SOLUTION_PATTERN: Final = "solution_pattern"
NODE_FAILURE_PATTERN: Final = "failure_pattern"
NODE_TRIGGER: Final = "trigger"
NODE_REUSE_BENEFIT: Final = "reuse_benefit"
NODE_ARCHIVE_SCOPE: Final = "archive_scope"
NODE_LIFECYCLE: Final = "lifecycle"

EDGE_MEMORY_EVENT: Final = "memory_event"
EDGE_MEMORY_LAYER: Final = "memory_layer"
EDGE_TASK_CONTEXT: Final = "task_context"
EDGE_AGENT_MESSAGE: Final = "agent_message"
EDGE_TRAJECTORY: Final = "trajectory"
EDGE_STATE_FLOW: Final = "state_flow"
EDGE_MEMORY_REUSE: Final = "memory_reuse"
EDGE_GAP: Final = "cognitive_gap"
EDGE_GAP_COVERAGE: Final = "gap_coverage"
EDGE_EVIDENCE: Final = "evidence"
EDGE_SUPPORTING_FACT: Final = "supporting_fact"
EDGE_OUTCOME: Final = "outcome"
EDGE_OUTCOME_FEEDBACK: Final = "outcome_feedback"
EDGE_TOOL_EXECUTION: Final = "tool_execution"
EDGE_SANDBOX_OUTCOME: Final = "sandbox_outcome"
EDGE_STRATEGY: Final = "strategy"
EDGE_EXPERIENCE: Final = "experience"
EDGE_EXPERIENCE_RULE: Final = "experience_rule"
EDGE_EXPERIENCE_SIGNATURE: Final = "experience_signature"
EDGE_REUSE_BENEFIT: Final = "reuse_benefit"
EDGE_ARCHIVE_SCOPE: Final = "archive_scope"

STRUCTURED_NODE_KINDS: Final = {
    NODE_MEMORY_TYPE,
    NODE_MEMORY_LAYER,
    NODE_EXPECTED_REUSE,
    NODE_ARTIFACT,
    NODE_STATE,
    NODE_STATE_REF,
    NODE_MEMORY_REF,
    NODE_GAP,
    NODE_SUPPORTING_FACT,
    NODE_EVIDENCE,
    NODE_ENTITY,
    NODE_AGENT,
    NODE_ACTION,
    NODE_MESSAGE,
    NODE_METRIC,
    NODE_OUTCOME,
    NODE_EXECUTION,
    NODE_TOOL,
    NODE_CAPABILITY,
    NODE_GROUP,
    NODE_TURN,
    NODE_DATASET,
    NODE_INTENT,
    NODE_STRATEGY,
    NODE_EXPERIENCE,
    NODE_EXPERIENCE_RULE,
    NODE_EXPERIENCE_SIGNATURE,
    NODE_PROBLEM_PATTERN,
    NODE_SOLUTION_PATTERN,
    NODE_FAILURE_PATTERN,
    NODE_TRIGGER,
    NODE_REUSE_BENEFIT,
    NODE_ARCHIVE_SCOPE,
    NODE_LIFECYCLE,
}

PROFILE_FIELDS: Final = (
    "intent",
    "expected_reuse",
    "question",
    "gaps",
    "artifacts",
    "keywords",
)
