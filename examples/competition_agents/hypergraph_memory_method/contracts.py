from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, Protocol, Required, TypedDict

from .layers import MEMORY_LAYERS


class MemoryEncoder(Protocol):
    """The caller owns model choice; vectors from different spaces never mix."""

    model_id: str

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class BaseMemoryInput(TypedDict, total=False):
    """Common caller-owned fields accepted by every memory layer."""

    source_agent: Required[str]
    task_topic: Required[str]
    summary: Required[str]
    memory_id: str
    created_at: str
    tags: list[str]
    evidence: list[str]
    entities: list[str]
    entity_aliases: dict[str, str]
    state_refs: list[str]
    knowledge_refs: list[str]
    source_memory_ids: list[str]
    task_type: str
    metadata: dict[str, Any]


class KnowledgeMemoryInput(BaseMemoryInput, total=False):
    """Fact, document fragment, or tool material scoped to one batch."""

    artifact_types: list[str]


class EpisodeMemoryInput(BaseMemoryInput, total=False):
    """Task output and evidence retained inside the current batch."""

    answer: str
    supporting_facts: list[Any]


class ExperienceMemoryInput(BaseMemoryInput, total=False):
    """Candidate reusable procedure; source memories are mandatory."""

    problem_pattern: Required[str]
    solution_pattern: Required[str]
    source_memory_ids: Required[list[str]]
    trigger_conditions: list[str]
    failure_pattern: str


class WriteReceiptRecord(TypedDict):
    memory_id: str
    local_memory_id: str
    memory_layer: str


class WriteReceipt(TypedDict):
    batch_id: str
    operation_id: str
    records: list[WriteReceiptRecord]


class RetrievalHit(TypedDict):
    memory_id: str
    memory_layer: str
    score: float
    verified_utility: float
    channel_ranks: dict[str, int]
    graph_score: float
    relation_paths: list[dict[str, Any]]
    memory: dict[str, Any]


class RetrievalResult(TypedDict):
    retrieval_id: str
    batch_id: str
    hits: list[RetrievalHit]
    diagnostics: dict[str, Any]


class MaintenanceReport(TypedDict, total=False):
    batch_id: Required[str]
    cleared_ids: Required[list[str]]
    promoted_ids: Required[list[str]]
    pending_ids: Required[list[str]]
    rejected_ids: Required[list[str]]
    superseded_ids: Required[list[str]]
    capacity_pruned_ids: Required[list[str]]
    merged: Required[list[dict[str, Any]]]
    merge_suggestions: Required[list[dict[str, Any]]]
    active_experience_count: int


@dataclass(frozen=True)
class BatchPolicy:
    clear_episodes: bool = True
    experience_scope: Literal["task_family", "dataset"] = "task_family"
    min_verified_successes: int = 1
    reject_after_failures: int = 2
    max_active_experiences: int = 1000
    merge_similarity: float = 0.97

    def __post_init__(self) -> None:
        if not isinstance(self.clear_episodes, bool):
            raise ValueError("clear_episodes must be a boolean")
        if self.experience_scope not in {"task_family", "dataset"}:
            raise ValueError("experience_scope must be task_family or dataset")
        for value in (
            self.min_verified_successes,
            self.reject_after_failures,
            self.max_active_experiences,
        ):
            if type(value) is not int or value < 1:
                raise ValueError("policy counts must be positive integers")
        if not 0.0 < self.merge_similarity <= 1.0:
            raise ValueError("merge_similarity must be in (0, 1]")


@dataclass(frozen=True)
class RetrievalRequest:
    batch_id: str
    query: str
    layers: tuple[str, ...] = ("episodic", "experience")
    bridge_layers: tuple[str, ...] = ()
    mode: Literal["keyword", "tag", "semantic", "hypergraph", "fusion"] = "hypergraph"
    top_k: int = 5
    seed_memory_ids: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    exclude_ids: tuple[str, ...] = ()
    layer_limits: dict[str, int] = field(default_factory=dict)
    candidate_limit: int = 64
    graph_hops: int = 2
    max_node_degree: int = 24
    max_evidence_steps: int = 6
    max_payload_bytes: int = 32768
    include_candidates: bool = False

    def __post_init__(self) -> None:
        if not self.batch_id.strip():
            raise ValueError("batch_id is required")
        if not self.query.strip() and not self.seed_memory_ids and not self.entities:
            raise ValueError("a query or explicit seeds are required")
        if not self.layers or set(self.layers) - set(MEMORY_LAYERS):
            raise ValueError("layers must explicitly select valid memory layers")
        if set(self.bridge_layers) - set(MEMORY_LAYERS):
            raise ValueError("bridge_layers must select valid memory layers")
        if self.mode not in {"keyword", "tag", "semantic", "hypergraph", "fusion"}:
            raise ValueError("unknown retrieval mode")
        if not 1 <= self.top_k <= self.candidate_limit <= 512:
            raise ValueError("require 1 <= top_k <= candidate_limit <= 512")
        if not 0 <= self.graph_hops <= 3 or not 2 <= self.max_node_degree <= 128:
            raise ValueError("graph limits are out of range")
        if self.max_evidence_steps < 1 or self.max_payload_bytes < 256:
            raise ValueError("projection budgets are too small")
        if set(self.layer_limits) - set(self.layers) or any(
            type(value) is not int or value < 0 for value in self.layer_limits.values()
        ):
            raise ValueError(
                "layer_limits must contain non-negative selected-layer quotas"
            )


@dataclass(frozen=True)
class TaskFeedback:
    batch_id: str
    task_id: str
    retrieved_ids: tuple[str, ...] = ()
    adopted_ids: tuple[str, ...] = ()
    answer_correct: bool | None = None
    memory_verdicts: dict[str, bool | None] = field(default_factory=dict)
    validation_source: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    retrieval_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.batch_id.strip() or not self.task_id.strip():
            raise ValueError("batch_id and task_id are required")
        values = [self.answer_correct, *self.memory_verdicts.values()]
        if any(value is not None and type(value) is not bool for value in values):
            raise ValueError("verdicts must be true, false or null")
        if (
            any(value is not None for value in values)
            and not self.validation_source.strip()
        ):
            raise ValueError("a validation source is required for correctness verdicts")


@dataclass(frozen=True)
class TaskMemoryCommit:
    """One retry-safe task memory transaction owned by the caller."""

    batch_id: str
    task_id: str
    operation_id: str
    episode: EpisodeMemoryInput | None = None
    experiences: tuple[ExperienceMemoryInput, ...] = ()
    feedback: TaskFeedback | None = None

    def __post_init__(self) -> None:
        if not self.batch_id.strip() or not self.task_id.strip():
            raise ValueError("batch_id and task_id are required")
        if not self.operation_id.strip():
            raise ValueError("operation_id is required")
        if self.feedback is not None and (
            self.feedback.batch_id != self.batch_id
            or self.feedback.task_id != self.task_id
        ):
            raise ValueError("feedback must belong to the committed task")
        if self.episode is None and not self.experiences and self.feedback is None:
            raise ValueError("task commit must contain memory or feedback")


@dataclass(frozen=True)
class ExperienceReview:
    """External review independent from task answer generation."""

    batch_id: str
    review_id: str
    memory_verdicts: dict[str, bool]
    validation_source: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.batch_id.strip() or not self.review_id.strip():
            raise ValueError("batch_id and review_id are required")
        if not self.memory_verdicts:
            raise ValueError("memory_verdicts must not be empty")
        if any(type(value) is not bool for value in self.memory_verdicts.values()):
            raise ValueError("experience review verdicts must be boolean")
        if not self.validation_source.strip():
            raise ValueError("validation_source is required")


@dataclass(frozen=True)
class MemoryRelationUpdate:
    """Versioned relationship changes for an existing memory unit."""

    batch_id: str
    memory_id: str
    operation_id: str
    add_source_memory_ids: tuple[str, ...] = ()
    remove_source_memory_ids: tuple[str, ...] = ()
    add_knowledge_refs: tuple[str, ...] = ()
    remove_knowledge_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.batch_id.strip() or not self.memory_id.strip():
            raise ValueError("batch_id and memory_id are required")
        if not self.operation_id.strip():
            raise ValueError("operation_id is required")
        changes = (
            self.add_source_memory_ids,
            self.remove_source_memory_ids,
            self.add_knowledge_refs,
            self.remove_knowledge_refs,
        )
        if not any(changes):
            raise ValueError("relation update must contain at least one change")
