from __future__ import annotations

from .benchmark import evaluate_retrieval_modes
from .builder import build_hypergraph_view, ensure_hypergraph_view, task_node_id
from .context import search_context
from .distill import archive_scope, distill_experience
from .evolution import (
    apply_decay,
    deduplicate_memory,
    experience_identity,
    maintain_memory,
    merge_experience_records,
    prepare_searchable_memory,
    prune_memory,
    retention_profile,
)
from .memory_model import (
    build_evidence_chain,
    canonical_search_document,
    compact_reuse_projection,
    compact_trace,
    infer_memory_kinds,
    normalize_memory_record,
    validate_evidence_chain,
)
from .retrieval import (
    HYPERGRAPH_RETRIEVAL_MODES,
    RETRIEVAL_MODE_VALUES,
    TEXT_RETRIEVAL_MODES,
    RetrievalMode,
    compare_retrieval_modes,
    search_keyword_memory,
    search_memory_by_mode,
    search_semantic_memory,
    search_tag_memory,
)
from .index import clear_hypergraph_index_cache
from .layers import (
    EPISODIC_LAYER,
    EXPERIENCE_LAYER,
    KNOWLEDGE_LAYER,
    MEMORY_LAYERS,
    MemoryLayer,
    memory_layer_for_record,
    normalize_memory_layer,
    partition_memory_by_layer,
)
from .retrieval_batches import (
    KNOWLEDGE_FOLLOWUP_BATCH,
    MEMORY_ROUTE_BATCH,
    build_retrieval_batches,
    resolve_followup_decision,
    update_retrieval_batch,
)
from .reuse_metrics import (
    evaluate_paired_memory_gain,
    evaluate_run_reuse,
    evaluate_task_reuse,
)
from .search import (
    configured_retrieval_mode,
    load_persistent_memory,
    memory_organization_for_mode,
    search_hypergraph_memory,
)
from .store import SQLiteHypergraphMemoryStore
from .text_memory import (
    compare_retrieval_sets,
    retrieval_metrics,
    search_text_context,
    search_text_memory,
)

__all__ = [
    "archive_scope",
    "evaluate_retrieval_modes",
    "build_hypergraph_view",
    "compare_retrieval_sets",
    "distill_experience",
    "ensure_hypergraph_view",
    "retrieval_metrics",
    "search_context",
    "search_hypergraph_memory",
    "configured_retrieval_mode",
    "memory_organization_for_mode",
    "RetrievalMode",
    "RETRIEVAL_MODE_VALUES",
    "TEXT_RETRIEVAL_MODES",
    "HYPERGRAPH_RETRIEVAL_MODES",
    "evaluate_run_reuse",
    "evaluate_task_reuse",
    "evaluate_paired_memory_gain",
    "build_evidence_chain",
    "canonical_search_document",
    "compact_reuse_projection",
    "compact_trace",
    "infer_memory_kinds",
    "load_persistent_memory",
    "normalize_memory_record",
    "validate_evidence_chain",
    "search_keyword_memory",
    "search_tag_memory",
    "search_semantic_memory",
    "search_memory_by_mode",
    "compare_retrieval_modes",
    "SQLiteHypergraphMemoryStore",
    "deduplicate_memory",
    "experience_identity",
    "merge_experience_records",
    "maintain_memory",
    "apply_decay",
    "prepare_searchable_memory",
    "prune_memory",
    "retention_profile",
    "clear_hypergraph_index_cache",
    "MemoryLayer",
    "KNOWLEDGE_LAYER",
    "EPISODIC_LAYER",
    "EXPERIENCE_LAYER",
    "MEMORY_LAYERS",
    "memory_layer_for_record",
    "normalize_memory_layer",
    "partition_memory_by_layer",
    "MEMORY_ROUTE_BATCH",
    "KNOWLEDGE_FOLLOWUP_BATCH",
    "build_retrieval_batches",
    "resolve_followup_decision",
    "update_retrieval_batch",
    "search_text_context",
    "search_text_memory",
    "task_node_id",
]
