from __future__ import annotations

import os
from typing import Any

from .distill import distill_experience
from .layers import memory_layer_for_record
from .memory_model import normalize_memory_record
from .schema import (
    EDGE_AGENT_MESSAGE,
    EDGE_ARCHIVE_SCOPE,
    EDGE_CLAIM_CONTRADICTION,
    EDGE_CLAIM_DEPENDENCY,
    EDGE_CLAIM_SUPPORT,
    EDGE_EVIDENCE,
    EDGE_EVIDENCE_CHAIN,
    EDGE_EVIDENCE_PROVENANCE,
    EDGE_EXPERIENCE,
    EDGE_EXPERIENCE_ABSTRACTION,
    EDGE_EXPERIENCE_RULE,
    EDGE_EXPERIENCE_SIGNATURE,
    EDGE_GAP,
    EDGE_GAP_COVERAGE,
    EDGE_MEMORY_EVENT,
    EDGE_MEMORY_LAYER,
    EDGE_MEMORY_REUSE,
    EDGE_KNOWLEDGE_GROUNDING,
    EDGE_OUTCOME,
    EDGE_OUTCOME_FEEDBACK,
    EDGE_REUSE_BENEFIT,
    EDGE_SANDBOX_OUTCOME,
    EDGE_STATE_FLOW,
    EDGE_STRATEGY,
    EDGE_SUPPORTING_FACT,
    EDGE_TASK_CONTEXT,
    EDGE_TOOL_EXECUTION,
    EDGE_TRAJECTORY,
    NODE_ACTION,
    NODE_AGENT,
    NODE_ARCHIVE_SCOPE,
    NODE_ARTIFACT,
    NODE_CAPABILITY,
    NODE_CLAIM,
    NODE_CONTEXT,
    NODE_DATASET,
    NODE_ENTITY,
    NODE_EVIDENCE,
    NODE_EVIDENCE_CHAIN,
    NODE_EVIDENCE_SOURCE,
    NODE_EXECUTION,
    NODE_EXPERIENCE,
    NODE_EXPERIENCE_RULE,
    NODE_EXPERIENCE_SIGNATURE,
    NODE_FAILURE_PATTERN,
    NODE_GAP,
    NODE_GROUP,
    NODE_KEYWORD,
    NODE_LIFECYCLE,
    NODE_MEMORY,
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
    NODE_TASK,
    NODE_TOOL,
    NODE_TRIGGER,
    NODE_TURN,
    NODE_VERSION,
)
from .text import limited_tokens, node_id, slug, stable_id


def task_node_id(task_topic: str) -> str:
    """Build a stable task node from task text when no task id is available."""

    return node_id(NODE_TASK, f"{stable_id(task_topic)}-{slug(task_topic)}")


def _edge(
    edge_type: str, memory_id: str, suffix: str, nodes: list[str]
) -> dict[str, Any]:
    unique_nodes = list(dict.fromkeys(node for node in nodes if node))
    edge_key = memory_id if suffix == "event" else f"{memory_id}:{suffix}"
    return {
        "hyperedge_id": node_id("hyperedge", edge_key),
        "type": edge_type,
        "nodes": unique_nodes,
    }


def _clean_tags(tags: list[str]) -> list[str]:
    return sorted(
        {str(tag).lower().replace("_", "-").strip() for tag in tags if str(tag).strip()}
    )


def _memory_facets(
    record: dict[str, Any], trace: dict[str, Any] | None = None
) -> list[str]:
    """Describe the payload facets carried by one memory record.

    Facets such as evidence, trajectory, and outcome are not primary storage
    layers. They remain indexed because retrieval scoring uses them.
    """

    layers: set[str] = set()
    memory_type = str(record.get("memory_type", "evidence") or "evidence")
    if memory_type == "experience":
        layers.add("experience")
    else:
        layers.add("evidence")

    trace = trace if isinstance(trace, dict) else record.get("trace")
    if isinstance(trace, dict):
        if (
            trace.get("messages")
            or trace.get("agent_outputs")
            or trace.get("cognitive_gaps")
        ):
            layers.add("trajectory")
        if (
            trace.get("execution_result")
            or trace.get("metrics")
            or trace.get("memory_hits")
        ):
            layers.add("outcome")

    if (
        record.get("problem_pattern")
        or record.get("solution_pattern")
        or record.get("failure_pattern")
    ):
        layers.add("experience")
    if isinstance(trace, dict) and trace.get("experience_rules"):
        layers.add("experience")
    if (
        record.get("supporting_evidence")
        or record.get("evidence")
        or record.get("supporting_facts")
    ):
        layers.add("evidence")
    if (
        record.get("reuse_count")
        or record.get("reuse_success_count")
        or record.get("reuse_benefits")
    ):
        layers.add("outcome")

    return sorted(layers)


def _memory_layers(
    record: dict[str, Any], trace: dict[str, Any] | None = None
) -> list[str]:
    """Return the single primary layer used by the three-layer architecture."""

    del trace
    return [memory_layer_for_record(record)]


def _memory_layer_edge(
    record: dict[str, Any], trace: dict[str, Any] | None = None
) -> tuple[list[str], dict[str, Any]]:
    memory_id = record["memory_id"]
    memory_node = node_id(NODE_MEMORY, memory_id)
    memory_type = str(record.get("memory_type", "evidence") or "evidence")
    layer_nodes = [
        node_id(NODE_MEMORY_TYPE, memory_type),
        *(node_id(NODE_MEMORY_LAYER, layer) for layer in _memory_layers(record, trace)),
    ]
    return layer_nodes, _edge(
        EDGE_MEMORY_LAYER, memory_id, "layer", [memory_node, *layer_nodes]
    )


def _cross_layer_structure(
    record: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Build optional links between the three primary memory layers."""

    memory_id = str(record["memory_id"])
    memory_node = node_id(NODE_MEMORY, memory_id)
    layer = memory_layer_for_record(record)
    nodes: list[str] = []
    edges: list[dict[str, Any]] = []

    knowledge_nodes = [
        node_id(NODE_CONTEXT, str(ref))
        for ref in record.get("knowledge_refs", []) or []
        if str(ref).strip()
    ]
    if knowledge_nodes:
        nodes.extend(knowledge_nodes)
        edges.append(
            _edge(
                EDGE_KNOWLEDGE_GROUNDING,
                memory_id,
                "knowledge-grounding",
                [
                    memory_node,
                    node_id(NODE_MEMORY_LAYER, layer),
                    node_id(NODE_MEMORY_LAYER, "knowledge"),
                    *knowledge_nodes,
                ],
            )
        )

    source_memory_nodes = [
        node_id(NODE_MEMORY, str(ref))
        for ref in record.get("source_memory_ids", []) or []
        if str(ref).strip() and str(ref) != memory_id
    ]
    if source_memory_nodes:
        nodes.extend(source_memory_nodes)
        edges.append(
            _edge(
                EDGE_EXPERIENCE_ABSTRACTION,
                memory_id,
                "experience-abstraction",
                [
                    memory_node,
                    node_id(NODE_MEMORY_LAYER, layer),
                    node_id(NODE_MEMORY_LAYER, "episodic"),
                    *source_memory_nodes,
                ],
            )
        )

    return list(dict.fromkeys(nodes)), edges


def _record_value_nodes(record: dict[str, Any], key: str, kind: str) -> list[str]:
    value = record.get(key)
    if value in (None, "", [], {}):
        return []
    if isinstance(value, list):
        return [node_id(kind, str(item)) for item in value if str(item).strip()]
    return [node_id(kind, str(value))]


def _experience_signature(record: dict[str, Any]) -> str:
    problem = str(record.get("problem_pattern") or "").strip()
    solution = str(record.get("solution_pattern") or "").strip()
    triggers = record.get("trigger_conditions") or []
    if not isinstance(triggers, list):
        triggers = [triggers]
    trigger_text = "|".join(
        sorted(str(trigger).strip() for trigger in triggers if str(trigger).strip())
    )
    if not any([problem, solution, trigger_text]):
        return ""
    return stable_id(f"{problem}|{solution}|{trigger_text}", length=16)


def _experience_signature_nodes(record: dict[str, Any]) -> list[str]:
    signature = _experience_signature(record)
    if not signature:
        return []
    return [
        node_id(NODE_EXPERIENCE_SIGNATURE, signature),
        *_record_value_nodes(record, "problem_pattern", NODE_PROBLEM_PATTERN),
        *_record_value_nodes(record, "solution_pattern", NODE_SOLUTION_PATTERN),
        *_record_value_nodes(record, "trigger_conditions", NODE_TRIGGER),
    ]


def _gap_coverage_values(
    record: dict[str, Any],
    trace: dict[str, Any] | None = None,
    distilled: dict[str, Any] | None = None,
) -> list[str]:
    """Infer which query gaps this memory can help cover.

    These labels intentionally reuse the query profile gap namespace, so search
    can match a query need directly to a memory capability.
    """

    gaps: set[str] = set()
    facets = set(_memory_facets(record, trace))
    if "evidence" in facets:
        gaps.update({"external-supporting-facts", "needs-reasoning"})
    if "experience" in facets:
        gaps.update(
            {"experience-distillation", "memory-grounded-reasoning", "partial-context"}
        )
    if "trajectory" in facets:
        gaps.add("agent-trajectory")
    if "outcome" in facets:
        gaps.add("outcome-feedback")

    memory_type = str(record.get("memory_type", "evidence") or "evidence")
    if memory_type == "experience":
        gaps.add("memory-grounded-reasoning")
    trace_has_experience_rules = isinstance(trace, dict) and bool(
        trace.get("experience_rules")
    )
    if (
        memory_type == "experience"
        or record.get("trigger_conditions")
        or record.get("problem_pattern")
        or record.get("solution_pattern")
        or trace_has_experience_rules
    ):
        gaps.add("experience-distillation")
    if (
        record.get("supporting_evidence")
        or record.get("supporting_facts")
        or record.get("evidence")
    ):
        gaps.add("external-supporting-facts")
    if record.get("reuse_count") or record.get("reuse_success_count"):
        gaps.add("memory-grounded-reasoning")
        gaps.add("outcome-feedback")

    trace = trace if isinstance(trace, dict) else record.get("trace")
    if isinstance(trace, dict):
        task_meta = (
            trace.get("task_meta", {})
            if isinstance(trace.get("task_meta"), dict)
            else {}
        )
        execution_result = (
            trace.get("execution_result", {})
            if isinstance(trace.get("execution_result"), dict)
            else {}
        )
        if trace.get("memory_hits") or trace.get("memory_context"):
            gaps.update(
                {"missing-memory", "partial-context", "memory-grounded-reasoning"}
            )
        if trace.get("evidence"):
            gaps.add("external-supporting-facts")
        if trace.get("messages") or trace.get("agent_outputs"):
            gaps.add("agent-trajectory")
        if trace.get("cognitive_gaps"):
            gaps.add("gap-aware-coordination")
        if (
            execution_result.get("sandbox_status")
            or execution_result.get("exit_code") is not None
        ):
            gaps.add("tool-execution")
        if execution_result:
            gaps.add("outcome-feedback")

    distilled = distilled or {}
    if distilled.get("strategies"):
        gaps.add("gap-aware-coordination")
    if distilled.get("experiences"):
        gaps.add("memory-grounded-reasoning")
    if distilled.get("reuse_benefits"):
        gaps.add("outcome-feedback")

    return sorted(gaps)


def _question_from_task_topic(task_topic: str) -> str:
    if "Question:" not in task_topic:
        return task_topic
    question = task_topic.split("Question:", 1)[1]
    if "\nContext:" in question:
        question = question.split("\nContext:", 1)[0]
    return question.strip() or task_topic


def _keyword_nodes(*texts: str, limit: int = 32) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for token in limited_tokens(text, limit=limit):
            if token not in seen:
                seen.add(token)
                tokens.append(token)
            if len(tokens) >= limit:
                break
        if len(tokens) >= limit:
            break
    return [node_id(NODE_KEYWORD, token) for token in tokens]


def _packet_list(value: Any) -> list[Any]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, list):
        return value
    return [value]


def _message_refs(message: dict[str, Any], key: str) -> list[str]:
    refs: list[str] = []
    payload = message.get("payload", {})
    if isinstance(payload, dict):
        refs.extend(str(ref) for ref in _packet_list(payload.get(key)))
        expanded_key = {
            "s": "state_refs",
            "m": "memory_refs",
            "cap": "capability_required",
        }.get(key)
        if expanded_key:
            refs.extend(str(ref) for ref in _packet_list(payload.get(expanded_key)))
    return [ref for ref in refs if ref]


def _evidence_chain_structure(
    record: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """将显式证据链映射为带顺序、来源和关系类型的超图结构。"""

    chain = record.get("evidence_chain")
    if not isinstance(chain, dict) or not chain.get("steps"):
        return [], [], {}

    memory_id = str(record["memory_id"])
    memory_node = node_id(NODE_MEMORY, memory_id)
    chain_id = str(chain.get("chain_id") or f"chain-{memory_id}")
    chain_node = node_id(NODE_EVIDENCE_CHAIN, chain_id)
    artifact_node = node_id(NODE_ARTIFACT, "evidence-chain")
    version_node = node_id(NODE_VERSION, str(chain.get("schema_version") or "1.0"))
    structured_nodes = [chain_node, artifact_node, version_node]
    hyperedges: list[dict[str, Any]] = []
    step_nodes: dict[str, str] = {}

    for step in chain.get("steps", []):
        if not isinstance(step, dict) or not step.get("step_id"):
            continue
        step_id = str(step["step_id"])
        kind = str(step.get("kind") or "evidence")
        node_kind = NODE_CLAIM if "claim" in kind else NODE_EVIDENCE
        step_node = node_id(node_kind, step_id)
        step_nodes[step_id] = step_node
        structured_nodes.append(step_node)

        title = str(step.get("title") or "").strip()
        if title:
            structured_nodes.append(node_id(NODE_ENTITY, title))
        structured_nodes.extend(
            _keyword_nodes(title, str(step.get("text") or ""), limit=12)
        )

        source = step.get("source") if isinstance(step.get("source"), dict) else {}
        source_value = "|".join(
            str(source.get(key) or "")
            for key in ("source_type", "source_id", "task_id", "turn_id")
        ).strip("|")
        if source_value:
            source_node = node_id(NODE_EVIDENCE_SOURCE, source_value)
            structured_nodes.append(source_node)
            hyperedges.append(
                _edge(
                    EDGE_EVIDENCE_PROVENANCE,
                    memory_id,
                    f"provenance:{stable_id(step_id, length=8)}",
                    [memory_node, chain_node, step_node, source_node],
                )
            )

    hyperedges.append(
        _edge(
            EDGE_EVIDENCE_CHAIN,
            memory_id,
            "evidence-chain",
            [
                memory_node,
                chain_node,
                artifact_node,
                version_node,
                *step_nodes.values(),
            ],
        )
    )

    for index, link in enumerate(chain.get("links", []), start=1):
        if not isinstance(link, dict):
            continue
        source_node = step_nodes.get(str(link.get("from") or ""))
        target_node = step_nodes.get(str(link.get("to") or ""))
        if not source_node or not target_node:
            continue
        relation = str(link.get("relation") or "depends_on")
        relation_node = node_id(NODE_RELATION, relation)
        structured_nodes.append(relation_node)
        if relation == "contradicts":
            edge_type = EDGE_CLAIM_CONTRADICTION
        elif relation == "supports":
            edge_type = EDGE_CLAIM_SUPPORT
        else:
            edge_type = EDGE_CLAIM_DEPENDENCY
        hyperedges.append(
            _edge(
                edge_type,
                memory_id,
                f"chain-link:{index}",
                [memory_node, chain_node, source_node, target_node, relation_node],
            )
        )

    chain_summary = {
        "chain_id": chain_id,
        "valid": bool(chain.get("valid")),
        "completeness": float(chain.get("completeness", 0.0) or 0.0),
        "evidence_count": int(chain.get("evidence_count", 0) or 0),
        "claim_count": int(chain.get("claim_count", 0) or 0),
        "issues": chain.get("issues", []),
    }
    return list(dict.fromkeys(structured_nodes)), hyperedges, chain_summary


def _finalize_hypergraph_view(
    record: dict[str, Any],
    view: dict[str, Any],
    *,
    persist: bool,
) -> dict[str, Any]:
    """在显式配置后通过原写入路径透明持久化，不改变调用侧接口。"""

    database_path = os.environ.get("HYPERGRAPH_MEMORY_DB", "").strip()
    enabled = os.environ.get("HYPERGRAPH_MEMORY_AUTOPERSIST", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if persist and enabled and database_path:
        from .store import SQLiteHypergraphMemoryStore

        with SQLiteHypergraphMemoryStore(database_path) as store:
            store.upsert(record, hypergraph=view)
    return view


def _legacy_hypergraph_view(record: dict[str, Any]) -> dict[str, Any]:
    memory_id = record["memory_id"]
    task_topic = record.get("task_topic", "")
    tags = _clean_tags(record.get("tags", []))
    memory_type = str(record.get("memory_type", "evidence"))
    if memory_type == "experience":
        trigger_text = " ".join(
            str(item) for item in record.get("trigger_conditions", []) or []
        )
        expanded_trigger_text = " ".join(
            str(item).replace("_", " ").replace(":", " ")
            for item in record.get("trigger_conditions", []) or []
        )
        task_topic = str(record.get("problem_pattern") or task_topic)
        keyword_source = " ".join(
            [
                str(record.get("problem_pattern", "")),
                str(record.get("solution_pattern", "")),
                str(record.get("failure_pattern", "")),
                trigger_text,
                expanded_trigger_text,
            ]
        )
    else:
        keyword_source = record.get("summary", "")

    memory_node = node_id(NODE_MEMORY, memory_id)
    memory_alias_nodes = _record_value_nodes(
        record, "merged_memory_ids", NODE_MEMORY_REF
    )
    task_node = task_node_id(task_topic)
    agent_node = node_id(NODE_AGENT, record.get("source_agent", "unknown"))
    tag_nodes = [node_id(NODE_TAG, tag) for tag in tags]
    keyword_nodes = _keyword_nodes(task_topic, keyword_source, " ".join(tags), limit=48)
    entity_nodes = _record_value_nodes(record, "entities", NODE_ENTITY)
    record_context_nodes = []
    for key, kind in (
        ("group_id", NODE_GROUP),
        ("turn_id", NODE_TURN),
        ("dataset_type", NODE_DATASET),
    ):
        value = record.get(key)
        if value:
            record_context_nodes.append(node_id(kind, str(value)))
    artifact_nodes = [
        node_id(NODE_ARTIFACT, str(value))
        for value in record.get("artifact_types", []) or []
        if str(value).strip()
    ]
    layer_nodes, layer_edge = _memory_layer_edge(record)
    gap_coverage_nodes = [
        node_id(NODE_GAP, value) for value in _gap_coverage_values(record)
    ]
    gap_coverage_edge = _edge(
        EDGE_GAP_COVERAGE,
        memory_id,
        "gap-coverage",
        [memory_node, node_id(NODE_ARTIFACT, "gap-coverage"), *gap_coverage_nodes],
    )
    signature_nodes = _experience_signature_nodes(record)
    signature_edge = _edge(
        EDGE_EXPERIENCE_SIGNATURE,
        memory_id,
        "experience-signature",
        [memory_node, node_id(NODE_ARTIFACT, "experience-signature"), *signature_nodes],
    )
    chain_nodes, chain_edges, chain_summary = _evidence_chain_structure(record)
    cross_layer_nodes, cross_layer_edges = _cross_layer_structure(record)
    hyperedge = _edge(
        EDGE_MEMORY_EVENT,
        memory_id,
        "event",
        [
            memory_node,
            task_node,
            agent_node,
            *tag_nodes,
            *keyword_nodes,
            *entity_nodes,
            *record_context_nodes,
            *artifact_nodes,
            *layer_nodes,
            *gap_coverage_nodes,
            *signature_nodes,
            *memory_alias_nodes,
            *chain_nodes,
            *cross_layer_nodes,
        ],
    )

    alias_edge = _edge(
        EDGE_MEMORY_REUSE,
        memory_id,
        "merged-aliases",
        [memory_node, *memory_alias_nodes],
    )

    return {
        "memory_node": memory_node,
        "memory_type": memory_type,
        "memory_layers": _memory_layers(record),
        "memory_facets": _memory_facets(record),
        "task_node": task_node,
        "task_label": task_topic,
        "agent_node": agent_node,
        "tag_nodes": tag_nodes,
        "record_context_nodes": record_context_nodes,
        "artifact_nodes": artifact_nodes,
        "layer_nodes": layer_nodes,
        "gap_coverage_nodes": gap_coverage_nodes,
        "experience_signature_nodes": signature_nodes,
        "keyword_nodes": keyword_nodes,
        "entity_nodes": entity_nodes,
        "evidence_chain_nodes": chain_nodes,
        "cross_layer_nodes": cross_layer_nodes,
        "evidence_chain": chain_summary,
        "event_hyperedge": hyperedge,
        "memory_alias_nodes": memory_alias_nodes,
        "hyperedges": [
            hyperedge,
            layer_edge,
            gap_coverage_edge,
            *([signature_edge] if signature_nodes else []),
            *([alias_edge] if memory_alias_nodes else []),
            *chain_edges,
            *cross_layer_edges,
        ],
    }


def build_hypergraph_view(
    record: dict[str, Any], *, persist: bool = True, _normalized: bool = False
) -> dict[str, Any]:
    """Build an explainable hypergraph view for one reusable memory record."""

    normalized = record if _normalized else normalize_memory_record(record)
    record.update(
        {key: value for key, value in normalized.items() if key != "hypergraph"}
    )
    trace = record.get("trace")
    if not isinstance(trace, dict):
        return _finalize_hypergraph_view(
            record,
            _legacy_hypergraph_view(record),
            persist=persist,
        )

    memory_id = record["memory_id"]
    task_topic = record.get("task_topic", "")
    task_meta = (
        trace.get("task_meta", {})
        if isinstance(trace.get("task_meta", {}), dict)
        else {}
    )
    question = str(task_meta.get("question") or _question_from_task_topic(task_topic))
    tags = _clean_tags(record.get("tags", []))

    memory_node = node_id(NODE_MEMORY, memory_id)
    memory_alias_nodes = _record_value_nodes(
        record, "merged_memory_ids", NODE_MEMORY_REF
    )
    task_node = node_id(
        NODE_TASK,
        task_meta.get("task_id") or f"{stable_id(task_topic)}-{slug(question)}",
    )
    agent_node = node_id(NODE_AGENT, record.get("source_agent", "unknown"))
    tag_nodes = [node_id(NODE_TAG, tag) for tag in tags]
    keyword_nodes = _keyword_nodes(
        question, record.get("summary", ""), " ".join(tags), limit=40
    )
    memory_type = str(record.get("memory_type", "evidence") or "evidence")
    memory_layers = _memory_layers(record, trace)
    memory_facets = _memory_facets(record, trace)
    layer_nodes, layer_edge = _memory_layer_edge(record, trace)
    distilled = distill_experience(record)
    gap_coverage_nodes = [
        node_id(NODE_GAP, value)
        for value in _gap_coverage_values(record, trace, distilled)
    ]
    gap_coverage_edge = _edge(
        EDGE_GAP_COVERAGE,
        memory_id,
        "gap-coverage",
        [memory_node, node_id(NODE_ARTIFACT, "gap-coverage"), *gap_coverage_nodes],
    )
    signature_nodes = _experience_signature_nodes(record)
    signature_edge = _edge(
        EDGE_EXPERIENCE_SIGNATURE,
        memory_id,
        "experience-signature",
        [memory_node, node_id(NODE_ARTIFACT, "experience-signature"), *signature_nodes],
    )
    chain_nodes, chain_edges, chain_summary = _evidence_chain_structure(record)
    cross_layer_nodes, cross_layer_edges = _cross_layer_structure(record)

    structured_nodes: list[str] = [
        *layer_nodes,
        *gap_coverage_nodes,
        *signature_nodes,
        *memory_alias_nodes,
        *chain_nodes,
        *cross_layer_nodes,
    ]
    entity_nodes = _record_value_nodes(record, "entities", NODE_ENTITY)
    structured_nodes.extend(entity_nodes)
    hyperedges: list[dict[str, Any]] = [
        layer_edge,
        gap_coverage_edge,
        *([signature_edge] if signature_nodes else []),
        *(
            [
                _edge(
                    EDGE_MEMORY_REUSE,
                    memory_id,
                    "merged-aliases",
                    [memory_node, *memory_alias_nodes],
                )
            ]
            if memory_alias_nodes
            else []
        ),
        *chain_edges,
        *cross_layer_edges,
    ]
    if entity_nodes:
        hyperedges.append(
            _edge(
                EDGE_EVIDENCE,
                memory_id,
                "entities",
                [memory_node, node_id(NODE_ARTIFACT, "entities"), *entity_nodes],
            )
        )

    task_context_nodes = [memory_node, task_node]
    for key, kind in (
        ("group_id", NODE_GROUP),
        ("turn_id", NODE_TURN),
        ("dataset_type", NODE_DATASET),
    ):
        value = task_meta.get(key) or record.get(key)
        if value:
            task_context_nodes.append(node_id(kind, str(value)))
    for value in record.get("artifact_types", []) or []:
        if not str(value).strip():
            continue
        artifact_node = node_id(NODE_ARTIFACT, str(value))
        structured_nodes.append(artifact_node)
        task_context_nodes.append(artifact_node)
    hyperedges.append(_edge(EDGE_TASK_CONTEXT, memory_id, "task", task_context_nodes))

    state_nodes = [memory_node]
    state_refs = trace.get("state_refs", {})
    if isinstance(state_refs, dict):
        for role, ref in state_refs.items():
            if not ref:
                continue
            role_node = node_id(NODE_STATE, str(role))
            ref_node = node_id(NODE_STATE_REF, str(ref))
            artifact_node = node_id(NODE_ARTIFACT, str(role))
            state_nodes.extend([role_node, ref_node, artifact_node])
            structured_nodes.extend([role_node, ref_node, artifact_node])
    for key in ("plan_ref", "evidence_ref", "answer_ref"):
        ref = (
            trace.get("execution_result", {}).get(key)
            if isinstance(trace.get("execution_result"), dict)
            else ""
        )
        if ref:
            role = key.removesuffix("_ref")
            role_node = node_id(NODE_STATE, role)
            ref_node = node_id(NODE_STATE_REF, str(ref))
            artifact_node = node_id(NODE_ARTIFACT, role)
            state_nodes.extend([role_node, ref_node, artifact_node])
            structured_nodes.extend([role_node, ref_node, artifact_node])
    if len(state_nodes) > 1:
        hyperedges.append(_edge(EDGE_STATE_FLOW, memory_id, "state", state_nodes))

    trajectory_nodes = [
        memory_node,
        task_node,
        node_id(NODE_ARTIFACT, "agent-trajectory"),
    ]
    for index, message in enumerate(trace.get("messages", []), start=1):
        if not isinstance(message, dict):
            continue
        message_node = node_id(NODE_MESSAGE, f"{memory_id}:{index}")
        sender_node = node_id(NODE_AGENT, message.get("from", "unknown"))
        receiver_node = node_id(NODE_AGENT, message.get("to", "unknown"))
        action_node = node_id(NODE_ACTION, message.get("action", "unknown"))
        message_nodes = [
            memory_node,
            message_node,
            sender_node,
            receiver_node,
            action_node,
        ]
        trajectory_nodes.extend([message_node, sender_node, receiver_node, action_node])
        for ref in _message_refs(message, "s"):
            ref_node = node_id(NODE_STATE_REF, ref)
            message_nodes.append(ref_node)
            trajectory_nodes.append(ref_node)
        for ref in _message_refs(message, "m"):
            ref_node = node_id(NODE_MEMORY_REF, ref)
            message_nodes.append(ref_node)
            trajectory_nodes.append(ref_node)
        for capability in _message_refs(message, "cap"):
            capability_node = node_id(NODE_CAPABILITY, capability)
            message_nodes.append(capability_node)
            trajectory_nodes.append(capability_node)
        structured_nodes.extend(message_nodes[1:])
        hyperedges.append(
            _edge(EDGE_AGENT_MESSAGE, memory_id, f"message:{index}", message_nodes)
        )
    if len(trajectory_nodes) > 3:
        structured_nodes.extend(trajectory_nodes[1:])
        hyperedges.append(
            _edge(EDGE_TRAJECTORY, memory_id, "trajectory", trajectory_nodes)
        )

    memory_reuse_nodes = [memory_node]
    for hit in trace.get("memory_hits", []):
        if not isinstance(hit, dict):
            continue
        hit_id = hit.get("memory_id")
        if hit_id:
            memory_reuse_nodes.append(node_id(NODE_MEMORY_REF, str(hit_id)))
    if len(memory_reuse_nodes) > 1:
        structured_nodes.extend(memory_reuse_nodes[1:])
        hyperedges.append(
            _edge(EDGE_MEMORY_REUSE, memory_id, "reuse", memory_reuse_nodes)
        )

    for index, gap in enumerate(trace.get("cognitive_gaps", []), start=1):
        if not isinstance(gap, dict):
            continue
        gap_type = gap.get("gap_type") or gap.get("type")
        if not gap_type:
            continue
        gap_nodes = [
            memory_node,
            node_id(NODE_GAP, str(gap_type)),
            node_id(NODE_AGENT, str(gap.get("agent", "unknown"))),
        ]
        for value in gap.get("unknown", []):
            gap_nodes.append(node_id(NODE_GAP, str(value)))
        for value in gap.get("needed", []):
            gap_nodes.append(node_id(NODE_ARTIFACT, str(value)))
        structured_nodes.extend(gap_nodes[1:])
        hyperedges.append(_edge(EDGE_GAP, memory_id, f"gap:{index}", gap_nodes))

    evidence_nodes = [memory_node, node_id(NODE_ARTIFACT, "evidence")]
    for index, evidence in enumerate(trace.get("evidence", []), start=1):
        evidence_node = node_id(
            NODE_EVIDENCE, f"{memory_id}:{index}:{stable_id(str(evidence), length=8)}"
        )
        evidence_nodes.append(evidence_node)
    if len(evidence_nodes) > 2:
        structured_nodes.extend(evidence_nodes[1:])
        hyperedges.append(_edge(EDGE_EVIDENCE, memory_id, "evidence", evidence_nodes))

    supporting_fact_nodes = [memory_node, node_id(NODE_ARTIFACT, "supporting-facts")]
    supporting_items = [
        *record.get("supporting_facts", []),
        *record.get("supporting_evidence", []),
    ]
    for index, fact in enumerate(supporting_items, start=1):
        title = ""
        sentence = ""
        if isinstance(fact, (list, tuple)) and fact:
            title = str(fact[0])
            sentence = str(fact[1]) if len(fact) > 1 else ""
        elif isinstance(fact, dict):
            title = str(fact.get("title", ""))
            sentence = str(fact.get("sentence", ""))
        if not title and not sentence:
            continue
        fact_node = node_id(NODE_SUPPORTING_FACT, f"{slug(title or sentence)}-{index}")
        supporting_fact_nodes.append(fact_node)
        if title:
            supporting_fact_nodes.append(node_id(NODE_ENTITY, title))
            supporting_fact_nodes.extend(_keyword_nodes(title, limit=8))
    if len(supporting_fact_nodes) > 2:
        structured_nodes.extend(supporting_fact_nodes[1:])
        hyperedges.append(
            _edge(
                EDGE_SUPPORTING_FACT,
                memory_id,
                "supporting-facts",
                supporting_fact_nodes,
            )
        )

    outcome_nodes = [memory_node]
    metrics = trace.get("metrics", {})
    if isinstance(metrics, dict):
        for key in (
            "message_count",
            "text_chars",
            "latent_transfer_count",
            "latent_bytes",
        ):
            if key in metrics:
                outcome_nodes.append(node_id(NODE_METRIC, key))
        if metrics:
            outcome_nodes.append(node_id(NODE_ARTIFACT, "metrics"))
    execution_result = trace.get("execution_result", {})
    if isinstance(execution_result, dict):
        for key in ("execution_mode", "retrieval_mode", "sandbox_status"):
            value = execution_result.get(key)
            if value:
                outcome_nodes.append(node_id(NODE_OUTCOME, f"{key}:{value}"))
        for key in ("exit_code", "stdout", "stderr", "tool_elapsed_ms"):
            if key in execution_result:
                outcome_nodes.append(node_id(NODE_EXECUTION, key))
    if len(outcome_nodes) > 1:
        structured_nodes.extend(outcome_nodes[1:])
        hyperedges.append(_edge(EDGE_OUTCOME, memory_id, "outcome", outcome_nodes))

    archive_scope_value = distilled.get("archive_scope", "")
    lifecycle_value = distilled.get("lifecycle_stage", "")
    archive_nodes = [memory_node]
    if archive_scope_value:
        archive_nodes.append(node_id(NODE_ARCHIVE_SCOPE, str(archive_scope_value)))
    if lifecycle_value:
        archive_nodes.append(node_id(NODE_LIFECYCLE, str(lifecycle_value)))
    if len(archive_nodes) > 1:
        structured_nodes.extend(archive_nodes[1:])
        hyperedges.append(
            _edge(EDGE_ARCHIVE_SCOPE, memory_id, "archive", archive_nodes)
        )

    strategy_nodes = [memory_node, node_id(NODE_ARTIFACT, "strategy")]
    for strategy in distilled.get("strategies", []):
        strategy_nodes.append(node_id(NODE_STRATEGY, str(strategy)))
    if len(strategy_nodes) > 2:
        structured_nodes.extend(strategy_nodes[1:])
        hyperedges.append(_edge(EDGE_STRATEGY, memory_id, "strategy", strategy_nodes))

    experience_nodes = [memory_node, node_id(NODE_ARTIFACT, "experience")]
    for experience in distilled.get("experiences", []):
        experience_nodes.append(node_id(NODE_EXPERIENCE, str(experience)))
    if len(experience_nodes) > 2:
        structured_nodes.extend(experience_nodes[1:])
        hyperedges.append(
            _edge(EDGE_EXPERIENCE, memory_id, "experience", experience_nodes)
        )

    benefit_nodes = [memory_node, node_id(NODE_ARTIFACT, "reuse-benefit")]
    for benefit in distilled.get("reuse_benefits", []):
        benefit_nodes.append(node_id(NODE_REUSE_BENEFIT, str(benefit)))
    if len(benefit_nodes) > 2:
        structured_nodes.extend(benefit_nodes[1:])
        hyperedges.append(
            _edge(EDGE_REUSE_BENEFIT, memory_id, "reuse-benefit", benefit_nodes)
        )

    rule_nodes = [
        memory_node,
        node_id(NODE_EXPERIENCE_RULE, "rule-card"),
        *_record_value_nodes(record, "problem_pattern", NODE_PROBLEM_PATTERN),
        *_record_value_nodes(record, "solution_pattern", NODE_SOLUTION_PATTERN),
        *_record_value_nodes(record, "failure_pattern", NODE_FAILURE_PATTERN),
        *_record_value_nodes(record, "trigger_conditions", NODE_TRIGGER),
        *_record_value_nodes(record, "supporting_evidence", NODE_SUPPORTING_FACT),
    ]
    for rule in trace.get("experience_rules", []):
        if not isinstance(rule, dict):
            continue
        rule_nodes.extend(
            _record_value_nodes(rule, "problem_pattern", NODE_PROBLEM_PATTERN)
        )
        rule_nodes.extend(
            _record_value_nodes(rule, "solution_pattern", NODE_SOLUTION_PATTERN)
        )
        rule_nodes.extend(
            _record_value_nodes(rule, "failure_pattern", NODE_FAILURE_PATTERN)
        )
        rule_nodes.extend(_record_value_nodes(rule, "trigger_conditions", NODE_TRIGGER))
        rule_nodes.extend(
            _record_value_nodes(rule, "supporting_evidence", NODE_SUPPORTING_FACT)
        )
        rule_memory_id = rule.get("memory_id")
        if rule_memory_id:
            rule_nodes.append(node_id(NODE_MEMORY_REF, str(rule_memory_id)))
    if len(rule_nodes) > 2:
        structured_nodes.extend(rule_nodes[1:])
        hyperedges.append(
            _edge(EDGE_EXPERIENCE_RULE, memory_id, "experience-rule", rule_nodes)
        )

    feedback_nodes = [
        memory_node,
        node_id(NODE_ARTIFACT, "outcome-feedback"),
    ]
    for key in ("reuse_count", "reuse_success_count", "merge_count", "confidence"):
        if key in record:
            feedback_nodes.append(node_id(NODE_METRIC, key))
    if isinstance(execution_result, dict):
        for key in ("exact_match", "experience_used_count"):
            if key in execution_result:
                feedback_nodes.append(node_id(NODE_METRIC, key))
        for memory_ref in execution_result.get("experience_used_memory_ids", []) or []:
            feedback_nodes.append(node_id(NODE_MEMORY_REF, str(memory_ref)))
    if isinstance(metrics, dict):
        for key in (
            "message_count",
            "text_chars",
            "latent_transfer_count",
            "latent_bytes",
            "transfer_latency_ms",
        ):
            if key in metrics:
                feedback_nodes.append(node_id(NODE_METRIC, key))
    if len(feedback_nodes) > 2:
        structured_nodes.extend(feedback_nodes[1:])
        hyperedges.append(
            _edge(EDGE_OUTCOME_FEEDBACK, memory_id, "outcome-feedback", feedback_nodes)
        )

    tool_trace = trace.get("tool_trace", {})
    if isinstance(tool_trace, dict) and tool_trace:
        tool_nodes = [
            memory_node,
            node_id(NODE_TOOL, str(tool_trace.get("tool", "unknown"))),
        ]
        for key in (
            "sandbox_status",
            "exit_code",
            "stdout",
            "stderr",
            "tool_elapsed_ms",
        ):
            if key in tool_trace:
                tool_nodes.append(node_id(NODE_EXECUTION, key))
        structured_nodes.extend(tool_nodes[1:])
        hyperedges.append(_edge(EDGE_TOOL_EXECUTION, memory_id, "tool", tool_nodes))
        if "sandbox_status" in tool_trace:
            sandbox_nodes = [
                memory_node,
                node_id(NODE_OUTCOME, f"sandbox:{tool_trace['sandbox_status']}"),
            ]
            structured_nodes.extend(sandbox_nodes[1:])
            hyperedges.append(
                _edge(EDGE_SANDBOX_OUTCOME, memory_id, "sandbox", sandbox_nodes)
            )

    event_nodes = [
        memory_node,
        task_node,
        agent_node,
        *tag_nodes,
        *keyword_nodes,
        *structured_nodes,
    ]
    event_hyperedge = _edge(EDGE_MEMORY_EVENT, memory_id, "event", event_nodes)

    view = {
        "memory_node": memory_node,
        "memory_type": memory_type,
        "memory_layers": memory_layers,
        "memory_facets": memory_facets,
        "task_node": task_node,
        "task_label": task_topic,
        "agent_node": agent_node,
        "tag_nodes": tag_nodes,
        "layer_nodes": layer_nodes,
        "gap_coverage_nodes": gap_coverage_nodes,
        "experience_signature_nodes": signature_nodes,
        "memory_alias_nodes": memory_alias_nodes,
        "keyword_nodes": keyword_nodes,
        "evidence_chain_nodes": chain_nodes,
        "cross_layer_nodes": cross_layer_nodes,
        "evidence_chain": chain_summary,
        "event_hyperedge": event_hyperedge,
        "hyperedges": [event_hyperedge, *hyperedges],
        "distilled_experience": distilled,
    }
    return _finalize_hypergraph_view(record, view, persist=persist)


def ensure_hypergraph_view(record: dict[str, Any]) -> dict[str, Any]:
    """Build a hypergraph view on demand for older records."""

    return record.get("hypergraph") or build_hypergraph_view(record)
