from __future__ import annotations

import json
import re
from copy import deepcopy
from functools import lru_cache
from typing import Any

from .entities import canonical_entity, extract_canonical_entities
from .layers import KNOWLEDGE_LAYER
from .text import limited_tokens, node_id, question_content_tokens, stable_id

ANSWER_LIKE_FIELDS = {
    "answer",
    "expected",
    "expected_answer",
    "gold",
    "reference_code",
}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _tokens(*values: Any) -> set[str]:
    return set(limited_tokens(" ".join(_text(value) for value in values), limit=96))


def _dataset_name(task_meta: dict[str, Any], dataset_type: str | None) -> str:
    value = (
        dataset_type
        or task_meta.get("dataset_type")
        or task_meta.get("source_dataset")
        or ""
    )
    value = _text(value).lower()
    if "topiocqa" in value:
        return "topiocqa"
    if "codeact" in value:
        return "codeact"
    if "continuous" in value:
        return "hotpot_continuous"
    if "multihop" in value or "hotpot" in value:
        return "hotpot"
    if task_meta.get("supporting_passage"):
        return "topiocqa"
    if task_meta.get("inputs") or task_meta.get("allowed_imports"):
        return "codeact"
    return "generic"


def _task_id(task_meta: dict[str, Any]) -> str:
    if task_meta.get("task_id"):
        return _text(task_meta["task_id"])
    group_id = _text(task_meta.get("group_id") or task_meta.get("chain_id"))
    turn_id = _text(task_meta.get("turn_id") or task_meta.get("turn"))
    if group_id and turn_id:
        return f"{group_id}:{turn_id}"
    return group_id or stable_id(str(task_meta), length=10)


def _context_id(task_id: str, kind: str, index: int, *parts: Any) -> str:
    return f"{task_id}:{kind}:{index:03d}:{stable_id('|'.join(_text(part) for part in parts), length=8)}"


def _unit(
    *,
    task_id: str,
    dataset_type: str,
    kind: str,
    index: int,
    text: str,
    title: str = "",
    section: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "context_id": _context_id(task_id, kind, index, title, section, text),
        "memory_layer": KNOWLEDGE_LAYER,
        "dataset_type": dataset_type,
        "kind": kind,
        "title": title,
        "section": section,
        "text": text.strip(),
        "metadata": metadata or {},
    }


def _normalize_hotpot(
    task_meta: dict[str, Any], dataset_type: str, task_id: str
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for doc_index, doc in enumerate(task_meta.get("context", []) or []):
        if not isinstance(doc, dict):
            continue
        title = _text(doc.get("title"))
        section = _text(doc.get("section") or doc.get("topic_section"))
        for sentence_index, sentence in enumerate(doc.get("sentences", []) or []):
            sentence_text = _text(sentence)
            if not sentence_text:
                continue
            units.append(
                _unit(
                    task_id=task_id,
                    dataset_type=dataset_type,
                    kind="sentence",
                    index=len(units),
                    title=title,
                    section=section,
                    text=sentence_text,
                    metadata={"doc_index": doc_index, "sentence_index": sentence_index},
                )
            )
    return units


def _normalize_topiocqa(
    task_meta: dict[str, Any], dataset_type: str, task_id: str
) -> list[dict[str, Any]]:
    units = _normalize_hotpot(task_meta, dataset_type, task_id)
    documents = task_meta.get("context", []) or []
    for position, unit in enumerate(units):
        metadata = unit["metadata"]
        document = documents[metadata["doc_index"]]
        index = metadata["sentence_index"]
        sentences = document.get("sentences", []) or []
        start, end = max(0, index - 1), min(len(sentences), index + 2)
        # A bounded overlapping window keeps a split statement with its continuation.
        unit["text"] = " ".join(
            _text(sentence) for sentence in sentences[start:end] if _text(sentence)
        )
        title, separator, section = unit["title"].partition("[SEP]")
        if separator:
            unit["title"] = title.strip()
            unit["section"] = unit["section"] or section.strip()
        metadata.update(
            {
                "source": document.get("source", "dataset_context"),
                "sentence_range": [start, end],
            }
        )
        unit["context_id"] = _context_id(
            task_id,
            unit["kind"],
            position,
            unit["title"],
            unit["section"],
            unit["text"],
        )
    history = (
        task_meta.get("conversation_context", []) or task_meta.get("context", []) or []
    )
    if isinstance(history, list):
        for index, item in enumerate(history):
            if isinstance(item, dict):
                continue
            text = _text(item)
            if not text:
                continue
            role = "dialogue_question" if index % 2 == 0 else "dialogue_answer"
            units.append(
                _unit(
                    task_id=task_id,
                    dataset_type=dataset_type,
                    kind=role,
                    index=len(units),
                    title=_text(task_meta.get("topic")),
                    section="dialogue_history",
                    text=text,
                    metadata={"history_index": index},
                )
            )
    passage = _text(task_meta.get("supporting_passage"))
    if passage:
        units.append(
            _unit(
                task_id=task_id,
                dataset_type=dataset_type,
                kind="supporting_passage",
                index=len(units),
                title=_text(task_meta.get("topic")),
                section=_text(task_meta.get("topic_section")),
                text=passage,
            )
        )
    seen: set[tuple[str, str, str, str]] = set()
    unique = []
    for unit in units:
        channel = "dialogue" if unit["kind"].startswith("dialogue_") else "knowledge"
        key = (channel, unit["title"], unit["section"], unit["text"])
        if key not in seen:
            seen.add(key)
            unique.append(unit)
    return unique


def _normalize_codeact(
    task_meta: dict[str, Any], dataset_type: str, task_id: str
) -> list[dict[str, Any]]:
    parts: list[str] = []
    if task_meta.get("task_type"):
        parts.append(f"task_type: {task_meta.get('task_type')}")
    if task_meta.get("inputs"):
        parts.append(f"inputs: {task_meta.get('inputs')}")
    if task_meta.get("allowed_imports"):
        parts.append(f"allowed_imports: {task_meta.get('allowed_imports')}")
    if task_meta.get("tags"):
        parts.append(f"tags: {task_meta.get('tags')}")
    if not parts:
        return []
    return [
        _unit(
            task_id=task_id,
            dataset_type=dataset_type,
            kind="tool_task_context",
            index=0,
            title=_text(task_meta.get("task_type")),
            section="codeact",
            text="; ".join(parts),
        )
    ]


def _normalize_generic(
    task_meta: dict[str, Any], dataset_type: str, task_id: str
) -> list[dict[str, Any]]:
    units = _normalize_hotpot(task_meta, dataset_type, task_id)
    if units:
        return units
    passage = _text(task_meta.get("supporting_passage"))
    if passage:
        return _normalize_topiocqa(task_meta, dataset_type, task_id)
    return []


def _context_units(
    task_meta: dict[str, Any], dataset_type: str, task_id: str
) -> list[dict[str, Any]]:
    # Provenance, not gold-answer comparison, identifies this invalid source.
    # Answer annotations must be replaced with actual source passages upstream.
    if (
        any(
            isinstance(doc, dict) and doc.get("source") == "official_answer_annotation"
            for doc in task_meta.get("context", []) or []
        )
        or task_meta.get("evidence_scope")
        == "official_answer_annotations_existing_project_adapter"
    ):
        raise ValueError(
            "Answer annotations cannot be used as retrieval evidence; "
            "provide source passages instead of official_answer_annotation."
        )
    if dataset_type in {"hotpot", "hotpot_continuous"}:
        return _normalize_hotpot(task_meta, dataset_type, task_id)
    if dataset_type == "topiocqa":
        return _normalize_topiocqa(task_meta, dataset_type, task_id)
    if dataset_type == "codeact":
        return _normalize_codeact(task_meta, dataset_type, task_id)
    return _normalize_generic(task_meta, dataset_type, task_id)


def _extract_entities(*texts: str, limit: int = 14) -> list[str]:
    return extract_canonical_entities(*texts, limit=limit)


def _entity_overlap(left: str, right: str) -> bool:
    """实体别名可以包含完整名称，不能匹配名字内部的字符片段。"""
    return bool(left and right and (
        re.search(r"(?<!\w)" + re.escape(left) + r"(?!\w)", right)
        or re.search(r"(?<!\w)" + re.escape(right) + r"(?!\w)", left)
    ))


@lru_cache(maxsize=2048)
def _context_features(
    title: str, section: str, text: str
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Reuse document features across turns without caching mutable caller data."""

    return (
        frozenset(_tokens(title, section, text)),
        frozenset(_extract_entities(title, section, text)),
        frozenset(_extract_entities(text)),
    )


@lru_cache(maxsize=32)
def _cached_context_graph(payload: str, dataset_type: str) -> dict[str, Any]:
    graph = _build_context_hypergraph(json.loads(payload), dataset_type)
    graph["lookup"] = _context_lookup(graph)
    return graph


def _context_lookup(graph):
    """倒排表随知识源缓存，不随问题重建。"""
    edges_by_context = {}
    for edge in graph["hyperedges"]:
        for node in edge.get("nodes", []):
            if node.startswith("context:"):
                edges_by_context.setdefault(node, []).append(edge)
    entities, text_entities, entity_contexts = {}, {}, {}
    for unit in graph["context_units"]:
        context_id = unit["context_id"]
        features = _context_features(unit.get("title", ""), unit.get("section", ""), unit["text"])
        entities[context_id], text_entities[context_id] = features[1:]
        for entity in entities[context_id]:
            entity_contexts.setdefault(entity, set()).add(context_id)
    return edges_by_context, entities, text_entities, entity_contexts


def select_connected_hits(ranked, top_k):
    """保留命中与它依赖的桥接证据，避免末次截取拆断证据链。"""
    by_id = {hit["context_id"]: hit for hit in ranked}
    chosen = {}
    for hit in ranked:
        bundle = {}

        def include(row, visiting):
            key = row["context_id"]
            if key in visiting or key in chosen or key in bundle:
                return True
            visiting = visiting | {key}
            for source in row.get("bridge_source_context_ids", []):
                if source not in by_id or not include(by_id[source], visiting):
                    return False
            bundle[key] = row
            return True

        if not include(hit, set()):
            continue
        if len(chosen) + len(bundle) <= top_k:
            chosen.update(bundle)
        if len(chosen) == top_k:
            break
    return list(chosen.values())


def clear_context_cache() -> None:
    """Release cached graphs and document features between independent runs."""

    _cached_context_graph.cache_clear()
    _context_features.cache_clear()


def _edge(edge_type: str, edge_key: str, nodes: list[str]) -> dict[str, Any]:
    return {
        "hyperedge_id": node_id("context_hyperedge", edge_key),
        "type": edge_type,
        "nodes": list(dict.fromkeys(node for node in nodes if node)),
    }


def _build_context_hypergraph(
    task_meta: dict[str, Any], dataset_type: str
) -> dict[str, Any]:
    task_id = _task_id(task_meta)
    question = _text(task_meta.get("question") or task_meta.get("task"))
    group_id = _text(task_meta.get("group_id") or task_meta.get("chain_id"))
    turn_id = _text(task_meta.get("turn_id") or task_meta.get("turn"))
    units = _context_units(task_meta, dataset_type, task_id)

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def add_node(kind: str, value: str, label: str | None = None) -> str:
        node = node_id(kind, value)
        nodes[node] = {
            "id": node,
            "kind": kind,
            "value": value,
            "label": label or value,
        }
        return node

    task_node = add_node("task", task_id)
    dataset_node = add_node("dataset", dataset_type)
    layer_node = add_node("memory_layer", KNOWLEDGE_LAYER)
    edges.append(
        _edge(
            "memory_layer",
            f"{task_id}:{KNOWLEDGE_LAYER}",
            [task_node, dataset_node, layer_node],
        )
    )
    question_node = (
        add_node(
            "question", f"{task_id}:{stable_id(question, length=8)}", question[:96]
        )
        if question
        else ""
    )
    membership_nodes = [dataset_node, task_node, question_node]
    if group_id:
        membership_nodes.append(add_node("group", group_id))
    if turn_id:
        membership_nodes.append(
            add_node("turn", f"{group_id}:{turn_id}" if group_id else turn_id, turn_id)
        )
    edges.append(
        _edge("context_task_membership", f"{task_id}:membership", membership_nodes)
    )

    for unit in units:
        context_node = add_node("context", unit["context_id"], unit["text"][:96])
        edges.append(
            _edge(
                "task_context",
                f"{unit['context_id']}:task",
                [task_node, question_node, context_node, layer_node],
            )
        )

        topic_nodes = [context_node]
        if unit.get("title"):
            topic_nodes.append(add_node("topic", unit["title"]))
        if unit.get("section"):
            topic_nodes.append(add_node("section", unit["section"]))
        if len(topic_nodes) > 1:
            edges.append(
                _edge("context_topic", f"{unit['context_id']}:topic", topic_nodes)
            )

        entity_nodes = [context_node]
        for entity in _extract_entities(
            unit.get("title", ""), unit.get("section", ""), unit.get("text", "")
        ):
            kind = "date" if any(char.isdigit() for char in entity) else "entity"
            entity_nodes.append(add_node(kind, entity))
        if len(entity_nodes) > 1:
            edges.append(
                _edge(
                    "context_entities", f"{unit['context_id']}:entities", entity_nodes
                )
            )

        keyword_nodes = [context_node]
        for token in limited_tokens(
            " ".join(
                [unit.get("title", ""), unit.get("section", ""), unit.get("text", "")]
            ),
            limit=18,
        ):
            keyword_nodes.append(add_node("keyword", token))
        if len(keyword_nodes) > 1:
            edges.append(
                _edge(
                    "context_keywords", f"{unit['context_id']}:keywords", keyword_nodes
                )
            )

        edges.append(
            _edge(
                "candidate_evidence",
                f"{unit['context_id']}:candidate",
                [question_node, context_node, *topic_nodes[1:]],
            )
        )

    return {
        "graph_type": "context_hypergraph",
        "memory_layer": KNOWLEDGE_LAYER,
        "memory_layers": [KNOWLEDGE_LAYER],
        "dataset_type": dataset_type,
        "task_id": task_id,
        "nodes": list(nodes.values()),
        "hyperedges": edges,
        "context_units": units,
        "eval_labels": {
            "gold_fields_present": sorted(
                key for key in ANSWER_LIKE_FIELDS if key in task_meta
            ),
            "supporting_fact_count": len(task_meta.get("supporting_facts", []) or []),
        },
    }


def _score_unit(
    unit: dict[str, Any],
    query_tokens: set[str],
    focus_tokens: set[str],
    query_entities: set[str],
) -> tuple[float, list[str], list[str]]:
    unit_tokens, unit_entities, _ = _context_features(
        unit.get("title", ""), unit.get("section", ""), unit.get("text", "")
    )
    combined = query_tokens | focus_tokens
    matched = sorted(unit_tokens & combined)
    title_overlap = _tokens(unit.get("title", "")) & combined
    section_overlap = _tokens(unit.get("section", "")) & combined
    matched_entities = sorted(unit_entities & query_entities)

    score = float(len(matched))
    score += 2.0 * len(title_overlap)
    score += 2.0 * len(section_overlap)
    score += 4.0 * len(matched_entities)
    if unit.get("kind") in {"supporting_passage", "tool_task_context"}:
        score += 1.0
    return score, matched, matched_entities


def _answer_signal_score(question_tokens: set[str], text: str) -> float:
    """按问题要求的属性奖励可能承载答案关系的句子。

    这里只读取问题形式和候选正文，不读取标准答案或 supporting_facts。
    它用于避免多跳桥实体的介绍句占满 top-k，而真正的属性句排在下一位。
    """

    lowered = _text(text).lower()
    score = 0.0
    if {"position", "role", "office", "job"} & question_tokens and re.search(
        r"\b(?:served|appointed|named|held|elected|became|minister|ambassador|"
        r"president|director|secretary|protocol)\b",
        lowered,
    ):
        # 需要跨越一次实体桥接后仍能抵消最终词面重排对片名词元的偏好。
        score += 12.0
    if {"when", "date", "year"} & question_tokens and re.search(
        r"\b(?:1[5-9]|20)\d{2}\b", lowered
    ):
        score += 4.0
    if {"where", "location", "country", "city", "place"} & question_tokens and re.search(
        r"\b(?:located|situated|based|born|in|at|near)\b", lowered
    ):
        score += 3.0
    if {"many", "much", "number", "percentage"} & question_tokens and re.search(
        r"\b\d+(?:\.\d+)?%?\b", lowered
    ):
        score += 3.0
    return score


def search_context(
    task_meta: dict[str, Any],
    query: str | None = None,
    *,
    dataset_type: str | None = None,
    planner_focus: list[str] | None = None,
    top_k: int = 4,
    exclude_kinds: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Search dataset context through a lightweight context hypergraph.

    This is the only public context API intended for agent integration. It
    normalizes dataset-specific fields, builds a per-task context hypergraph,
    and returns compact evidence hits with matched nodes and hyperedges.
    Gold answers, expected results, and reference code are never used as online
    retrieval context. Structured TopiOCQA documents use overlapping, at-most
    three-sentence windows. `exclude_kinds` filters candidates before ranking.
    """

    from .prepared_knowledge import configured_knowledge
    prepared = configured_knowledge()
    if prepared is not None:
        return prepared.search_graph(query or _text(task_meta.get("question")), top_k)
    if top_k <= 0:
        return []
    resolved_dataset = _dataset_name(task_meta, dataset_type)
    # Do not retain evaluation labels or unrelated execution traces in cache.
    graph_meta = {
        key: task_meta[key]
        for key in (
            "group_id",
            "chain_id",
            "context",
            "conversation_context",
            "supporting_passage",
            "topic",
            "topic_section",
            "task_type",
            "inputs",
            "allowed_imports",
            "tags",
            "evidence_scope",
        )
        if key in task_meta
    }
    # 工具任务本身是知识来源；一般问答的问题与轮次不是知识图的一部分。
    if resolved_dataset == "codeact":
        graph_meta["task"] = task_meta.get("task", "")
    source_key = stable_id(json.dumps(graph_meta, ensure_ascii=False, sort_keys=True), length=24)
    graph_meta["task_id"] = "source-" + source_key
    payload = json.dumps(graph_meta, ensure_ascii=False, sort_keys=True)
    graph = (
        _cached_context_graph(payload, resolved_dataset)
        if len(payload) <= 262144
        else _build_context_hypergraph(graph_meta, resolved_dataset)
    )
    units = [
        unit
        for unit in graph["context_units"]
        if unit["kind"] not in (exclude_kinds or ())
    ]
    requested_tokens = question_content_tokens(_text(task_meta.get("question")))
    query_text = " ".join(
        [_text(task_meta.get("question")), _text(task_meta.get("task")), _text(query)]
    )
    query_tokens = _tokens(query_text)
    focus_tokens = _tokens(" ".join(planner_focus or []))
    query_entities = set(
        extract_canonical_entities(
            query_text,
            " ".join(planner_focus or []),
        )
    )

    edges_by_context, unit_entities, unit_text_entities, entity_contexts = (
        graph["lookup"] if "lookup" in graph else _context_lookup(graph)
    )

    scored_units: list[dict[str, Any]] = []
    for unit in units:
        score, matched_tokens, matched_entities = _score_unit(
            unit,
            query_tokens,
            focus_tokens,
            query_entities,
        )
        score += 4.0 * len(requested_tokens & _tokens(unit.get("text", "")))
        answer_signal_score = _answer_signal_score(
            _tokens(_text(task_meta.get("question"))), unit.get("text", "")
        )
        scored_units.append(
            {
                "unit": unit,
                "base_score": score,
                "matched_tokens": matched_tokens,
                "matched_entities": matched_entities,
                "answer_signal_score": answer_signal_score,
            }
        )

    seed_count = max(2, min(4, top_k))
    seeds = sorted(
        (item for item in scored_units if item["base_score"] > 0),
        key=lambda item: (-item["base_score"], item["unit"]["context_id"]),
    )[:seed_count]
    bridge_sources: dict[str, list[tuple[str, float]]] = {}
    document_entities = {canonical_entity(unit.get("title", "")) for unit in units}
    for seed in seeds:
        context_id = seed["unit"]["context_id"]
        title_entity = canonical_entity(seed["unit"].get("title", ""))
        for entity in unit_text_entities.get(context_id, set()):
            overlaps_query = any(
                _entity_overlap(entity, query_entity)
                for query_entity in query_entities
            )
            overlaps_title = bool(
                title_entity
                and (
                    _entity_overlap(entity, title_entity)
                )
            )
            if (
                overlaps_query
                or overlaps_title
                or (len(entity.split()) < 2 and entity not in document_entities)
                or len(entity_contexts.get(entity, set())) < 2
            ):
                continue
            bridge_sources.setdefault(entity, []).append(
                (context_id, float(seed["base_score"]))
            )

    hits: list[dict[str, Any]] = []
    for item in scored_units:
        unit = item["unit"]
        context_id = unit["context_id"]
        candidate_bridges = sorted(
            unit_entities.get(context_id, set()) & set(bridge_sources)
        )
        expansion_score = 0.0
        bridge_paths: list[tuple[str, str]] = []
        title_entity = canonical_entity(unit.get("title", ""))
        for entity in candidate_bridges:
            sources = [
                source for source in bridge_sources[entity] if source[0] != context_id
            ]
            if not sources:
                continue
            source_id, source_score = max(sources, key=lambda source: source[1])
            expansion_score += source_score * 0.65
            if entity == title_entity:
                expansion_score += 4.0
            bridge_paths.append((entity, source_id))
        title_grounded = bool(
            _tokens(unit.get("title", "")) & (query_tokens | focus_tokens)
        )
        signal_grounded = bool(
            bridge_paths or item["matched_entities"] or title_grounded
        )
        applied_answer_signal = float(item["answer_signal_score"]) * (
            1.0 if signal_grounded else 0.25
        )
        score = float(item["base_score"]) + expansion_score + applied_answer_signal
        if score <= 0:
            continue
        matched_tokens = item["matched_tokens"]
        matched_entities = item["matched_entities"]
        context_node = node_id("context", unit["context_id"])
        matched_edges = edges_by_context.get(context_node, [])[:6]
        matched_nodes = [
            context_node,
            *[node_id("keyword", token) for token in matched_tokens[:8]],
        ]
        matched_nodes.extend(
            node_id("entity", entity) for entity in matched_entities[:6]
        )
        matched_nodes.extend(
            node_id("entity", entity) for entity, _ in bridge_paths[:4]
        )
        if unit.get("title") and _tokens(unit["title"]) & (query_tokens | focus_tokens):
            matched_nodes.append(node_id("topic", unit["title"]))
        if unit.get("section") and _tokens(unit["section"]) & (
            query_tokens | focus_tokens
        ):
            matched_nodes.append(node_id("section", unit["section"]))
        hits.append(
            {
                "context_id": unit["context_id"],
                "memory_layer": KNOWLEDGE_LAYER,
                "dataset_type": resolved_dataset,
                "kind": unit["kind"],
                "title": unit.get("title", ""),
                "section": unit.get("section", ""),
                "text": unit["text"],
                "metadata": unit.get("metadata", {}),
                "context_score": round(score, 4),
                "lexical_score": round(float(item["base_score"]), 4),
                "graph_expansion_score": round(expansion_score, 4),
                "answer_signal_score": round(applied_answer_signal, 4),
                "raw_answer_signal_score": round(
                    float(item["answer_signal_score"]), 4
                ),
                "bridge_entities": [entity for entity, _ in bridge_paths],
                "bridge_source_context_ids": list(
                    dict.fromkeys(source_id for _, source_id in bridge_paths)
                ),
                "matched_nodes": list(dict.fromkeys(matched_nodes))[:10],
                "matched_hyperedges": [edge["hyperedge_id"] for edge in matched_edges],
                "reasoning_path": [
                    {
                        "query_node": node_id(
                            "entity" if token in matched_entities else "keyword",
                            token,
                        ),
                        "hyperedge_id": matched_edges[0]["hyperedge_id"]
                        if matched_edges
                        else "",
                        "context_id": unit["context_id"],
                    }
                    for token in [*matched_entities[:2], *matched_tokens[:3]][:3]
                ]
                + [
                    {
                        "query_node": node_id("context", source_context_id),
                        "via_entity": node_id("entity", entity),
                        "context_id": unit["context_id"],
                    }
                    for entity, source_context_id in bridge_paths[:2]
                ],
            }
        )

    ranked = sorted(hits, key=lambda hit: (-hit["context_score"], hit["context_id"]))
    return deepcopy(select_connected_hits(ranked, top_k))
