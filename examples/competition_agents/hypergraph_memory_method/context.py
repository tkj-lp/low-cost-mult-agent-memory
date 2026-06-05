from __future__ import annotations

import re
from typing import Any

from .text import limited_tokens, node_id, stable_id


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
    value = dataset_type or task_meta.get("dataset_type") or task_meta.get("source_dataset") or ""
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
        "dataset_type": dataset_type,
        "kind": kind,
        "title": title,
        "section": section,
        "text": text.strip(),
        "metadata": metadata or {},
    }


def _normalize_hotpot(task_meta: dict[str, Any], dataset_type: str, task_id: str) -> list[dict[str, Any]]:
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


def _normalize_topiocqa(task_meta: dict[str, Any], dataset_type: str, task_id: str) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    history = task_meta.get("conversation_context", []) or task_meta.get("context", []) or []
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
    return units


def _normalize_codeact(task_meta: dict[str, Any], dataset_type: str, task_id: str) -> list[dict[str, Any]]:
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


def _normalize_generic(task_meta: dict[str, Any], dataset_type: str, task_id: str) -> list[dict[str, Any]]:
    units = _normalize_hotpot(task_meta, dataset_type, task_id)
    if units:
        return units
    passage = _text(task_meta.get("supporting_passage"))
    if passage:
        return _normalize_topiocqa(task_meta, dataset_type, task_id)
    return []


def _context_units(task_meta: dict[str, Any], dataset_type: str, task_id: str) -> list[dict[str, Any]]:
    if dataset_type in {"hotpot", "hotpot_continuous"}:
        return _normalize_hotpot(task_meta, dataset_type, task_id)
    if dataset_type == "topiocqa":
        return _normalize_topiocqa(task_meta, dataset_type, task_id)
    if dataset_type == "codeact":
        return _normalize_codeact(task_meta, dataset_type, task_id)
    return _normalize_generic(task_meta, dataset_type, task_id)


def _extract_entities(*texts: str, limit: int = 14) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    patterns = [
        r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b(?:[A-Z][a-zA-Z0-9'\-]*(?:\s+|$)){1,5}",
    ]
    for text in texts:
        for pattern in patterns:
            for match in re.findall(pattern, text):
                value = " ".join(match.split())
                key = value.lower()
                if len(value) < 3 or key in seen or key in {"the", "and"}:
                    continue
                seen.add(key)
                found.append(value)
                if len(found) >= limit:
                    return found
    return found


def _edge(edge_type: str, edge_key: str, nodes: list[str]) -> dict[str, Any]:
    return {
        "hyperedge_id": node_id("context_hyperedge", edge_key),
        "type": edge_type,
        "nodes": list(dict.fromkeys(node for node in nodes if node)),
    }


def _build_context_hypergraph(task_meta: dict[str, Any], dataset_type: str) -> dict[str, Any]:
    task_id = _task_id(task_meta)
    question = _text(task_meta.get("question") or task_meta.get("task"))
    group_id = _text(task_meta.get("group_id") or task_meta.get("chain_id"))
    turn_id = _text(task_meta.get("turn_id") or task_meta.get("turn"))
    units = _context_units(task_meta, dataset_type, task_id)

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def add_node(kind: str, value: str, label: str | None = None) -> str:
        node = node_id(kind, value)
        nodes[node] = {"id": node, "kind": kind, "value": value, "label": label or value}
        return node

    task_node = add_node("task", task_id)
    dataset_node = add_node("dataset", dataset_type)
    question_node = add_node("question", f"{task_id}:{stable_id(question, length=8)}", question[:96]) if question else ""
    membership_nodes = [dataset_node, task_node, question_node]
    if group_id:
        membership_nodes.append(add_node("group", group_id))
    if turn_id:
        membership_nodes.append(add_node("turn", f"{group_id}:{turn_id}" if group_id else turn_id, turn_id))
    edges.append(_edge("context_task_membership", f"{task_id}:membership", membership_nodes))

    for unit in units:
        context_node = add_node("context", unit["context_id"], unit["text"][:96])
        edges.append(_edge("task_context", f"{unit['context_id']}:task", [task_node, question_node, context_node]))

        topic_nodes = [context_node]
        if unit.get("title"):
            topic_nodes.append(add_node("topic", unit["title"]))
        if unit.get("section"):
            topic_nodes.append(add_node("section", unit["section"]))
        if len(topic_nodes) > 1:
            edges.append(_edge("context_topic", f"{unit['context_id']}:topic", topic_nodes))

        entity_nodes = [context_node]
        for entity in _extract_entities(unit.get("title", ""), unit.get("section", ""), unit.get("text", "")):
            kind = "date" if any(char.isdigit() for char in entity) else "entity"
            entity_nodes.append(add_node(kind, entity))
        if len(entity_nodes) > 1:
            edges.append(_edge("context_entities", f"{unit['context_id']}:entities", entity_nodes))

        keyword_nodes = [context_node]
        for token in limited_tokens(" ".join([unit.get("title", ""), unit.get("section", ""), unit.get("text", "")]), limit=18):
            keyword_nodes.append(add_node("keyword", token))
        if len(keyword_nodes) > 1:
            edges.append(_edge("context_keywords", f"{unit['context_id']}:keywords", keyword_nodes))

        edges.append(_edge("candidate_evidence", f"{unit['context_id']}:candidate", [question_node, context_node, *topic_nodes[1:]]))

    return {
        "graph_type": "context_hypergraph",
        "dataset_type": dataset_type,
        "task_id": task_id,
        "nodes": list(nodes.values()),
        "hyperedges": edges,
        "context_units": units,
        "eval_labels": {
            "gold_fields_present": sorted(key for key in ANSWER_LIKE_FIELDS if key in task_meta),
            "supporting_fact_count": len(task_meta.get("supporting_facts", []) or []),
        },
    }


def _score_unit(unit: dict[str, Any], query_tokens: set[str], focus_tokens: set[str]) -> tuple[float, list[str]]:
    unit_tokens = _tokens(unit.get("title", ""), unit.get("section", ""), unit.get("text", ""))
    combined = query_tokens | focus_tokens
    matched = sorted(unit_tokens & combined)
    title_overlap = _tokens(unit.get("title", "")) & combined
    section_overlap = _tokens(unit.get("section", "")) & combined
    text_lower = _text(unit.get("text")).lower()

    score = float(len(matched))
    score += 2.0 * len(title_overlap)
    score += 2.0 * len(section_overlap)
    if unit.get("kind") in {"supporting_passage", "tool_task_context"}:
        score += 1.0
    for marker in ("served as", "was named", "held", "position", "government", "protocol", "ambassador", "diplomat"):
        if marker in text_lower:
            score += 2.0
    if "chief" in text_lower and "protocol" in text_lower:
        score += 5.0
    if "sequel" in text_lower or "retitled" in text_lower:
        score -= 3.0
    return score, matched


def search_context(
    task_meta: dict[str, Any],
    query: str | None = None,
    *,
    dataset_type: str | None = None,
    planner_focus: list[str] | None = None,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Search dataset context through a lightweight context hypergraph.

    This is the only public context API intended for agent integration. It
    normalizes dataset-specific fields, builds a per-task context hypergraph,
    and returns compact evidence hits with matched nodes and hyperedges.
    Gold answers, expected results, and reference code are never used as online
    retrieval context.
    """

    resolved_dataset = _dataset_name(task_meta, dataset_type)
    graph = _build_context_hypergraph(task_meta, resolved_dataset)
    query_text = " ".join([_text(task_meta.get("question")), _text(task_meta.get("task")), _text(query)])
    query_tokens = _tokens(query_text)
    focus_tokens = _tokens(" ".join(planner_focus or []))

    edges_by_context: dict[str, list[dict[str, Any]]] = {}
    for edge in graph["hyperedges"]:
        for node in edge.get("nodes", []):
            if node.startswith("context:"):
                edges_by_context.setdefault(node, []).append(edge)

    hits: list[dict[str, Any]] = []
    for unit in graph["context_units"]:
        score, matched_tokens = _score_unit(unit, query_tokens, focus_tokens)
        if score <= 0:
            continue
        context_node = node_id("context", unit["context_id"])
        matched_edges = edges_by_context.get(context_node, [])[:6]
        matched_nodes = [context_node, *[node_id("keyword", token) for token in matched_tokens[:8]]]
        if unit.get("title") and _tokens(unit["title"]) & (query_tokens | focus_tokens):
            matched_nodes.append(node_id("topic", unit["title"]))
        if unit.get("section") and _tokens(unit["section"]) & (query_tokens | focus_tokens):
            matched_nodes.append(node_id("section", unit["section"]))
        hits.append(
            {
                "context_id": unit["context_id"],
                "dataset_type": resolved_dataset,
                "kind": unit["kind"],
                "title": unit.get("title", ""),
                "section": unit.get("section", ""),
                "text": unit["text"],
                "context_score": round(score, 4),
                "matched_nodes": list(dict.fromkeys(matched_nodes))[:10],
                "matched_hyperedges": [edge["hyperedge_id"] for edge in matched_edges],
                "reasoning_path": [
                    {
                        "query_node": node_id("keyword", token),
                        "hyperedge_id": matched_edges[0]["hyperedge_id"] if matched_edges else "",
                        "context_id": unit["context_id"],
                    }
                    for token in matched_tokens[:3]
                ],
            }
        )

    return sorted(hits, key=lambda hit: (-hit["context_score"], hit["context_id"]))[:top_k]
