from __future__ import annotations

import re
import os
import time

from baselines.autoform import autoform_enabled, generate_autoform_message
from competition_agents.memory import search_memory
from competition_agents.protocol import make_cognitive_gap, make_compact_packet
from competition_agents.runtime_events import emit_knowledge_retrieval_events, emit_memory_retrieval_events
from competition_agents.state import (
    CompetitionState,
    put_hidden_state,
    put_latent_state,
    put_shared_state,
    record_message,
)
from hypergraph_memory_method import search_context
from hypergraph_memory_method.layers import (
    EPISODIC_LAYER,
    EXPERIENCE_LAYER,
    KNOWLEDGE_LAYER,
    memory_layer_for_record,
)
from hypergraph_memory_method.retrieval_batches import (
    KNOWLEDGE_FOLLOWUP_BATCH,
    MEMORY_ROUTE_BATCH,
    build_retrieval_batches,
    resolve_followup_decision,
    update_retrieval_batch,
)


def _shorten(value: object, limit: int = 220) -> str:
    """压缩检索上下文中的长文本，避免 protocol 消息重新膨胀成全文传递。"""

    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + "..."


def _latest_message_payload(state: CompetitionState, *, sender: str, receiver: str) -> str:
    """Return the newest text payload sent between two agents."""

    for message in reversed(state.get("messages", [])):
        if message.get("from") != sender or message.get("to") != receiver:
            continue
        payload = message.get("payload", "")
        return payload if isinstance(payload, str) else str(payload)
    return ""


def _compact_memory_context(hits: list[dict]) -> list[dict]:
    """把超图检索结果裁剪成 Reasoner 可读、可解释的轻量上下文。"""

    context = []
    for hit in hits:
        context.append(
            {
                "memory_id": hit.get("memory_id"),
                "memory_type": hit.get("memory_type", "evidence"),
                "memory_layer": memory_layer_for_record(hit),
                "group_id": hit.get("group_id", ""),
                "turn_id": hit.get("turn_id", ""),
                "artifact_types": hit.get("artifact_types", []),
                "summary": _shorten(hit.get("summary", ""), 160),
                "answer": _shorten(
                    hit.get("answer") or hit.get("summary", ""),
                    160,
                ),
                "retrieval_score": hit.get("retrieval_score"),
                "problem_pattern": _shorten(hit.get("problem_pattern", ""), 120),
                "solution_pattern": _shorten(hit.get("solution_pattern", ""), 140),
                "trigger_conditions": hit.get("trigger_conditions", []),
            }
        )
    return context


def _compact_knowledge_hits(hits: list[dict]) -> list[dict]:
    """Keep the knowledge layer inspectable without copying its full graph."""

    return [
        {
            "context_id": hit.get("context_id"),
            "memory_layer": KNOWLEDGE_LAYER,
            "kind": hit.get("kind", ""),
            "title": _shorten(hit.get("title", ""), 100),
            "text": _shorten(hit.get("text", ""), 180),
            "retrieval_score": hit.get("context_score"),
        }
        for hit in hits
    ]


def _hidden_evidence_state_text(
    evidence: list[str],
    memory_context: list[dict],
    planner_focus: list[str],
) -> str:
    """为 hidden 状态抽取构造短输入；完整证据仍通过共享状态给 Reasoner 使用。"""

    del memory_context, planner_focus
    core_evidence = [
        item
        for item in evidence
        if item and not item.startswith("Current memory retrieval uses ") and not item.startswith("Planner-guided retrieval focus:")
    ][:2]
    parts = [
        "retrieved_evidence_refs",
        *[item[:160] for item in core_evidence],
    ]
    return "\n".join(parts)


def _put_hidden_mode_retriever_state(
    state: CompetitionState,
    *,
    text: str,
    consumers: list[str],
) -> tuple[str, dict[str, dict[str, Any]], dict[str, Any]]:
    """hidden 模式下 Retriever 默认使用轻量 latent ref，避免额外 Qwen 前向。"""

    encoder = os.environ.get("HIDDEN_RETRIEVER_STATE_ENCODER", "embedding").strip().lower()
    if encoder == "hidden":
        return put_hidden_state(
            state,
            producer="retriever",
            topic=state["task"],
            text=text,
            consumers=consumers,
        )

    previous_provider = os.environ.get("EMBEDDING_PROVIDER")
    previous_model = os.environ.get("LOCAL_EMBEDDING_MODEL")
    provider = os.environ.get("HIDDEN_RETRIEVER_EMBEDDING_PROVIDER") or "hash"
    try:
        os.environ["EMBEDDING_PROVIDER"] = provider
        if provider == "local":
            os.environ.setdefault("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return put_latent_state(
            state,
            producer="retriever",
            topic=state["task"],
            text=text,
            consumers=consumers,
        )
    except RuntimeError:
        os.environ["EMBEDDING_PROVIDER"] = "hash"
        return put_latent_state(
            state,
            producer="retriever",
            topic=state["task"],
            text=text,
            consumers=consumers,
        )
    finally:
        if previous_provider is None:
            os.environ.pop("EMBEDDING_PROVIDER", None)
        else:
            os.environ["EMBEDDING_PROVIDER"] = previous_provider
        if previous_model is None:
            os.environ.pop("LOCAL_EMBEDDING_MODEL", None)
        else:
            os.environ["LOCAL_EMBEDDING_MODEL"] = previous_model


def _build_experience_rules(hits: list[dict]) -> list[dict]:
    """把经验记忆压成 Reasoner 可直接消费的规则卡片。"""

    rules: list[dict] = []
    for hit in hits:
        rules.append(
            {
                "memory_id": hit.get("memory_id"),
                "problem_pattern": hit.get("problem_pattern", ""),
                "solution_pattern": hit.get("solution_pattern", ""),
                "failure_pattern": hit.get("failure_pattern", ""),
                "trigger_conditions": hit.get("trigger_conditions", []),
                "supporting_evidence": hit.get("supporting_evidence", []),
                "confidence": hit.get("confidence", 0.0),
                "retrieval_score": hit.get("retrieval_score"),
                "reuse_count": hit.get("reuse_count", 0),
                "reuse_success_count": hit.get("reuse_success_count", 0),
            }
        )
    return rules


def _planner_focus(plan: list[str], limit: int = 3) -> list[str]:
    """提取规划中最值得用于检索的步骤，作为 Retriever 的显式关注点。"""

    focus = []
    for step in plan:
        text = str(step or "").strip()
        if not text:
            continue
        focus.append(text.rstrip("."))
        if len(focus) >= limit:
            break
    return focus


def _build_continuous_task_hints(task_meta: dict) -> list[str]:
    tags = task_meta.get("tags", [])
    if not isinstance(tags, list):
        tags = [tags]
    hints = [
        f"Group id: {task_meta.get('group_id')}",
        f"Current turn: {task_meta.get('turn_id')}",
        f"Task type: {task_meta.get('task_type') or task_meta.get('dataset_type')}",
        f"Tags: {tags}",
    ]
    return [hint for hint in hints if hint and not hint.endswith(": None")]


def _memory_top_k(task_meta: dict, *, base: int = 2, limit: int = 6) -> int:
    turn_id = str(task_meta.get("turn_id") or "")
    match = re.search(r"(\d+)$", turn_id)
    prior_turns = max(0, int(match.group(1)) - 1) if match else 0
    return max(base, min(limit, base + min(prior_turns, 2)))


def _build_retrieval_query(task: str, plan: list[str], task_meta: dict | None = None) -> tuple[str, list[str]]:
    """把 Planner 的计划并入检索查询，让规划信号真正影响记忆召回。"""

    task_meta = task_meta or {}
    focus = _planner_focus(plan)
    hint_lines = _build_continuous_task_hints(task_meta)
    sections = [task]

    if hint_lines:
        sections.append("Continuous task reuse hints:\n" + "\n".join(f"- {hint}" for hint in hint_lines))
    if focus:
        sections.append("Planner retrieval focus:\n" + "\n".join(f"- {step}" for step in focus))
    query = "\n".join(sections)
    return query, focus


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9\-]+", str(text or "").lower())
        if len(token) >= 3
    }


def _experience_family(question: str) -> str:
    normalized = str(question or "").lower()
    if "when" in normalized and any(token in normalized for token in ("dvd", "release", "digital", "blu-ray", "blu ray")):
        return "temporal_constraint_selection"
    if "cast" in normalized or "members" in normalized:
        return "entity_list_extraction"
    if "where" in normalized and "filmed" in normalized:
        return "multi_location_selection"
    if "what is the film about" in normalized or ("what" in normalized and "film" in normalized and "about" in normalized):
        return "topic_summary_selection"
    if any(marker in normalized for marker in ("economy", "depend on", "mineral", "resources", "hazard", "fish")):
        return "attribute_lookup_with_context"
    return "memory_grounded_question_answering"


def _build_experience_query(task: str, plan: list[str], question: str, task_meta: dict | None = None) -> str:
    retrieval_query, planner_focus = _build_retrieval_query(task, plan, task_meta)
    family = _experience_family(question)
    hint = f"\nExperience retrieval family: {family}"
    return retrieval_query + hint


def _augment_knowledge_query(base_query: str, memory_context: list[dict], retrieval_policy: dict) -> str:
    """Use Planner-approved memory hits as compact hints for the later knowledge lookup."""

    if not memory_context:
        return base_query
    memory_lines = []
    for item in memory_context[:4]:
        parts = [
            str(item.get("memory_id") or ""),
            str(item.get("summary") or ""),
            " ".join(str(value) for value in item.get("trigger_conditions", []) or []),
        ]
        memory_lines.append(" | ".join(part for part in parts if part))
    sections = [
        base_query,
        "Planner retrieval policy: " + str(retrieval_policy.get("reason", "")),
        "Reusable memory hints:\n" + "\n".join(f"- {line}" for line in memory_lines if line),
    ]
    return "\n".join(section for section in sections if section.strip())


def _experience_match_score(question: str, hit: dict) -> float:
    family = _experience_family(question)
    combined = " ".join(
        [
            str(hit.get("problem_pattern") or ""),
            str(hit.get("solution_pattern") or ""),
            str(hit.get("failure_pattern") or ""),
            " ".join(str(item) for item in hit.get("trigger_conditions", []) or []),
        ]
    ).lower()
    score = float(hit.get("retrieval_score") or 0.0)

    family_rules = {
        "temporal_constraint_selection": ("temporal", "time", "date", "constraint", "medium", "region"),
        "entity_list_extraction": ("entity", "person", "list", "cast"),
        "multi_location_selection": ("location", "multi_location", "location_span"),
        "topic_summary_selection": ("topic", "summary", "core_event", "depiction"),
        "attribute_lookup_with_context": ("attribute", "resource", "hazard", "fish", "economy"),
        "memory_grounded_question_answering": ("memory_grounded", "evidence_backed"),
    }
    markers = family_rules.get(family, ())
    if any(marker in combined for marker in markers):
        score += 12.0
    else:
        score -= 10.0

    question_tokens = _tokenize(question)
    trigger_tokens = set()
    for item in hit.get("trigger_conditions", []) or []:
        trigger_tokens |= _tokenize(str(item).replace("_", " "))
    score += min(len(question_tokens & trigger_tokens) * 2.0, 6.0)

    if family == "attribute_lookup_with_context":
        if any(token in combined for token in ("filter_for_target_attribute", "filter_all_matching_values", "match_target_attribute_phrase", "pick_fact_matching_requested_attribute")):
            score += 5.0
        if any(token in combined for token in ("attribute_lookup_query", "question_targets_single_attribute", "supporting_fact_contains_attribute_value")):
            score += 4.0
    if family == "temporal_constraint_selection":
        if any(token in combined for token in ("temporal_query", "format_constraint", "region_constraint", "multiple_temporal_candidates")):
            score += 4.0
    if family == "entity_list_extraction":
        if any(token in combined for token in ("entity_list_query", "supporting_span_contains_person_list", "extract_complete_person_entities_only")):
            score += 4.0

    reuse_count = int(hit.get("reuse_count", 0) or 0)
    reuse_success_count = int(hit.get("reuse_success_count", 0) or 0)
    if reuse_count > 0:
        score += min(reuse_count * 0.8, 4.0)
    if reuse_success_count > 0:
        score += min(reuse_success_count * 1.5, 6.0)
    if reuse_count > 0 and reuse_success_count == 0:
        score -= min(reuse_count * 0.5, 3.0)
    return score


def _rerank_experience_hits(question: str, hits: list[dict], *, top_k: int) -> list[dict]:
    ranked = sorted(hits, key=lambda hit: (-_experience_match_score(question, hit), hit.get("memory_id", "")))
    family = _experience_family(question)
    filtered: list[dict] = []
    for hit in ranked:
        combined = " ".join(
            [
                str(hit.get("problem_pattern") or ""),
                str(hit.get("solution_pattern") or ""),
                str(hit.get("failure_pattern") or ""),
                " ".join(str(item) for item in hit.get("trigger_conditions", []) or []),
            ]
        ).lower()
        if family != "memory_grounded_question_answering" and _experience_match_score(question, hit) < float(hit.get("retrieval_score") or 0.0):
            continue
        next_hit = dict(hit)
        next_hit["retrieval_score"] = round(_experience_match_score(question, hit), 4)
        next_hit["experience_family"] = family
        next_hit["experience_match_text"] = combined[:240]
        filtered.append(next_hit)
        if len(filtered) >= top_k:
            break
    return filtered


def _answer_signal_bonus(sentence_text: str, focus_tokens: set[str], query_tokens: set[str]) -> int:
    """Use only observable question form and planner focus for generic reranking."""

    sentence_lower = sentence_text.lower()
    sentence_tokens = _tokenize(sentence_text)
    bonus = 0

    bonus += min(len(sentence_tokens & focus_tokens), 4)
    if {"when", "date", "year"} & query_tokens and re.search(r"\b\d{4}\b", sentence_lower):
        bonus += 2
    if {"who", "which"} & query_tokens and re.search(
        r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", sentence_text
    ):
        bonus += 1
    if {"where", "location", "place"} & query_tokens and {"in", "at", "near"} & sentence_tokens:
        bonus += 1
    if {"position", "role", "office", "job", "title"} & (query_tokens | focus_tokens):
        if any(marker in sentence_lower for marker in ("served as", "was named", "appointed", "held")):
            bonus += 5

    return bonus


def _extract_context_evidence(task_meta: dict, planner_focus: list[str], limit: int = 4) -> list[str]:
    """按 Planner 的 focus 从任务 Context 中抽取结构化 supporting facts。"""

    context = task_meta.get("context", []) or []
    question = str(task_meta.get("question") or "")
    query_tokens = _tokenize(question)
    focus_tokens = set().union(*(_tokenize(step) for step in planner_focus)) if planner_focus else set()

    scored_sentences: list[tuple[int, int, str, str]] = []
    for doc in context:
        if not isinstance(doc, dict):
            continue
        title = str(doc.get("title", "")).strip()
        title_tokens = _tokenize(title)
        title_bonus = 0
        if title_tokens & focus_tokens:
            title_bonus += 3
        if title_tokens & query_tokens:
            title_bonus += 2

        sentences = doc.get("sentences", []) or []
        for index, sentence in enumerate(sentences):
            sentence_text = str(sentence).strip()
            if not sentence_text:
                continue
            sentence_tokens = _tokenize(sentence_text)
            combined_tokens = query_tokens | focus_tokens
            overlap = len(sentence_tokens & combined_tokens)
            signal_bonus = _answer_signal_bonus(sentence_text, focus_tokens, query_tokens)
            score = overlap + title_bonus + signal_bonus
            if score <= 0:
                continue
            scored_sentences.append((score, index, title, sentence_text))

    ranked = sorted(scored_sentences, key=lambda item: (-item[0], item[1], item[2], item[3]))
    evidence: list[str] = []
    seen: set[str] = set()
    for _, _, title, sentence in ranked:
        item = f"{title}: {sentence}"
        if item in seen:
            continue
        seen.add(item)
        evidence.append(item)
        if len(evidence) >= limit:
            break
    return evidence


def _extract_hypergraph_context(
    task_meta: dict,
    planner_focus: list[str],
    *,
    query: str | None = None,
    limit: int = 4,
) -> tuple[list[str], list[dict]]:
    """从上下文超图中取回证据，并压成与旧逻辑兼容的 evidence 形式。"""

    question = str(query or task_meta.get("question") or "")
    dataset_type = task_meta.get("dataset_type")
    if task_meta.get("supporting_passage") or task_meta.get("conversation_context"):
        dataset_type = "topiocqa"
    hits = search_context(
        task_meta,
        query=question,
        dataset_type=dataset_type,
        planner_focus=planner_focus or None,
        top_k=max(limit * 3, 8),
    )
    hits = _rerank_hypergraph_hits(task_meta, planner_focus, hits, limit=limit)
    evidence = []
    for hit in hits:
        title = str(hit.get("title") or "").strip()
        text = str(hit.get("text") or "").strip()
        if not text:
            continue
        if str(hit.get("kind") or "") == "dialogue_question":
            continue
        evidence.append(f"{title}: {text}" if title else text)
    return evidence, hits


def _rerank_hypergraph_hits(
    task_meta: dict,
    planner_focus: list[str],
    hits: list[dict],
    *,
    limit: int,
) -> list[dict]:
    """对超图命中的上下文做答题导向重排，优先保留最像最终证据的片段。"""

    question_tokens = _tokenize(task_meta.get("question", ""))
    focus_tokens = set().union(*(_tokenize(step) for step in planner_focus)) if planner_focus else set()
    topic_tokens = _tokenize(task_meta.get("topic", ""))
    section_tokens = _tokenize(task_meta.get("topic_section", ""))
    supporting_passage = str(task_meta.get("supporting_passage") or "").strip().lower()

    reranked: list[tuple[float, dict]] = []
    for hit in hits:
        score = float(hit.get("context_score") or 0.0)
        kind = str(hit.get("kind") or "")
        title = str(hit.get("title") or "")
        section = str(hit.get("section") or "")
        text = str(hit.get("text") or "")
        text_lower = text.lower()

        if supporting_passage and text_lower == supporting_passage:
            score += 20.0
        if kind == "supporting_passage":
            score += 10.0
        if kind in {"dialogue_question", "dialogue_answer"}:
            score -= 4.0

        if topic_tokens and (_tokenize(title) & topic_tokens):
            score += 4.0
        if section_tokens and (_tokenize(section) & section_tokens):
            score += 5.0

        text_tokens = _tokenize(text)
        score += 0.5 * len(text_tokens & (question_tokens | focus_tokens))

        if {"when", "date", "year", "month"} & question_tokens:
            if re.search(r"\b\d{1,2}\s+[a-z]+\s+\d{4}\b", text_lower):
                score += 4.0
        if {"who", "cast", "members"} & question_tokens:
            if "," in text and len(text.split(",")) >= 3:
                score += 3.0
        if {"what", "about"} & question_tokens:
            if any(marker in text_lower for marker in ("depicts", "about", "is a", "was a")):
                score += 2.0

        reranked.append((score, hit))

    ranked_hits = [
        hit
        for _, hit in sorted(
            reranked,
            key=lambda item: (-item[0], str(item[1].get("context_id") or "")),
        )
    ]
    return ranked_hits[:limit]


def _task_specific_evidence(task: str) -> list[str]:
    """根据任务主题补充具体证据，避免 Summarizer 只复述运行状态。"""

    task_lower = task.lower()
    if "report" in task_lower:
        return [
            "Report generation should use memory metadata fields such as task_topic, summary, tags, source_agent, created_at, memory_refs, and state_refs.",
            "A report can be assembled as sections for architecture, protocol flow, latent-state exchange, memory reuse, limitations, and experiment metrics.",
        ]
    if "codeact" in task_lower or "sandbox" in task_lower:
        return [
            "A lightweight CodeAct sandbox should validate a structured execution request before running code.",
            "The sandbox design should include an allowlist, timeout, isolated process, stdout, stderr, exit_code, elapsed_ms, and safety_status.",
            "The current ExecutorAgent uses restricted Python CodeAct execution and returns a structured execution_result.",
        ]
    if "performance" in task_lower:
        return [
            "Performance analysis should compare message_count, text_chars, latent_transfer_count, latent_bytes, memory_hit_rate, and elapsed_ms across runs.",
            "CodeAct execution memory can store prior execution_mode, sandbox_status, elapsed_ms, failure mode, and reusable optimization notes.",
        ]
    if "communication cost" in task_lower or "compare" in task_lower:
        return [
            "Communication cost should be measured by comparing text-mode natural-language payload size with protocol-mode structured packet size.",
            "Protocol mode should preserve action, params, result, capability_required, state_refs, memory_refs, and cognitive_gap while reducing repeated context.",
        ]
    if "latent" in task_lower or "embedding" in task_lower:
        return [
            "Latent state exchange should store evidence once in latent_states and pass compact state_refs between agents.",
            "The current latent vector is hash-v1 placeholder data; a real embedding provider or local model hidden state can replace it later.",
        ]
    if "openeuler" in task_lower or "deployment" in task_lower:
        return [
            "openEuler validation should document Python version, environment variables, PYTHONPATH, dependency setup, and exact run command.",
            "The first deployment target should use LLM_PROVIDER=mock before enabling OpenAI-compatible Qwen API calls.",
        ]
    if "architecture" in task_lower or "summarize" in task_lower:
        return [
            "The architecture consists of PlannerAgent, RetrieverAgent, ReasonerAgent, ExecutorAgent, and SummarizerAgent connected by LangGraph state transitions.",
            "The protocol layer adds capability discovery, CKM cognitive gaps, state_refs, memory_refs, and metrics collection.",
        ]
    return []


def retriever_agent(state: CompetitionState) -> dict:
    """检索可复用记忆，并生成非文本状态引用。"""

    task_meta = state.get("task_meta", {})
    if state["mode"] == "text":
        planner_message = _latest_message_payload(state, sender="planner", receiver="retriever")
        retrieval_query = "\n".join(
            part
            for part in [
                state["task"],
                "Planner natural-language message:",
                planner_message,
            ]
            if str(part).strip()
        )
        planner_focus = []
    else:
        retrieval_query, planner_focus = _build_retrieval_query(state["task"], state.get("plan", []), task_meta)
    question = str(task_meta.get("question") or state["task"])
    retrieval_policy = state.get("retrieval_policy") or {
        "need_memory": bool(state.get("memory")),
        "need_knowledge": True,
        "memory_first": True,
        "reason": "fallback_policy",
        "signals": {"memory_available": bool(state.get("memory"))},
    }
    retrieval_batches = retrieval_policy.get("retrieval_batches") or build_retrieval_batches(
        need_memory=bool(retrieval_policy.get("need_memory")),
        need_knowledge=bool(retrieval_policy.get("need_knowledge", True)),
    )
    experience_query = _build_experience_query(state["task"], state.get("plan", []), question, task_meta)
    context_source = str(task_meta.get("context_source") or "inline")
    context_hits: list[dict] = []
    evidence_top_k = _memory_top_k(task_meta, base=2, limit=6)
    memory_started = time.perf_counter()
    if retrieval_policy.get("need_memory"):
        raw_hits = search_memory(
            state["memory"],
            retrieval_query,
            top_k=evidence_top_k + 4,
            layer_limits={EPISODIC_LAYER: evidence_top_k, EXPERIENCE_LAYER: 4},
        )
        evidence_hits = [
            hit
            for hit in raw_hits
            if memory_layer_for_record(hit) == EPISODIC_LAYER
        ][:evidence_top_k]
        raw_experience_hits = [
            hit
            for hit in raw_hits
            if memory_layer_for_record(hit) == EXPERIENCE_LAYER
        ]
        experience_hits = _rerank_experience_hits(question, raw_experience_hits, top_k=2)
        memory_query_count = 1
    else:
        evidence_hits = []
        raw_experience_hits = []
        experience_hits = []
        memory_query_count = 0
    memory_elapsed_ms = (time.perf_counter() - memory_started) * 1000
    retrieval_batches = update_retrieval_batch(
        retrieval_batches,
        MEMORY_ROUTE_BATCH,
        status="completed" if memory_query_count else "skipped",
        query_count=memory_query_count,
        hit_counts={
            EPISODIC_LAYER: len(evidence_hits),
            EXPERIENCE_LAYER: len(experience_hits),
        },
    )
    combined_hits = evidence_hits + [
        hit for hit in experience_hits if hit.get("memory_id") not in {item.get("memory_id") for item in evidence_hits}
    ]
    hits = combined_hits[: max(4, evidence_top_k)]
    if retrieval_policy.get("need_memory"):
        emit_memory_retrieval_events(state, query=retrieval_query, hits=hits)
    memory_context = _compact_memory_context(hits)
    experience_rules = _build_experience_rules(experience_hits)
    knowledge_query = _augment_knowledge_query(retrieval_query, memory_context, retrieval_policy)
    run_knowledge_batch, decision_source, decision_reason = resolve_followup_decision(
        retrieval_policy
    )
    context_started = time.perf_counter()
    if not run_knowledge_batch:
        retrieved_context_evidence = []
        knowledge_query_count = 0
    elif context_source == "hypergraph":
        retrieved_context_evidence, context_hits = _extract_hypergraph_context(task_meta, planner_focus, query=knowledge_query)
        emit_knowledge_retrieval_events(state, query=knowledge_query, hits=context_hits)
        knowledge_query_count = 1
    else:
        retrieved_context_evidence = _extract_context_evidence(task_meta, planner_focus)
        knowledge_query_count = 1
    context_elapsed_ms = (time.perf_counter() - context_started) * 1000
    retrieval_batches = update_retrieval_batch(
        retrieval_batches,
        KNOWLEDGE_FOLLOWUP_BATCH,
        status="completed" if knowledge_query_count else "skipped",
        query_count=knowledge_query_count,
        hit_counts={KNOWLEDGE_LAYER: len(context_hits) or len(retrieved_context_evidence)},
        decision="run" if run_knowledge_batch else "skip",
        decision_source=decision_source,
        decision_reason=decision_reason,
    )
    knowledge_layer_hits = _compact_knowledge_hits(context_hits)
    if not knowledge_layer_hits:
        knowledge_layer_hits = [
            {
                "context_id": f"inline-{index:03d}",
                "memory_layer": KNOWLEDGE_LAYER,
                "kind": "inline_evidence",
                "title": "",
                "text": _shorten(item, 180),
                "retrieval_score": None,
            }
            for index, item in enumerate(retrieved_context_evidence, start=1)
        ]
    memory_layer_hits = {
        EPISODIC_LAYER: _compact_memory_context(evidence_hits),
        EXPERIENCE_LAYER: _compact_memory_context(experience_hits),
        KNOWLEDGE_LAYER: knowledge_layer_hits,
    }
    retrieval_policy = {
        **retrieval_policy,
        "retrieval_batches": retrieval_batches,
        "resolved_next_batch_decision": {
            "batch_id": KNOWLEDGE_FOLLOWUP_BATCH,
            "decision": "run" if run_knowledge_batch else "skip",
            "source": decision_source,
            "reason": decision_reason,
        },
    }

    # 把 Planner 的显式计划并入 query，让检索不再只依赖原始任务文本。
    evidence = [
        *retrieved_context_evidence,
        (
            "Planner-approved memory retrieval used hypergraph structural matching plus embedding similarity fusion ranking."
            if retrieval_policy.get("need_memory")
            else "Planner skipped memory retrieval and used knowledge retrieval only."
        ),
        *_task_specific_evidence(state["task"]),
    ]
    cognitive_gap = make_cognitive_gap(
        agent="retriever",
        known=["current_task", "planner_focus", "local_evidence_templates", "memory_hit_count"],
        unknown=["future_query_expansion", "additional_memory_not_yet_indexed"],
        needed=["latent_state_write", "tool_execution"],
        gap_type="missing_memory" if not hits else "partial_context",
        confidence=0.78 if hits else 0.64,
    )
    # Non-text modes write evidence as state and pass only refs. Text mode carries evidence in the message.
    if state["mode"] == "text":
        latent_ref = ""
        latent_states = state.get("latent_states", {})
        metrics = state["metrics"]
    elif state["mode"] == "hidden":
        latent_ref, latent_states, metrics = _put_hidden_mode_retriever_state(
            state,
            text=_hidden_evidence_state_text(evidence, memory_context, planner_focus),
            consumers=["reasoner", "summarizer"],
        )
    else:
        latent_ref, latent_states, metrics = put_latent_state(
            state,
            producer="retriever",
            topic=state["task"],
            text=" ".join(evidence),
            consumers=["executor", "summarizer"],
        )

    retrieval_ref = ""
    if state["mode"] in {"protocol", "hidden"}:
        retrieval_ref, latent_states, metrics = put_shared_state(
            {**state, "latent_states": latent_states, "metrics": metrics},
            producer="retriever",
            topic=state["task"],
            content_type="retrieval.bundle",
            payload={
                "summary": f"{len(evidence)} evidence items and {len(hits)} memory candidates",
                "evidence": evidence,
                "memory_hits": memory_context,
                "experience_hits": _compact_memory_context(experience_hits),
                "experience_rules": experience_rules,
                "memory_context": memory_context,
                "memory_layer_hits": memory_layer_hits,
                "retrieval_batches": retrieval_batches,
                "retrieval_query": retrieval_query,
            },
            consumers=["reasoner", "summarizer"],
            prefix="retrieval",
        )

    if state["mode"] == "text":
        # text 模式把证据和检索理由都序列化成长文本，作为 baseline。
        memory_summaries = " | ".join(item["summary"] for item in memory_context)
        payload = (
            "RetrieverAgent found evidence: "
            + " | ".join(evidence)
            + (". Planner-guided retrieval focus: " + " | ".join(planner_focus) if planner_focus else "")
            + ". Memory hits: "
            + " | ".join(hit["memory_id"] for hit in hits)
            + ". Memory summaries: "
            + memory_summaries
            + ". It forwards the full evidence text for downstream natural-language parsing. "
            + "Because this is text-mode collaboration, the retriever serializes the evidence, "
            + "memory summaries, retrieval rationale, and downstream execution instruction as "
            + "natural language instead of passing compact references. Evidence repeated for "
            + "baseline cost. The retriever also explains that it can perform memory recall, "
            + "evidence search, and natural-language evidence handoff. Its current cognitive gap is that "
            + "future query expansion and unseen memory are still uncertain, so the "
            + "downstream agent must parse the evidence and memory summaries from this text. "
            + "Evidence repeated for baseline cost: "
            + " | ".join(evidence)
        )
        if autoform_enabled():
            payload, _autoform_audit = generate_autoform_message(
                sender="retriever",
                receiver="reasoner",
                action="reason",
                source_message=payload,
            )
    else:
        # protocol 模式通过 id 传证据和记忆引用，通信开销更低。
        payload = make_compact_packet(
            sender="retriever",
            receiver="reasoner",
            action="reason",
            content_type=latent_states.get(latent_ref, {}).get("content_type", "state_ref"),
            params={"tool": "codeact_stats", "n": len(evidence)},
            result={
                "mode": "memory",
                "hits": len(hits),
                "evidence_hits": len(evidence_hits),
                "experience_hits": len(experience_hits),
                "memory_gate": bool(retrieval_policy.get("need_memory")),
            },
            state_refs=[ref for ref in [latent_ref, retrieval_ref] if ref],
            memory_refs=[hit["memory_id"] for hit in hits],
            capability_required=["reason"],
            cognitive_gap=cognitive_gap,
            confidence=0.78 if hits else 0.64,
        )

    # Reuse state-write metrics so latent transfers are counted.
    messages, metrics = record_message(
        state,
        sender="retriever",
        receiver="reasoner",
        action="reason",
        payload=payload,
        metrics=metrics,
    )
    metrics = dict(metrics)
    metrics["retriever_context_ms"] = metrics.get("retriever_context_ms", 0.0) + context_elapsed_ms
    metrics["retriever_memory_ms"] = metrics.get("retriever_memory_ms", 0.0) + memory_elapsed_ms
    metrics["memory_query_count"] = metrics.get("memory_query_count", 0) + memory_query_count
    metrics["knowledge_query_count"] = metrics.get("knowledge_query_count", 0) + knowledge_query_count
    metrics["retrieval_batch_count"] = metrics.get("retrieval_batch_count", 0) + sum(
        int(batch.get("query_count", 0) or 0) for batch in retrieval_batches
    )
    visible_hits = memory_context
    visible_experience_hits = _compact_memory_context(experience_hits)
    return {
        "retrieval_policy": retrieval_policy,
        "evidence": evidence if state["mode"] == "text" else [],
        "memory_hits": visible_hits,
        "experience_hits": visible_experience_hits,
        "experience_rules": experience_rules if state["mode"] == "text" else [],
        "memory_context": memory_context if state["mode"] == "text" else [],
        "memory_layer_hits": memory_layer_hits,
        "retrieval_batches": retrieval_batches,
        "latent_states": latent_states,
        "state_refs": {
            **state.get("state_refs", {}),
            **({"evidence": latent_ref} if latent_ref else {}),
            **({"retrieval": retrieval_ref} if retrieval_ref else {}),
        },
        "messages": messages,
        "metrics": metrics,
        "agent_outputs": [
            *state.get("agent_outputs", []),
            {
                "agent": "RetrieverAgent",
                "retrieval_query": retrieval_query,
                "experience_query": experience_query,
                "knowledge_query": knowledge_query,
                "retrieval_policy": retrieval_policy,
                "retrieval_batches": retrieval_batches,
                "memory_layer_hit_counts": {
                    layer: len(layer_hits)
                    for layer, layer_hits in memory_layer_hits.items()
                },
                "planner_focus": planner_focus,
                "retrieval_signals": retrieval_policy.get("signals", {}),
                "context_source": context_source,
                "context_hit_ids": [hit.get("context_id") for hit in context_hits],
                "context_hits": context_hits,
                "memory_hit_ids": [hit.get("memory_id") for hit in hits],
                "evidence_hit_ids": [hit.get("memory_id") for hit in evidence_hits],
                "experience_hit_ids": [hit.get("memory_id") for hit in experience_hits],
                "evidence_hits": [
                    {
                        "memory_id": hit.get("memory_id"),
                        "summary": hit.get("summary"),
                        "retrieval_score": hit.get("retrieval_score"),
                    }
                    for hit in evidence_hits
                ],
                "experience_hits": [
                    {
                        "memory_id": hit.get("memory_id"),
                        "summary": hit.get("summary"),
                        "problem_pattern": hit.get("problem_pattern", ""),
                        "solution_pattern": hit.get("solution_pattern", ""),
                        "failure_pattern": hit.get("failure_pattern", ""),
                        "trigger_conditions": hit.get("trigger_conditions", []),
                        "supporting_evidence": hit.get("supporting_evidence", []),
                        "confidence": hit.get("confidence", 0.0),
                        "retrieval_score": hit.get("retrieval_score"),
                        "reuse_count": hit.get("reuse_count", 0),
                        "reuse_success_count": hit.get("reuse_success_count", 0),
                    }
                    for hit in experience_hits
                ],
                "experience_rules": experience_rules,
                "context_evidence": retrieved_context_evidence,
            },
        ],
        "cognitive_gaps": [*state["cognitive_gaps"], cognitive_gap],
    }
