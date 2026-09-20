from __future__ import annotations

import json
import re
import os
import time

from baselines.autoform import autoform_enabled, generate_autoform_message
from competition_agents.memory import search_memory
from competition_agents.handoff import historical_qa, memory_object, decide_qa_retrieval, json_object
from competition_agents.embedding import embed_text
from competition_agents.latent_hidden import extract_last_hidden_state
from competition_agents.latent_hidden.adapter import cosine_similarity
from competition_agents.protocol import make_cognitive_gap, make_compact_packet
from competition_agents.runtime_events import emit_knowledge_retrieval_events, emit_memory_retrieval_events
from competition_agents.state import (
    CompetitionState,
    put_hidden_state,
    put_latent_state,
    put_shared_state,
    record_message,
    resolve_state_ref,
    received_state_ref,
)
from hypergraph_memory_method import (
    KNOWLEDGE_FOLLOWUP_BATCH,
    MEMORY_ROUTE_BATCH,
    search_context,
    update_retrieval_batch,
)
from hypergraph_memory_method.text import question_content_tokens


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
                "group_id": hit.get("group_id", ""),
                "turn_id": hit.get("turn_id", ""),
                "artifact_types": hit.get("artifact_types", []),
                "summary": _shorten(hit.get("summary", ""), 160),
                "history_qa": historical_qa(hit) if hit.get("memory_type", "evidence") == "evidence" else None,
                "retrieval_score": hit.get("retrieval_score"),
                "problem_pattern": _shorten(hit.get("problem_pattern", ""), 120),
                "solution_pattern": _shorten(hit.get("solution_pattern", ""), 140),
                "trigger_conditions": hit.get("trigger_conditions", []),
            }
        )
    return context


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
    semantic_payload: dict | None = None,
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
            semantic_payload=semantic_payload,
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
            semantic_payload=semantic_payload,
        )
    except RuntimeError:
        os.environ["EMBEDDING_PROVIDER"] = "hash"
        return put_latent_state(
            state,
            producer="retriever",
            topic=state["task"],
            text=text,
            consumers=consumers,
            semantic_payload=semantic_payload,
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


def _reuse_specs(task_meta: dict) -> list[str]:
    return [str(value) for value in task_meta.get("expected_reuse", []) or [] if str(value).strip()]


def _reuse_artifacts(expected_reuse: list[str]) -> list[str]:
    artifacts: list[str] = []
    for value in expected_reuse:
        _, _, artifact = value.partition(".")
        if artifact and artifact not in artifacts:
            artifacts.append(artifact)
    return artifacts


def _build_continuous_task_hints(task_meta: dict) -> list[str]:
    tags = task_meta.get("tags", [])
    if not isinstance(tags, list):
        tags = [tags]
    hints = [
        f"Group id: {task_meta.get('group_id')}",
        f"Topic: {task_meta.get('topic')}",
        f"Topic section: {task_meta.get('topic_section')}",
        f"Current turn: {task_meta.get('turn_id')}",
        f"Task type: {task_meta.get('task_type') or task_meta.get('dataset_type')}",
        f"Tags: {tags}",
    ]
    return [hint for hint in hints if hint and not hint.endswith(": None")]


def _memory_top_k(task_meta: dict, *, base: int = 2, limit: int = 6) -> int:
    del task_meta
    return max(base, min(limit, 3))


def _build_retrieval_query(task: str, plan: list[str], task_meta: dict | None = None) -> tuple[str, list[str]]:
    """把 Planner 的计划并入检索查询，让规划信号真正影响记忆召回。"""

    task_meta = task_meta or {}
    focus = _planner_focus(plan)
    hint_lines = _build_continuous_task_hints(task_meta)
    # `task` is a mode-dependent runtime prompt. Retrieval must instead begin
    # from the same user question in every collaboration mode.
    question = str(task_meta.get("question") or "").strip()
    sections = [f"Question: {question}" if question else task]

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
    if re.search(r"\b(when|what year|what date)\b", normalized):
        return "temporal_constraint_selection"
    if re.search(r"\b(list|name|mention|members|what kinds|what types|which)\b", normalized):
        return "entity_list_extraction"
    if re.search(r"\b(where|which place|which country|which city|location)\b", normalized):
        return "location_selection"
    if re.search(r"\b(who|whose|which person)\b", normalized):
        return "person_or_role_selection"
    if re.search(r"\b(what|how|is|are|was|were|does|do|has|have)\b", normalized):
        return "attribute_lookup_with_context"
    return "memory_grounded_question_answering"


def _build_experience_query(task: str, plan: list[str], question: str, task_meta: dict | None = None) -> str:
    retrieval_query, planner_focus = _build_retrieval_query(task, plan, task_meta)
    family = _experience_family(question)
    hint = f"\nExperience retrieval family: {family}"
    return retrieval_query + hint


def _augment_knowledge_query(
    base_query: str,
    memory_context: list[dict],
    retrieval_policy: dict,
    task_meta: dict | None = None,
) -> str:
    """Build a standalone knowledge query from public metadata and recalled QA state."""

    task_meta = task_meta or {}
    topic = str(task_meta.get("topic") or "").strip()
    memory_lines = []
    # A dataset-provided current topic is a safer reference anchor than an
    # unverified answer produced earlier in the same batch. When no topic is
    # available, retain recent QA as an explicitly unverified reference hint.
    if not topic:
        for item in memory_context[:2]:
            qa = item.get("history_qa") or historical_qa(item)
            parts = [
                str(item.get("memory_id") or ""),
                "Question: " + str(qa.get("question") or ""),
                "Unverified answer reference: " + str(qa.get("answer") or ""),
            ]
            memory_lines.append(" | ".join(part for part in parts if part))
    sections = [
        base_query,
        f"Current topic: {topic}",
        f"Requested section or attribute: {task_meta.get('topic_section') or task_meta.get('task_type') or ''}",
        "Planner retrieval policy: " + str(retrieval_policy.get("reason", "")),
        (
            "Previous conversational QA for reference resolution only:\n"
            + "\n".join(f"- {line}" for line in memory_lines if line)
            if memory_lines
            else ""
        ),
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
        "location_selection": ("location", "place", "country", "city", "location_span"),
        "person_or_role_selection": ("person", "identity", "role", "position"),
        "attribute_lookup_with_context": ("attribute", "property", "description", "status", "relation"),
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
    """Score generic answer-shape compatibility without benchmark-specific facts."""

    sentence_lower = sentence_text.lower()
    bonus = 0
    all_query_tokens = query_tokens | focus_tokens
    if {"when", "date", "year"} & all_query_tokens and re.search(r"\b(?:19|20)\d{2}\b", sentence_lower):
        bonus += 4
    if {"where", "location", "country", "city"} & all_query_tokens and re.search(
        r"\b(?:in|at|near|from|located)\b", sentence_lower
    ):
        bonus += 3
    if {"who", "person", "role", "position"} & all_query_tokens and re.search(
        r"\b(?:is|was|became|served|appointed|named|held)\b", sentence_lower
    ):
        bonus += 3
    if {"list", "kinds", "types", "members", "examples"} & all_query_tokens and sentence_text.count(",") >= 2:
        bonus += 3
    if {"many", "much", "number", "percentage"} & all_query_tokens and re.search(
        r"\b\d+(?:\.\d+)?%?\b", sentence_lower
    ):
        bonus += 3

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
            scored_sentences.append((score, -index, title, sentence_text))

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
    plan_state_payload: dict | None = None,
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
        exclude_kinds={"dialogue_question", "dialogue_answer"},
    )
    # Dialogue turns belong to episodic QA memory. Keep the knowledge channel
    # limited to knowledge nodes so prior answers cannot overwrite current facts.
    hits = [
        hit
        for hit in hits
        if str(hit.get("kind") or "") not in {"dialogue_question", "dialogue_answer"}
    ]
    # Nested snippets of the same section should not consume separate evidence slots.
    normalized_texts = [" ".join(str(hit.get("text") or "").split()) for hit in hits]
    hits = [
        hit for index, hit in enumerate(hits)
        if not any(
            normalized_texts[index]
            and len(normalized_texts[index]) < len(normalized_texts[other_index])
            and normalized_texts[index] in normalized_texts[other_index]
            and (hit.get("title"), hit.get("section")) == (other.get("title"), other.get("section"))
            for other_index, other in enumerate(hits)
        )
    ]
    hits = _rerank_hypergraph_hits(
        task_meta,
        planner_focus,
        hits,
        limit=limit,
        plan_state_payload=plan_state_payload,
        query_text=question,
    )
    evidence = []
    for hit in hits:
        title = str(hit.get("title") or "").strip()
        text = str(hit.get("text") or "").strip()
        if not text:
            continue
        evidence.append(f"{title}: {text}" if title else text)
    return evidence, hits


def _rerank_hypergraph_hits(
    task_meta: dict,
    planner_focus: list[str],
    hits: list[dict],
    *,
    limit: int,
    plan_state_payload: dict | None = None,
    query_text: str = "",
) -> list[dict]:
    """对超图命中的上下文做答题导向重排，优先保留最像最终证据的片段。"""

    question_tokens = _tokenize(
        f"{task_meta.get('question', '')} {query_text}"
    )
    focus_tokens = set().union(*(_tokenize(step) for step in planner_focus)) if planner_focus else set()
    topic_tokens = _tokenize(task_meta.get("topic", ""))
    section_tokens = _tokenize(task_meta.get("topic_section", ""))
    requested_tokens = question_content_tokens(str(task_meta.get("question") or ""))

    latent_query_vector = []
    latent_vector_kind = ""
    plan_content_type = str((plan_state_payload or {}).get("content_type") or "")
    embedding_provider = str((plan_state_payload or {}).get("embedding_provider") or "").lower()
    if plan_content_type == "latent.llm_hidden_state":
        latent_vector_kind = "hidden"
        latent_query_vector = [
            float(value) for value in (plan_state_payload or {}).get("pooled_vector", [])
        ]
    elif plan_content_type == "latent.embedding" and embedding_provider not in {"", "hash", "hash-v1"}:
        latent_vector_kind = "embedding"
        latent_query_vector = [
            float(value) for value in (plan_state_payload or {}).get("vector", [])
        ]

    reranked: list[tuple[float, dict]] = []
    for hit in hits:
        score = float(hit.get("context_score") or 0.0)
        kind = str(hit.get("kind") or "")
        title = str(hit.get("title") or "")
        section = str(hit.get("section") or "")
        text = str(hit.get("text") or "")
        text_lower = text.lower()
        latent_similarity = 0.0
        if latent_query_vector and text:
            try:
                if latent_vector_kind == "hidden":
                    candidate_state, _ = extract_last_hidden_state(text)
                    candidate_vector = candidate_state.pooled_vector
                else:
                    candidate_vector = embed_text(text).vector
                latent_similarity = cosine_similarity(
                    latent_query_vector,
                    candidate_vector,
                )
                # Hidden similarity is a tie-breaker after lexical/plan
                # relevance, never a substitute for question-grounded match.
                score += 0.25 * max(latent_similarity, 0.0)
            except RuntimeError:
                latent_similarity = 0.0

        if kind in {"dialogue_question", "dialogue_answer"}:
            score -= 4.0

        if topic_tokens and (_tokenize(title) & topic_tokens):
            score += 4.0
        if section_tokens and (_tokenize(section) & section_tokens):
            score += 5.0

        text_tokens = _tokenize(text)
        score += 4.0 * len(text_tokens & requested_tokens)
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

        reranked.append(
            (
                score,
                {
                    **hit,
                    **(
                        {"latent_similarity": round(latent_similarity, 6)}
                        if latent_query_vector
                        else {}
                    ),
                },
            )
        )

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


def _planned_knowledge_steps(
    retrieval_policy: dict,
    *,
    fallback_query: str,
) -> list[dict]:
    raw_steps = retrieval_policy.get("knowledge_steps", []) or []
    steps: list[dict] = []
    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            continue
        query = str(raw_step.get("query") or raw_step.get("goal") or "").strip()
        if not query:
            continue
        steps.append(
            {
                "step": int(raw_step.get("step") or index),
                "goal": str(raw_step.get("goal") or query).strip(),
                "query": query,
                "depends_on": raw_step.get("depends_on"),
            }
        )
    return steps or [
        {
            "step": 1,
            "goal": "Retrieve evidence needed to answer the question",
            "query": fallback_query,
            "depends_on": None,
        }
    ]


def _dedupe_records(records: list[dict], *, id_key: str) -> list[dict]:
    selected: list[dict] = []
    seen: set[str] = set()
    for record in records:
        record_id = str(record.get(id_key) or "")
        fingerprint = record_id or repr(sorted(record.items()))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append(record)
    return selected


def _dedupe_text(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if str(value).strip()))


def retriever_agent(state: CompetitionState) -> dict:
    """检索可复用记忆，并生成非文本状态引用。"""

    task_meta = state.get("task_meta", {})
    metrics = state["metrics"]
    plan = state.get("plan", [])
    retrieval_policy = state.get("retrieval_policy") or {}
    plan_state_payload: dict = {}
    plan_ref = received_state_ref(state, sender="planner", receiver="retriever") if state["mode"] in {"protocol", "hidden"} else ""
    if state["mode"] in {"protocol", "hidden"} and plan_ref:
        plan_state_payload, metrics = resolve_state_ref(
            {**state, "metrics": metrics},
            plan_ref,
            consumer="retriever",
            expected_content_types={"latent.embedding", "latent.llm_hidden_state"},
        )
        semantic_payload = plan_state_payload.get("semantic_payload", {})
        if not plan_state_payload or not semantic_payload:
            raise ValueError("Planner handoff state is missing or invalid")
        if isinstance(semantic_payload, dict):
            resolved_plan = semantic_payload.get("plan")
            if isinstance(resolved_plan, list) and resolved_plan:
                plan = [str(step) for step in resolved_plan]
            resolved_policy = semantic_payload.get("retrieval_policy")
            if isinstance(resolved_policy, dict) and resolved_policy:
                retrieval_policy = resolved_policy
    if state["mode"] == "text":
        delivered_plan = json_object(_latest_message_payload(state, sender="planner", receiver="retriever"))
        if delivered_plan and isinstance(delivered_plan.get("retrieval_policy"), dict):
            retrieval_policy = delivered_plan["retrieval_policy"]
            plan = delivered_plan.get("plan", [])
        else:
            # Compressed plans may lose JSON syntax. Keep the delivered text as
            # retrieval guidance without silently recovering the original plan.
            plan = [_latest_message_payload(state, sender="planner", receiver="retriever")]
            retrieval_policy = {"query_type": "unknown", "need_knowledge": True,
                                "planner_delivery_status": "unstructured"}
    retrieval_query, planner_focus = _build_retrieval_query(state["task"], plan, task_meta)
    question = str(task_meta.get("question") or state["task"])
    retrieval_policy = decide_qa_retrieval(
        retrieval_policy, question,
        any(hit.get("memory_type", "evidence") == "evidence" for hit in state.get("memory", [])),
    )
    experience_query = str(
        (retrieval_policy.get("experience_retrieval") or {}).get("query")
        or _build_experience_query(state["task"], plan, question, task_meta)
    )
    context_source = str(task_meta.get("context_source") or "inline")
    context_hits: list[dict] = []
    evidence_top_k = _memory_top_k(task_meta, base=2, limit=6)
    memory_started = time.perf_counter()
    need_qa_memory = bool(
        retrieval_policy.get("need_qa_memory", retrieval_policy.get("need_memory"))
    )
    experience_hits = list(state.get("experience_hits", []))
    if need_qa_memory:
        qa_query = str(retrieval_policy.get("qa_memory_query") or retrieval_query)
        evidence_hits = search_memory(
            state["memory"],
            qa_query,
            top_k=evidence_top_k,
            memory_types=["evidence"],
            memory_layers=["episodic"],
            layer_limits={"episodic": evidence_top_k},
        )
        memory_query_count = 1
    else:
        evidence_hits = []
        memory_query_count = 0
    memory_elapsed_ms = (time.perf_counter() - memory_started) * 1000
    combined_hits = evidence_hits + [
        hit for hit in experience_hits if hit.get("memory_id") not in {item.get("memory_id") for item in evidence_hits}
    ]
    hits = combined_hits[: max(4, evidence_top_k)]
    if need_qa_memory:
        emit_memory_retrieval_events(
            state,
            query=qa_query,
            hits=evidence_hits,
            memory_layer="episodic",
        )
    memory_context = _compact_memory_context(hits)
    experience_rules = _build_experience_rules(experience_hits)
    knowledge_steps = _planned_knowledge_steps(
        retrieval_policy,
        fallback_query=_augment_knowledge_query(
            retrieval_query,
            memory_context,
            retrieval_policy,
            task_meta,
        ),
    )
    context_started = time.perf_counter()
    if not retrieval_policy.get("need_knowledge", True):
        retrieved_context_evidence = []
        knowledge_query_count = 0
    else:
        retrieved_context_evidence = []
        prior_findings: list[str] = []
        for step in knowledge_steps:
            step_query = _augment_knowledge_query(
                str(step["query"]),
                memory_context,
                retrieval_policy,
                task_meta,
            )
            if prior_findings:
                step_query += "\nPrior step findings: " + " | ".join(prior_findings[-2:])[:420]
            step_focus = [str(step.get("goal") or step_query)]
            if context_source == "hypergraph":
                step_evidence, step_hits = _extract_hypergraph_context(
                    task_meta,
                    step_focus,
                    query=step_query,
                    limit=3,
                    plan_state_payload=plan_state_payload,
                )
                annotated_hits = [
                    {**hit, "retrieval_step": step.get("step"), "step_goal": step.get("goal")}
                    for hit in step_hits
                ]
                context_hits.extend(annotated_hits)
                emit_knowledge_retrieval_events(state, query=step_query, hits=annotated_hits)
            else:
                step_evidence = _extract_context_evidence(task_meta, step_focus, limit=3)
            retrieved_context_evidence.extend(step_evidence)
            prior_findings.extend(step_evidence[:2])
        retrieved_context_evidence = _dedupe_text(retrieved_context_evidence)
        context_hits = _dedupe_records(context_hits, id_key="context_id")
        knowledge_query_count = len(knowledge_steps)
    knowledge_query = "\n\n".join(str(step["query"]) for step in knowledge_steps)
    context_elapsed_ms = (time.perf_counter() - context_started) * 1000

    retrieval_batches = list(state.get("retrieval_batches", []))
    retrieval_batches = update_retrieval_batch(
        retrieval_batches,
        MEMORY_ROUTE_BATCH,
        status="completed",
        query_count=1 + memory_query_count,
        hit_counts={"episodic": len(evidence_hits), "experience": len(experience_hits)},
        decision="run",
        decision_source="retriever",
        decision_reason=str(retrieval_policy.get("qa_memory_reason") or "experience_lookup_completed"),
    )
    retrieval_batches = update_retrieval_batch(
        retrieval_batches,
        KNOWLEDGE_FOLLOWUP_BATCH,
        status="completed" if knowledge_query_count else "skipped",
        query_count=knowledge_query_count,
        hit_counts={"knowledge": len(context_hits) or len(retrieved_context_evidence)},
        decision="run" if knowledge_query_count else "skip",
        decision_source="planner",
        decision_reason=str(retrieval_policy.get("reason") or "planner_retrieval_plan"),
    )

    # 把 Planner 的显式计划并入 query，让检索不再只依赖原始任务文本。
    evidence_top_k = _memory_top_k(task_meta, base=2, limit=6)
    evidence = [
        *retrieved_context_evidence,
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
    stage_content = memory_object(evidence, evidence_hits, experience_rules)
    reasoning_input = json.dumps(stage_content, ensure_ascii=False, separators=(",", ":"))
    state_payload = {
        "memory_object": stage_content,
        "reasoning_input": reasoning_input,
        "evidence": evidence,
        "memory_hits": hits,
        "experience_hits": experience_hits,
        "experience_rules": experience_rules,
        "memory_context": memory_context,
        "retrieval_batches": retrieval_batches,
        "context_hits": context_hits,
    }
    working_state = {**state, "metrics": metrics}
    if state["mode"] == "text":
        latent_ref = ""
        latent_states = state.get("latent_states", {})
    elif state["mode"] == "hidden":
        latent_ref, latent_states, metrics = _put_hidden_mode_retriever_state(
            working_state,
            text=_hidden_evidence_state_text(evidence, memory_context, planner_focus),
            consumers=["reasoner", "summarizer"],
            semantic_payload=state_payload,
        )
    else:
        latent_ref, latent_states, metrics = put_latent_state(
            working_state,
            producer="retriever",
            topic=state["task"],
            text=" ".join(evidence),
            consumers=["reasoner", "executor", "summarizer"],
            semantic_payload=state_payload,
        )

    retrieval_ref = ""
    if state["mode"] in {"protocol", "hidden"}:
        retrieval_ref, latent_states, metrics = put_shared_state(
            {**state, "latent_states": latent_states, "metrics": metrics},
            producer="retriever",
            topic=state["task"],
            content_type="retrieval.layered_result",
            payload={
                "summary": (
                    f"experience={len(experience_hits)}, episodic={len(evidence_hits)}, "
                    f"knowledge={len(context_hits) or len(retrieved_context_evidence)}"
                ),
                "batches": retrieval_batches,
                "layer_hit_ids": {
                    "experience": [hit.get("memory_id") for hit in experience_hits],
                    "episodic": [hit.get("memory_id") for hit in evidence_hits],
                    "knowledge": [hit.get("context_id") for hit in context_hits],
                },
            },
            consumers=["reasoner", "summarizer"],
            scope="task",
            prefix="retrieval",
        )

    if state["mode"] == "text":
        # Text baseline carries the same selected evidence once, using natural language.
        payload = reasoning_input
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
            # The primary ref is always evidence; the optional second ref lets
            # Reasoner consume the exact Planner state used by Retriever.
            state_refs=[ref for ref in (latent_ref, plan_ref) if ref],
            capability_required=["reason"],
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
    metrics["retrieval_batch_count"] = metrics.get("retrieval_batch_count", 0) + memory_query_count + knowledge_query_count
    metrics["memory_gate_count"] = metrics.get("memory_gate_count", 0) + 1
    metrics["memory_gate_skip_count"] = metrics.get("memory_gate_skip_count", 0) + int(not need_qa_memory)
    metrics["latent_rerank_count"] = metrics.get("latent_rerank_count", 0) + sum(
        1 for hit in context_hits if "latent_similarity" in hit
    )
    return {
        "retrieval_policy": retrieval_policy,
        "evidence": evidence,
        "memory_hits": hits,
        "experience_hits": experience_hits,
        "experience_rules": experience_rules,
        "memory_context": memory_context,
        "retrieval_batches": retrieval_batches,
        "memory_layer_hits": {
            "experience": experience_hits,
            "episodic": evidence_hits,
            "knowledge": context_hits,
        },
        "latent_states": latent_states,
        "state_refs": {
            **state.get("state_refs", {}),
            "evidence": latent_ref,
            **({"retrieval_result": retrieval_ref} if retrieval_ref else {}),
        },
        "messages": messages,
        "metrics": metrics,
        "agent_outputs": [
            *state.get("agent_outputs", []),
            {
                "agent": "RetrieverAgent",
                "memory_object": stage_content,
                "retrieval_query": retrieval_query,
                "experience_query": experience_query,
                "knowledge_query": knowledge_query,
                "knowledge_steps": knowledge_steps,
                "retrieval_policy": retrieval_policy,
                "planner_focus": planner_focus,
                "plan_state_ref": plan_ref,
                "plan_state_resolved": bool(plan_state_payload),
                "latent_rerank_count": sum(
                    1 for hit in context_hits if "latent_similarity" in hit
                ),
                "expected_reuse": _reuse_specs(task_meta),
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
