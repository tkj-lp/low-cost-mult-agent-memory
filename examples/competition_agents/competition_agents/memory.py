from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Iterable

from competition_agents.embedding import embed_text
from competition_agents.llm import get_llm_client
from hypergraph_memory_method import build_hypergraph_view, search_hypergraph_memory
from hypergraph_memory_method.distill import distill_experience


def _question_text(task_topic: str, trace: dict[str, Any] | None) -> str:
    task_meta = trace.get("task_meta", {}) if isinstance(trace, dict) else {}
    question = str(task_meta.get("question") or "").strip()
    return question or task_topic


def _normalized_text(text: str) -> str:
    lowered = str(text or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9\s\-]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _answer_matches(expected_answer: str, answer_candidate: str, final_answer: str) -> bool:
    expected_norm = _normalized_text(expected_answer)
    answer_norm = _normalized_text(answer_candidate)
    final_norm = _normalized_text(final_answer)
    if not expected_norm:
        return False
    return expected_norm in answer_norm or expected_norm in final_norm


def _case_feedback(trace: dict[str, Any] | None, summary: str) -> tuple[str, str, bool | None]:
    if not isinstance(trace, dict):
        return "unknown", "", None

    task_meta = trace.get("task_meta", {})
    expected_answer = str(task_meta.get("expected_answer") or "").strip()
    expected_structured = task_meta.get("expected")
    execution_result = trace.get("execution_result", {})
    answer_candidate = str(execution_result.get("answer_candidate") or "").strip()
    final_answer = str(summary or "").strip()

    if expected_structured is not None and execution_result.get("result") is not None:
        exact_match = execution_result.get("result") == expected_structured
        return (
            ("success", "Structured result matches expected output.", True)
            if exact_match
            else ("failure", "Structured result does not match expected output.", False)
        )

    if expected_answer:
        exact_match = _answer_matches(expected_answer, answer_candidate, final_answer)
        if exact_match:
            return "success", "Predicted answer overlaps expected answer.", True
        return "failure", "Predicted answer does not match expected answer.", False

    limitations = execution_result.get("limitations", [])
    if limitations:
        return "unknown", str(limitations[0]), None
    return "unknown", "", None


def _build_distillation_case(task_topic: str, summary: str, trace: dict[str, Any] | None) -> dict[str, Any]:
    task_meta = trace.get("task_meta", {}) if isinstance(trace, dict) else {}
    execution_result = trace.get("execution_result", {}) if isinstance(trace, dict) else {}
    outcome, outcome_note, exact_match = _case_feedback(trace, summary)
    answer_candidate = str(execution_result.get("answer_candidate") or "").strip()
    evidence = trace.get("evidence", []) if isinstance(trace, dict) else []
    memory_hits = trace.get("memory_hits", []) if isinstance(trace, dict) else []

    return {
        "question": _question_text(task_topic, trace),
        "dataset_type": task_meta.get("dataset_type"),
        "group_id": task_meta.get("group_id"),
        "gold": task_meta.get("expected_answer") if task_meta.get("expected_answer") is not None else task_meta.get("expected"),
        "answer_candidate": answer_candidate,
        "reasoning_summary": execution_result.get("reasoning_summary", ""),
        "experience_hit_count": len(
            [hit for hit in memory_hits if str(hit.get("memory_type", "evidence")) == "experience"]
        ),
        "evidence": [str(item) for item in evidence[:3]],
        "limitations": execution_result.get("limitations", []),
        "exact_match": exact_match,
        "case_outcome": outcome,
        "outcome_note": outcome_note,
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = str(text or "").strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(stripped[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _sanitize_pattern(value: Any, fallback: str) -> str:
    text = _normalized_text(str(value or ""))
    if not text:
        return fallback
    return text.replace(" ", "_")[:80]


def _looks_like_task_label(pattern: str) -> bool:
    text = _normalized_text(pattern.replace("_", " "))
    if not text:
        return True
    label_markers = {
        "question",
        "query",
        "lookup",
        "summary",
        "intro",
        "span",
        "event",
        "topic",
        "main",
        "film",
        "movie",
        "cast",
        "location",
        "resource",
        "hazard",
    }
    strategy_markers = {
        "prefer",
        "avoid",
        "choose",
        "select",
        "match",
        "verify",
        "constrain",
        "ground",
        "filter",
        "disambiguate",
        "resolve",
        "return",
    }
    tokens = set(_word_tokens(text))
    if not tokens:
        return True
    if tokens & strategy_markers:
        return False
    return len(tokens & label_markers) >= max(1, len(tokens) // 2)


def _word_tokens(text: str) -> list[str]:
    return [token for token in _normalized_text(text).split() if token]


def _looks_over_specific(pattern: str, question: str) -> bool:
    tokens = _word_tokens(pattern.replace("_", " "))
    question_tokens = set(_word_tokens(question))
    content_tokens = [token for token in tokens if len(token) >= 5]
    if len(tokens) > 8:
        return True
    if len(pattern) > 64:
        return True
    if len(content_tokens) >= 4 and len([token for token in content_tokens if token in question_tokens]) >= 3:
        return True
    return False


def _abstract_pattern_fallback(field: str, case: dict[str, Any], current: str) -> str:
    question = _normalized_text(str(case.get("question") or ""))
    outcome = str(case.get("case_outcome") or "")
    if "when" in question and any(token in question for token in ("dvd", "release", "digital", "blu ray")):
        mapping = {
            "problem_pattern": "multiple_temporal_candidates_with_constraints",
            "solution_pattern": "choose_answer_span_matching_all_constraints",
            "failure_pattern": "confuse_competing_date_candidates",
        }
        return mapping[field]
    if "cast" in question:
        mapping = {
            "problem_pattern": "entity_list_question_with_target_type",
            "solution_pattern": "extract_complete_person_entities_only",
            "failure_pattern": "return_topic_instead_of_entity_list",
        }
        return mapping[field]
    if "where was it filmed" in question:
        mapping = {
            "problem_pattern": "joint_answer_requires_multiple_locations",
            "solution_pattern": "return_all_linked_location_spans",
            "failure_pattern": "return_partial_location_answer",
        }
        return mapping[field]
    if "what is the film about" in question or ("what" in question and "film" in question and "about" in question):
        mapping = {
            "problem_pattern": "definition_sentence_vs_metadata_conflict",
            "solution_pattern": "prefer_definition_sentence_over_metadata",
            "failure_pattern": "select_release_or_production_information",
        }
        return mapping[field]
    if any(marker in question for marker in ("mineral", "fish", "economy", "hazard")):
        mapping = {
            "problem_pattern": "attribute_selection_among_related_facts",
            "solution_pattern": "pick_fact_matching_requested_attribute",
            "failure_pattern": "mix_multiple_attributes_into_one_answer",
        }
        return mapping[field]
    if field == "failure_pattern" and outcome == "failure":
        return "repeat_unverified_answer_pattern"
    generic = {
        "problem_pattern": "memory_grounded_question_answering_pattern",
        "solution_pattern": "select_evidence_backed_answer_span",
        "failure_pattern": "overfit_to_surface_task_description",
    }
    return generic[field]


def _sanitize_triggers(value: Any) -> list[str]:
    raw_items: list[Any]
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = re.split(r"[\n,;|]+", value)
    else:
        return []
    triggers: list[str] = []
    for item in raw_items:
        text = _normalized_text(str(item or ""))
        if not text:
            continue
        triggers.append(text.replace(" ", "_")[:60])
    return triggers[:6]


def _generic_trigger_only(triggers: Iterable[str]) -> bool:
    normalized = [str(item or "").strip() for item in triggers if str(item or "").strip()]
    return not normalized or set(normalized) <= {"memory_grounded_qa"}


def _effective_trigger_conditions(
    case: dict[str, Any],
    current: list[str],
    hint_catalog: dict[str, Any],
) -> list[str]:
    normalized = _unique_keep_order(str(item) for item in current if str(item or "").strip())
    if _generic_trigger_only(normalized):
        normalized = list(hint_catalog.get("trigger_candidates", []) or [])

    question = _normalized_text(str(case.get("question") or ""))
    derived: list[str] = []
    if "when" in question:
        derived.append("temporal_query")
    if any(token in question for token in ("dvd", "blu ray", "blu-ray", "digital", "format")):
        derived.append("format_constraint")
    if any(token in question for token in ("united kingdom", "uk", "united states", "region")):
        derived.append("region_constraint")
    if any(token in question for token in ("what is", "about")):
        derived.append("definition_or_topic_query")
    if any(token in question for token in ("cast", "members")):
        derived.append("entity_list_query")
    if "where" in question and "filmed" in question:
        derived.append("multi_location_query")
    if any(token in question for token in ("economy", "depend on", "mineral", "resource", "hazard", "fish")):
        derived.append("attribute_lookup_query")
    return _unique_keep_order([*normalized, *derived])[:8]


def _experience_hint_catalog(case: dict[str, Any]) -> dict[str, Any]:
    question = _normalized_text(str(case.get("question") or ""))

    if "when" in question and any(token in question for token in ("dvd", "release", "digital", "blu ray")):
        return {
            "family": "temporal_constraint_selection",
            "problem_candidates": [
                "multiple_temporal_candidates_with_constraints",
                "time_slot_selection_under_medium_and_region_constraints",
            ],
            "solution_candidates": [
                "choose_answer_span_matching_all_constraints",
                "select_time_candidate_satisfying_medium_and_region",
            ],
            "failure_candidates": [
                "confuse_competing_date_candidates",
                "ignore_medium_or_region_constraint",
            ],
            "trigger_candidates": [
                "question_involves_time",
                "medium_constraint_present",
                "region_constraint_present",
            ],
        }
    if "cast" in question or "members" in question:
        return {
            "family": "entity_list_extraction",
            "problem_candidates": [
                "entity_list_question_with_target_type",
                "list_answer_requires_entity_type_filtering",
            ],
            "solution_candidates": [
                "extract_complete_person_entities_only",
                "return_named_entities_matching_target_type",
            ],
            "failure_candidates": [
                "return_topic_instead_of_entity_list",
                "drop_required_entities_from_list_answer",
            ],
            "trigger_candidates": [
                "question_requests_people",
                "supporting_span_contains_person_list",
            ],
        }
    if "where was it filmed" in question or ("where" in question and "filmed" in question):
        return {
            "family": "multi_location_selection",
            "problem_candidates": [
                "joint_answer_requires_multiple_locations",
                "location_answer_requires_joint_span_selection",
            ],
            "solution_candidates": [
                "return_all_linked_location_spans",
                "keep_joint_location_answer_complete",
            ],
            "failure_candidates": [
                "return_partial_location_answer",
                "drop_secondary_location_from_joint_answer",
            ],
            "trigger_candidates": [
                "question_requests_location",
                "supporting_span_contains_multiple_locations",
            ],
        }
    if "what is the film about" in question or ("what" in question and "film" in question and "about" in question):
        return {
            "family": "topic_summary_selection",
            "problem_candidates": [
                "definition_sentence_vs_metadata_conflict",
                "topic_question_competes_with_background_metadata",
            ],
            "solution_candidates": [
                "prefer_definition_sentence_over_metadata",
                "select_core_topic_description_only",
            ],
            "failure_candidates": [
                "add_irrelevant_background_details",
                "select_release_or_production_information",
            ],
            "trigger_candidates": [
                "question_requests_main_topic",
                "supporting_span_contains_definition_or_depiction",
            ],
        }
    if any(marker in question for marker in ("economy", "depend on", "mineral", "resources", "hazard", "fish")):
        return {
            "family": "attribute_lookup_with_context",
            "problem_candidates": [
                "attribute_selection_among_related_facts",
                "target_attribute_requires_fact_alignment",
            ],
            "solution_candidates": [
                "pick_fact_matching_requested_attribute",
                "return_only_target_attribute_phrase",
            ],
            "failure_candidates": [
                "mix_multiple_attributes_into_one_answer",
                "return_related_fact_instead_of_target_attribute",
            ],
            "trigger_candidates": [
                "question_targets_single_attribute",
                "supporting_fact_contains_attribute_value",
            ],
        }
    return {
        "family": "memory_grounded_question_answering",
        "problem_candidates": ["memory_grounded_question_answering_pattern"],
        "solution_candidates": ["select_evidence_backed_answer_span"],
        "failure_candidates": ["overfit_to_surface_task_description"],
        "trigger_candidates": ["memory_grounded_qa"],
    }


def _fallback_rule_card(case: dict[str, Any], distilled: dict[str, Any]) -> dict[str, Any]:
    question = _normalized_text(str(case.get("question") or ""))
    if "when" in question and any(token in question for token in ("dvd", "release", "digital", "blu ray")):
        return {
            "should_write": True,
            "problem_pattern": "multiple_temporal_candidates_with_constraints",
            "solution_pattern": "choose_date_matching_all_constraints",
            "failure_pattern": "confuse_digital_release_with_disc_release",
            "trigger_conditions": ["question_when", "release_date", "medium_constraint"],
            "abstraction_note": "When multiple release dates exist, follow the medium and region constraint before selecting a date.",
        }
    if "cast" in question:
        return {
            "should_write": True,
            "problem_pattern": "entity_list_question_with_target_type",
            "solution_pattern": "extract_complete_person_entities_only",
            "failure_pattern": "return_title_instead_of_people",
            "trigger_conditions": ["question_cast", "person_list"],
            "abstraction_note": "For cast questions, prioritize the sentence enumerating people and output person names only.",
        }
    if "what is the film about" in question or ("what" in question and "film" in question and "about" in question):
        return {
            "should_write": True,
            "problem_pattern": "definition_sentence_vs_metadata_conflict",
            "solution_pattern": "prefer_definition_sentence_over_metadata",
            "failure_pattern": "select_release_or_production_information",
            "trigger_conditions": ["question_main_topic", "definition_sentence_present"],
            "abstraction_note": "When the question asks what something is about, prefer the defining sentence over release, production, or setting metadata.",
        }
    return {
        "should_write": True,
        "problem_pattern": "memory_grounded_qa_case",
        "solution_pattern": "reason_from_retrieved_evidence",
        "failure_pattern": "overfit_to_surface_question_words",
        "trigger_conditions": ["memory_grounded_qa"],
        "abstraction_note": "Prefer evidence-backed answer selection over repeating surface phrasing from the question.",
    }


def _distill_rule_card(case: dict[str, Any], distilled: dict[str, Any]) -> dict[str, Any]:
    hint_catalog = _experience_hint_catalog(case)
    prompt = (
        "You are an experience distillation module for self-evolving multi-agent memory.\n"
        "Your job is to distill one reusable experience rule from a single finished task case.\n"
        "Experience memory is not a question-type label. It must capture why the case succeeded or failed, what strategy worked, what mistake should be avoided, and when that lesson should be reused.\n"
        "Base the rule only on the provided case. Do not use external knowledge.\n"
        "Do not restate the answer. Do not copy the question text. Avoid task-specific wording.\n"
        "Your patterns must be abstract and reusable across similar tasks.\n"
        "Do not collapse unrelated question types into the same generic pattern.\n"
        "Different question families should yield different reusable rules.\n"
        "The problem_pattern should name the transferable reasoning difficulty, not the surface topic.\n"
        "The solution_pattern should name the transferable solving strategy, not the retrieval action or task title.\n"
        "The failure_pattern should name the concrete historical error that the strategy avoids.\n"
        "Bad problem_pattern example: how_to_retrieve_specific_release_information_for_a_film_in_a_given_region\n"
        "Bad problem_pattern example: main_event_summary_from_intro_span\n"
        "Good problem_pattern example: multiple_temporal_candidates_with_constraints\n"
        "Good problem_pattern example: definition_sentence_vs_metadata_conflict\n"
        "Bad solution_pattern example: retrieve_film_release_information\n"
        "Bad solution_pattern example: answer_intro_question\n"
        "Good solution_pattern example: choose_answer_span_matching_all_constraints\n"
        "Good solution_pattern example: prefer_definition_sentence_over_metadata\n"
        "Bad failure_pattern example: intro_question_type\n"
        "Good failure_pattern example: select_release_or_production_information\n"
        "Keep each pattern under 8 tokens and prefer high-level error or strategy motifs.\n"
        "If the case does not contain a transferable lesson, return should_write=false.\n"
        "If case_outcome is failure, the failure_pattern must describe the failure.\n"
        "If case_outcome is success, focus on a reusable strategy and the main error it avoids.\n"
        "Use gold and exact_match as feedback signals when deciding whether the lesson is reliable.\n"
        "If the result was correct only because the answer was directly copied and no reusable reasoning lesson is visible, return should_write=false.\n"
        "Prefer strategic verbs such as prefer, avoid, choose, verify, match, constrain, filter, disambiguate.\n"
        "Prefer one of the hinted abstract families when it truly matches the case, but infer from the case itself.\n"
        "Return strict JSON with keys:\n"
        "should_write, problem_pattern, solution_pattern, failure_pattern, trigger_conditions, abstraction_note\n"
        "Use short English snake_case labels for the pattern fields.\n"
        f"Case: {json.dumps(case, ensure_ascii=False)}\n"
        f"Trace meta: {json.dumps({'strategies': distilled.get('strategies', []), 'experiences': distilled.get('experiences', [])}, ensure_ascii=False)}\n"
        f"Hint family: {json.dumps(hint_catalog, ensure_ascii=False)}\n"
    )
    raw = get_llm_client("summarizer").generate(prompt)
    parsed = _extract_json_object(raw)
    if parsed is None:
        fallback = _fallback_rule_card(case, distilled)
        return {
            **fallback,
            "distillation_raw_output": raw,
            "distillation_mode": "fallback",
        }

    should_write = bool(parsed.get("should_write", True))
    result = {
        "should_write": should_write,
        "problem_pattern": _sanitize_pattern(parsed.get("problem_pattern"), "memory_grounded_qa_case"),
        "solution_pattern": _sanitize_pattern(parsed.get("solution_pattern"), "reason_from_retrieved_evidence"),
        "failure_pattern": _sanitize_pattern(parsed.get("failure_pattern"), "overfit_to_surface_question_words"),
        "trigger_conditions": _sanitize_triggers(parsed.get("trigger_conditions")),
        "abstraction_note": str(parsed.get("abstraction_note") or "").strip(),
        "distillation_raw_output": raw,
        "distillation_mode": "llm",
    }
    for field in ("problem_pattern", "solution_pattern", "failure_pattern"):
        if _looks_over_specific(str(result[field]), str(case.get("question") or "")):
            result[field] = _abstract_pattern_fallback(field, case, str(result[field]))
    if _looks_like_task_label(str(result["problem_pattern"])) and _looks_like_task_label(str(result["solution_pattern"])):
        fallback = _fallback_rule_card(case, distilled)
        result.update(
            {
                "should_write": fallback["should_write"],
                "problem_pattern": fallback["problem_pattern"],
                "solution_pattern": fallback["solution_pattern"],
                "failure_pattern": fallback["failure_pattern"],
                "trigger_conditions": fallback["trigger_conditions"],
                "abstraction_note": fallback["abstraction_note"],
                "distillation_mode": "fallback_from_task_label",
            }
        )
    if not result["trigger_conditions"]:
        result["trigger_conditions"] = list(hint_catalog.get("trigger_candidates", []) or [])
    result["trigger_conditions"] = _effective_trigger_conditions(case, result["trigger_conditions"], hint_catalog)
    if not result["abstraction_note"]:
        result["abstraction_note"] = "Reusable lesson distilled from the current task case."
    return result


def _experience_source(case: dict[str, Any]) -> str:
    outcome = str(case.get("case_outcome") or "unknown")
    if outcome == "success":
        return "success-derived"
    if outcome == "failure":
        return "failure-derived"
    return "outcome-unknown"


def _experience_rule_card(
    *,
    problem_pattern: str,
    solution_pattern: str,
    failure_pattern: str,
    trigger_conditions: list[str],
    supporting_evidence: list[str],
    confidence: float,
    abstraction_note: str,
) -> dict[str, Any]:
    return {
        "problem_pattern": problem_pattern,
        "solution_pattern": solution_pattern,
        "failure_pattern": failure_pattern,
        "trigger_conditions": trigger_conditions,
        "supporting_evidence": supporting_evidence,
        "confidence": confidence,
        "abstraction_note": abstraction_note,
    }


def _source_case(case: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(case, dict):
        return {}
    return {
        "question": case.get("question", ""),
        "group_id": case.get("group_id"),
        "dataset_type": case.get("dataset_type"),
        "case_outcome": case.get("case_outcome"),
        "exact_match": case.get("exact_match"),
    }


def _unique_keep_order(values: Iterable[Any]) -> list[Any]:
    seen: set[str] = set()
    ordered: list[Any] = []
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def _experience_signature(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("problem_pattern") or "").strip(),
        str(record.get("solution_pattern") or "").strip(),
        str(record.get("failure_pattern") or "").strip(),
    )


def _trigger_overlap(left: Iterable[str], right: Iterable[str]) -> int:
    return len(set(str(item) for item in left) & set(str(item) for item in right))


def _find_mergeable_experience(
    memory: list[dict],
    *,
    problem_pattern: str,
    solution_pattern: str,
    failure_pattern: str,
    trigger_conditions: list[str],
) -> int | None:
    target_signature = (
        str(problem_pattern or "").strip(),
        str(solution_pattern or "").strip(),
        str(failure_pattern or "").strip(),
    )
    for index, record in enumerate(memory):
        if str(record.get("memory_type", "evidence")) != "experience":
            continue
        signature = _experience_signature(record)
        problem = str(record.get("problem_pattern") or "").strip()
        solution = str(record.get("solution_pattern") or "").strip()
        failure = str(record.get("failure_pattern") or "").strip()
        if signature == target_signature:
            return index
        if solution == target_signature[1] and failure == target_signature[2]:
            return index
        if solution == target_signature[1] and _trigger_overlap(record.get("trigger_conditions", []), trigger_conditions) >= 1:
            return index
        if problem == target_signature[0] and _trigger_overlap(record.get("trigger_conditions", []), trigger_conditions) >= 2:
            return index
    return None


def _merge_experience_record(
    record: dict[str, Any],
    *,
    source_agent: str,
    task_topic: str,
    summary: str,
    tags: list[str],
    problem_pattern: str,
    solution_pattern: str,
    failure_pattern: str,
    trigger_conditions: list[str],
    supporting_evidence: list[str],
    confidence: float,
    archive_scope: str,
    lifecycle_stage: str,
    strategies: list[str],
    experiences: list[str],
    reuse_benefits: list[str],
    experience_rule_card: dict[str, Any] | None,
    experience_source: str,
    distillation_case: dict[str, Any] | None,
    abstraction_note: str,
) -> dict[str, Any]:
    merged = dict(record)
    merged["source_agent"] = source_agent
    merged["task_topic"] = task_topic
    merged["summary"] = summary
    merged["tags"] = _unique_keep_order([*(record.get("tags", []) or []), *tags])
    merged["supporting_evidence"] = _unique_keep_order(
        [*(record.get("supporting_evidence", []) or []), *supporting_evidence]
    )[:6]
    merged["problem_pattern"] = problem_pattern or record.get("problem_pattern", "")
    merged["solution_pattern"] = solution_pattern or record.get("solution_pattern", "")
    merged["failure_pattern"] = failure_pattern or record.get("failure_pattern", "")
    merged["trigger_conditions"] = _unique_keep_order(
        [*(record.get("trigger_conditions", []) or []), *trigger_conditions]
    )[:8]
    merged["confidence"] = max(float(record.get("confidence", 0.0) or 0.0), float(confidence or 0.0))
    merged["archive_scope"] = archive_scope or record.get("archive_scope", "task")
    merged["lifecycle_stage"] = lifecycle_stage or record.get("lifecycle_stage", "episodic")
    merged["strategies"] = _unique_keep_order([*(record.get("strategies", []) or []), *strategies])
    merged["experiences"] = _unique_keep_order([*(record.get("experiences", []) or []), *experiences])
    merged["reuse_benefits"] = _unique_keep_order([*(record.get("reuse_benefits", []) or []), *reuse_benefits])
    merged["experience_source"] = experience_source or record.get("experience_source", "")
    merged["abstraction_note"] = abstraction_note or record.get("abstraction_note", "")
    merged["source_case"] = _source_case(distillation_case) or record.get("source_case", {})
    merged["merge_count"] = int(record.get("merge_count", 0) or 0) + 1
    merged["experience_rule_card"] = experience_rule_card or _experience_rule_card(
        problem_pattern=merged["problem_pattern"],
        solution_pattern=merged["solution_pattern"],
        failure_pattern=merged["failure_pattern"],
        trigger_conditions=merged["trigger_conditions"],
        supporting_evidence=merged["supporting_evidence"],
        confidence=merged["confidence"],
        abstraction_note=merged["abstraction_note"],
    )
    embedding = embed_text(
        _embedding_text(
            task_topic=task_topic,
            summary=summary,
            memory_type="experience",
            problem_pattern=merged["problem_pattern"],
            solution_pattern=merged["solution_pattern"],
            failure_pattern=merged["failure_pattern"],
            trigger_conditions=merged["trigger_conditions"],
        )
    )
    merged["embedding"] = embedding.vector
    merged["embedding_provider"] = embedding.provider
    merged["embedding_model"] = embedding.model
    merged["hypergraph"] = build_hypergraph_view(merged)
    return merged


def _rule_card_summary(experience_rule_card: dict[str, Any]) -> str:
    trigger_text = ", ".join(experience_rule_card["trigger_conditions"]) if experience_rule_card["trigger_conditions"] else "general"
    return (
        "experience_rule_card: "
        f"problem={experience_rule_card['problem_pattern']}; "
        f"solution={experience_rule_card['solution_pattern']}; "
        f"avoid={experience_rule_card['failure_pattern']}; "
        f"triggers={trigger_text}"
    )


def refine_memory(
    *,
    task_topic: str,
    summary: str,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """基于任务案例做大模型经验提炼，而不是按题型模板直接生规则。"""

    distilled = distill_experience(
        {
            "memory_id": "draft-experience",
            "task_topic": task_topic,
            "summary": summary,
            "trace": trace or {},
        }
    )
    case = _build_distillation_case(task_topic, summary, trace)
    llm_rule = _distill_rule_card(case, distilled)
    supporting_evidence = case.get("evidence", [])
    hit_count = len(trace.get("memory_hits", [])) if isinstance(trace, dict) else 0
    confidence = 0.8 if hit_count else 0.65
    experience_rule_card = _experience_rule_card(
        problem_pattern=llm_rule["problem_pattern"],
        solution_pattern=llm_rule["solution_pattern"],
        failure_pattern=llm_rule["failure_pattern"],
        trigger_conditions=llm_rule["trigger_conditions"],
        supporting_evidence=supporting_evidence,
        confidence=confidence,
        abstraction_note=llm_rule["abstraction_note"],
    )
    refined_summary = _rule_card_summary(experience_rule_card)

    return {
        "memory_type": "experience",
        "should_write": llm_rule["should_write"],
        "summary": refined_summary,
        "problem_pattern": experience_rule_card["problem_pattern"],
        "solution_pattern": experience_rule_card["solution_pattern"],
        "failure_pattern": experience_rule_card["failure_pattern"],
        "trigger_conditions": experience_rule_card["trigger_conditions"],
        "supporting_evidence": supporting_evidence,
        "confidence": confidence,
        "archive_scope": distilled.get("archive_scope", "task"),
        "lifecycle_stage": distilled.get("lifecycle_stage", "episodic"),
        "strategies": distilled.get("strategies", []),
        "experiences": distilled.get("experiences", []),
        "reuse_benefits": distilled.get("reuse_benefits", []),
        "distilled_experience": distilled,
        "experience_source": _experience_source(case),
        "distillation_case": case,
        "experience_rule_card": experience_rule_card,
        "abstraction_note": experience_rule_card["abstraction_note"],
        "distillation_mode": llm_rule["distillation_mode"],
        "distillation_raw_output": llm_rule["distillation_raw_output"],
    }


def _embedding_text(
    *,
    task_topic: str,
    summary: str,
    memory_type: str,
    problem_pattern: str,
    solution_pattern: str,
    failure_pattern: str,
    trigger_conditions: list[str],
) -> str:
    if memory_type == "experience":
        parts = [
            memory_type,
            problem_pattern,
            solution_pattern,
            failure_pattern,
            " ".join(trigger_conditions),
        ]
        return " ".join(part for part in parts if part)

    parts = [task_topic, summary, memory_type]
    if problem_pattern:
        parts.append(problem_pattern)
    if solution_pattern:
        parts.append(solution_pattern)
    if failure_pattern:
        parts.append(failure_pattern)
    if trigger_conditions:
        parts.append(" ".join(trigger_conditions))
    return " ".join(part for part in parts if part)


def add_memory(
    memory: list[dict],
    *,
    source_agent: str,
    task_topic: str,
    summary: str,
    tags: list[str],
    trace: dict[str, Any] | None = None,
    memory_type: str = "evidence",
    problem_pattern: str = "",
    solution_pattern: str = "",
    failure_pattern: str = "",
    trigger_conditions: list[str] | None = None,
    supporting_evidence: list[str] | None = None,
    confidence: float = 0.0,
    archive_scope: str = "task",
    lifecycle_stage: str = "episodic",
    strategies: list[str] | None = None,
    experiences: list[str] | None = None,
    reuse_benefits: list[str] | None = None,
    experience_rule_card: dict[str, Any] | None = None,
    experience_source: str = "",
    distillation_case: dict[str, Any] | None = None,
    abstraction_note: str = "",
    distillation_mode: str = "",
    distillation_raw_output: str = "",
) -> list[dict]:
    """向共享记忆追加一条记录，兼容 evidence 与 experience 两类记忆。"""

    normalized_trigger_conditions = trigger_conditions or []
    if memory_type == "experience":
        merge_index = _find_mergeable_experience(
            memory,
            problem_pattern=problem_pattern,
            solution_pattern=solution_pattern,
            failure_pattern=failure_pattern,
            trigger_conditions=normalized_trigger_conditions,
        )
        if merge_index is not None:
            merged = _merge_experience_record(
                memory[merge_index],
                source_agent=source_agent,
                task_topic=task_topic,
                summary=summary,
                tags=tags,
                problem_pattern=problem_pattern,
                solution_pattern=solution_pattern,
                failure_pattern=failure_pattern,
                trigger_conditions=normalized_trigger_conditions,
                supporting_evidence=supporting_evidence or [],
                confidence=confidence,
                archive_scope=archive_scope,
                lifecycle_stage=lifecycle_stage,
                strategies=strategies or [],
                experiences=experiences or [],
                reuse_benefits=reuse_benefits or [],
                experience_rule_card=experience_rule_card,
                experience_source=experience_source,
                distillation_case=distillation_case,
                abstraction_note=abstraction_note,
            )
            updated_memory = list(memory)
            updated_memory[merge_index] = merged
            return updated_memory

    embedding = embed_text(
        _embedding_text(
            task_topic=task_topic,
            summary=summary,
            memory_type=memory_type,
            problem_pattern=problem_pattern,
            solution_pattern=solution_pattern,
            failure_pattern=failure_pattern,
            trigger_conditions=normalized_trigger_conditions,
        )
    )
    record = {
        "memory_id": f"mem-{len(memory) + 1:04d}",
        "memory_type": memory_type,
        "source_agent": source_agent,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_topic": task_topic,
        "summary": summary,
        "tags": tags,
        "embedding": embedding.vector,
        "embedding_provider": embedding.provider,
        "embedding_model": embedding.model,
        "entities": [],
        "supporting_facts": [],
        "supporting_evidence": supporting_evidence or [],
        "evidence": [],
        "reasoning_path": [],
        "state_refs": [],
        "answer": summary,
        "reuse_count": 0,
        "reuse_success_count": 0,
        "problem_pattern": problem_pattern,
        "solution_pattern": solution_pattern,
        "failure_pattern": failure_pattern,
        "trigger_conditions": normalized_trigger_conditions,
        "confidence": confidence,
        "archive_scope": archive_scope,
        "lifecycle_stage": lifecycle_stage,
        "strategies": strategies or [],
        "experiences": experiences or [],
        "reuse_benefits": reuse_benefits or [],
        "experience_rule_card": experience_rule_card
        or _experience_rule_card(
            problem_pattern=problem_pattern,
            solution_pattern=solution_pattern,
            failure_pattern=failure_pattern,
            trigger_conditions=normalized_trigger_conditions,
            supporting_evidence=supporting_evidence or [],
            confidence=confidence,
            abstraction_note=abstraction_note,
        ),
        "experience_source": experience_source,
        "abstraction_note": abstraction_note,
        "source_case": _source_case(distillation_case),
    }
    if trace is not None:
        record["trace"] = trace
        state_refs = trace.get("state_refs", {})
        if isinstance(state_refs, dict):
            record["state_refs"] = [ref for ref in state_refs.values() if ref]
        record["evidence"] = trace.get("evidence", [])
        task_meta = trace.get("task_meta", {})
        if isinstance(task_meta, dict):
            record["supporting_facts"] = task_meta.get("supporting_facts", [])
    record["hypergraph"] = build_hypergraph_view(record)
    return [*memory, record]


def increment_reuse_counts(
    memory: list[dict],
    memory_ids: Iterable[str],
    *,
    successful_memory_ids: Iterable[str] | None = None,
) -> list[dict]:
    """命中历史记忆后累计 `reuse_count`，并在成功复用时更新 `reuse_success_count`。"""

    hit_ids = {str(memory_id) for memory_id in memory_ids if memory_id}
    successful_ids = {str(memory_id) for memory_id in (successful_memory_ids or []) if memory_id}
    if not hit_ids:
        return memory

    updated: list[dict] = []
    for record in memory:
        if str(record.get("memory_id")) in hit_ids:
            next_record = dict(record)
            next_record["reuse_count"] = int(record.get("reuse_count", 0)) + 1
            if str(record.get("memory_id")) in successful_ids:
                next_record["reuse_success_count"] = int(record.get("reuse_success_count", 0)) + 1
            updated.append(next_record)
        else:
            updated.append(record)
    return updated


def search_memory(
    memory: list[dict],
    query: str,
    *,
    top_k: int,
    memory_types: Iterable[str] | None = None,
) -> list[dict]:
    """使用超图索引检索记忆，并支持按 `memory_type` 过滤。"""

    if not memory:
        return []

    allowed_types = {str(memory_type) for memory_type in memory_types} if memory_types else None
    searchable_memory = (
        [record for record in memory if str(record.get("memory_type", "evidence")) in allowed_types]
        if allowed_types
        else memory
    )
    if not searchable_memory:
        return []

    return search_hypergraph_memory(searchable_memory, query, top_k=top_k)
