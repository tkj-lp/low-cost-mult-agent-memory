from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Iterable

from competition_agents.embedding import embed_text
from competition_agents.llm import get_llm_client
from hypergraph_memory_method import build_hypergraph_view, search_hypergraph_memory
from hypergraph_memory_method.distill import distill_experience, lifecycle_stage
from hypergraph_memory_method.entities import extract_canonical_entities
from hypergraph_memory_method.layers import memory_layer_for_record, normalize_memory_layer


def _question_text(task_topic: str, trace: dict[str, Any] | None) -> str:
    task_meta = trace.get("task_meta", {}) if isinstance(trace, dict) else {}
    question = str(task_meta.get("question") or "").strip()
    return question or task_topic


def _normalized_text(text: str) -> str:
    lowered = str(text or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9\s\-]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _case_feedback(trace: dict[str, Any] | None, summary: str) -> tuple[str, str, bool | None]:
    if not isinstance(trace, dict):
        return "unknown", "", None

    execution_result = trace.get("execution_result", {})

    limitations = execution_result.get("limitations", [])
    if limitations:
        return "unknown", str(limitations[0]), None
    return "unknown", "", None


def _build_distillation_case(task_topic: str, summary: str, trace: dict[str, Any] | None) -> dict[str, Any]:
    task_meta = trace.get("task_meta", {}) if isinstance(trace, dict) else {}
    execution_result = trace.get("execution_result", {}) if isinstance(trace, dict) else {}
    outcome, outcome_note, exact_match = _case_feedback(trace, summary)
    answer_candidate = str(execution_result.get("answer_candidate") or "").strip()
    if not answer_candidate and execution_result.get("result") is not None:
        answer_candidate = json.dumps(execution_result.get("result"), ensure_ascii=False, sort_keys=True)
    evidence = trace.get("evidence", []) if isinstance(trace, dict) else []
    memory_hits = trace.get("memory_hits", []) if isinstance(trace, dict) else []
    question = _question_text(task_topic, trace)

    return {
        "question": question,
        "dataset_type": task_meta.get("dataset_type"),
        "group_id": task_meta.get("group_id"),
        "task_type": task_meta.get("task_type"),
        "tags": task_meta.get("tags", []),
        "answer_candidate": answer_candidate,
        "reasoning_summary": execution_result.get("reasoning_summary", ""),
        "experience_hit_count": len(
            [hit for hit in memory_hits if str(hit.get("memory_type", "evidence")) == "experience"]
        ),
        "evidence": _clean_supporting_evidence(evidence, question=question)[:3],
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


_STABLE_TRIGGER_ENUMS = {
    "temporal_query",
    "format_constraint",
    "region_constraint",
    "entity_list_query",
    "multi_location_query",
    "definition_or_topic_query",
    "attribute_lookup_query",
    "reference_resolution_query",
    "statistical_computation_query",
    "summary_statistics_query",
    "formula_computation_query",
    "table_aggregation_query",
    "table_transformation_query",
    "string_extraction_query",
    "date_computation_query",
    "evidence_extraction_query",
    "evidence_span_extraction_query",
    "memory_reuse_query",
}


def _looks_like_question_text(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    return stripped.endswith("?") or lowered.startswith(
        (
            "what ",
            "when ",
            "where ",
            "which ",
            "who ",
            "whom ",
            "whose ",
            "can you ",
            "does ",
            "do ",
            "did ",
            "is ",
            "are ",
            "was ",
            "were ",
        )
    )


def _clean_supporting_evidence(evidence: Iterable[Any], *, question: str) -> list[str]:
    cleaned: list[str] = []
    question_norm = _normalized_text(question)
    noise_markers = (
        "planner guided retrieval focus",
        "planner-guided retrieval focus",
        "current memory retrieval uses",
        "question:",
        "task:",
        "answer:",
    )
    for item in evidence:
        text = str(item or "").strip()
        if not text:
            continue
        normalized = _normalized_text(text)
        if not normalized:
            continue
        if any(marker in normalized for marker in noise_markers):
            continue
        rhs = text.split(":", 1)[1].strip() if ":" in text else text
        rhs_norm = _normalized_text(rhs)
        if _looks_like_question_text(rhs):
            continue
        if rhs_norm and rhs_norm == question_norm:
            continue
        if len(rhs_norm.split()) < 3:
            continue
        cleaned.append(text)
    return _unique_keep_order(cleaned)


def _normalize_trigger_label(trigger: str) -> str:
    raw = str(trigger or "").strip()
    if not raw:
        return ""
    compact = raw.replace(":", "_").replace("-", "_").strip().lower()
    if compact in _STABLE_TRIGGER_ENUMS:
        return compact

    text = _normalized_text(trigger).replace("-", " ")
    if not text:
        return ""
    if any(token in text for token in ("when", "date", "time", "temporal")):
        return "temporal_query"
    if any(token in text for token in ("dvd", "digital", "blu ray", "format", "medium")):
        return "format_constraint"
    if any(token in text for token in ("united kingdom", "united states", "uk", "region", "country")):
        return "region_constraint"
    if any(token in text for token in ("people", "person", "cast", "members", "person list", "entity list")):
        return "entity_list_query"
    if any(token in text for token in ("location", "filmed", "place", "multiple locations")):
        return "multi_location_query"
    if any(token in text for token in ("main topic", "definition", "depiction", "subject", "film about")):
        return "definition_or_topic_query"
    if any(token in text for token in ("economy", "depend on", "mineral", "resource", "hazard", "fish", "attribute")):
        return "attribute_lookup_query"
    if any(
        token in text
        for token in (
            "latter",
            "former",
            "reference",
            "pronoun",
            "that list",
            "that process",
            "that substance",
            "that experiment",
            "earlier list",
            "first hormone",
            "third hormone",
            "last hormone",
            "ordinal",
        )
    ):
        return "reference_resolution_query"
    if any(token in text for token in ("summary statistics", "multiple statistical", "mean", "median", "variance", "stddev")):
        return "summary_statistics_query"
    if any(token in text for token in ("formula", "growth rate", "saving rate", "hit rate", "derive", "ratio")):
        return "formula_computation_query"
    if any(token in text for token in ("statistic", "variance", "median", "mean", "average", "stddev", "arithmetic")):
        return "statistical_computation_query"
    if any(token in text for token in ("table transformation", "group by", "filter sort", "max row")):
        return "table_transformation_query"
    if any(token in text for token in ("table", "rows", "group", "aggregate", "sort", "filter")):
        return "table_aggregation_query"
    if any(token in text for token in ("text", "string", "keyword", "word count", "extract numbers")):
        return "string_extraction_query"
    if any(token in text for token in ("date", "days between", "weekday", "isoformat")):
        return "date_computation_query"
    if any(token in text for token in ("evidence span", "answer span", "span extraction")):
        return "evidence_span_extraction_query"
    if any(token in text for token in ("evidence list", "supported", "answer key", "served as")):
        return "evidence_extraction_query"
    if any(token in text for token in ("reuse", "memory hits", "hit rate", "source id", "threshold")):
        return "memory_reuse_query"
    return ""


def _derive_codeact_triggers(case: dict[str, Any]) -> list[str]:
    question = _normalized_text(str(case.get("question") or ""))
    task_type = _normalized_text(str(case.get("task_type") or ""))
    tags = [_normalized_text(str(tag)) for tag in case.get("tags", []) if str(tag).strip()]
    evidence = [_normalized_text(str(item)) for item in case.get("evidence", []) if str(item).strip()]

    derived: list[str] = []
    if any(tag in {"statistics"} for tag in tags) or any(
        token in task_type for token in ("stats_summary", "population_variance", "sum_average")
    ):
        derived.append("summary_statistics_query")
    if any(tag in {"metrics"} for tag in tags) or any(
        token in task_type for token in ("saving_rate", "growth_rate", "hit_rate")
    ):
        derived.append("formula_computation_query")
    if any(tag in {"statistics", "metrics"} for tag in tags) or any(
        token in task_type for token in ("stats", "variance", "average", "hit_rate", "saving_rate")
    ):
        derived.append("statistical_computation_query")
    if any(tag in {"table", "aggregation"} for tag in tags) or any(
        token in task_type for token in ("group_sum", "filter_sort", "max_row")
    ):
        derived.append("table_transformation_query")
        derived.append("table_aggregation_query")
    if any(tag in {"string", "keyword"} for tag in tags) or any(
        token in task_type for token in ("word_count", "top_keywords", "extract_numbers_sum")
    ):
        derived.append("string_extraction_query")
    if "date" in tags or any(token in task_type for token in ("days_between", "weekday")):
        derived.append("date_computation_query")
    if "evidence" in tags or "evidence_answer" in task_type:
        derived.append("evidence_span_extraction_query")
        derived.append("evidence_extraction_query")
    if "memory_reuse" in tags or any(token in task_type for token in ("reuse", "hit_rate", "trend")):
        derived.append("memory_reuse_query")
    if any("memory[" in item for item in evidence):
        derived.append("memory_reuse_query")
    if "served as" in question or "series" in question:
        derived.append("evidence_extraction_query")
    return [item for item in _unique_keep_order(derived) if item][:6]


def _generic_trigger_only(triggers: Iterable[str]) -> bool:
    normalized = [str(item or "").strip() for item in triggers if str(item or "").strip()]
    return not normalized or set(normalized) <= {"memory_grounded_qa"}


def _effective_trigger_conditions(
    case: dict[str, Any],
    current: list[str],
    hint_catalog: dict[str, Any],
) -> list[str]:
    normalized = _unique_keep_order(
        _normalize_trigger_label(str(item))
        for item in current
        if _normalize_trigger_label(str(item))
    )
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
    if "what is" in question or "film about" in question or "main topic" in question:
        derived.append("definition_or_topic_query")
    if any(token in question for token in ("cast", "members")):
        derived.append("entity_list_query")
    if "where" in question and "filmed" in question:
        derived.append("multi_location_query")
    if any(token in question for token in ("economy", "depend on", "mineral", "resource", "hazard", "fish")):
        derived.append("attribute_lookup_query")
    if any(
        token in question
        for token in (
            "latter",
            "former",
            "that list",
            "that process",
            "that substance",
            "that experiment",
            "earlier catabolic-hormone list",
            "first hormone",
            "third hormone",
            "last hormone",
        )
    ):
        derived.append("reference_resolution_query")
    if str(case.get("dataset_type") or "") == "codeact":
        derived.extend(_derive_codeact_triggers(case))
    merged = [
        *[label for label in _unique_keep_order(_normalize_trigger_label(item) for item in normalized) if label],
        *[label for label in derived if label],
    ]
    return _unique_keep_order(merged)[:8]


def _normalize_trigger_conditions(triggers: Iterable[str] | None) -> list[str]:
    if not triggers:
        return []
    normalized: list[str] = []
    for item in triggers:
        label = _normalize_trigger_label(str(item))
        if label:
            normalized.append(label)
    return _unique_keep_order(normalized)[:8]


def _experience_hint_catalog(case: dict[str, Any]) -> dict[str, Any]:
    question = _normalized_text(str(case.get("question") or ""))
    task_type = _normalized_text(str(case.get("task_type") or ""))
    tags = [_normalized_text(str(tag)) for tag in case.get("tags", []) if str(tag).strip()]

    if str(case.get("dataset_type") or "") == "codeact":
        if any(tag in {"statistics"} for tag in tags) or any(
            token in task_type for token in ("stats_summary", "population_variance", "sum_average")
        ):
            return {
                "family": "codeact_summary_statistics",
                "problem_candidates": [
                    "numerical_summary_with_multiple_metrics",
                    "multi_metric_statistics_from_shared_input",
                ],
                "solution_candidates": [
                    "compute_all_metrics_in_one_execution_step",
                    "reuse_single_dataset_for_joint_statistics",
                ],
                "failure_candidates": [
                    "compute_metrics_separately_without_consistency_check",
                    "drop_required_metric_from_structured_result",
                ],
                "trigger_candidates": [
                    "summary_statistics_query",
                    "statistical_computation_query",
                ],
            }
        if any(tag in {"metrics"} for tag in tags) or any(
            token in task_type for token in ("saving_rate", "growth_rate", "hit_rate")
        ):
            return {
                "family": "codeact_formula_computation",
                "problem_candidates": [
                    "evidence_based_calculation_with_formula",
                    "formula_application_under_structured_constraints",
                ],
                "solution_candidates": [
                    "verify_formula_against_evidence_before_computation",
                    "compute_from_declared_variables_only",
                ],
                "failure_candidates": [
                    "compute_without_verifying_formula_source",
                    "mix_formula_inputs_or_denominator",
                ],
                "trigger_candidates": [
                    "formula_computation_query",
                    "statistical_computation_query",
                ],
            }
        if any(tag in {"table", "aggregation"} for tag in tags) or any(
            token in task_type for token in ("group_sum", "filter_sort", "max_row")
        ):
            return {
                "family": "codeact_table_transformation",
                "problem_candidates": [
                    "structured_table_transformation_with_selection_rules",
                    "row_level_filter_sort_or_group_aggregation",
                ],
                "solution_candidates": [
                    "apply_transformation_rules_before_projection",
                    "preserve_requested_ordering_and_grouping_constraints",
                ],
                "failure_candidates": [
                    "return_unordered_or_partial_table_result",
                    "project_fields_before_finishing_transformation",
                ],
                "trigger_candidates": [
                    "table_transformation_query",
                    "table_aggregation_query",
                ],
            }
        if any(tag in {"string", "keyword"} for tag in tags) or any(
            token in task_type for token in ("word_count", "top_keywords", "extract_numbers_sum")
        ):
            return {
                "family": "codeact_string_extraction",
                "problem_candidates": [
                    "string_token_extraction_with_count_or_rank",
                    "text_to_structured_value_extraction",
                ],
                "solution_candidates": [
                    "normalize_text_before_token_selection",
                    "extract_then_rank_or_aggregate_tokens",
                ],
                "failure_candidates": [
                    "count_raw_surface_forms_without_normalization",
                    "rank_before_finishing_extraction",
                ],
                "trigger_candidates": ["string_extraction_query"],
            }
        if "date" in tags or any(token in task_type for token in ("days_between", "weekday")):
            return {
                "family": "codeact_date_computation",
                "problem_candidates": [
                    "calendar_value_computation_from_structured_date_input",
                    "date_arithmetic_or_calendar_lookup",
                ],
                "solution_candidates": [
                    "convert_to_date_object_before_computation",
                    "apply_calendar_operation_on_normalized_date_values",
                ],
                "failure_candidates": [
                    "treat_date_as_plain_string_during_computation",
                    "skip_date_normalization_before_calendar_lookup",
                ],
                "trigger_candidates": ["date_computation_query"],
            }
        if "evidence" in tags or "evidence_answer" in task_type:
            return {
                "family": "codeact_evidence_span_extraction",
                "problem_candidates": [
                    "evidence_span_selection_under_pattern_constraint",
                    "supported_answer_extraction_from_evidence_list",
                ],
                "solution_candidates": [
                    "match_answer_span_to_target_pattern",
                    "extract_only_supported_span_from_evidence",
                ],
                "failure_candidates": [
                    "return_full_evidence_sentence_instead_of_span",
                    "choose_unsupported_span_from_related_evidence",
                ],
                "trigger_candidates": [
                    "evidence_span_extraction_query",
                    "evidence_extraction_query",
                ],
            }
        if "memory_reuse" in tags or any(
            token in task_type for token in ("reuse_previous_result", "reuse_threshold_check", "reuse_trend")
        ):
            return {
                "family": "codeact_memory_reuse",
                "problem_candidates": [
                    "cross_task_value_reuse_with_structured_lookup",
                    "prior_result_reuse_under_threshold_or_scale_constraint",
                ],
                "solution_candidates": [
                    "resolve_source_result_before_post_processing",
                    "reuse_prior_structured_value_then_apply_local_check",
                ],
                "failure_candidates": [
                    "compute_from_current_input_without_loading_memory",
                    "reuse_wrong_field_from_prior_result",
                ],
                "trigger_candidates": ["memory_reuse_query"],
            }

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
                "temporal_query",
                "format_constraint",
                "region_constraint",
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
                "entity_list_query",
                "person_list_evidence",
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
                "multi_location_query",
                "multiple_location_evidence",
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
                "definition_or_topic_query",
                "definition_sentence_evidence",
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
                "attribute_lookup_query",
                "attribute_value_evidence",
            ],
        }
    return {
        "family": "memory_grounded_question_answering",
        "problem_candidates": ["memory_grounded_question_answering_pattern"],
        "solution_candidates": ["select_evidence_backed_answer_span"],
        "failure_candidates": ["overfit_to_surface_task_description"],
        "trigger_candidates": [],
    }


def _hint_based_rule_card(case: dict[str, Any], hint_catalog: dict[str, Any]) -> dict[str, Any]:
    outcome = str(case.get("case_outcome") or "unknown")
    problem_candidates = list(hint_catalog.get("problem_candidates", []) or [])
    solution_candidates = list(hint_catalog.get("solution_candidates", []) or [])
    failure_candidates = list(hint_catalog.get("failure_candidates", []) or [])
    trigger_candidates = list(hint_catalog.get("trigger_candidates", []) or [])

    problem_pattern = problem_candidates[0] if problem_candidates else "memory_grounded_qa_case"
    solution_pattern = solution_candidates[0] if solution_candidates else "reason_from_retrieved_evidence"
    if outcome == "failure" and len(failure_candidates) >= 2:
        failure_pattern = failure_candidates[1]
    else:
        failure_pattern = failure_candidates[0] if failure_candidates else "overfit_to_surface_question_words"

    abstraction_note = {
        "temporal_constraint_selection": "遇到多个候选时间槽位时，先按介质和地区约束过滤，再选择最终日期。",
        "entity_list_extraction": "遇到实体列表题时，先按目标实体类型过滤，再检查是否遗漏必要项。",
        "multi_location_selection": "遇到联合地点答案时，先确认是否需要多个地点共同构成答案，再保持完整返回。",
        "topic_summary_selection": "遇到“是什么/讲什么”类问题时，优先定义句，不要被元数据或背景信息带偏。",
        "attribute_lookup_with_context": "遇到属性抽取题时，先锁定被问属性，再只返回与该属性直接匹配的事实短语。",
        "codeact_summary_statistics": "一次执行里完成同一输入上的多个统计量，避免拆散计算导致结构不一致。",
        "codeact_formula_computation": "先确认公式和变量来源，再按声明变量执行计算，避免直接套用错误分母或字段。",
        "codeact_table_transformation": "先完成筛选、排序、分组等变换，再做字段投影和结果输出。",
        "codeact_string_extraction": "先规范化文本或数字片段，再做计数、排序或聚合，避免基于原始表面形式直接统计。",
        "codeact_date_computation": "先把日期转成可计算对象，再做日期差值或星期判断。",
        "codeact_evidence_span_extraction": "先匹配目标模式，再从证据里截取最小支持答案片段，不返回整句。",
        "codeact_memory_reuse": "先加载历史结构化结果，再读取正确字段并做本地缩放、阈值判断或后处理。",
        "memory_grounded_question_answering": "优先依据证据做答案选择，不要直接复述题面或表面词。",
    }.get(
        str(hint_catalog.get("family") or ""),
        "从当前任务案例中提炼可复用的策略，并明确它规避的历史错误模式。",
    )

    return {
        "should_write": True,
        "problem_pattern": problem_pattern,
        "solution_pattern": solution_pattern,
        "failure_pattern": failure_pattern,
        "trigger_conditions": trigger_candidates,
        "abstraction_note": abstraction_note,
    }


def _fallback_rule_card(case: dict[str, Any], distilled: dict[str, Any]) -> dict[str, Any]:
    hint_catalog = _experience_hint_catalog(case)
    hint_rule = _hint_based_rule_card(case, hint_catalog)
    question = _normalized_text(str(case.get("question") or ""))
    if "when" in question and any(token in question for token in ("dvd", "release", "digital", "blu ray")):
        return {
            "should_write": True,
            "problem_pattern": "multiple_temporal_candidates_with_constraints",
            "solution_pattern": "choose_date_matching_all_constraints",
            "failure_pattern": "confuse_digital_release_with_disc_release",
            "trigger_conditions": ["temporal_query", "format_constraint", "region_constraint"],
            "abstraction_note": "When multiple release dates exist, follow the medium and region constraint before selecting a date.",
        }
    if "cast" in question:
        return {
            "should_write": True,
            "problem_pattern": "entity_list_question_with_target_type",
            "solution_pattern": "extract_complete_person_entities_only",
            "failure_pattern": "return_title_instead_of_people",
            "trigger_conditions": ["entity_list_query"],
            "abstraction_note": "For cast questions, prioritize the sentence enumerating people and output person names only.",
        }
    if "what is the film about" in question or ("what" in question and "film" in question and "about" in question):
        return {
            "should_write": True,
            "problem_pattern": "definition_sentence_vs_metadata_conflict",
            "solution_pattern": "prefer_definition_sentence_over_metadata",
            "failure_pattern": "select_release_or_production_information",
            "trigger_conditions": ["definition_or_topic_query"],
            "abstraction_note": "When the question asks what something is about, prefer the defining sentence over release, production, or setting metadata.",
        }
    return hint_rule


def _case_lacks_transferable_lesson(case: dict[str, Any], hint_catalog: dict[str, Any]) -> bool:
    family = str(hint_catalog.get("family") or "")
    evidence = list(case.get("evidence", []) or [])
    experience_hit_count = int(case.get("experience_hit_count") or 0)
    outcome = str(case.get("case_outcome") or "unknown")
    question = _normalized_text(str(case.get("question") or ""))
    triggers = set(_derive_codeact_triggers(case) if str(case.get("dataset_type") or "") == "codeact" else [])
    if str(case.get("dataset_type") or "") != "codeact":
        triggers |= {
            label
            for label in _effective_trigger_conditions(case, [], hint_catalog)
            if label
        }

    if str(case.get("dataset_type") or "") == "codeact":
        return False
    if outcome == "failure":
        return False
    if experience_hit_count > 0:
        return False
    if len(evidence) > 1:
        return False
    if family == "temporal_constraint_selection" and not any(
        token in question for token in ("dvd", "digital", "blu ray", "blu-ray", "format", "medium", "disc")
    ):
        return True
    if family != "memory_grounded_question_answering":
        return False
    if any(token in question for token in ("when", "where", "who", "which", "what year", "what date")):
        return True
    return False


def _distill_rule_card(case: dict[str, Any], distilled: dict[str, Any]) -> dict[str, Any]:
    hint_catalog = _experience_hint_catalog(case)
    distillation_mode = os.environ.get("COMPETITION_MEMORY_DISTILLATION_MODE", "").strip().lower()
    summarizer_provider = (
        os.environ.get("LLM_SUMMARIZER_PROVIDER")
        or os.environ.get("LLM_PROVIDER")
        or ""
    ).strip().lower()
    if (
        distillation_mode in {
        "fallback",
        "local",
        "local_fallback",
        "rule",
        "rule_based",
        }
        or (not distillation_mode and summarizer_provider == "local_hf")
    ):
        fallback = _fallback_rule_card(case, distilled)
        if _case_lacks_transferable_lesson(case, hint_catalog):
            fallback["should_write"] = False
            if not fallback.get("abstraction_note"):
                fallback["abstraction_note"] = (
                    "This case is too close to direct evidence copying and does not add a stable reusable strategy."
                )
        return {
            **fallback,
            "distillation_raw_output": "",
            "distillation_mode": "local_fallback",
        }

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
        "If the case is a simple fact lookup with one obvious evidence span and no reusable strategy beyond copying, return should_write=false.\n"
        "Prefer strategic verbs such as prefer, avoid, choose, verify, match, constrain, filter, disambiguate.\n"
        "Prefer one of the hinted abstract families when it truly matches the case, but infer from the case itself.\n"
        "The final rule should read like a reusable decision rule, not like a task title.\n"
        "If your candidate patterns look like a task name, replace them with a more strategic abstraction.\n"
        "Return strict JSON with keys:\n"
        "should_write, problem_pattern, solution_pattern, failure_pattern, trigger_conditions, abstraction_note\n"
        "Use short English snake_case labels for the pattern fields.\n"
        f"Case: {json.dumps(case, ensure_ascii=False)}\n"
        f"Trace meta: {json.dumps({'strategies': distilled.get('strategies', []), 'experiences': distilled.get('experiences', [])}, ensure_ascii=False)}\n"
        f"Hint family: {json.dumps(hint_catalog, ensure_ascii=False)}\n"
    )
    try:
        raw = get_llm_client("summarizer").generate(prompt)
    except Exception as exc:
        fallback = _fallback_rule_card(case, distilled)
        result = {
            **fallback,
            "distillation_raw_output": f"{type(exc).__name__}: {exc}",
            "distillation_mode": "llm_error_fallback",
        }
        if _case_lacks_transferable_lesson(case, hint_catalog):
            result["should_write"] = False
            if not result.get("abstraction_note"):
                result["abstraction_note"] = (
                    "This case is too close to direct evidence copying and does not add a stable reusable strategy."
                )
        return result

    parsed = _extract_json_object(raw)
    if parsed is None:
        fallback = _fallback_rule_card(case, distilled)
        result = {
            **fallback,
            "distillation_raw_output": raw,
            "distillation_mode": "fallback",
        }
        if _case_lacks_transferable_lesson(case, hint_catalog):
            result["should_write"] = False
            if not result.get("abstraction_note"):
                result["abstraction_note"] = (
                    "This case is too close to direct evidence copying and does not add a stable reusable strategy."
                )
        return result

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
    if _case_lacks_transferable_lesson(case, hint_catalog):
        result["should_write"] = False
        if not result["abstraction_note"]:
            result["abstraction_note"] = "This case is too close to direct evidence copying and does not add a stable reusable strategy."
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
        if problem == target_signature[0] and solution == target_signature[1]:
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
    source_memory_ids: list[str],
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
        [
            *_normalize_trigger_conditions(record.get("trigger_conditions", []) or []),
            *_normalize_trigger_conditions(trigger_conditions),
        ]
    )[:8]
    merged["confidence"] = max(float(record.get("confidence", 0.0) or 0.0), float(confidence or 0.0))
    merged["archive_scope"] = archive_scope or record.get("archive_scope", "task")
    merged["lifecycle_stage"] = lifecycle_stage or record.get("lifecycle_stage", "episodic")
    merged["strategies"] = _unique_keep_order([*(record.get("strategies", []) or []), *strategies])
    merged["experiences"] = _unique_keep_order([*(record.get("experiences", []) or []), *experiences])
    merged["reuse_benefits"] = _unique_keep_order([*(record.get("reuse_benefits", []) or []), *reuse_benefits])
    merged["experience_source"] = experience_source or record.get("experience_source", "")
    merged["source_memory_ids"] = _unique_keep_order(
        [*(record.get("source_memory_ids", []) or []), *source_memory_ids]
    )
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
    trigger_text = ", ".join(experience_rule_card["trigger_conditions"]) if experience_rule_card["trigger_conditions"] else "none"
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
    source_memory_ids: list[str] | None = None,
    distillation_case: dict[str, Any] | None = None,
    abstraction_note: str = "",
    distillation_mode: str = "",
    distillation_raw_output: str = "",
) -> list[dict]:
    """向共享记忆追加一条记录，兼容 evidence 与 experience 两类记忆。"""

    normalized_trigger_conditions = _normalize_trigger_conditions(trigger_conditions or [])
    task_meta = trace.get("task_meta", {}) if isinstance(trace, dict) and isinstance(trace.get("task_meta"), dict) else {}
    artifact_types = ["experience", "strategy"] if memory_type == "experience" else ["answer", "evidence"]
    trace_evidence = trace.get("evidence", []) if isinstance(trace, dict) else []
    if supporting_evidence or trace_evidence:
        artifact_types.append("supporting-facts")
    artifact_types = list(dict.fromkeys(artifact_types))
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
                source_memory_ids=source_memory_ids or [],
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
        "entities": extract_canonical_entities(
            task_topic,
            summary,
            *(supporting_evidence or []),
            *trace_evidence,
            explicit=[task_meta.get("topic"), task_meta.get("topic_section")],
        ),
        "supporting_facts": [],
        "supporting_evidence": supporting_evidence or [],
        "evidence": [],
        "reasoning_path": [],
        "state_refs": [],
        "answer": summary,
        "artifact_types": artifact_types,
        "task_id": str(task_meta.get("task_id") or ""),
        "group_id": str(task_meta.get("group_id") or task_meta.get("chain_id") or ""),
        "turn_id": str(task_meta.get("turn_id") or ""),
        "dataset_type": str(task_meta.get("dataset_type") or ""),
        "task_type": str(task_meta.get("task_type") or ""),
        "retrieval_count": 0,
        "reuse_count": 0,
        "reuse_success_count": 0,
        "reuse_failure_count": 0,
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
        "source_memory_ids": source_memory_ids or [],
        "abstraction_note": abstraction_note,
        "source_case": _source_case(distillation_case)
        or {
            key: value
            for key, value in {
                "task_id": task_meta.get("task_id"),
                "group_id": task_meta.get("group_id") or task_meta.get("chain_id"),
                "turn_id": task_meta.get("turn_id"),
                "dataset_type": task_meta.get("dataset_type"),
                "question": task_meta.get("question"),
            }.items()
            if value not in (None, "", [], {})
        },
    }
    if trace is not None:
        record["trace"] = trace
        state_refs = trace.get("state_refs", {})
        if isinstance(state_refs, dict):
            record["state_refs"] = [ref for ref in state_refs.values() if ref]
        record["evidence"] = trace.get("evidence", [])
        task_meta = trace.get("task_meta", {})
    record["hypergraph"] = build_hypergraph_view(record)
    return [*memory, record]


def increment_reuse_counts(
    memory: list[dict],
    memory_ids: Iterable[str],
    *,
    successful_memory_ids: Iterable[str] | None = None,
) -> list[dict]:
    """兼容旧接口：把传入记录视为既召回又进入推理。"""

    memory_ids = list(memory_ids)
    updated = record_memory_usage(
        memory,
        retrieved_memory_ids=memory_ids,
        adopted_memory_ids=memory_ids,
    )
    successful_ids = list(successful_memory_ids or [])
    return (
        apply_memory_outcome(updated, successful_ids, successful=True)
        if successful_ids
        else updated
    )


def record_memory_usage(
    memory: list[dict],
    *,
    retrieved_memory_ids: Iterable[str],
    adopted_memory_ids: Iterable[str],
) -> list[dict]:
    """分别记录候选召回与进入推理上下文，避免把命中直接算成复用。"""

    retrieved_ids = {str(memory_id) for memory_id in retrieved_memory_ids if memory_id}
    adopted_ids = {str(memory_id) for memory_id in adopted_memory_ids if memory_id}
    touched_ids = retrieved_ids | adopted_ids
    if not touched_ids:
        return memory

    now = datetime.now(timezone.utc).isoformat()
    updated: list[dict] = []
    for record in memory:
        memory_id = str(record.get("memory_id"))
        if memory_id in touched_ids:
            next_record = dict(record)
            next_record.setdefault("retrieval_count", 0)
            next_record.setdefault("reuse_count", 0)
            next_record.setdefault("reuse_success_count", 0)
            next_record.setdefault("reuse_failure_count", 0)
            if memory_id in retrieved_ids:
                next_record["retrieval_count"] = int(record.get("retrieval_count", 0)) + 1
                next_record["last_retrieved_at"] = now
            if memory_id in adopted_ids:
                next_record["reuse_count"] = int(record.get("reuse_count", 0)) + 1
                next_record["last_reused_at"] = now
            next_record["updated_at"] = now
            updated.append(next_record)
        else:
            updated.append(record)
    return updated


def apply_memory_outcome(
    memory: list[dict],
    adopted_memory_ids: Iterable[str],
    *,
    successful: bool,
) -> list[dict]:
    """在任务完成并由评测器确认结果后更新复用成败与生命周期。"""

    adopted_ids = {str(memory_id) for memory_id in adopted_memory_ids if memory_id}
    if not adopted_ids:
        return memory

    now = datetime.now(timezone.utc).isoformat()
    updated: list[dict] = []
    for record in memory:
        memory_id = str(record.get("memory_id"))
        if memory_id not in adopted_ids:
            updated.append(record)
            continue
        next_record = dict(record)
        next_record.setdefault("retrieval_count", 0)
        next_record.setdefault("reuse_count", 0)
        next_record.setdefault("reuse_success_count", 0)
        next_record.setdefault("reuse_failure_count", 0)
        if successful:
            next_record["reuse_success_count"] = int(record.get("reuse_success_count", 0)) + 1
        else:
            next_record["reuse_failure_count"] = int(record.get("reuse_failure_count", 0)) + 1

        next_record["lifecycle_stage"] = lifecycle_stage(next_record)
        next_record["last_evaluated_at"] = now
        next_record["updated_at"] = now
        next_record["record_version"] = int(record.get("record_version", 1) or 1) + 1
        updated.append(next_record)
    return updated


def _extract_query_structure(query: str) -> tuple[str, set[str], set[str], str]:
    task_type = ""
    group_id = ""
    tags: set[str] = set()
    triggers: set[str] = set()
    for raw_line in str(query or "").splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if lowered.startswith("task type:"):
            task_type = line.split(":", 1)[1].strip().lower()
        elif lowered.startswith("group id:"):
            group_id = line.split(":", 1)[1].strip().lower()
        elif lowered.startswith("tags:"):
            payload = line.split(":", 1)[1].strip()
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, list):
                    tags |= {str(item).strip().lower() for item in parsed if str(item).strip()}
            except json.JSONDecodeError:
                tags |= {token.strip().lower() for token in payload.split(",") if token.strip()}
    triggers |= set(_normalize_trigger_conditions(_sanitize_triggers(query)))
    return task_type, tags, triggers, group_id


def _rerank_memory_hits(hits: list[dict], query: str) -> list[dict]:
    query_task_type, query_tags, query_triggers, query_group_id = _extract_query_structure(query)
    if not any([query_task_type, query_tags, query_triggers, query_group_id]):
        return hits

    rescored: list[tuple[float, dict[str, Any]]] = []
    for index, hit in enumerate(hits):
        score = float(hit.get("retrieval_score", 0.0) or 0.0)
        memory_task_type = str(hit.get("task_type") or "").strip().lower()
        memory_tags = {
            str(item).strip().lower()
            for item in hit.get("tags", []) or []
            if str(item).strip()
        }
        memory_triggers = {
            str(item).strip()
            for item in hit.get("trigger_conditions", []) or []
            if str(item).strip()
        }
        memory_group = str(hit.get("group_id") or "").strip().lower()
        if query_task_type and memory_task_type == query_task_type:
            score += 8.0
        if query_tags and memory_tags:
            score += len(query_tags & memory_tags) * 3.0
        if query_triggers and memory_triggers:
            score += len(query_triggers & memory_triggers) * 4.0
        if query_group_id and memory_group == query_group_id:
            score += 4.0
        rescored.append((score, {**hit, "retrieval_score": round(score, 4), "_rank_index": index}))

    rescored.sort(
        key=lambda item: (
            -item[0],
            item[1]["_rank_index"],
        )
    )
    return [{k: v for k, v in hit.items() if k != "_rank_index"} for _, hit in rescored]


def search_memory(
    memory: list[dict],
    query: str,
    *,
    top_k: int,
    memory_types: Iterable[str] | None = None,
    type_limits: dict[str, int] | None = None,
    memory_layers: Iterable[str] | None = None,
    layer_limits: dict[str, int] | None = None,
) -> list[dict]:
    """使用超图索引检索记忆，并支持类型与三层语义过滤。"""

    if not memory:
        return []

    effective_type_limits = dict(type_limits or {})
    for layer, limit in (layer_limits or {}).items():
        normalized_layer = normalize_memory_layer(layer)
        if normalized_layer == "episodic":
            effective_type_limits.setdefault("evidence", int(limit))
        elif normalized_layer == "experience":
            effective_type_limits.setdefault("experience", int(limit))

    allowed_types = {str(memory_type) for memory_type in memory_types} if memory_types else None
    if effective_type_limits:
        allowed_types = {*allowed_types} if allowed_types else set()
        allowed_types.update(str(memory_type) for memory_type in effective_type_limits)
    allowed_layers = (
        {normalize_memory_layer(layer) for layer in memory_layers}
        if memory_layers
        else None
    )
    if layer_limits:
        allowed_layers = {*allowed_layers} if allowed_layers else set()
        allowed_layers.update(normalize_memory_layer(layer) for layer in layer_limits)

    searchable_memory = [
        record
        for record in memory
        if (not allowed_types or str(record.get("memory_type", "evidence")) in allowed_types)
        and (not allowed_layers or memory_layer_for_record(record) in allowed_layers)
    ]
    if not searchable_memory:
        return []

    hits = search_hypergraph_memory(
        searchable_memory,
        query,
        top_k=top_k,
        memory_types=allowed_types,
        type_limits=effective_type_limits or None,
    )
    ranked_hits = _rerank_memory_hits(hits, query)
    selected: list[dict] = []
    layer_counts: dict[str, int] = {}
    normalized_limits = {
        normalize_memory_layer(layer): max(0, int(limit))
        for layer, limit in (layer_limits or {}).items()
    }
    for hit in ranked_hits:
        enriched = dict(hit)
        layer = memory_layer_for_record(enriched)
        enriched["memory_layer"] = layer
        if layer in normalized_limits and layer_counts.get(layer, 0) >= normalized_limits[layer]:
            continue
        selected.append(enriched)
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        if len(selected) >= top_k:
            break
    return selected
