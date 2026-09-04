from __future__ import annotations

from agents.reasoner.agent import (
    _inferred_adopted_memory_ids,
    _reuse_previous_answer_if_needed,
)


MEMORY_CONTEXT = [
    {
        "memory_id": "mem-0001",
        "memory_type": "evidence",
        "answer": "Chief of Protocol",
        "summary": "Chief of Protocol",
    }
]


def test_explicit_cross_turn_restatement_uses_prior_answer() -> None:
    answer, memory_id = _reuse_previous_answer_if_needed(
        "Reuse the previous evidence chain. Which answer was supported by the facts?",
        "The answer was supported by the facts in the previous evidence chain.",
        MEMORY_CONTEXT,
    )

    assert answer == "Chief of Protocol"
    assert memory_id == "mem-0001"


def test_valid_short_model_answer_is_not_overridden() -> None:
    answer, memory_id = _reuse_previous_answer_if_needed(
        "Using the same supporting facts, answer again.",
        "United States ambassador",
        MEMORY_CONTEXT,
    )

    assert answer == "United States ambassador"
    assert memory_id == ""


def test_new_question_never_uses_prior_answer_fallback() -> None:
    answer, memory_id = _reuse_previous_answer_if_needed(
        "What government position did she hold?",
        "I cannot determine the requested position from the supplied material.",
        MEMORY_CONTEXT,
    )

    assert answer.startswith("I cannot determine")
    assert memory_id == ""


def test_adoption_requires_answer_overlap_not_just_materialization() -> None:
    assert _inferred_adopted_memory_ids(
        "Chief of Protocol",
        MEMORY_CONTEXT,
    ) == ["mem-0001"]
    assert _inferred_adopted_memory_ids(
        "United States ambassador",
        MEMORY_CONTEXT,
    ) == []
