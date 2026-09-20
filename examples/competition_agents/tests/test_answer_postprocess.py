from __future__ import annotations

from competition_agents.llm.openai_compatible import (
    _answer_repair_reason,
    _coerce_answer_candidate,
)
from run_demo import answer_token_f1, answers_match


def test_answer_cleanup_removes_only_output_wrapper() -> None:
    answer = _coerce_answer_candidate(
        "Final answer: A reusable answer span",
        ["Evidence contains a different phrase."],
        question="What is requested?",
    )

    assert answer == "A reusable answer span"


def test_answer_cleanup_does_not_inject_fact_from_evidence() -> None:
    answer = _coerce_answer_candidate(
        "Model-selected candidate",
        ["Evidence: benchmark-specific gold answer"],
        question="Which value is correct?",
    )

    assert answer == "Model-selected candidate"


def test_grounded_location_is_not_rejected_by_incomplete_relation_patterns() -> None:
    reason = _answer_repair_reason(
        task="Question: where is this band from?\nResolved topic: U2",
        evidence=[
            "U2 are an Irish rock band from Dublin, formed in 1976.",
            "The song was released in the Netherlands and West Germany.",
        ],
        answer="Dublin",
    )

    assert reason == ""


def test_answers_match_accepts_referential_conversational_gold() -> None:
    assert answers_match(
        "An oribi is a small antelope found in eastern, southern and western Africa.",
        "It is a small antelope.",
    )


def test_answers_match_rejects_partial_topic_overlap() -> None:
    assert not answers_match("Dunkirk", "The Dunkirk evacuation of World War II")


def test_answer_token_f1_credits_partial_but_not_exact_answer() -> None:
    prediction = "eastern, southern and western Africa"
    gold = "Eastern, southern and western Africa from savannahs and grasslands"

    score = answer_token_f1(prediction, gold)

    assert 0.5 < score < 1.0
    assert not answers_match(prediction, gold)


def test_answers_match_accepts_superset_with_all_expected_list_entities() -> None:
    gold = "Some species include alpha bird, beta fish, gamma deer, and delta cat"
    prediction = "alpha bird, beta fish, gamma deer, delta cat, epsilon fox"

    assert answers_match(prediction, gold)


def test_answers_match_accepts_entity_without_appositive_description() -> None:
    assert answers_match(
        "sample bush",
        "Sample bush, an ornamental species of the example family",
    )
