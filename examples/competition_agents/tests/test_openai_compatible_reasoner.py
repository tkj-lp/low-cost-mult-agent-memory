from __future__ import annotations

from competition_agents.llm.openai_compatible import (
    _answer_repair_reason,
    _coerce_answer_candidate,
    _extractive_fallback,
)


def test_coerce_answer_candidate_preserves_grounded_model_answer() -> None:
    answer = "Eastern, southern and western Africa"
    evidence = [
        "The species is found in eastern, southern and western Africa.",
    ]

    assert _coerce_answer_candidate(answer, evidence) == answer


def test_coerce_answer_candidate_normalizes_whitespace() -> None:
    assert _coerce_answer_candidate("  Red   sunbird bush  ", []) == "Red sunbird bush"


def test_coerce_answer_candidate_is_dataset_independent() -> None:
    candidate = "Candidate chosen by the model"
    evidence = ["A passage containing a tempting alternative answer."]

    assert _coerce_answer_candidate(
        candidate,
        evidence,
        question="What attribute is requested?",
    ) == candidate


def test_validator_rejects_topic_repetition() -> None:
    assert _answer_repair_reason(
        task="Question: Where is it located?\nResolved topic: Example Park",
        evidence=["Example Park is located in Example Country."],
        answer="Example Park",
    ) == "the answer merely repeats the resolved topic"


def test_validator_rejects_hyphenated_topic_repetition() -> None:
    assert _answer_repair_reason(
        task="Question: Where is it located?\nResolved topic: Example Sample Park",
        evidence=["Example-Sample Park is located in Example Country."],
        answer="Example-Sample Park",
    ) == "the answer merely repeats the resolved topic"


def test_validator_accepts_grounded_location() -> None:
    assert not _answer_repair_reason(
        task="Question: Where is it located?\nResolved topic: Example Park",
        evidence=["Example Park is located in Example Country."],
        answer="Example Country",
    )


def test_validator_rejects_ungrounded_long_list() -> None:
    assert _answer_repair_reason(
        task="Question: Which species are present?\nResolved topic: Example Park",
        evidence=["Species include alpha fox and beta deer."],
        answer="alpha fox, invented bird, invented fish, invented cat",
    ) == "most answer terms are not grounded in evidence"


def test_validator_rejects_entity_outside_requested_location_relation() -> None:
    assert _answer_repair_reason(
        task="Question: Where is this park located?\nResolved topic: Example Park",
        evidence=[
            "Example Park is in Example Country. "
            "Its highest point is Sample Mountain."
        ],
        answer="Sample Mountain",
    ) == "the answer is mentioned in evidence but is not a location-relation value"


def test_extractive_fallback_handles_generic_definition_location_and_list() -> None:
    assert _extractive_fallback(
        task="Question: What is an example unit?\nResolved topic: Example Unit",
        evidence=["The example unit is a compact measurement device found in laboratories."],
    ) == "a compact measurement device"
    assert _extractive_fallback(
        task="Question: Where is this site located?\nResolved topic: Example Site",
        evidence=["Example Site is a protected site in Example Country."],
    ) == "Example Country"
    assert _extractive_fallback(
        task="Question: Which species live there?\nResolved topic: Example Site",
        evidence=["Species include alpha bird, beta fish, and gamma deer."],
    ) == "alpha bird, beta fish, and gamma deer"


def test_validator_rejects_attribute_label_but_keeps_grounded_specific_entity() -> None:
    task = "Question: Tell me about its diet.\nResolved topic: Example Animal"
    evidence = ["The example animal eats fresh grasses and flowers."]
    assert _answer_repair_reason(task=task, evidence=evidence, answer="diet")

    species_task = "Question: And plant species?\nResolved topic: Example Park"
    species_evidence = ["A population of the red sample bush was recorded in the park."]
    assert not _answer_repair_reason(
        task=species_task,
        evidence=species_evidence,
        answer="red sample bush",
    )


def test_validator_requires_complete_generic_answer_shapes() -> None:
    assert _answer_repair_reason(
        task="Question: Which species live there?\nResolved topic: Example Park",
        evidence=["Species include alpha bird, beta fish, and gamma deer."],
        answer="three species",
    )
    assert _answer_repair_reason(
        task="Question: How does it look?\nResolved topic: Example Object",
        evidence=["It has a narrow body, a red surface, and two long handles."],
        answer="red surface",
    ) == "the descriptive answer omits the requested evidence details"
    assert _answer_repair_reason(
        task="Question: Is it protected?\nResolved topic: Example Site",
        evidence=["The site is protected under national law."],
        answer="Yes",
    ) == "the yes/no answer lacks an evidence-backed qualification"
