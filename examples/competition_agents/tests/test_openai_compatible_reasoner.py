from __future__ import annotations

from competition_agents.llm.openai_compatible import _coerce_answer_candidate


def test_coerce_answer_candidate_prefers_single_position_from_evidence() -> None:
    """当模型并列输出多个职位时，应收束成单一答案。"""

    answer_candidate = (
        "United States ambassador to Ghana and to Czechoslovakia and "
        "Chief of Protocol of the United States"
    )
    evidence = [
        (
            "Shirley Temple: As an adult, she was named United States ambassador to Ghana "
            "and to Czechoslovakia and also served as Chief of Protocol of the United States."
        )
    ]

    assert _coerce_answer_candidate(answer_candidate, evidence) == "Chief of Protocol"


def test_coerce_answer_candidate_keeps_normal_single_answer() -> None:
    """正常单答案不应被无关改写。"""

    answer_candidate = "Chief of Protocol"
    evidence = [
        "Shirley Temple: As an adult, she also served as Chief of Protocol of the United States."
    ]

    assert _coerce_answer_candidate(answer_candidate, evidence) == "Chief of Protocol"
