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


def test_coerce_answer_candidate_recovers_full_catabolism_hormone_list() -> None:
    """列表题应优先返回证据中的完整集合，而不是只截前半句。"""

    answer_candidate = "cortisol, glucagon, and adrenaline"
    evidence = [
        (
            "Catabolism: Classic catabolic hormones include cortisol, glucagon, and adrenaline. "
            "Additional hormones with catabolic effects include cytokines, orexin, and melatonin."
        )
    ]

    assert _coerce_answer_candidate(
        answer_candidate,
        evidence,
        question="which hormones are related to catabolism?",
    ) == "Cortisol, glucagon, adrenaline, cytokines, orexin, and melatonin"


def test_coerce_answer_candidate_prefers_yes_for_yes_no_reference_question() -> None:
    """是非题不要误把相关实体短语当成最终答案。"""

    answer_candidate = "adult cow pancreas"
    evidence = [
        "Insulin: Researchers later succeeded in extracting insulin from the adult cow pancreas.",
    ]

    assert _coerce_answer_candidate(
        answer_candidate,
        evidence,
        question="could that substance also be extracted from an animal organ?",
    ) == "Yes"


def test_coerce_answer_candidate_does_not_treat_information_request_as_yes_no() -> None:
    """can you / could you 这类信息请求不应被当成是非题。"""

    answer_candidate = "Fionn Whitehead, Tom Glynn-Carney, Jack Lowden, Harry Styles"
    evidence = [
        (
            "Dunkirk (2017 film): Its ensemble cast includes Fionn Whitehead, Tom Glynn-Carney, "
            "Jack Lowden, Harry Styles, Aneurin Barnard, James D'Arcy, Barry Keoghan, Kenneth Branagh, "
            "Cillian Murphy, Mark Rylance, and Tom Hardy."
        )
    ]

    assert _coerce_answer_candidate(
        answer_candidate,
        evidence,
        question="can you mention a few members of the cast?",
    ) == "Fionn Whitehead, Tom Glynn-Carney, Jack Lowden, Harry Styles"
