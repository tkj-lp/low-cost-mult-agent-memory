from __future__ import annotations

from competition_agents.llm.openai_compatible import _coerce_answer_candidate


def test_government_position_question_prefers_evidence_span() -> None:
    """职位题应从证据中抽取职位短语，避免本地小模型只返回人物名。"""

    answer = _coerce_answer_candidate(
        "Shirley Temple",
        [
            (
                "Shirley Temple: As an adult, she was named United States ambassador "
                "to Ghana and to Czechoslovakia and also served as Chief of Protocol "
                "of the United States."
            )
        ],
        question="What government position was held by the woman who portrayed Corliss Archer?",
    )

    assert answer == "Chief of Protocol"
