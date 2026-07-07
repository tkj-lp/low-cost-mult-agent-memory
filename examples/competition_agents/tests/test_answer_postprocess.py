from __future__ import annotations

from competition_agents.llm.openai_compatible import _coerce_answer_candidate
from run_demo import answers_match


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


def test_dvd_release_question_extracts_full_uk_date() -> None:
    answer = _coerce_answer_candidate(
        "December 2017",
        [
            (
                "Dunkirk (2017 film): \"Dunkirk\" was released digitally on "
                "12 December 2017, and on 4K Ultra HD, Blu-ray, and DVD on "
                "18 December in the United Kingdom and 19 December in the United States."
            )
        ],
        question="when was the 2017 film dunkirk released on dvd in the united kingdom?",
    )

    assert answer == "18 December 2017"


def test_film_about_question_extracts_depicted_event() -> None:
    answer = _coerce_answer_candidate(
        "Dunkirk",
        [
            (
                "Dunkirk (2017 film): Dunkirk is a 2017 war film written, directed, "
                "and produced by Christopher Nolan that depicts the Dunkirk evacuation "
                "of World War II."
            )
        ],
        question="what is the film about?",
    )

    assert answer == "The Dunkirk evacuation of World War II"
    assert not answers_match("Dunkirk", "The Dunkirk evacuation of World War II")


def test_cast_question_extracts_first_four_cast_members() -> None:
    answer = _coerce_answer_candidate(
        "Tom Hanks, Leonardo DiCaprio, Brad Pitt",
        [
            (
                "Dunkirk (2017 film): Its ensemble cast includes Fionn Whitehead, "
                "Tom Glynn-Carney, Jack Lowden, Harry Styles, Aneurin Barnard, "
                "James D'Arcy, Barry Keoghan, Kenneth Branagh, Cillian Murphy, "
                "Mark Rylance, and Tom Hardy."
            )
        ],
        question="can you mention a few members of the cast?",
    )

    assert answer == "Fionn Whitehead, Tom Glynn-Carney, Jack Lowden, Harry Styles"


def test_fish_question_extracts_complete_fish_list() -> None:
    answer = _coerce_answer_candidate(
        "Common types of fish found in the ocean include tuna, salmon, mackerel, and cod.",
        [
            (
                "North Sea: Cod, haddock, whiting, saithe, plaice, sole, mackerel, "
                "herring, pouting, sprat, and sandeel are all very common in the "
                "North Sea and are fished commercially."
            )
        ],
        question="what kinds of fish are commonly found in this sea?",
    )

    assert answer == "Cod, haddock, whiting, saithe, plaice, sole, mackerel, herring, pouting, sprat, and sandeel"
