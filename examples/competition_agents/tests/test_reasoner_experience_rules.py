from __future__ import annotations

from agents.reasoner.agent import reasoner_agent
from competition_agents.state import initial_state


def test_reasoner_carries_and_reports_experience_rules() -> None:
    """Reasoner 应显式消费 experience_rules，并把使用痕迹写回 execution_result。"""

    state = initial_state("when was dunkirk released on dvd in the united kingdom?", "protocol", [])
    state["plan"] = [
        "Retrieve the film release sentence.",
        "Check the United Kingdom constraint.",
        "Select the DVD release date.",
    ]
    state["evidence"] = [
        'Dunkirk (2017 film): "Dunkirk" was released digitally on 12 December 2017, and on DVD on 18 December in the United Kingdom.',
    ]
    state["memory_hits"] = [
        {"memory_id": "mem-0001", "memory_type": "evidence", "summary": "18 December 2017"},
    ]
    state["experience_hits"] = [
        {
            "memory_id": "mem-0002",
            "memory_type": "experience",
            "summary": "经验规则卡片：problem=temporal_slot_selection; solution=choose_date_matching_constraints",
            "problem_pattern": "temporal_slot_selection",
            "solution_pattern": "choose_date_matching_constraints",
            "failure_pattern": "wrong_date_slot",
            "trigger_conditions": ["question:when", "constraint:dvd_release"],
        }
    ]
    state["experience_rules"] = [
        {
            "memory_id": "mem-0002",
            "problem_pattern": "temporal_slot_selection",
            "solution_pattern": "choose_date_matching_constraints",
            "failure_pattern": "wrong_date_slot",
            "trigger_conditions": ["question:when", "constraint:dvd_release"],
            "supporting_evidence": ['Dunkirk (2017 film): DVD on 18 December in the United Kingdom.'],
            "confidence": 0.9,
        }
    ]
    state["memory_context"] = [
        {
            "memory_id": "mem-0002",
            "memory_type": "experience",
            "summary": "经验规则卡片",
            "solution_pattern": "choose_date_matching_constraints",
        }
    ]

    result = reasoner_agent(state)

    assert result["execution_result"]["experience_used_count"] == 1
    assert result["execution_result"]["experience_used_memory_ids"] == ["mem-0002"]
    assert result["execution_result"]["experience_rules"][0]["solution_pattern"] == "choose_date_matching_constraints"
