from __future__ import annotations

import json
from types import SimpleNamespace

from agents.reasoner.agent import reasoner_agent
from competition_agents.handoff import memory_object
from competition_agents.protocol import make_compact_packet
from competition_agents.state import initial_state, put_latent_state, record_message


def test_reasoner_carries_and_reports_experience_rules() -> None:
    """Reasoner 应显式消费 experience_rules，并把使用痕迹写回 execution_result。"""

    state = initial_state("when was dunkirk released on dvd in the united kingdom?", "text", [])
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
    reasoning_input = json.dumps(
        memory_object(state["evidence"], state["memory_hits"], state["experience_rules"]),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    messages, metrics = record_message(
        state,
        sender="retriever",
        receiver="reasoner",
        action="reason",
        payload=reasoning_input,
    )
    state.update(messages=messages, metrics=metrics)

    result = reasoner_agent(state)

    assert result["execution_result"]["experience_used_count"] == 1
    assert result["execution_result"]["experience_used_memory_ids"] == ["mem-0002"]
    assert result["execution_result"]["experience_rules"][0]["solution_pattern"] == "choose_date_matching_constraints"


def test_reasoner_prefers_state_pool_payload_over_graph_semantic_fields(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class CapturingReasoner:
        def reason(self, **kwargs):
            captured.update(kwargs)
            return {
                "answer_candidate": "state-pool answer",
                "reasoning_summary": "resolved references",
                "limitations": [],
            }

    monkeypatch.setattr("agents.reasoner.agent.get_llm_client", lambda role: CapturingReasoner())
    monkeypatch.setattr(
        "competition_agents.state.embed_text",
        lambda text: SimpleNamespace(
            vector=[1.0, 0.0] if text == "correct plan" else [0.8, 0.2],
            provider="local",
            model="test-embedding",
        ),
    )
    state = initial_state("Question: test", "protocol", [])
    plan_ref, latent_states, metrics = put_latent_state(
        state,
        producer="planner",
        topic="test",
        text="correct plan",
        consumers=["reasoner"],
        semantic_payload={"plan": ["correct plan"]},
    )
    state = {**state, "latent_states": latent_states, "metrics": metrics}
    evidence_ref, latent_states, metrics = put_latent_state(
        state,
        producer="retriever",
        topic="test",
        text="correct evidence",
        consumers=["reasoner"],
        semantic_payload={
            "reasoning_input": json.dumps(
                memory_object(["correct evidence"], [], []),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "evidence": ["correct evidence"],
            "memory_hits": [],
            "experience_hits": [],
            "experience_rules": [],
            "memory_context": [],
        },
    )
    state.update(
        {
            "plan": ["wrong graph plan"],
            "evidence": ["wrong graph evidence"],
            "latent_states": latent_states,
            "metrics": metrics,
            "state_refs": {"plan": plan_ref, "evidence": evidence_ref},
        }
    )
    packet = make_compact_packet(
        sender="retriever",
        receiver="reasoner",
        action="reason",
        content_type="latent.embedding",
        state_refs=[evidence_ref, plan_ref],
    )
    messages, metrics = record_message(
        state,
        sender="retriever",
        receiver="reasoner",
        action="reason",
        payload=packet,
        metrics=metrics,
    )
    state.update(messages=messages, metrics=metrics)

    result = reasoner_agent(state)

    assert captured["evidence"] == ["correct evidence"]
    assert result["metrics"]["state_store_read_count"] == 2
    assert result["metrics"]["latent_use_count"] == 2
    assert result["execution_result"]["latent_alignment_score"] is not None
