from __future__ import annotations

from competition_agents.state import initial_state, put_latent_state, resolve_state_ref
from competition_agents.state_store import StateStore, clear_state_store


def test_state_store_resolves_payload_without_exposing_record_wrapper() -> None:
    store = StateStore()
    store.put("retrieval-1", {"content_type": "retrieval.bundle", "evidence": ["fact"]})

    assert store.resolve("retrieval-1", expected_content_type="retrieval.bundle") == {
        "content_type": "retrieval.bundle",
        "evidence": ["fact"],
    }
    assert store.resolve("retrieval-1", expected_content_type="latent.embedding") is None


def test_state_store_releases_task_scope_and_expired_items() -> None:
    store = StateStore()
    store.put("task-state", {"value": 1}, scope="task", ttl_seconds=60)
    store.put("session-state", {"value": 2}, scope="session", ttl_seconds=60)
    store.put("expired-state", {"value": 3}, scope="task", ttl_seconds=0)

    assert store.resolve("expired-state") is None
    assert store.clear_scope("task") == 1
    assert store.resolve("task-state") is None
    assert store.resolve("session-state") == {"value": 2}


def test_state_ref_resolution_tracks_authorized_data_plane_reads() -> None:
    clear_state_store()
    state = initial_state("task", "protocol")
    state_ref, latent_states, metrics = put_latent_state(
        state,
        producer="planner",
        topic="task",
        text="retrieve evidence",
        consumers=["retriever"],
        semantic_payload={"plan": ["retrieve evidence"]},
    )
    state = {**state, "latent_states": latent_states, "metrics": metrics}

    denied, denied_metrics = resolve_state_ref(
        state,
        state_ref,
        consumer="summarizer",
    )
    payload, read_metrics = resolve_state_ref(
        state,
        state_ref,
        consumer="retriever",
        expected_content_types={"latent.embedding"},
    )

    assert denied == {}
    assert denied_metrics["state_store_read_count"] == 0
    assert payload["semantic_payload"]["plan"] == ["retrieve evidence"]
    assert read_metrics["state_store_read_count"] == 1
    assert read_metrics["state_store_read_bytes"] > 0
    assert read_metrics["latent_use_count"] == 1
