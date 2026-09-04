from __future__ import annotations

from competition_agents.state_store import StateStore


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
