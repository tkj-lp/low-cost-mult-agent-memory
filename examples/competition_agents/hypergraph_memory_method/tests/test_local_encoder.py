from __future__ import annotations

import os

import pytest

from hypergraph_memory_method import (
    MemoryService,
    RetrievalRequest,
    SentenceTransformerEncoder,
)


@pytest.mark.skipif(
    not os.environ.get("MEMORY_TEST_MODEL_PATH"),
    reason="local model integration is opt-in",
)
def test_real_local_encoder_can_recall_and_reuse_query_cache(tmp_path):
    encoder = SentenceTransformerEncoder(
        os.environ["MEMORY_TEST_MODEL_PATH"],
        model_id="local-integration-test:minilm:normalized",
    )
    with MemoryService(tmp_path / "local.db", encoder=encoder) as service:
        service.begin_batch(
            "local-check",
            task_family="continuous_qa",
            dataset_id="local-semantic-fixture",
        )
        receipt = service.write_knowledge(
            "local-check",
            [
                {
                    "source_agent": "fixture",
                    "task_topic": "vehicle service",
                    "summary": "Inspect the car tires and change the engine oil regularly.",
                },
                {
                    "source_agent": "fixture",
                    "task_topic": "food preparation",
                    "summary": "Mix flour, bananas and eggs before baking the bread.",
                },
                {
                    "source_agent": "fixture",
                    "task_topic": "weather",
                    "summary": "A thunderstorm brought heavy rain to the coast.",
                },
            ],
            operation_id="seed",
        )
        request = RetrievalRequest(
            "local-check",
            "automobile maintenance",
            layers=("knowledge",),
            mode="fusion",
            top_k=1,
        )
        first = service.retrieve(request)
        second = service.retrieve(request)
        assert first["hits"][0]["memory_id"] == receipt["records"][0]["memory_id"]
        assert first["hits"] == second["hits"]
        assert second["diagnostics"]["query_embedding_cache_hit"] is True
        assert second["diagnostics"]["vector_cache_hit"] is True
