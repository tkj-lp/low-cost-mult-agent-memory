from __future__ import annotations

from hypergraph_memory_method.context import search_context


def test_context_hypergraph_expands_across_shared_bridge_entity() -> None:
    task_meta = {
        "task_id": "chain-1:turn-1",
        "group_id": "chain-1",
        "turn_id": "turn-1",
        "dataset_type": "continuous",
        "question": (
            "What government position was held by the woman who portrayed "
            "Corliss Archer in Kiss and Tell?"
        ),
        "context": [
            {
                "title": "Kiss and Tell",
                "sentences": [
                    "Kiss and Tell starred Shirley Temple as Corliss Archer.",
                    "The film is an American comedy.",
                ],
            },
            {
                "title": "Shirley Temple",
                "sentences": [
                    "Shirley Temple was an actress and diplomat.",
                    "She served as Chief of Protocol of the United States.",
                ],
            },
            {
                "title": "Unrelated government office",
                "sentences": ["This government position was created in 2003."],
            },
        ],
        "expected_answer": "an intentionally wrong label",
        "supporting_facts": [["Unrelated government office", 0]],
    }

    hits = search_context(task_meta, top_k=4)
    target = next(hit for hit in hits if "Chief of Protocol" in hit["text"])

    assert target["graph_expansion_score"] > 0
    assert "shirley temple" in target["bridge_entities"]
    assert target["bridge_source_context_ids"]


def test_context_labels_cannot_change_online_ranking() -> None:
    base = {
        "task_id": "task-1",
        "dataset_type": "multihop",
        "question": "Which role did Ada Lovelace hold?",
        "context": [
            {
                "title": "Ada Lovelace",
                "sentences": ["Ada Lovelace was an English mathematician and writer."],
            },
            {
                "title": "Distractor",
                "sentences": ["A role can describe a government office."],
            },
        ],
    }
    changed_labels = {
        **base,
        "expected_answer": "Distractor",
        "supporting_facts": [["Distractor", 0]],
    }

    base_ids = [hit["context_id"] for hit in search_context(base, top_k=2)]
    changed_ids = [
        hit["context_id"] for hit in search_context(changed_labels, top_k=2)
    ]

    assert changed_ids == base_ids
