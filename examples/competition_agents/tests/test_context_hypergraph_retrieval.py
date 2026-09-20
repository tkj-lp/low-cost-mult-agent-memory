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


def test_topiocqa_indexes_batch_documents_with_section_and_provenance() -> None:
    task = {
        "task_id": "sample:turn-2",
        "dataset_type": "topiocqa",
        "question": "Which category includes this object?",
        "topic": "Sample object",
        "supporting_passage": "The sample object has a smooth surface.",
        "context": [{"title": "Sample object [SEP] Classification", "source": "source_document",
                     "sentences": ["The sample object belongs to category Delta."]}],
    }
    hits = search_context(task, top_k=4)
    target = next(hit for hit in hits if "category Delta" in hit["text"])
    assert target["kind"] == "sentence"
    assert target["title"] == "Sample object"
    assert target["section"] == "Classification"
    assert target["metadata"]["source"] == "source_document"
    assert target["metadata"]["sentence_range"] == [0, 1]


def test_topiocqa_windows_keep_adjacent_context_but_not_entire_document() -> None:
    task = {
        "task_id": "sample:turn-3", "dataset_type": "topiocqa", "question": "Where is ZoneBeta protected?",
        "context": [{"title": "Reserve", "sentences": [
            "Unrelated opening.", "Protection is provided in the following reserves:",
            "ZoneBeta reserve;", "ZoneGamma reserve.", "Unrelated ending.",
        ]}],
    }
    hits = search_context(task, top_k=10)
    target = next(hit for hit in hits if hit["metadata"].get("sentence_index") == 2)
    assert "Protection is provided" in target["text"]
    assert "ZoneGamma" in target["text"]
    assert "Unrelated opening" not in target["text"]
    assert "Unrelated ending" not in target["text"]


def test_topiocqa_reference_labels_cannot_change_knowledge_hits() -> None:
    task = {"task_id": "sample:turn-4", "dataset_type": "topiocqa", "question": "Which category?",
            "context": [{"title": "Example", "sentences": ["Category Delta includes the sample."]}]}
    changed = {**task, "reference_answers": ["secret"], "expected_answer": "secret",
               "reference_annotations": [{"Answer": "secret", "Rationale": "Do not index this label"}]}
    base_hits = search_context(task)
    assert base_hits == search_context(changed)
    assert all("secret" not in hit["text"] for hit in base_hits)


def test_topiocqa_deduplicates_document_and_supporting_passage() -> None:
    task = {"task_id": "sample:turn-5", "dataset_type": "topiocqa", "question": "Which category?",
            "topic": "Example", "topic_section": "Classification", "supporting_passage": "Category Delta.",
            "context": [{"title": "Example [SEP] Classification", "sentences": ["Category Delta."]}]}
    hits = search_context(task, top_k=10)
    assert len(hits) == 1


def test_excluded_history_cannot_displace_knowledge_before_top_k() -> None:
    task = {"task_id": "sample:turn-6", "dataset_type": "topiocqa", "question": "Which category?",
            "conversation_context": ["Which category?", "Which category?"],
            "context": [{"title": "Example", "sentences": ["Category Delta includes the sample."]}]}
    hits = search_context(task, top_k=1, exclude_kinds={"dialogue_question", "dialogue_answer"})
    assert len(hits) == 1
    assert hits[0]["kind"] == "sentence"


def test_question_terms_outweigh_conflicting_section_hint() -> None:
    task = {"task_id": "sample:turn-7", "dataset_type": "topiocqa", "question": "What material is used?",
            "topic": "Sample", "context": [
                {"title": "Sample [SEP] Introduction", "sentences": ["Sample is an example object."]},
                {"title": "Sample [SEP] Construction", "sentences": ["The material used is titanium."]},
            ]}
    hits = search_context(task, query="Sample Introduction", top_k=1)
    assert "titanium" in hits[0]["text"]
