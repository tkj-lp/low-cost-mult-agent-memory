from run_demo import evaluate_task_answer, task_meta_from_item
from prepare_official_references import rebuild


def test_multiple_references_accept_valid_short_answer():
    task = {"expected_answer": "Alpha and additional details", "reference_answers": ["Alpha and additional details", "Alpha"]}
    assert evaluate_task_answer("Alpha", task) == (True, 1.0)


def test_strict_em_is_not_substring_matching():
    task = {"reference_answers": ["Alpha beta"]}
    em, f1 = evaluate_task_answer("Alpha", task)
    assert em is False
    assert abs(f1 - 2 / 3) < 1e-9


def test_reference_labels_are_not_agent_metadata():
    assert "reference_answers" not in task_meta_from_item({"reference_answers": ["secret gold"]})
    assert "reference_annotations" not in task_meta_from_item({"reference_annotations": [{"Answer": "secret gold"}]})


def test_conversion_matches_questions_and_retains_reference_evidence():
    groups = [{"group_id": "topiocqa-official-001", "source_dataset": "TopiOCQA", "original_conversation_no": 999,
               "tasks": [{"question": "Which category?", "turn_id": "turn-1"}], "context": []}]
    rows = [{"Conversation_no": 7, "Turn_no": 1, "Question": "Which category?", "Answer": "original label",
             "Topic": "Example", "Topic_section": "Introduction", "Rationale": "Original evidence",
             "Additional_answers": [{"Answer": "alternate label", "Topic": "Example", "Topic_section": "Classification",
                                     "Rationale": "Classification evidence"}]}]
    group = rebuild(groups, rows, add_evidence=True)[0]
    assert group["original_conversation_no"] == 7
    assert group["tasks"][0]["reference_answers"] == ["original label", "alternate label"]
    assert group["context"][1]["sentences"] == ["Classification evidence"]
    assert all("Answer" not in item for item in group["context"])
