from __future__ import annotations

import json
from pathlib import Path

import reassess_reuse


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ANNOTATIONS = DATA / "qa_reuse_annotations.json"


def test_factoid_annotations_cover_six_groups():
    payload = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    groups = payload["groups"]
    assert set(groups) == {
        "topiocqa-factoid-ranked-001",
        "topiocqa-factoid-ranked-002",
        "topiocqa-factoid-ranked-003",
        "qrecc-factoid-ranked-001",
        "qrecc-factoid-ranked-002",
        "qrecc-factoid-ranked-003",
    }
    for group_id, turns in groups.items():
        assert len(turns) == 10
        for turn_id, entry in turns.items():
            assert turn_id.startswith("turn-")
            assert isinstance(entry["sources"], list)
            assert entry["note"]


def test_annotate_groups_writes_turn_answer_specs():
    annotations = reassess_reuse.load_annotations()
    groups = [
        {
            "group_id": "topiocqa-factoid-ranked-001",
            "tasks": [
                {"turn_id": "turn-1", "question": "where is this band from ?"},
                {"turn_id": "turn-6", "question": "who is the lead singer of the band ?"},
                {"turn_id": "turn-7", "question": "what is he's full name ?"},
            ],
        }
    ]
    annotated = reassess_reuse.annotate_groups(groups, annotations)
    assert annotated[0]["tasks"][0]["expected_reuse"] == []
    assert annotated[0]["tasks"][2]["expected_reuse"] == ["turn-6.answer"]
    assert annotated[0]["tasks"][2]["reuse_annotation_version"] == annotations["version"]


def test_reassess_run_scores_only_historical_qa_hits():
    annotations = reassess_reuse.load_annotations()
    question = annotations["groups"]["topiocqa-factoid-ranked-001"]["turn-7"]["question"]
    run = {
        "task_results": [
            {
                "task_id": "topiocqa-factoid-ranked-001:turn-7",
                "group_id": "topiocqa-factoid-ranked-001",
                "turn_id": "turn-7",
                "question": question,
                "exact_match": True,
                "memory_hits": [
                    {
                        "memory_id": "qa-6",
                        "memory_type": "evidence",
                        "summary": "Bono",
                        "source_case": {
                            "group_id": "topiocqa-factoid-ranked-001",
                            "turn_id": "turn-6",
                            "question": annotations["groups"]["topiocqa-factoid-ranked-001"]["turn-6"]["question"],
                        },
                    },
                    {
                        "memory_id": "exp-1",
                        "memory_type": "experience",
                        "summary": "prefer short answers",
                        "source_case": {
                            "group_id": "topiocqa-factoid-ranked-001",
                            "turn_id": "turn-2",
                        },
                    },
                ],
                "execution_result": {
                    "memory_used_ids": ["qa-6", "exp-1"],
                    "evidence_used_memory_ids": ["qa-6"],
                    "experience_used_memory_ids": ["exp-1"],
                    "cited_memory_ids": ["qa-6"],
                    "memory_citations": [
                        {"memory_id": "qa-6"},
                        {"memory_id": "exp-1"},
                    ],
                },
            }
        ]
    }
    report = reassess_reuse.reassess_run(run, annotations)
    assert report is not None
    assert report["available"] is True
    assert report["metric_scope"] == "historical_qa_only"
    summary = report["summary"]
    assert summary["expected_reuse_task_count"] == 1
    assert summary["expected_artifact_recall"] == 1.0
    assert summary["memory_precision"] == 1.0
    assert summary["correct_reuse_rate"] == 1.0
    row = report["rows"][0]
    assert row["scored_expected_reuse"] == ["turn-6.answer"]
    assert row["matched_expected_reuse"] == ["turn-6.answer"]
    assert row["used_memory_ids"] == ["qa-6"]
    assert all(detail["memory_id"] != "exp-1" for detail in row["hit_details"])
