"""Attach evaluation-only QA dependencies and rescore saved experiments."""

import argparse
import hashlib
import json
from pathlib import Path

from hypergraph_memory_method.layers import memory_layer_for_record
from hypergraph_memory_method.reuse_metrics import evaluate_run_reuse

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"
ANNOTATIONS = DATA / "qa_reuse_annotations.json"


def load_annotations():
    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    for path in DATA.glob("official_factoid_ranked_*_00?_10turns.json"):
        for group in json.loads(path.read_text(encoding="utf-8")):
            for task in group["tasks"]:
                annotations["groups"][group["group_id"]][task["turn_id"]]["question"] = task["question"]
    return annotations


def annotate_groups(groups, annotations):
    for group in groups:
        entries = annotations["groups"].get(group.get("group_id"))
        if entries is None:
            continue
        available = {t["turn_id"] for t in group["tasks"]}
        for task in group["tasks"]:
            entry = entries[task["turn_id"]]
            current = int(task["turn_id"].split("-")[-1])
            if any(source >= current or f"turn-{source}" not in available for source in entry["sources"]):
                raise ValueError("Invalid historical source in annotation")
            task["expected_reuse"] = [f"turn-{source}.answer" for source in entry["sources"]]
            task["reuse_annotation_note"] = entry["note"]
            task["reuse_annotation_version"] = annotations["version"]
    return groups


def reassess_run(run, annotations):
    tasks = run.get("task_results", [])
    if not tasks or any(t.get("group_id") not in annotations["groups"] for t in tasks):
        return None
    prepared = []
    for task in tasks:
        entry = annotations["groups"][task["group_id"]].get(task["turn_id"])
        if entry is None:
            raise ValueError("Unknown task turn")
        if task.get("question") != entry["question"]:
            raise ValueError("Saved question does not match the annotated dataset")
        qa_hits = [h for h in task.get("memory_hits", [])
                   if memory_layer_for_record(h) == "episodic"
                   and h.get("memory_type") != "failed_experience"]
        qa_ids = {h.get("memory_id") for h in qa_hits}
        execution = dict(task.get("execution_result", {}))
        for key in ("memory_used_ids", "evidence_used_memory_ids", "experience_used_memory_ids", "cited_memory_ids"):
            execution[key] = [i for i in execution.get(key, []) if i in qa_ids]
        execution["memory_citations"] = [c for c in execution.get("memory_citations", [])
                                        if c.get("memory_id") in qa_ids]
        prepared.append({**task, "memory_hits": qa_hits, "execution_result": execution,
                         "expected_reuse": [f"turn-{s}.answer" for s in entry["sources"]]})
    report = evaluate_run_reuse({"task_results": prepared})
    # Missing source provenance cannot be treated as a measured retrieval failure.
    untraceable = any(not detail["source_turn"] for row in report["rows"]
                      if row["requires_memory_reuse"] for detail in row["hit_details"])
    report["available"] = not untraceable and all("memory_hits" in t for t in tasks)
    report["metric_scope"] = "historical_qa_only"
    report["annotation_version"] = annotations["version"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "run_result")
    parser.add_argument("--annotate-datasets", action="store_true")
    args = parser.parse_args()
    annotations = load_annotations()
    if args.annotate_datasets:
        for path in DATA.glob("official_factoid_ranked_*10turns.json"):
            groups = json.loads(path.read_text(encoding="utf-8"))
            groups = annotate_groups(groups, annotations)
            path.write_text(json.dumps(groups, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print("Annotated:", path.name)
    paths = [args.results] if args.results.is_file() else args.results.rglob("frontend_*.json")
    count = 0
    for path in paths:
        raw = path.read_bytes()
        payload = json.loads(raw)
        reports = [reassess_run(run, annotations) for run in payload.get("runs", [])]
        if not any(report is not None for report in reports):
            continue
        stamp = path.stat()
        sidecar = {"source_size": stamp.st_size, "source_mtime_ns": stamp.st_mtime_ns,
                   "source_sha256": hashlib.sha256(raw).hexdigest(), "reports": reports}
        path.with_suffix(".reuse-evaluation").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        count += 1
        print("Reassessed:", path.name)
    print("Saved QA reuse reports:", count)


if __name__ == "__main__":
    main()
