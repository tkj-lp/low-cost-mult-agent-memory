"""Restore official TopiOCQA multi-reference labels without task-specific rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path


SOURCE_URL = (
    "https://huggingface.co/datasets/McGill-NLP/TopiOCQA/resolve/"
    "543f3fc424da32c8b024fb5ae182946860be5517/data/topiocqa_valid.jsonl"
)
SOURCE_SHA256 = "a1598add3b55eb093d8642adb73bc7b7e380af9a80eb4c4c0fc0a48a35b2d667"


def rebuild(groups: list[dict], rows: list[dict], *, add_evidence: bool) -> list[dict]:
    conversations = defaultdict(list)
    for row in rows:
        conversations[row["Conversation_no"]].append(row)
    for conversation in conversations.values():
        conversation.sort(key=lambda row: int(row["Turn_no"]))

    for group in groups:
        if str(group.get("source_dataset", "")).lower() != "topiocqa":
            continue
        tasks = group["tasks"]
        questions = [task["question"].strip() for task in tasks]
        matches = [
            conversation for conversation in conversations.values()
            if [row["Question"].strip() for row in conversation[:len(tasks)]] == questions
        ]
        if len(matches) != 1:
            raise ValueError(f"{group['group_id']}: expected one exact conversation match, got {len(matches)}")
        conversation = matches[0]
        group["original_conversation_no"] = conversation[0]["Conversation_no"]
        group["original_turn_count"] = len(conversation)
        group["selected_turn_count"] = len(tasks)
        group["reference_source_url"] = SOURCE_URL
        group["reference_source_sha256"] = SOURCE_SHA256
        group["evaluation_protocol"] = "max_over_official_references_strict_em_token_f1"
        if add_evidence:
            group["evidence_scope"] = "existing_passages_plus_official_annotated_rationales_not_full_wikipedia"
        context = group.setdefault("context", [])
        evidence_keys = {(item["title"], sentence) for item in context for sentence in item.get("sentences", [])}
        for task, row in zip(tasks, conversation):
            annotations = [row, *row.get("Additional_answers", [])]
            task["expected_answer"] = row["Answer"]
            task["reference_answers"] = list(dict.fromkeys(
                annotation["Answer"].strip() for annotation in annotations
                if annotation.get("Answer", "").strip()
            ))
            task["reference_annotations"] = [
                {key: annotation.get(key) for key in ("Answer", "Rationale", "Topic", "Topic_section")}
                for annotation in annotations
            ]
            task["original_turn_no"] = row["Turn_no"]
            task["conversation_context"] = row.get("Context", [])
            task["topic"] = row.get("Topic")
            task["topic_section"] = row.get("Topic_section")
            task["rationale"] = row.get("Rationale")
            if not add_evidence:
                continue
            for annotation in annotations:
                rationale = (annotation.get("Rationale") or "").strip()
                if not rationale or rationale.upper() == "UNANSWERABLE":
                    continue
                title = f"{annotation.get('Topic', '')} [SEP] {annotation.get('Topic_section', '')}"
                if (title, rationale) in evidence_keys:
                    continue
                context.append({
                    "title": title,
                    "sentences": [rationale],
                    "source": "official_annotated_rationale",
                })
                evidence_keys.add((title, rationale))
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Local official topiocqa_valid.jsonl; otherwise downloaded")
    parser.add_argument("--labels-only", action="store_true", help="Repair references without extending evidence")
    args = parser.parse_args()
    if args.source:
        raw = args.source.read_bytes()
    else:
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
            raw = response.read()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Official source checksum mismatch; refusing to change datasets")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    directory = Path(__file__).resolve().parent / "data"
    combined = directory / "official_topiocqa_qrecc_6chains_10turns.json"
    groups = rebuild(json.loads(combined.read_text(encoding="utf-8")), rows, add_evidence=not args.labels_only)
    outputs = {combined: groups}
    for group in groups:
        suffix = group["group_id"].split("-official-")[-1]
        name = str(group["source_dataset"]).lower()
        outputs[directory / f"official_{name}_{suffix}_10turns.json"] = [group]
    # Validate everything before replacing files and retain the complete old inputs.
    backup = directory / "backups" / datetime.now().strftime("official_references_%Y%m%d_%H%M%S_%f")
    backup.mkdir(parents=True)
    for path in outputs:
        if path.exists():
            shutil.copy2(path, backup / path.name)
    for path, content in outputs.items():
        path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for group in groups:
        print(f"{group['group_id']}: original_conversation_no={group.get('original_conversation_no')}, tasks={len(group['tasks'])}")
    print(f"Updated {len(outputs)} files. Backup: {backup}")
    print("QReCC inputs preserved; only TopiOCQA references were restored from the official source.")
    print("Existing experiment result files were NOT changed.")


if __name__ == "__main__":
    main()
