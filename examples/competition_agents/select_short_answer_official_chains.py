"""Build small-model-friendly, short-answer official conversational QA chains.

The selection is deterministic and is based only on official questions/answers,
never on any model prediction. Each selected chain contains its first ten
consecutive turns and every reference answer must be a short, single answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


TOPIOCQA_URL = (
    "https://huggingface.co/datasets/McGill-NLP/TopiOCQA/resolve/"
    "543f3fc424da32c8b024fb5ae182946860be5517/data/topiocqa_valid.jsonl"
)
QRECC_URL = (
    "https://huggingface.co/datasets/slupart/qrecc/resolve/"
    "35dd1d9b6306182f19a763b497e06dba9b3d48c9/qrecc_test.json"
)
TURN_COUNT = 10
MAX_ANSWER_TOKENS = 5
MULTI_ANSWER_MARKERS = re.compile(r"[,;:/]|\b(?:and|or)\b", re.IGNORECASE)
OPEN_ENDED_QUESTION = re.compile(
    r"\b(?:describe|explain|tell me|why|how does|how do|list|mention|kinds|types)\b",
    re.IGNORECASE,
)


def download_json(url: str) -> tuple[list[dict[str, Any]], str]:
    with urllib.request.urlopen(url, timeout=180) as response:
        raw = response.read()
    digest = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")
    if text.lstrip().startswith("["):
        return json.loads(text), digest
    return [json.loads(line) for line in text.splitlines() if line.strip()], digest


def tokens(value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9.'-]*", value)


def answer_for(row: dict[str, Any]) -> str:
    return str(row.get("Answer") or row.get("Truth_answer") or "").strip()


def rewrite_for(row: dict[str, Any]) -> str:
    return str(row.get("Rewrite") or row.get("Truth_rewrite") or "")


def is_short_single_answer(answer: str) -> bool:
    answer = str(answer or "").strip()
    word_tokens = tokens(answer)
    if not answer or not 1 <= len(word_tokens) <= MAX_ANSWER_TOKENS:
        return False
    if MULTI_ANSWER_MARKERS.search(answer):
        return False
    return answer.lower() not in {"unknown", "unanswerable", "n/a"}


def is_simple_question(question: str) -> bool:
    return bool(str(question).strip()) and not OPEN_ENDED_QUESTION.search(str(question))


def group_rows(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["Conversation_no"])].append(row)
    conversations = []
    for conversation_id, conversation in grouped.items():
        del conversation_id
        conversation.sort(key=lambda row: int(row["Turn_no"]))
        conversations.append(conversation)
    return conversations


def qualifying_turn_count(conversation: list[dict[str, Any]]) -> int:
    return sum(
        is_short_single_answer(answer_for(row)) and is_simple_question(str(row.get("Question") or ""))
        for row in conversation
    )


def continuous_windows(conversation: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [conversation[index:index + TURN_COUNT] for index in range(len(conversation) - TURN_COUNT + 1)]


def continuity_score(conversation: list[dict[str, Any]]) -> int:
    questions = " ".join(str(row.get("Question") or "").lower() for row in conversation[:TURN_COUNT])
    return len(re.findall(r"\b(?:it|its|they|them|this|that|these|those|he|she|where|when|who)\b", questions))


def topic_diversity(conversation: list[dict[str, Any]]) -> int:
    return len({str(row.get("Topic") or "").strip() for row in conversation[:TURN_COUNT] if row.get("Topic")})


def select(rows: list[dict[str, Any]], count: int) -> list[list[dict[str, Any]]]:
    candidates = [
        window
        for conversation in group_rows(rows)
        for window in continuous_windows(conversation)
    ]
    candidates.sort(
        key=lambda conversation: (
            -qualifying_turn_count(conversation),
            -continuity_score(conversation),
            -topic_diversity(conversation),
            int(conversation[0]["Conversation_no"]),
        )
    )
    selected: list[list[dict[str, Any]]] = []
    selected_conversations: set[str] = set()
    for candidate in candidates:
        conversation_id = str(candidate[0]["Conversation_no"])
        if conversation_id in selected_conversations:
            continue
        selected.append(candidate)
        selected_conversations.add(conversation_id)
        if len(selected) == count:
            return selected
    raise ValueError(f"Only {len(selected)} distinct conversations, need {count}.")


def topiocqa_group(index: int, conversation: list[dict[str, Any]], source_sha256: str) -> dict[str, Any]:
    tasks = []
    context = []
    for turn_index, row in enumerate(conversation[:TURN_COUNT], start=1):
        rationale = str(row.get("Rationale") or "").strip()
        title = f"{row.get('Topic', '')} [SEP] {row.get('Topic_section', '')}".strip()
        if rationale:
            context.append({"title": title, "sentences": [rationale], "source": "official_annotated_rationale"})
        references = [answer_for(row)]
        references.extend(
            str(item.get("Answer") or "").strip()
            for item in row.get("Additional_answers", []) or []
        )
        tasks.append({
            "turn_id": f"turn-{turn_index}",
            "original_turn_no": row.get("Turn_no"),
            "task_type": "continuous",
            "question": row["Question"],
            "expected_answer": answer_for(row),
            "reference_answers": list(dict.fromkeys(answer for answer in references if answer)),
            "expected_reuse": [],
            "topic": row.get("Topic"),
            "topic_section": row.get("Topic_section"),
            "conversation_context": row.get("Context", []),
            "supporting_passage": rationale,
            "rationale": rationale,
        })
    return {
        "group_id": f"topiocqa-factoid-ranked-{index:03d}",
        "source_dataset": "TopiOCQA",
        "source_file": "topiocqa_valid.jsonl",
        "original_conversation_no": conversation[0]["Conversation_no"],
        "original_turn_count": len(conversation),
        "selected_turn_count": TURN_COUNT,
        "selection_note": (
            f"ranked by short single-answer turns: {qualifying_turn_count(conversation)}/{TURN_COUNT}; "
            "complete 10-turn continuous window retained"
        ),
        "selection_protocol": selection_protocol(),
        "topic": " -> ".join(dict.fromkeys(str(row.get("Topic") or "") for row in conversation[:TURN_COUNT])),
        "context": context,
        "supporting_facts": [],
        "reference_source_url": TOPIOCQA_URL,
        "reference_source_sha256": source_sha256,
        "evaluation_protocol": "normalized_strict_em_and_token_f1",
        "evidence_scope": "official_annotated_rationales",
        "tasks": tasks,
    }


def qrecc_group(index: int, conversation: list[dict[str, Any]], source_sha256: str) -> dict[str, Any]:
    tasks = []
    context = []
    for turn_index, row in enumerate(conversation[:TURN_COUNT], start=1):
        answer = answer_for(row)
        title = f"QReCC conversation {row['Conversation_no']} turn {row['Turn_no']}"
        # QReCC provides answer provenance URLs but not passage text. This follows
        # the existing project adapter's deterministic answer-text evidence format.
        context.append({"title": title, "sentences": [answer], "source": "official_answer_annotation"})
        tasks.append({
            "turn_id": f"turn-{turn_index}",
            "original_turn_no": row.get("Turn_no"),
            "task_type": "continuous",
            "question": row["Question"],
            "rewrite": rewrite_for(row),
            "expected_answer": answer,
            "reference_answers": [answer],
            "expected_reuse": [],
            "topic": title,
            "topic_section": "",
            "conversation_context": row.get("Context", []),
            "supporting_passage": answer,
            "answer_url": row.get("Answer_URL", ""),
        })
    return {
        "group_id": f"qrecc-factoid-ranked-{index:03d}",
        "source_dataset": "QReCC",
        "source_file": "qrecc_test.json",
        "original_conversation_no": conversation[0]["Conversation_no"],
        "original_turn_count": len(conversation),
        "selected_turn_count": TURN_COUNT,
        "selection_note": (
            f"ranked by short single-answer turns: {qualifying_turn_count(conversation)}/{TURN_COUNT}; "
            "complete 10-turn continuous window retained"
        ),
        "selection_protocol": selection_protocol(),
        "topic": f"QReCC conversation {conversation[0]['Conversation_no']}",
        "context": context,
        "supporting_facts": [],
        "reference_source_url": QRECC_URL,
        "reference_source_sha256": source_sha256,
        "evaluation_protocol": "normalized_strict_em_and_token_f1",
        "evidence_scope": "official_answer_annotations_existing_project_adapter",
        "tasks": tasks,
    }


def selection_protocol() -> dict[str, Any]:
    return {
        "turn_count": TURN_COUNT,
        "answer_token_range": [1, MAX_ANSWER_TOKENS],
        "exclude_multi_answer_markers": "comma, semicolon, colon, slash, and, or",
        "exclude_open_ended_question_markers": "describe, explain, tell me, why, how does, how do, list, mention, kinds, types",
        "ranking": "qualifying_turn_count_desc, continuity_signal_desc, topic_diversity_desc, conversation_no_asc",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    topi_rows, topi_sha = download_json(TOPIOCQA_URL)
    qrecc_rows, qrecc_sha = download_json(QRECC_URL)
    topi = [topiocqa_group(index, conversation, topi_sha) for index, conversation in enumerate(select(topi_rows, args.count), 1)]
    qrecc = [qrecc_group(index, conversation, qrecc_sha) for index, conversation in enumerate(select(qrecc_rows, args.count), 1)]
    groups = [*topi, *qrecc]
    for group in groups:
        print(f"{group['group_id']}: conversation={group['original_conversation_no']} topic={group['topic']}")
        print("  answers=" + " | ".join(task["expected_answer"] for task in group["tasks"]))
    if args.dry_run:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for group in topi:
        path = args.output_dir / f"official_factoid_ranked_topiocqa_{group['group_id'].rsplit('-', 1)[-1]}_10turns.json"
        path.write_text(json.dumps([group], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for group in qrecc:
        path = args.output_dir / f"official_factoid_ranked_qrecc_{group['group_id'].rsplit('-', 1)[-1]}_10turns.json"
        path.write_text(json.dumps([group], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    combined = args.output_dir / "official_factoid_ranked_topiocqa_qrecc_6chains_10turns.json"
    combined.write_text(json.dumps(groups, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(groups)} chains to {args.output_dir}")


if __name__ == "__main__":
    main()
