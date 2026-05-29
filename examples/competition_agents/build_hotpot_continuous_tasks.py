from __future__ import annotations

"""从 HotpotQA dev distractor 原始数据构造连续任务数据集。"""

import argparse
import json
from pathlib import Path
from typing import Any


SELECTED_SAMPLE_IDS = [
    "5a8c7595554299585d9e36b6",
    "5a85ea095542994775f606a8",
]


def context_as_dicts(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """把 HotpotQA 原始 context 转成当前 runner 使用的字段格式。"""

    return [{"title": title, "sentences": sentences} for title, sentences in sample["context"]]


def supporting_sentences(sample: dict[str, Any]) -> list[list[str]]:
    """把 supporting_facts 的句子下标解析成真实证据句。"""

    contexts = {title: sentences for title, sentences in sample["context"]}
    facts = []
    for title, sentence_index in sample["supporting_facts"]:
        facts.append([title, contexts[title][sentence_index].strip()])
    return facts


def evidence_text(facts: list[list[str]]) -> str:
    return " ".join(f"{title}: {sentence}" for title, sentence in facts)


def title_chain(facts: list[list[str]]) -> str:
    titles = []
    for title, _ in facts:
        if title not in titles:
            titles.append(title)
    return " -> ".join(titles)


def build_tasks(sample: dict[str, Any], facts: list[list[str]]) -> list[dict[str, Any]]:
    """围绕同一个真实 HotpotQA 样本生成 10 个强关联 turn。"""

    answer = sample["answer"]
    question = sample["question"]
    chain = title_chain(facts)
    evidence = evidence_text(facts)
    return [
        {
            "turn_id": "turn-1",
            "task_type": "original_multihop_qa",
            "question": question,
            "expected_reuse": [],
            "expected_answer": answer,
        },
        {
            "turn_id": "turn-2",
            "task_type": "memory_reuse_followup",
            "question": (
                f"Reuse the previous evidence chain. Which answer was supported by the facts about {chain}? "
                "Return only the answer."
            ),
            "expected_reuse": ["turn-1.answer", "turn-1.supporting_facts"],
            "expected_answer": answer,
        },
        {
            "turn_id": "turn-3",
            "task_type": "evidence_grounded_reanswer",
            "question": (
                "Using the same supporting facts, answer the original question again and return only the final "
                f"answer. Original question: {question}"
            ),
            "expected_reuse": ["turn-1.supporting_facts", "turn-2.answer"],
            "expected_answer": answer,
        },
        {
            "turn_id": "turn-4",
            "task_type": "chain_title_reuse",
            "question": (
                f"The reusable evidence chain is {chain}. What final answer does this chain support? "
                "Return only the answer."
            ),
            "expected_reuse": ["turn-1.answer", "turn-1.supporting_facts", "turn-3.answer"],
            "expected_answer": answer,
        },
        {
            "turn_id": "turn-5",
            "task_type": "evidence_text_reuse",
            "question": f"Answer from the cited evidence chain and return only the answer. Evidence: {evidence}",
            "expected_reuse": ["turn-1.supporting_facts", "turn-4.answer"],
            "expected_answer": answer,
        },
        {
            "turn_id": "turn-6",
            "task_type": "memory_check",
            "question": (
                f"A later task asks to reuse the memory built from {chain}. Which answer should be reused for "
                "the original question? Return only the answer."
            ),
            "expected_reuse": ["turn-1.answer", "turn-2.answer", "turn-5.supporting_facts"],
            "expected_answer": answer,
        },
        {
            "turn_id": "turn-7",
            "task_type": "supporting_fact_consistency",
            "question": (
                "Check whether these supporting facts still support the same answer, then return only that "
                f"answer. Supporting facts: {evidence}"
            ),
            "expected_reuse": ["turn-1.supporting_facts", "turn-5.answer", "turn-6.answer"],
            "expected_answer": answer,
        },
        {
            "turn_id": "turn-8",
            "task_type": "compressed_memory_reuse",
            "question": (
                f"Use the compressed memory for {chain} instead of recomputing the full context. What is the "
                "answer to the original question? Return only the answer."
            ),
            "expected_reuse": ["turn-1.answer", "turn-4.chain", "turn-7.answer"],
            "expected_answer": answer,
        },
        {
            "turn_id": "turn-9",
            "task_type": "memory_to_summary_answer",
            "question": (
                "Summarize the reusable evidence chain in one short phrase, but make the final output only the "
                f"answer. Chain: {chain}. Evidence: {evidence}"
            ),
            "expected_reuse": ["turn-1.supporting_facts", "turn-8.answer"],
            "expected_answer": answer,
        },
        {
            "turn_id": "turn-10",
            "task_type": "final_reuse_regression",
            "question": (
                f"Final reuse regression for this chain. Reuse all previous turns for {chain}; what answer "
                "should be returned? Return only the answer."
            ),
            "expected_reuse": ["turn-1.answer", "turn-1.supporting_facts", "turn-5.answer", "turn-9.answer"],
            "expected_answer": answer,
        },
    ]


def build_groups(source: Path) -> list[dict[str, Any]]:
    data = json.loads(source.read_text(encoding="utf-8"))
    samples = {sample["_id"]: sample for sample in data}
    groups = []
    for index, sample_id in enumerate(SELECTED_SAMPLE_IDS, start=1):
        sample = samples[sample_id]
        facts = supporting_sentences(sample)
        groups.append(
            {
                "group_id": f"chain-{index:03d}",
                "source_sample_id": sample["_id"],
                "source_dataset": source.name,
                "topic": title_chain(facts),
                "original_question": sample["question"],
                "answer": sample["answer"],
                "type": sample.get("type"),
                "level": sample.get("level"),
                "context": context_as_dicts(sample),
                "supporting_facts": facts,
                "tasks": build_tasks(sample, facts),
            }
        )
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the continuous HotpotQA task dataset.")
    parser.add_argument(
        "--source",
        default="examples/competition_agents/data/hotpot_dev_distractor_v1.json",
        help="HotpotQA dev distractor JSON file.",
    )
    parser.add_argument(
        "--output",
        default="examples/competition_agents/data/hotpot_continuous_tasks.json",
        help="Output continuous task JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = build_groups(Path(args.source))
    output = Path(args.output)
    output.write_text(json.dumps(groups, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output} with {len(groups)} groups and {sum(len(group['tasks']) for group in groups)} turns")


if __name__ == "__main__":
    main()
