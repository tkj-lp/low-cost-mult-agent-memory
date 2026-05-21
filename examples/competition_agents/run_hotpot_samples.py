from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from competition_agents.embedding import get_embedding_provider
from competition_agents.graph import build_graph
from competition_agents.llm.client import get_llm_provider
from competition_agents.state import initial_state

MODES = ("text", "protocol")

SAMPLES: list[dict[str, Any]] = [
    {
        "id": "sample-1",
        "question": "Alan Turing worked at an institution during World War II. Which country is that institution located in?",
        "context": [
            {
                "title": "Alan Turing",
                "sentences": [
                    "Alan Turing was a British mathematician and computer scientist.",
                    "During World War II, Turing worked at Bletchley Park.",
                    "He contributed to breaking the Enigma cipher.",
                ],
            },
            {
                "title": "Bletchley Park",
                "sentences": [
                    "Bletchley Park is an estate in Milton Keynes, Buckinghamshire.",
                    "It was the central site for British codebreakers during World War II.",
                    "Bletchley Park is located in the United Kingdom.",
                ],
            },
        ],
        "answer": "United Kingdom",
        "supporting_facts": [
            ["Alan Turing", "During World War II, Turing worked at Bletchley Park."],
            ["Bletchley Park", "Bletchley Park is located in the United Kingdom."],
        ],
    },
    {
        "id": "sample-2",
        "question": 'The author of "Pride and Prejudice" was born in which country?',
        "context": [
            {
                "title": "Pride and Prejudice",
                "sentences": [
                    "Pride and Prejudice is a novel by Jane Austen.",
                    "It was first published in 1813.",
                ],
            },
            {
                "title": "Jane Austen",
                "sentences": [
                    "Jane Austen was an English novelist.",
                    "She was born in Steventon, Hampshire, England.",
                    "Her novels often critique the British landed gentry.",
                ],
            },
        ],
        "answer": "England",
        "supporting_facts": [
            ["Pride and Prejudice", "Pride and Prejudice is a novel by Jane Austen."],
            ["Jane Austen", "She was born in Steventon, Hampshire, England."],
        ],
    },
    {
        "id": "sample-3",
        "question": 'The film "Spirited Away" was directed by a filmmaker from which country?',
        "context": [
            {
                "title": "Spirited Away",
                "sentences": [
                    "Spirited Away is an animated fantasy film directed by Hayao Miyazaki.",
                    "The film was produced by Studio Ghibli.",
                ],
            },
            {
                "title": "Hayao Miyazaki",
                "sentences": [
                    "Hayao Miyazaki is a Japanese animator, filmmaker, and manga artist.",
                    "He co-founded Studio Ghibli.",
                ],
            },
        ],
        "answer": "Japan",
        "supporting_facts": [
            ["Spirited Away", "Spirited Away is an animated fantasy film directed by Hayao Miyazaki."],
            ["Hayao Miyazaki", "Hayao Miyazaki is a Japanese animator, filmmaker, and manga artist."],
        ],
    },
    {
        "id": "sample-4",
        "question": "The Eiffel Tower is located in a city that is the capital of which country?",
        "context": [
            {
                "title": "Eiffel Tower",
                "sentences": [
                    "The Eiffel Tower is a wrought-iron tower located in Paris.",
                    "It is one of the most recognizable structures in the world.",
                ],
            },
            {
                "title": "Paris",
                "sentences": [
                    "Paris is the capital and most populous city of France.",
                    "The city is located on the Seine River.",
                ],
            },
        ],
        "answer": "France",
        "supporting_facts": [
            ["Eiffel Tower", "The Eiffel Tower is located in Paris."],
            ["Paris", "Paris is the capital and most populous city of France."],
        ],
    },
    {
        "id": "sample-5",
        "question": "The company that developed the iPhone was co-founded by which person also associated with Pixar?",
        "context": [
            {
                "title": "iPhone",
                "sentences": [
                    "The iPhone is a line of smartphones designed and marketed by Apple Inc.",
                    "The first iPhone was announced in 2007.",
                ],
            },
            {
                "title": "Apple Inc.",
                "sentences": [
                    "Apple Inc. was co-founded by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
                    "The company is headquartered in Cupertino, California.",
                ],
            },
            {
                "title": "Steve Jobs",
                "sentences": [
                    "Steve Jobs was an American entrepreneur and business magnate.",
                    "He was also associated with Pixar Animation Studios.",
                ],
            },
        ],
        "answer": "Steve Jobs",
        "supporting_facts": [
            ["iPhone", "The iPhone is a line of smartphones designed and marketed by Apple Inc."],
            ["Apple Inc.", "Apple Inc. was co-founded by Steve Jobs, Steve Wozniak, and Ronald Wayne."],
            ["Steve Jobs", "He was also associated with Pixar Animation Studios."],
        ],
    },
]


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^(answer|final answer)\s*:\s*", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def build_task(sample: dict[str, Any]) -> str:
    return (
        "HotpotQA-style multi-hop QA task.\n"
        "Answer using only the provided Context. Return the shortest answer string.\n\n"
        f"Question: {sample['question']}\n"
        f"Context: {json.dumps(sample['context'], ensure_ascii=False)}\n"
    )


def print_step_outputs(state: dict[str, Any]) -> None:
    print("raw agent outputs:")
    for output in state.get("agent_outputs", []):
        print(f"  {output['agent']}:")
        raw_output = output.get("raw_output", "")
        if raw_output:
            for line in str(raw_output).splitlines():
                print(f"    {line}")
        else:
            print("    <empty>")


def print_latent_states(state: dict[str, Any]) -> None:
    latent_states = state.get("latent_states", {})
    if not latent_states:
        return
    print("latent states:")
    for state_id, latent_state in latent_states.items():
        print(
            "  "
            f"{state_id}: "
            f"producer={latent_state.get('producer')}, "
            f"provider={latent_state.get('embedding_provider')}, "
            f"model={latent_state.get('embedding_model')}, "
            f"shape={latent_state.get('shape')}, "
            f"checksum={latent_state.get('checksum')}"
        )


def parse_modes() -> list[str]:
    raw_modes = os.environ.get("HOTPOT_MODES", "text,protocol")
    modes = [mode.strip().lower() for mode in raw_modes.split(",") if mode.strip()]
    invalid = [mode for mode in modes if mode not in MODES]
    if invalid:
        raise ValueError(f"Unsupported HOTPOT_MODES value(s): {invalid}. Use text and/or protocol.")
    return modes or ["protocol"]


def run_mode(mode: str, *, verbose: bool, show_latent: bool) -> dict[str, Any]:
    graph = build_graph()
    memory: list[dict[str, Any]] = []
    correct = 0
    total_elapsed = 0.0
    totals = {
        "message_count": 0,
        "text_chars": 0,
        "latent_transfer_count": 0,
        "latent_bytes": 0,
        "memory_hits": 0,
    }

    print(f"\n=== mode: {mode} ===")
    for sample in SAMPLES:
        started = time.perf_counter()
        state = graph.invoke(initial_state(build_task(sample), mode, memory))
        elapsed_ms = (time.perf_counter() - started) * 1000
        total_elapsed += elapsed_ms
        memory = state["memory"]
        prediction = state["execution_result"].get("answer_candidate", state["final_answer"])
        is_correct = normalize_answer(prediction) == normalize_answer(sample["answer"])
        correct += int(is_correct)

        metrics = state["metrics"]
        totals["message_count"] += metrics["message_count"]
        totals["text_chars"] += metrics["text_chars"]
        totals["latent_transfer_count"] += metrics["latent_transfer_count"]
        totals["latent_bytes"] += metrics["latent_bytes"]
        totals["memory_hits"] += len(state["memory_hits"])

        print(f"\ncase: {sample['id']}")
        print(f"question: {sample['question']}")
        if verbose:
            print_step_outputs(state)
        if show_latent:
            print_latent_states(state)
        print(f"final_answer: {prediction}")
        print(f"gold: {sample['answer']}")
        print(f"exact_match: {is_correct}")
        print(
            "metrics: "
            f"messages={metrics['message_count']}, "
            f"text_chars={metrics['text_chars']}, "
            f"latent_transfers={metrics['latent_transfer_count']}, "
            f"latent_bytes={metrics['latent_bytes']}, "
            f"memory_hits={len(state['memory_hits'])}, "
            f"elapsed_ms={elapsed_ms:.2f}"
        )

    return {
        "mode": mode,
        "samples": len(SAMPLES),
        "correct": correct,
        "exact_match": correct / len(SAMPLES),
        "elapsed_ms": total_elapsed,
        **totals,
    }


def print_comparison(summaries: list[dict[str, Any]]) -> None:
    print("\n=== hotpot comparison summary ===")
    print(
        "mode | samples | exact_match | messages | text_chars | "
        "latent_transfers | latent_bytes | memory_hits | elapsed_ms"
    )
    for summary in summaries:
        print(
            f"{summary['mode']} | "
            f"{summary['samples']} | "
            f"{summary['exact_match']:.2%} | "
            f"{summary['message_count']} | "
            f"{summary['text_chars']} | "
            f"{summary['latent_transfer_count']} | "
            f"{summary['latent_bytes']} | "
            f"{summary['memory_hits']} | "
            f"{summary['elapsed_ms']:.2f}"
        )

    by_mode = {summary["mode"]: summary for summary in summaries}
    if "text" in by_mode and "protocol" in by_mode:
        text = by_mode["text"]
        protocol = by_mode["protocol"]
        text_chars_delta = text["text_chars"] - protocol["text_chars"]
        text_chars_pct = (text_chars_delta / text["text_chars"] * 100) if text["text_chars"] else 0.0
        elapsed_delta = text["elapsed_ms"] - protocol["elapsed_ms"]
        elapsed_pct = (elapsed_delta / text["elapsed_ms"] * 100) if text["elapsed_ms"] else 0.0
        print("\n=== protocol vs text ===")
        print(f"text_chars_delta={text_chars_delta} ({text_chars_pct:.2f}%)")
        print(f"latent_transfer_delta={protocol['latent_transfer_count'] - text['latent_transfer_count']}")
        print(f"latent_bytes_delta={protocol['latent_bytes'] - text['latent_bytes']}")
        print(f"elapsed_ms_delta={elapsed_delta:.2f} ({elapsed_pct:.2f}%)")


def main() -> None:
    verbose = os.environ.get("HOTPOT_VERBOSE", "1").lower() not in {"0", "false", "no"}
    show_latent = os.environ.get("HOTPOT_SHOW_LATENT", "1").lower() not in {"0", "false", "no"}
    modes = parse_modes()
    print(
        "LLM providers: "
        f"default={get_llm_provider()}, "
        f"planner={get_llm_provider('planner')}, "
        f"reasoner={get_llm_provider('reasoner')}, "
        f"summarizer={get_llm_provider('summarizer')}, "
        f"embedding={get_embedding_provider()}"
    )
    print(f"modes={','.join(modes)}")
    summaries = [run_mode(mode, verbose=verbose, show_latent=show_latent) for mode in modes]
    print_comparison(summaries)


if __name__ == "__main__":
    main()
