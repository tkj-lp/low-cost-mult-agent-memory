from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXAMPLE_ROOT.parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from competition_agents.memory import add_memory
from hypergraph_memory_method import (
    compare_retrieval_sets,
    search_context,
    search_hypergraph_memory,
    search_text_context,
    search_text_memory,
)


DEFAULT_DATASET = EXAMPLE_ROOT / "data" / "topiocqa_two_chains_10turns_refined.json"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _short(value: Any, limit: int = 180) -> str:
    text = " ".join(_text(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _load_items(dataset_file: Path, limit: int) -> list[dict[str, Any]]:
    data = json.loads(dataset_file.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("tasks", "items", "data", "examples"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a list or contain a list field: {dataset_file}")
    items: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        tasks = item.get("tasks")
        if isinstance(tasks, list):
            chain_fields = {key: value for key, value in item.items() if key != "tasks"}
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                merged = dict(chain_fields)
                merged.update(task)
                if not merged.get("chain_id") and merged.get("group_id"):
                    merged["chain_id"] = merged["group_id"]
                items.append(merged)
        else:
            items.append(item)
    return items[:limit] if limit > 0 else items


def _chain_id(item: dict[str, Any]) -> str:
    return _text(item.get("chain_id") or item.get("group_id") or item.get("conversation_no") or "default")


def _turn_id(item: dict[str, Any]) -> str:
    value = item.get("turn_id") or item.get("turn") or item.get("conversation_turn")
    text = _text(value)
    return text if text.startswith("turn-") else f"turn-{text or '0'}"


def _task_id(item: dict[str, Any]) -> str:
    return _text(item.get("task_id")) or f"{_chain_id(item)}:{_turn_id(item)}"


def _question(item: dict[str, Any]) -> str:
    return _text(item.get("question") or item.get("task") or item.get("original_question"))


def _gold(item: dict[str, Any]) -> str:
    return _text(item.get("answer") or item.get("expected_answer") or item.get("gold"))


def _task_text(item: dict[str, Any], context_hits: list[dict[str, Any]]) -> str:
    context = "\n".join(f"- {hit.get('title', '')}: {hit.get('text', '')}" for hit in context_hits)
    return (
        "TopiOCQA continuous QA task.\n"
        f"Question: {_question(item)}\n"
        f"Topic: {_text(item.get('topic'))}\n"
        f"Topic section: {_text(item.get('topic_section'))}\n"
        f"Retrieved context:\n{context}"
    )


def _task_meta(item: dict[str, Any]) -> dict[str, Any]:
    meta = dict(item)
    meta.update(
        {
            "task_id": _task_id(item),
            "group_id": _chain_id(item),
            "turn_id": _turn_id(item),
            "dataset_type": "topiocqa",
            "question": _question(item),
        }
    )
    return meta


def _same_chain_memory_ids(memory: list[dict[str, Any]], chain_id: str) -> set[str]:
    ids: set[str] = set()
    for record in memory:
        trace = record.get("trace", {})
        task_meta = trace.get("task_meta", {}) if isinstance(trace, dict) else {}
        if isinstance(task_meta, dict):
            record_chain = _text(task_meta.get("group_id") or task_meta.get("chain_id"))
            if record_chain == chain_id:
                ids.add(_text(record.get("memory_id")))
    return {item for item in ids if item}


def _supporting_passage_hit(item: dict[str, Any], hits: list[dict[str, Any]]) -> bool:
    supporting = _text(item.get("supporting_passage"))
    if not supporting:
        return False
    return any(_text(hit.get("text")) == supporting or _short(supporting, 80) in _text(hit.get("text")) for hit in hits)


def _print_context_hits(title: str, hits: list[dict[str, Any]]) -> None:
    print(f"\n  {title}: {len(hits)}")
    for index, hit in enumerate(hits, start=1):
        print(
            f"    {index}. id={hit.get('context_id')} "
            f"kind={hit.get('kind')} score={hit.get('context_score', hit.get('retrieval_score'))}"
        )
        print(f"       title={hit.get('title')} section={hit.get('section')}")
        print(f"       text={_short(hit.get('text'), 220)}")
        if hit.get("matched_nodes"):
            print(f"       matched={hit.get('matched_nodes')[:8]}")
        if hit.get("reasoning_path"):
            print(f"       path={hit.get('reasoning_path')[:3]}")


def _print_memory_hits(title: str, hits: list[dict[str, Any]]) -> None:
    print(f"\n  {title}: {len(hits)}")
    for index, hit in enumerate(hits, start=1):
        print(
            f"    {index}. id={hit.get('memory_id')} "
            f"backend={hit.get('retrieval_backend', 'hypergraph_memory')} "
            f"score={hit.get('retrieval_score')}"
        )
        print(f"       summary={_short(hit.get('summary'), 240)}")
        if hit.get("matched_nodes"):
            print(f"       matched_nodes={hit.get('matched_nodes')[:10]}")
        if hit.get("matched_hyperedges"):
            print(f"       matched_hyperedges={hit.get('matched_hyperedges')[:6]}")
        if hit.get("reasoning_path"):
            print(f"       path={hit.get('reasoning_path')[:3]}")


def _add_dataset_memory(memory: list[dict[str, Any]], item: dict[str, Any], context_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    context_text = "; ".join(_short(hit.get("text"), 160) for hit in context_hits)
    summary = (
        f"TopiOCQA answer memory. question={_question(item)} "
        f"answer={_gold(item)} context={context_text}"
    )
    trace = {
        "task_meta": _task_meta(item),
        "evidence": [hit.get("text", "") for hit in context_hits],
        "memory_hits": [],
        "state_refs": {},
        "metrics": {},
    }
    return add_memory(
        memory,
        source_agent="dataset_memory_baseline",
        task_topic=_question(item),
        summary=summary,
        tags=["topiocqa", "text_baseline", _chain_id(item)],
        trace=trace,
    )


def _resolve_experiment_mode(experiment_mode: str, fallback_mode: str) -> str:
    if experiment_mode == "baseline_text":
        os.environ["EMBEDDING_PROVIDER"] = "hash"
        return "text"
    if experiment_mode == "protocol_api":
        os.environ["EMBEDDING_PROVIDER"] = "hash"
        return "protocol"
    if experiment_mode == "protocol_api_embedding":
        os.environ["EMBEDDING_PROVIDER"] = "openai_compatible"
        return "protocol"
    return fallback_mode


def _add_agent_memory(
    graph: Any,
    memory: list[dict[str, Any]],
    item: dict[str, Any],
    context_hits: list[dict[str, Any]],
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from competition_agents.state import initial_state

    result = graph.invoke(initial_state(_task_text(item, context_hits), mode, memory, task_meta=_task_meta(item)))
    return result.get("memory", memory), result


def run(args: argparse.Namespace) -> dict[str, Any]:
    items = _load_items(Path(args.dataset_file), args.limit)
    memory: list[dict[str, Any]] = []
    mode = _resolve_experiment_mode(args.experiment_mode, args.mode)
    graph = None
    if args.memory_source == "agent":
        from competition_agents.graph import build_graph

        graph = build_graph()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    print("=== TopiOCQA context and memory retrieval comparison ===")
    print(f"dataset_file: {args.dataset_file}")
    print(f"memory_source: {args.memory_source}")
    print(f"experiment_mode: {args.experiment_mode or '<developer-mode>'}")
    print(f"mode: {mode}")
    print(f"llm_provider: {os.environ.get('LLM_PROVIDER', '')}")
    print(f"embedding_provider: {os.environ.get('EMBEDDING_PROVIDER', '')}")

    for item_index, item in enumerate(items, start=1):
        question = _question(item)
        chain_id = _chain_id(item)
        expected_memory_ids = _same_chain_memory_ids(memory, chain_id)

        print("\n" + "=" * 88)
        print(f"task {item_index}: {_task_id(item)}")
        print(f"question: {question}")
        print(f"gold: {_gold(item)}")

        text_context_hits = search_text_context(_task_meta(item), query=question, dataset_type="topiocqa", top_k=args.context_top_k)
        graph_context_hits = search_context(_task_meta(item), query=question, dataset_type="topiocqa", top_k=args.context_top_k)
        _print_context_hits("plain text context hits", text_context_hits)
        _print_context_hits("hypergraph context hits", graph_context_hits)

        memory_query = (
            f"Question: {question}\n"
            "Retrieved context:\n"
            + "\n".join(f"- {hit.get('title', '')}: {hit.get('text', '')}" for hit in graph_context_hits)
        )
        text_memory_hits = search_text_memory(memory, memory_query, top_k=args.memory_top_k) if memory else []
        graph_memory_hits = search_hypergraph_memory(memory, memory_query, top_k=args.memory_top_k) if memory else []
        _print_memory_hits("plain text memory hits", text_memory_hits)
        _print_memory_hits("hypergraph memory hits", graph_memory_hits)

        comparison = compare_retrieval_sets(
            text_hits=text_memory_hits,
            hypergraph_hits=graph_memory_hits,
            expected_ids=expected_memory_ids,
            top_k=args.memory_top_k,
        )
        row = {
            "task_id": _task_id(item),
            "chain_id": chain_id,
            "question": question,
            "gold": _gold(item),
            "text_context_hit_count": len(text_context_hits),
            "hypergraph_context_hit_count": len(graph_context_hits),
            "text_context_supporting_hit": _supporting_passage_hit(item, text_context_hits),
            "hypergraph_context_supporting_hit": _supporting_passage_hit(item, graph_context_hits),
            "expected_memory_ids": sorted(expected_memory_ids),
            "text_memory_hit_ids": [hit.get("memory_id") for hit in text_memory_hits],
            "hypergraph_memory_hit_ids": [hit.get("memory_id") for hit in graph_memory_hits],
            "memory_comparison": comparison,
        }
        rows.append(row)

        print("\n  memory comparison metrics:")
        print(json.dumps(comparison, ensure_ascii=False, indent=4))

        if args.memory_source == "agent":
            assert graph is not None
            memory, agent_result = _add_agent_memory(graph, memory, item, graph_context_hits, mode)
            print("\n  agent result:")
            print(f"    final_answer={_short(agent_result.get('final_answer') or agent_result.get('summary'), 240)}")
            print(f"    memory_size_after_write={len(memory)}")
        else:
            memory = _add_dataset_memory(memory, item, graph_context_hits)
            print("\n  dataset baseline memory write:")
            print(f"    memory_size_after_write={len(memory)} latest_memory_id={memory[-1].get('memory_id')}")

    total_elapsed_ms = (time.perf_counter() - started) * 1000.0
    summary = {
        "dataset_type": "topiocqa",
        "tasks": len(rows),
        "memory_source": args.memory_source,
        "experiment_mode": args.experiment_mode or "<developer-mode>",
        "mode": mode,
        "text_context_supporting_hit_rate": sum(1 for row in rows if row["text_context_supporting_hit"]) / len(rows)
        if rows
        else 0.0,
        "hypergraph_context_supporting_hit_rate": sum(1 for row in rows if row["hypergraph_context_supporting_hit"])
        / len(rows)
        if rows
        else 0.0,
        "text_memory_hit_rate": sum(1 for row in rows if row["text_memory_hit_ids"]) / len(rows) if rows else 0.0,
        "hypergraph_memory_hit_rate": sum(1 for row in rows if row["hypergraph_memory_hit_ids"]) / len(rows) if rows else 0.0,
        "elapsed_ms": round(total_elapsed_ms, 4),
    }

    report = {"summary": summary, "rows": rows}
    print("\n=== retrieval_compare_json ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TopiOCQA text retrieval with hypergraph retrieval.")
    parser.add_argument("--dataset-file", default=str(DEFAULT_DATASET))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--context-top-k", type=int, default=3)
    parser.add_argument("--memory-top-k", type=int, default=3)
    parser.add_argument("--mode", choices=["text", "protocol"], default="protocol")
    parser.add_argument(
        "--experiment-mode",
        choices=["", "baseline_text", "protocol_api", "protocol_api_embedding"],
        default="protocol_api_embedding",
        help="protocol_api_embedding maps to protocol mode and openai_compatible embedding.",
    )
    parser.add_argument(
        "--memory-source",
        choices=["agent", "dataset"],
        default="agent",
        help="agent uses the real multi-agent graph; dataset writes a deterministic baseline memory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
