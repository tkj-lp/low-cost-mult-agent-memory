from __future__ import annotations

import time

from competition_agents.graph import build_graph
from competition_agents.state import initial_state


TASKS = [
    # 用 10 个相关任务验证系统能连续稳定运行。
    "Design a LangGraph multi-agent protocol with shared memory",
    "Reuse the previous LangGraph protocol memory for openEuler deployment",
    "Compare text-mode and protocol-mode communication cost",
    "Design latent state exchange with embedding references",
    "Reuse latent state exchange memory for retrieval planning",
    "Design shared memory metadata for competition evaluation",
    "Reuse shared memory metadata for report generation",
    "Plan CodeAct tool execution in a lightweight sandbox",
    "Reuse CodeAct execution memory for performance analysis",
    "Summarize the complete multi-agent competition architecture",
]


def run_mode(mode: str) -> dict:
    """按指定协作模式运行全部任务，并汇总指标。"""

    graph = build_graph()
    # memory 会跨任务复用，用来模拟历史经验沉淀。
    memory: list[dict] = []
    total_elapsed = 0.0
    total_memory_hits = 0
    total_metrics = {
        "message_count": 0,
        "text_chars": 0,
        "latent_transfer_count": 0,
        "latent_bytes": 0,
    }

    print(f"\n=== {mode} mode ===")
    for index, task in enumerate(TASKS, start=1):
        # 每个任务有新的短期状态，但会继承长期共享记忆。
        started = time.perf_counter()
        result = graph.invoke(initial_state(task, mode, memory))
        elapsed = time.perf_counter() - started
        total_elapsed += elapsed
        memory = result["memory"]
        metrics = result["metrics"]
        memory_hits = len(result["memory_hits"])
        total_memory_hits += memory_hits
        # 累加指标，用于最后比较 text 和 protocol 两种模式。
        for key in total_metrics:
            total_metrics[key] += metrics[key]

        print(f"task {index}: {task}")
        print(f"answer: {result['final_answer']}")
        print("message path:")
        for message in result["messages"]:
            # 网页端可以直接用 from/to/action 画通信链路。
            print(
                f"  {message['from']} -> {message['to']} "
                f"action={message['action']} chars={message['char_cost']}"
            )
        print(
            "metrics: "
            f"messages={metrics['message_count']}, "
            f"text_chars={metrics['text_chars']}, "
            f"latent_transfers={metrics['latent_transfer_count']}, "
            f"latent_bytes={metrics['latent_bytes']}, "
            f"memory_hits={memory_hits}, "
            f"elapsed_ms={elapsed * 1000:.2f}"
        )

    memory_hit_rate = total_memory_hits / len(TASKS)
    print(
        "summary: "
        f"tasks={len(TASKS)}, "
        f"messages={total_metrics['message_count']}, "
        f"text_chars={total_metrics['text_chars']}, "
        f"latent_transfers={total_metrics['latent_transfer_count']}, "
        f"latent_bytes={total_metrics['latent_bytes']}, "
        f"memory_hits={total_memory_hits}, "
        f"memory_hit_rate={memory_hit_rate:.2f}, "
        f"elapsed_ms={total_elapsed * 1000:.2f}"
    )

    return {
        "mode": mode,
        "tasks": len(TASKS),
        "metrics": total_metrics,
        "memory_hits": total_memory_hits,
        "memory_hit_rate": memory_hit_rate,
        "elapsed_ms": total_elapsed * 1000,
    }


def print_comparison(text_result: dict, protocol_result: dict) -> None:
    """输出实验报告里最重要的对比指标。"""

    text_chars = text_result["metrics"]["text_chars"]
    protocol_chars = protocol_result["metrics"]["text_chars"]
    text_elapsed = text_result["elapsed_ms"]
    protocol_elapsed = protocol_result["elapsed_ms"]
    char_saving = (text_chars - protocol_chars) / text_chars if text_chars else 0.0
    speedup = (text_elapsed - protocol_elapsed) / text_elapsed if text_elapsed else 0.0

    print("\n=== comparison ===")
    print(f"text chars: {text_chars}")
    print(f"protocol chars: {protocol_chars}")
    print(f"average character saving: {char_saving:.2%}")
    print(f"text elapsed ms: {text_elapsed:.2f}")
    print(f"protocol elapsed ms: {protocol_elapsed:.2f}")
    print(f"elapsed improvement: {speedup:.2%}")
    print(f"text memory hit rate: {text_result['memory_hit_rate']:.2f}")
    print(f"protocol memory hit rate: {protocol_result['memory_hit_rate']:.2f}")


def main() -> None:
    # 两种模式运行同一批任务，保证对比公平。
    text_result = run_mode("text")
    protocol_result = run_mode("protocol")
    print_comparison(text_result, protocol_result)


if __name__ == "__main__":
    main()
