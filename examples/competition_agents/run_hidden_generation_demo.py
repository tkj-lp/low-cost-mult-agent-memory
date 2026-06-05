from __future__ import annotations

import argparse
import json
import os
import time

from competition_agents.latent_hidden import (
    generate_text_with_last_hidden_state,
    put_hidden_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用同一个本地小模型同时生成答案并提取最后一层 hidden state。")
    parser.add_argument(
        "--prompt",
        default="请用一句话解释多智能体共享记忆为什么可以降低重复通信开销。",
        help="输入提示词。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=48,
        help="最多生成多少个新 token。",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="是否开启采样生成。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    result, tensor = generate_text_with_last_hidden_state(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
    )

    latent_states: dict[str, dict] = {}
    state_ref, latent_states = put_hidden_tensor(
        latent_states,
        producer="local_reasoner",
        topic=args.prompt,
        result=result.hidden,
        tensor=tensor,
        consumers=["summarizer"],
    )

    elapsed_ms = (time.perf_counter() - started) * 1000
    summary = {
        "model": result.hidden.model_name,
        "prompt": result.prompt,
        "generated_text": result.generated_text,
        "state_ref": state_ref,
        "hidden_meta": {
            "layer": result.hidden.layer,
            "shape": result.hidden.shape,
            "pooling": result.hidden.pooling,
            "token_count": result.hidden.token_count,
            "dtype": result.hidden.dtype,
            "device": result.hidden.device,
            "checksum": result.hidden.checksum,
        },
        "elapsed_ms": elapsed_ms,
    }

    print("=== local hidden generation demo ===")
    print(f"LOCAL_HIDDEN_MODEL={os.environ.get('LOCAL_HIDDEN_MODEL', 'Qwen/Qwen2.5-0.5B-Instruct')}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
