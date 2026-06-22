from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hypergraph_memory_method.reuse_metrics import evaluate_run_reuse


MARKER = "=== run_result_json ==="


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")) or raw.count(b"\x00") > len(raw) // 4:
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def load_run_result(path: Path) -> dict[str, Any]:
    text = _read_text(path)
    payload = text.split(MARKER, 1)[1].strip() if MARKER in text else text.strip()
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("run result must be a JSON object")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate memory precision and correct reuse from run_result_json.")
    parser.add_argument("run_result", type=Path)
    parser.add_argument("--run-index", type=int, default=0)
    args = parser.parse_args()

    payload = load_run_result(args.run_result)
    runs = payload.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise ValueError("run result does not contain runs")
    report = evaluate_run_reuse(runs[args.run_index])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
