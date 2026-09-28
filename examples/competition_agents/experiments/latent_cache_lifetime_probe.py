"""按角色使用期限回收计算缓存；三题独立对照，不改变正式方案。"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from latent_cache_position_probe import PositionProbeChat, select_cache_positions
from latent_alignment_probe import (
    BASE, MODEL, NativeLocalChat, QueryReadout, answer_coverage, digest, dump,
    event, frozen_evidence, model_digest, verify,
)


POLICIES = ("full", "immediate", "consumer_lifetime")
ROLES = ("planner", "retriever", "reasoner", "summarizer")
PERSISTENT = frozenset(("common_instruction", "question", "external_facts", "answer"))


@dataclass
class CacheBlock:
    start: int
    length: int
    kind: str
    owner: str
    turn: int
    expires_after: int | None


def expiration(kind, owner):
    """期限来自明确工作流，不声称自动判断隐状态中的语义是否已无用。"""
    if kind in PERSISTENT:
        return None
    if kind == "role_instruction":
        return min(ROLES.index(owner) + 1, 3)
    if kind in ("latent", "boundary"):
        return 3
    raise ValueError("未声明生命周期的状态不得静默回收。")


def reclaimable(block, *, policy, completed_role, turn):
    if block.kind in PERSISTENT or policy == "full":
        return False
    if policy == "immediate":
        return block.kind == "role_instruction"
    if policy != "consumer_lifetime":
        raise ValueError("未知缓存回收策略。")
    return block.expires_after is not None and (
        block.turn < turn or block.turn == turn and block.expires_after <= ROLES.index(completed_role)
    )


class LifetimeProbeChat(PositionProbeChat):
    """统一输入、反馈和位置处理，只改变计算状态的保留期限。"""

    def __init__(self, bundle, *, policy, **kwargs):
        if policy not in POLICIES:
            raise ValueError("未知缓存回收策略。")
        self.policy = policy
        self.active_role = None
        # 父类只提供正确的位置与计量，本类统一负责回收。
        super().__init__(bundle, variant="full", **kwargs)

    def reset(self):
        super().reset()
        self.blocks = []
        self.turn = 0
        self.expected_role = 0

    def _forward(self, *, ids=None, embeds=None):
        start = self.next_absolute_position
        stored = len(self.cached_kinds)
        count = int((ids if ids is not None else embeds).shape[1])
        kind = self.pending_kind
        if kind == "completed_role_instruction":
            kind = "role_instruction"
        elif kind == "external_question":
            kind = "question"
        elif kind is None:
            kind = "latent" if embeds is not None else (
                "question" if self.active_role == "planner" else "answer"
            )
        expires = expiration(kind, self.active_role)
        value = super()._forward(ids=ids, embeds=embeds)
        self.metrics.setdefault("causal_attention_pairs_per_layer_head", 0)
        self.metrics["causal_attention_pairs_per_layer_head"] += count * stored + count * (count + 1) // 2
        block = CacheBlock(start, count, kind, self.active_role, self.turn, expires)
        previous = self.blocks[-1] if self.blocks else None
        if previous is not None and previous.start + previous.length == start and (
            previous.kind, previous.owner, previous.turn, previous.expires_after
        ) == (block.kind, block.owner, block.turn, block.expires_after):
            previous.length += count
        else:
            self.blocks.append(block)
        return value

    def handoff_size(self):
        report = super().handoff_size()
        metadata = {"policy": self.policy, "turn": self.turn, "expected_role": self.expected_role,
                    "blocks": [asdict(block) for block in self.blocks]}
        size = len(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        report["lifetime_metadata_bytes"] = size
        report["payload_bytes"] += size
        report["measurement"] = "完整接续状态的逻辑快照，含生命周期元数据；不是实测通信量。"
        return report

    def call(self, role, **kwargs):
        started = time.perf_counter()
        if role == "planner" and self.scope != kwargs.get("scope"):
            self.reset()
            self.scope = kwargs.get("scope")
        if role != ROLES[self.expected_role]:
            raise ValueError("消费者尚未完成，不能跳过角色并提前回收其依赖状态。")
        if role == "planner":
            self.turn += 1
        self.active_role = role
        answer = super().call(role, **kwargs)
        record = self.calls[-1]
        before_bytes = self.state_bytes()
        started_reclaim = time.perf_counter()
        retained, removed, keep, offset = [], [], [], 0
        for block in self.blocks:
            if reclaimable(block, policy=self.policy, completed_role=role, turn=self.turn):
                removed.append(block)
            else:
                retained.append(block)
                keep.extend(range(offset, offset + block.length))
            offset += block.length
        if offset != len(self.cached_kinds):
            raise ValueError("生命周期分段与物理缓存长度不一致。")
        if removed:
            peak = select_cache_positions(self.past, torch.tensor(keep, dtype=torch.long, device=self.bundle.device))
            self.call_cache_peak_bytes = max(self.call_cache_peak_bytes, peak)
            self.cached_kinds = [self.cached_kinds[i] for i in keep]
            self.blocks = retained
        reclaim_ms = (time.perf_counter() - started_reclaim) * 1000
        self.expected_role = (self.expected_role + 1) % len(ROLES)
        retained_counts = {}
        for block in self.blocks:
            retained_counts[block.kind] = retained_counts.get(block.kind, 0) + block.length
        record["state_after_bytes"] = self.state_bytes()
        record["cache_peak_bytes_including_compaction"] = self.call_cache_peak_bytes
        record["lifetime"] = {
            "policy": self.policy, "turn": self.turn, "milliseconds": reclaim_ms,
            "before_bytes": before_bytes, "after_bytes": self.state_bytes(),
            "retained_positions_by_kind": retained_counts,
            "removed_blocks": [asdict(block) for block in removed],
            "removed_positions": sum(block.length for block in removed),
            "next_absolute_position": self.next_absolute_position,
        }
        if role != "summarizer":
            record["handoff"] = self.handoff_size()
        record["elapsed_ms"] = (time.perf_counter() - started) * 1000
        return answer


def run_policy(bundle, readout, replay, policy, repeat, output, max_tokens):
    client = LifetimeProbeChat(bundle, policy=policy, latent_steps=2, max_tokens=max_tokens,
                               query_mode="calibrated", query_adapter=readout)
    tasks = []
    for task in replay:
        started = time.perf_counter()
        first = len(client.calls)
        prediction, error = "", None
        try:
            client.call("planner", question=task["question"], scope=task["group_id"])
            client.call("retriever", tool=lambda _: {
                "facts": copy.deepcopy(task["facts"]),
                "audit": {"evidence_replay": True, "source_ids": task["source_ids"], "knowledge_searches": 0}})
            client.call("reasoner")
            prediction = client.call("summarizer")
            if client.calls[-1]["finish_reason"] != "stop":
                error = "达到共同输出上限，未完整结束。"
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
        tasks.append({"id": task["id"], "question": task["question"], "prediction": prediction,
                      "error": error, "elapsed_ms": (time.perf_counter() - started) * 1000,
                      "calls": copy.deepcopy(client.calls[first:])})
        dump(output / f"{repeat}-{policy}.json", {"policy": policy, "repeat": repeat, "tasks": tasks})
        event("task_done", policy=policy, repeat=repeat, task=task["id"], prediction=prediction,
              seconds=round(tasks[-1]["elapsed_ms"] / 1000, 3), error=error)
    return {"policy": policy, "repeat": repeat, "tasks": tasks}


def summarize(runs, references):
    result = []
    for policy in POLICIES:
        selected = [r for r in runs if r["policy"] == policy]
        totals = []
        for run in selected:
            calls = [c for task in run["tasks"] for c in task["calls"]]
            scores = [answer_coverage(t["prediction"], references[t["id"]])["coverage"] for t in run["tasks"]]
            totals.append({
                "seconds": sum(t["elapsed_ms"] for t in run["tasks"]) / 1000,
                "coverage": statistics.mean(scores), "full_answers": sum(s == 1 for s in scores),
                "incomplete_tasks": sum(t["error"] is not None for t in run["tasks"]),
                "input_tokens": sum(c["input_tokens"] for c in calls),
                "output_tokens": sum(c["output_tokens"] for c in calls),
                "forward_positions": sum(c["forward_positions"] for c in calls),
                "causal_attention_pairs_per_layer_head": sum(c["causal_attention_pairs_per_layer_head"] for c in calls),
                "latent_steps": sum(c["latent_steps"] for c in calls),
                "facts_reused": sum(c["facts_reused_without_encoding"] for c in calls),
                "reclamation_ms": sum(c["lifetime"]["milliseconds"] for c in calls),
                "removed_positions": sum(c["lifetime"]["removed_positions"] for c in calls),
                "cache_peak_bytes_including_compaction": max(c["cache_peak_bytes_including_compaction"] for c in calls),
                "retained_cache_peak_bytes": max(c["state_after_bytes"] for c in calls),
                "final_cache_bytes": calls[-1]["state_after_bytes"],
                "cumulative_logical_snapshot_bytes": sum(c.get("handoff", {}).get("payload_bytes", 0) for c in calls),
                "network_transfer_bytes": None,
            })
        times = [t["seconds"] for t in totals]
        result.append({"policy": policy, "totals": totals,
                       "mean_seconds": statistics.mean(times), "median_seconds": statistics.median(times),
                       "sample_sd_seconds": statistics.stdev(times) if len(times) > 1 else None,
                       "range_seconds": [min(times), max(times)],
                       "predictions": [[t["prediction"] for t in r["tasks"]] for r in selected]})
    return result


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 1:
        raise ValueError("重复次数与输出上限必须为正数。")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    lock = verify()
    questions = BASE / "data/continuous_topiocqa_153_v1/questions.json"
    knowledge = BASE / "data/prepared_knowledge/topiocqa_153_v2"
    archive = BASE.parents[1] / "run_result/Qwen2.5-1.5B-Instruct/validation_v4_ranked153_latent_b71781b6.json"
    replay = frozen_evidence(archive, questions, knowledge)
    orders = [list(POLICIES[i % 3:] + POLICIES[:i % 3]) for i in range(args.repeats)]
    plan = {
        "scope": "固定真实证据的三题状态生命周期对照，不含模型加载、真实检索、经验写回或网络传输。",
        "model": MODEL, "threads": 4, "seed": 42, "latent_steps": 2, "max_tokens": args.max_tokens,
        "orders": orders, "policies": list(POLICIES),
        "lifetime_rule": "角色提示保留至直接下游角色完成；连续隐状态和内部边界保留至本题总结完成；问题、事实、公共规则和最终答案跨题保留。",
        "caveat": "期限来自工作流声明，不是自动推断语义依赖；保留原位置不保证被回收的信息无影响。",
        "fixed_baseline_version": lock["version"], "fixed_file_count": len(lock["files"]),
        "questions_sha256": digest(questions), "knowledge_manifest_sha256": digest(knowledge / "manifest.json"),
        "evidence_archive_sha256": digest(archive), "experiment_sha256": digest(Path(__file__)),
        "position_helper_sha256": digest(HERE / "latent_cache_position_probe.py"),
        "network_transfer_measured": False,
        "attention_metric": "每层每头的逻辑因果注意力位置对数；不是实际硬件运算数或实测内存读取量。",
        "statistical_scope": "三轮用于观察同三题的计时波动；不是九个独立任务，不做泛化显著性宣称。",
    }
    dump(output / "plan.json", plan)
    dump(output / "frozen_evidence.json", replay)
    event("model_loading")
    started = time.perf_counter()
    bundle = NativeLocalChat.from_pretrained(MODEL, max_tokens=args.max_tokens, threads=4, seed=42).bundle
    if bundle.device != "cpu":
        raise ValueError("当前计时仅核验过处理器环境。")
    before = model_digest(bundle.model)
    setup = time.perf_counter() - started
    readout = QueryReadout.load(BASE / "data/prepared_knowledge/topiocqa_153_readout_v2/readout.pt", bundle.device)
    event("model_loaded", setup_seconds=round(setup, 2))
    warm_fact = json.loads((BASE / "data/prepared_knowledge/topiocqa_119_v2/facts.json").read_text(encoding="utf-8"))[0]
    for policy in POLICIES:
        client = LifetimeProbeChat(bundle, policy=policy, latent_steps=2, max_tokens=1,
                                   query_mode="calibrated", query_adapter=readout)
        client.call("planner", question="What is known about " + warm_fact["subject"] + "?", scope="warmup")
        for role in ROLES[1:]:
            client.call(role)
    runs = []
    for repeat, order in enumerate(orders):
        for policy in order:
            event("policy_start", repeat=repeat, policy=policy)
            runs.append(run_policy(bundle, readout, replay, policy, repeat, output, args.max_tokens))
    references = {r["id"]: r["reference_answers"] for r in json.loads(questions.with_name("references.json").read_text(encoding="utf-8"))}
    after = model_digest(bundle.model)
    verify()
    inputs = [[c["inputs"] for t in r["tasks"] for c in t["calls"]] for r in runs]
    checks = {
        "fixed_baselines_unchanged": True, "base_weights_unchanged": before == after,
        "base_weights_sha256": before, "new_text_inputs_identical": all(v == inputs[0] for v in inputs),
        "only_final_role_generates_text": all(c["output_tokens"] == 0 for r in runs for t in r["tasks"]
                                               for c in t["calls"] if c["role"] != "summarizer"),
        "no_persistent_blocks_reclaimed": all(b["kind"] not in PERSISTENT for r in runs for t in r["tasks"]
                                               for c in t["calls"] for b in c["lifetime"]["removed_blocks"]),
        "all_attempts_finished": all(t["error"] is None for r in runs for t in r["tasks"]),
    }
    result = summarize(runs, references)
    dump(output / "summary.json", {"plan": plan, "checks": checks, "setup_seconds": setup, "policies": result})
    dump(output / "runs.json", runs)
    event("finished", checks=checks, policies=[{"policy": r["policy"], "mean_seconds": r["mean_seconds"],
          "full_answers": [t["full_answers"] for t in r["totals"]]} for r in result])


if __name__ == "__main__":
    main()
