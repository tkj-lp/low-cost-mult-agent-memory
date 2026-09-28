"""独立检查角色交接裁剪的位置一致性；不注册或改动正式协作入口。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from transformers.cache_utils import DynamicLayer

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from latent_alignment_probe import (
    BASE, MODEL, QueryReadout, NativeLocalChat, answer_coverage, digest, dump,
    event, frozen_evidence, model_digest, verify,
)
from competition_agents.llm.experimental_state_chat import cache_bytes
from competition_agents.llm.latent_only_chat import LatentOnlyChat


VARIANTS = ("full", "pruned_reindexed", "pruned_absolute")
UPSTREAM_COMMIT = "9a9e4d331eb11430bd9e64754c6b252b06d73031"


def select_cache_positions(past, keep):
    """物理删除缓存位置，保留键中既有的旋转编码；不隐式修改逻辑位置。"""
    if keep.ndim != 1 or keep.dtype != torch.long or keep.numel() == 0:
        raise ValueError("保留位置必须是一维非空整数索引。")
    length = int(past.get_seq_length())
    if int(keep[0]) < 0 or int(keep[-1]) >= length or not bool((keep[1:] > keep[:-1]).all()):
        raise ValueError("保留位置必须递增、不重复且在缓存范围内。")
    if any(type(layer) is not DynamicLayer or layer.get_seq_length() != length for layer in past.layers):
        raise ValueError("本实验只支持各层等长的完整注意力动态缓存。")
    peak_bytes = cache_bytes(past)
    for layer in past.layers:
        keys = layer.keys.index_select(-2, keep.to(layer.keys.device)).contiguous()
        values = layer.values.index_select(-2, keep.to(layer.values.device)).contiguous()
        # 旧层与新层短暂共存的缓存张量峰值也计入，避免仅报告裁剪后的大小。
        added = keys.numel() * keys.element_size() + values.numel() * values.element_size()
        peak_bytes = max(peak_bytes, cache_bytes(past) + added)
        layer.keys, layer.values = keys, values
    return peak_bytes


class PositionProbeChat(LatentOnlyChat):
    """同样的角色输入及反馈，仅在角色完成后裁剪其提示缓存。"""

    def __init__(self, bundle, *, variant, **kwargs):
        if variant not in VARIANTS:
            raise ValueError("未知位置对照方案。")
        self.variant = variant
        super().__init__(bundle, **kwargs)

    def reset(self):
        super().reset()
        self.next_absolute_position = 0
        self.cached_kinds = []
        self.pending_kind = None
        self.call_cache_peak_bytes = 0

    def _append(self, text, category):
        kind = category
        if category == "instruction":
            kind = "common_instruction" if self.past is None else "completed_role_instruction"
        previous = self.pending_kind
        self.pending_kind = kind
        try:
            return super()._append(text, category)
        finally:
            self.pending_kind = previous

    def _forward(self, *, ids=None, embeds=None):
        values = ids if ids is not None else embeds
        count = int(values.shape[1])
        stored = int(self.past.get_seq_length()) if self.past is not None else 0
        if stored != len(self.cached_kinds):
            raise ValueError("缓存位置记录与实际存储长度不一致。")
        position = stored if self.variant == "pruned_reindexed" else self.next_absolute_position
        kwargs = {
            "past_key_values": self.past, "use_cache": True, "return_dict": True,
            "attention_mask": torch.ones((1, stored + count), dtype=torch.long, device=values.device),
            # 存储槽位决定因果掩码，原始逻辑位置决定旋转编码；两者不能混用。
            "cache_position": torch.arange(stored, stored + count, device=values.device),
            "position_ids": torch.arange(position, position + count, device=values.device).unsqueeze(0),
            "input_ids" if ids is not None else "inputs_embeds": values,
        }
        started = time.perf_counter()
        output = self.bundle.model.model(**kwargs)
        self.metrics["model_forward_ms"] += (time.perf_counter() - started) * 1000
        self.metrics["forward_positions"] += count
        self.past = output.past_key_values
        self.last_hidden = output.last_hidden_state[:, -1:, :]
        self.next_absolute_position += count
        kind = self.pending_kind or ("latent" if embeds is not None else "question_or_answer")
        self.cached_kinds.extend([kind] * count)
        self.call_cache_peak_bytes = max(self.call_cache_peak_bytes, self.state_bytes())
        if ids is not None:
            self.seen_ids.extend(ids[0].tolist())
        return output.last_hidden_state

    def handoff_size(self):
        report = super().handoff_size()
        # 接收方接续计算必需的额外标量，按八字节整数计量。
        report["position_metadata_bytes"] = 8
        report["next_absolute_position"] = self.next_absolute_position
        report["payload_bytes"] += 8
        return report

    def call(self, role, **kwargs):
        started = time.perf_counter()
        self.call_cache_peak_bytes = self.state_bytes()
        answer = super().call(role, **kwargs)
        record = self.calls[-1]
        before_bytes, before_length = self.state_bytes(), len(self.cached_kinds)
        prune_started = time.perf_counter()
        remove = self.variant != "full" and "completed_role_instruction" in self.cached_kinds
        if remove:
            keep_list = [i for i, kind in enumerate(self.cached_kinds) if kind != "completed_role_instruction"]
            keep = torch.tensor(keep_list, dtype=torch.long, device=self.bundle.device)
            temporary_peak = select_cache_positions(self.past, keep)
            self.call_cache_peak_bytes = max(self.call_cache_peak_bytes, temporary_peak)
            self.cached_kinds = [self.cached_kinds[i] for i in keep_list]
        prune_ms = (time.perf_counter() - prune_started) * 1000
        record["state_after_bytes"] = self.state_bytes()
        record["cache_peak_bytes_including_compaction"] = self.call_cache_peak_bytes
        record["pruning"] = {
            "milliseconds": prune_ms, "before_bytes": before_bytes, "after_bytes": self.state_bytes(),
            "removed_positions": before_length - len(self.cached_kinds),
            "stored_positions": len(self.cached_kinds), "next_absolute_position": self.next_absolute_position,
        }
        if role != "summarizer":
            record["handoff"] = self.handoff_size()
        # 包含裁剪、额外计量与状态交接准备，不能把新增开销排除在比较之外。
        record["elapsed_ms"] = (time.perf_counter() - started) * 1000
        return answer


def run_variant(bundle, readout, replay, variant, repeat, output, max_tokens):
    client = PositionProbeChat(bundle, variant=variant, latent_steps=2, max_tokens=max_tokens,
                               query_mode="calibrated", query_adapter=readout)
    tasks = []
    for task in replay:
        started = time.perf_counter()
        first_call = len(client.calls)
        prediction, error = "", None

        def tool(_query):
            return {"facts": copy.deepcopy(task["facts"]),
                    "audit": {"evidence_replay": True, "source_ids": task["source_ids"], "knowledge_searches": 0}}

        try:
            client.call("planner", question=task["question"], scope=task["group_id"])
            client.call("retriever", tool=tool)
            client.call("reasoner")
            prediction = client.call("summarizer")
            if client.calls[-1]["finish_reason"] != "stop":
                error = "达到共同输出上限，未完整结束。"
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
        tasks.append({"id": task["id"], "question": task["question"], "prediction": prediction,
                      "error": error, "elapsed_ms": (time.perf_counter() - started) * 1000,
                      "calls": copy.deepcopy(client.calls[first_call:])})
        dump(output / f"{repeat}-{variant}.json", {"variant": variant, "repeat": repeat, "tasks": tasks})
        event("task_done", variant=variant, repeat=repeat, task=task["id"], prediction=prediction,
              seconds=round(tasks[-1]["elapsed_ms"] / 1000, 3), error=error)
    return {"variant": variant, "repeat": repeat, "tasks": tasks}


def summarize(runs, references):
    reports = []
    for variant in VARIANTS:
        selected = [r for r in runs if r["variant"] == variant]
        totals = []
        for run in selected:
            calls = [c for t in run["tasks"] for c in t["calls"]]
            coverages = [answer_coverage(t["prediction"], references[t["id"]])["coverage"] for t in run["tasks"]]
            totals.append({
                "seconds": sum(t["elapsed_ms"] for t in run["tasks"]) / 1000,
                "coverage": statistics.mean(coverages), "full_answers": sum(c == 1 for c in coverages),
                "incomplete_tasks": sum(t["error"] is not None for t in run["tasks"]),
                "input_tokens": sum(c["input_tokens"] for c in calls),
                "output_tokens": sum(c["output_tokens"] for c in calls),
                "forward_positions": sum(c["forward_positions"] for c in calls),
                "latent_steps": sum(c["latent_steps"] for c in calls),
                "facts_reused": sum(c["facts_reused_without_encoding"] for c in calls),
                "pruning_ms": sum(c["pruning"]["milliseconds"] for c in calls),
                "pruned_positions": sum(c["pruning"]["removed_positions"] for c in calls),
                "cache_peak_bytes_including_compaction": max(c["cache_peak_bytes_including_compaction"] for c in calls),
                "retained_cache_peak_bytes_at_role_end": max(c["state_after_bytes"] for c in calls),
                "handoff_snapshot_bytes": sum(c.get("handoff", {}).get("payload_bytes", 0) for c in calls),
            })
        seconds = [t["seconds"] for t in totals]
        reports.append({"variant": variant, "totals": totals, "median_seconds": statistics.median(seconds),
                        "range_seconds": [min(seconds), max(seconds)],
                        "predictions": [[t["prediction"] for t in r["tasks"]] for r in selected]})
    return reports


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
    question_file = BASE / "data/continuous_topiocqa_153_v1/questions.json"
    reference_file = question_file.with_name("references.json")
    knowledge = BASE / "data/prepared_knowledge/topiocqa_153_v2"
    archive = BASE.parents[1] / "run_result/Qwen2.5-1.5B-Instruct/validation_v4_ranked153_latent_b71781b6.json"
    replay = frozen_evidence(archive, question_file, knowledge)
    orders = [list(VARIANTS[i % 3:] + VARIANTS[:i % 3]) for i in range(args.repeats)]
    plan = {
        "scope": "固定真实证据的缓存交接机制对照；不计真实检索、经验写回、模型加载，不是全流程成绩。",
        "model": MODEL, "threads": 4, "seed": 42, "max_tokens": args.max_tokens, "latent_steps": 2,
        "orders": orders, "upstream_commit": UPSTREAM_COMMIT,
        "upstream_scope": "官方实现可选裁剪分支的位置缺失；不是默认完整缓存分支的错误。",
        "selection_rule": "每个角色完成后仅删除其角色提示缓存；保留共同规则、问题、事实、连续隐步骤、边界与最终回答。",
        "selection_caveat": "已完成角色提示是否可删是待验证假设；隐状态不保证完整保存被删提示。",
        "position_control": "相同裁剪规则分别用剩余长度、原始逻辑长度生成后续位置；未重训映射。",
        "fixed_baseline_version": lock["version"], "fixed_baseline_file_count": len(lock["files"]),
        "questions_sha256": digest(question_file), "knowledge_manifest_sha256": digest(knowledge / "manifest.json"),
        "evidence_archive_sha256": digest(archive), "experiment_code_sha256": digest(Path(__file__)),
        "network_measured": False,
        "payload_scope": "各次完整接续快照之和，含保留下来的所有层缓存、辅助张量和必要元数据；非实际网络流量。",
    }
    dump(output / "plan.json", plan)
    dump(output / "frozen_evidence.json", replay)
    event("model_loading")
    started = time.perf_counter()
    bundle = NativeLocalChat.from_pretrained(MODEL, max_tokens=args.max_tokens, threads=4, seed=42).bundle
    if bundle.device != "cpu":
        raise ValueError("本实验的计时实现仅核验过处理器环境。")
    before_hash = model_digest(bundle.model)
    setup_seconds = time.perf_counter() - started
    readout = QueryReadout.load(BASE / "data/prepared_knowledge/topiocqa_153_readout_v2/readout.pt", bundle.device)
    event("model_loaded", setup_seconds=round(setup_seconds, 2))
    warm_fact = json.loads((BASE / "data/prepared_knowledge/topiocqa_119_v2/facts.json").read_text(encoding="utf-8"))[0]
    for variant in VARIANTS:
        client = PositionProbeChat(bundle, variant=variant, latent_steps=2, max_tokens=1,
                                   query_mode="calibrated", query_adapter=readout)
        client.call("planner", question="What is known about " + warm_fact["subject"] + "?", scope="warmup")
        client.call("summarizer")
    runs = []
    for repeat, order in enumerate(orders):
        for variant in order:
            event("variant_start", repeat=repeat, variant=variant)
            runs.append(run_variant(bundle, readout, replay, variant, repeat, output, args.max_tokens))
    # 全部生成完毕后才读标准答案；不按本次三题表现选择裁剪比例或更改规则。
    references = {r["id"]: r["reference_answers"] for r in json.loads(reference_file.read_text(encoding="utf-8"))}
    after_hash = model_digest(bundle.model)
    verify()
    inputs = [[c["inputs"] for t in r["tasks"] for c in t["calls"]] for r in runs]
    checks = {
        "fixed_baselines_unchanged": True, "base_weights_unchanged": before_hash == after_hash,
        "base_weights_sha256": before_hash, "new_inputs_identical": all(v == inputs[0] for v in inputs),
        "only_final_role_generates_text": all(c["output_tokens"] == 0 for r in runs for t in r["tasks"]
                                               for c in t["calls"] if c["role"] != "summarizer"),
        "all_attempts_finished": all(t["error"] is None for r in runs for t in r["tasks"]),
    }
    summary = {"plan": plan, "checks": checks, "setup_seconds": setup_seconds,
               "variants": summarize(runs, references)}
    dump(output / "summary.json", summary)
    dump(output / "runs.json", runs)
    event("finished", checks=checks, variants=[{"variant": v["variant"], "seconds": v["median_seconds"],
          "full_answers": [t["full_answers"] for t in v["totals"]]} for v in summary["variants"]])


if __name__ == "__main__":
    main()
