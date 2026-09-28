"""把缓存生命周期方案接回真实检索和经验写回，只复测既定三题。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from latent_cache_lifetime_probe import (
    BASE, MODEL, POLICIES, ROLES, LifetimeProbeChat, NativeLocalChat, digest, dump,
    event, model_digest, summarize, verify,
)
from competition_agents.methods.latent import LatentCollaboration


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    lock = verify()
    questions_path = BASE / "data/continuous_topiocqa_153_v1/questions.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if len(questions) != 3:
        raise ValueError("此验证仅使用已固定的前三题。")
    plan = {
        "scope": "真实超图检索、事实计算复用、经验读取写回和四角色的功能核验，每组一次三题；不据此比较稳定速度。",
        "model": MODEL, "threads": 4, "seed": 42, "max_tokens": 128,
        "policies": list(POLICIES), "questions_sha256": digest(questions_path),
        "experiment_sha256": digest(HERE / "latent_cache_lifetime_probe.py"),
        "live_check_sha256": digest(Path(__file__)), "fixed_files": len(lock["files"]),
        "calibration": "使用原有校准，无重训、无更改检索阈值；校准有效性不因源码哈希相同而自动保证。",
        "experience": "每组独立新建空经验库；组内跨题保留事实与经验。",
        "setup_excluded": "模型加载、静态知识索引和原有校准门限的准备；经验在线写回计时。",
        "network_transfer_measured": False,
    }
    dump(output / "plan.json", plan)
    event("model_loading")
    native = NativeLocalChat.from_pretrained(MODEL, max_tokens=128, threads=4, seed=42)
    if native.bundle.device != "cpu":
        raise ValueError("本次验证固定为处理器环境。")
    before = model_digest(native.bundle.model)
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True, device="cpu")
    calibration = BASE / "data/prepared_knowledge/topiocqa_153_readout_v2"
    warm_fact = json.loads((BASE / "data/prepared_knowledge/topiocqa_119_v2/facts.json").read_text(encoding="utf-8"))[0]
    runs, setup_records = [], []
    for policy in POLICIES:
        started = time.perf_counter()
        runtime = LatentCollaboration(native, {
            "calibration": calibration, "knowledge": BASE / "data/prepared_knowledge/topiocqa_153_v2",
            "encoder": encoder, "experience_db": output / (policy + "-experience.sqlite"),
        })
        readout = runtime.client.query_adapter
        # 只给实验实例替换计算状态控制器；生产工厂、检索工具和模型对象保持原实现。
        runtime.client = LifetimeProbeChat(native.bundle, policy=policy, latent_steps=2, max_tokens=128,
                                           query_mode="calibrated", query_adapter=readout)
        setup_records.append({"policy": policy, "seconds": time.perf_counter() - started, "gate": runtime.gate})
        warm = LifetimeProbeChat(native.bundle, policy=policy, latent_steps=2, max_tokens=1,
                                  query_mode="calibrated", query_adapter=readout)
        warm.call("planner", question="What is known about " + warm_fact["subject"] + "?", scope="warmup")
        for role in ROLES[1:]:
            warm.call(role)
        tasks = []
        for task in questions:
            result = runtime.run(task["question"], history_scope=task["group_id"])
            tasks.append({"id": task["id"], **copy.deepcopy(result), "elapsed_ms": result["metrics"]["elapsed_ms"]})
            dump(output / f"{policy}.json", {"policy": policy, "repeat": 0, "tasks": tasks})
            event("live_task_done", policy=policy, task=task["id"], prediction=result["prediction"],
                  error=result["error"], searches=result["metrics"]["knowledge_searches"],
                  reused=result["metrics"]["facts_reused_without_encoding"],
                  source_ids=result["retrieval_audit"].get("source_ids", []))
        runs.append({"policy": policy, "repeat": 0, "tasks": tasks})
    references = {r["id"]: r["reference_answers"] for r in json.loads(questions_path.with_name("references.json").read_text(encoding="utf-8"))}
    after = model_digest(native.bundle.model)
    verify()
    checks = {
        "fixed_baselines_unchanged": True, "weights_unchanged": before == after,
        "weights_sha256": before, "all_tasks_finished": all(t["error"] is None for r in runs for t in r["tasks"]),
        "same_selected_sources": all([t["retrieval_audit"].get("source_ids", []) for t in r["tasks"]] ==
                                     [t["retrieval_audit"].get("source_ids", []) for t in runs[0]["tasks"]] for r in runs),
    }
    dump(output / "runs.json", runs)
    dump(output / "summary.json", {"plan": plan, "checks": checks, "setups": setup_records,
                                   "policies": summarize(runs, references)})
    event("live_finished", checks=checks)


if __name__ == "__main__":
    main()
