"""固定三题和证据的隐状态机制对照；不修改或注册正式协作入口。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

import torch

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from competition_agents.answer_evaluation import answer_coverage
from competition_agents.latent_hidden.calibrated_query import QueryReadout
from competition_agents.llm.latent_only_chat import LatentOnlyChat, ROLES
from competition_agents.llm.native_local_chat import NativeLocalChat
from tools.verify_locked_baselines import verify


VARIANTS = ("posterior8", "zero_steps", "paper_linear", "shared_residual", "role_residual")
INTERNAL_ROLES = ("planner", "retriever", "reasoner")
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event(phase, **values):
    print(json.dumps({"phase": phase, **values}, ensure_ascii=False), flush=True)


def unit(value):
    return torch.nn.functional.normalize(value.float(), dim=-1, eps=1e-6)


class LinearAlignment:
    """复现论文公开实现的岭回归映射及输入词向量平均幅度归一。"""

    def __init__(self, matrix, target_norm):
        self.matrix, self.target_norm = matrix, target_norm

    @classmethod
    def from_model(cls, model, ridge=1e-5):
        input_weight = model.get_input_embeddings().weight.detach().float()
        output_weight = model.get_output_embeddings().weight.detach().float()
        gram = output_weight.T @ output_weight
        # 共享权重时复用同一个乘积，仍求解完整岭回归，不以恒等映射冒充。
        rhs = gram if input_weight.data_ptr() == output_weight.data_ptr() else output_weight.T @ input_weight
        regularized = gram + ridge * torch.eye(gram.shape[0], device=gram.device)
        matrix = torch.linalg.solve(regularized, rhs)
        return cls(matrix, input_weight.norm(dim=1).mean())

    def __call__(self, hidden):
        return (unit(hidden.float() @ self.matrix) * self.target_norm).to(hidden.dtype)


class ResidualAlignment:
    """在固定映射旁增加按角色拟合的小修正；最大修正为基准幅度的四分之一。"""

    def __init__(self, base, adapters, *, maximum=0.25):
        self.base, self.adapters, self.maximum = base, adapters, maximum

    def __call__(self, hidden, role):
        left, right = self.adapters[role] if role in self.adapters else self.adapters["shared"]
        raw = (unit(hidden) @ left) @ right
        norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        correction = raw * (self.maximum / norm).clamp(max=1.)
        aligned = unit(self.base(hidden)) + correction
        return (unit(aligned) * self.base.target_norm).to(hidden.dtype)


def fit_low_rank(features, targets, *, rank=8, ridge=0.01):
    """只拟合小矩阵；训练目标来自同一冻结模型，不使用问答标签。"""
    features = unit(features)
    _, _, vh = torch.linalg.svd(features, full_matrices=False)
    left = vh[:min(rank, len(vh))].T.contiguous()
    z = features @ left
    right = torch.linalg.solve(z.T @ z + ridge * torch.eye(z.shape[1], device=z.device), z.T @ targets)
    return left, right


class ProbeChat(LatentOnlyChat):
    """仅替换连续反馈；角色提示、缓存接续、事实输入和最终生成继承固定版本。"""

    def __init__(self, bundle, *, alignment, observer=None, **kwargs):
        super().__init__(bundle, **kwargs)
        self.alignment, self.observer = alignment, observer
        self.active_role = None

    def call(self, role, **kwargs):
        self.active_role = role
        return super().call(role, **kwargs)

    def _work(self):
        for _ in range(self.latent_steps):
            if self.observer is not None:
                self.observer(self.active_role, self.last_hidden.detach())
            started = time.perf_counter()
            if isinstance(self.alignment, ResidualAlignment):
                vector = self.alignment(self.last_hidden, self.active_role)
            else:
                vector = self.alignment(self.last_hidden)
            self.metrics["alignment_ms"] += (time.perf_counter() - started) * 1000
            self.metrics["latent_vector_bytes"] += vector.numel() * vector.element_size()
            self._forward(embeds=vector)
            self.metrics["latent_steps"] += 1


def frozen_evidence(archive_path, questions_path, knowledge_path):
    """从已有真实检索轨迹提取证据，禁止把预测或标准答案送入回放。"""
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    source = {f["id"]: f for f in json.loads((knowledge_path / "facts.json").read_text(encoding="utf-8"))}
    projected = {}
    replay = []
    tasks = archive["runs"][0]["task_results"]
    if len(questions) != 3 or len(tasks) != 3:
        raise ValueError("此实验只接受已固定的前三道连续题。")
    for question, task in zip(questions, tasks):
        if question["question"] != task["question"] or question["id"] != task["task_id"]:
            raise ValueError("检索轨迹与固定题目不一致。")
        call = next(c for c in task["calls"] if c["role"] == "retriever")
        fresh = []
        for entry in call["inputs"]:
            if entry["category"] == "external_facts":
                tail = entry["text"].split("New original source facts:\n", 1)[1]
                values, _ = json.JSONDecoder().raw_decode(tail)
                fresh.extend(values)
        ids = call["tool"]["fresh_source_ids"]
        if len(ids) != len(fresh):
            raise ValueError("来源记录与首次输入条数不一致。")
        for key, fact in zip(ids, fresh):
            original = source[key]
            if any(fact[k] != original[k] for k in ("subject", "object", "polarity")):
                raise ValueError("回放证据与固定知识库不一致。")
            projected[key] = fact
        selected = call["tool"]["source_ids"]
        replay.append({**question, "facts": [copy.deepcopy(projected[k]) for k in selected],
                       "source_ids": selected})
    return replay


def calibration_split(facts_path, evaluation_knowledge, *, train_count=16, valid_count=8):
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    evaluation = json.loads((evaluation_knowledge / "facts.json").read_text(encoding="utf-8"))
    evaluation_documents = {f["document_id"] for f in evaluation}
    if evaluation_documents & {f["document_id"] for f in facts}:
        raise ValueError("校准来源文档不得与三题知识库重合。")
    documents = sorted({f["document_id"] for f in facts})
    validation_documents = set(documents[:2])
    training = sorted((f for f in facts if f["document_id"] not in validation_documents), key=lambda f: f["id"])
    validation = sorted((f for f in facts if f["document_id"] in validation_documents), key=lambda f: f["id"])
    if len(training) < train_count or len(validation) < valid_count:
        raise ValueError("独立来源校准样本不足。")
    return [("train", f) for f in training[:train_count]] + [("validation", f) for f in validation[:valid_count]]


def collect_calibration(bundle, alignment, split):
    rows = []
    input_weight = bundle.model.get_input_embeddings().weight
    for index, (partition, fact) in enumerate(split):
        def observe(role, hidden):
            # 离线用完整词表期望作为自监督参照；线上小修正不计算词表。
            scores = bundle.model.get_output_embeddings()(hidden).float()
            expectation = scores.softmax(dim=-1) @ input_weight.float()
            rows.append({"role": role, "partition": partition, "source_id": fact["id"],
                         "hidden": hidden.reshape(-1).float().clone(),
                         "target": unit(expectation).reshape(-1).clone()})

        client = ProbeChat(bundle, alignment=alignment, observer=observe, latent_steps=2,
                           max_tokens=1024, query_mode="legacy")
        question = "What is known about " + fact["subject"] + "?"
        projected = {k: fact[k] for k in ("id", "subject", "relation", "object", "polarity", "qualifiers")}
        client.call("planner", question=question, scope="calibration-" + str(index))
        client.call("retriever", tool=lambda _: {"facts": [projected], "audit": {"calibration_only": True}})
        client.call("reasoner")
        event("calibration_sample", completed=index + 1, total=len(split), partition=partition)
    return rows


def calibrate_residuals(base, rows, *, rank=8, ridge=0.01, maximum=0.25):
    adapters = {}
    for group in (*INTERNAL_ROLES, "shared"):
        selected = [r for r in rows if r["partition"] == "train" and (group == "shared" or r["role"] == group)]
        hidden = torch.stack([r["hidden"] for r in selected])
        target = torch.stack([r["target"] for r in selected])
        residual = target - unit(base(hidden))
        adapters[group] = fit_low_rank(hidden, residual, rank=rank, ridge=ridge)
    shared = ResidualAlignment(base, {"shared": adapters["shared"]}, maximum=maximum)
    role = ResidualAlignment(base, {r: adapters[r] for r in INTERNAL_ROLES}, maximum=maximum)
    report = {}
    for partition in ("train", "validation"):
        report[partition] = {}
        for role_name in INTERNAL_ROLES:
            selected = [r for r in rows if r["partition"] == partition and r["role"] == role_name]
            hidden = torch.stack([r["hidden"] for r in selected])
            target = torch.stack([r["target"] for r in selected])
            report[partition][role_name] = {
                "samples": len(selected),
                "base_cosine": float((unit(base(hidden)) * target).sum(-1).mean()),
                "shared_cosine": float((unit(shared(hidden, role_name)) * target).sum(-1).mean()),
                "role_cosine": float((unit(role(hidden, role_name)) * target).sum(-1).mean()),
            }
    return shared, role, adapters, report


def model_digest(model):
    checksum = hashlib.sha256()
    for name, parameter in model.named_parameters():
        checksum.update(name.encode())
        checksum.update(memoryview(parameter.detach().cpu().contiguous().numpy()))
    return checksum.hexdigest()


def run_variant(bundle, readout, replay, variant, alignments, output_path, repeat, *, max_tokens=1024):
    settings = dict(latent_steps=0 if variant == "zero_steps" else 2, max_tokens=max_tokens,
                    query_mode="calibrated", query_adapter=readout)
    if variant in ("posterior8", "zero_steps"):
        client = LatentOnlyChat(bundle, **settings)
    else:
        client = ProbeChat(bundle, alignment=alignments[variant], **settings)
    tasks = []
    for task in replay:
        call_start = len(client.calls)
        started = time.perf_counter()
        error, prediction = None, ""
        def tool(_query):
            return {"facts": copy.deepcopy(task["facts"]),
                    "audit": {"evidence_replay": True, "source_ids": task["source_ids"],
                              "knowledge_searches": 0}}
        try:
            client.call("planner", question=task["question"], scope=task["group_id"])
            client.call("retriever", tool=tool)
            client.call("reasoner")
            prediction = client.call("summarizer")
            if client.calls[-1]["finish_reason"] != "stop":
                error = "最终输出达到共同长度上限。"
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
        elapsed_ms = (time.perf_counter() - started) * 1000
        calls = copy.deepcopy(client.calls[call_start:])
        record = {"id": task["id"], "question": task["question"], "prediction": prediction,
                  "error": error, "elapsed_ms": elapsed_ms, "calls": calls,
                  "evidence_sha256": hashlib.sha256(json.dumps(task["facts"], sort_keys=True).encode()).hexdigest()}
        tasks.append(record)
        dump(output_path, {"variant": variant, "repeat": repeat, "tasks": tasks})
        event("task_done", variant=variant, repeat=repeat, task=task["id"],
              seconds=round(elapsed_ms / 1000, 3), prediction=prediction, error=error)
    return {"variant": variant, "repeat": repeat, "tasks": tasks}


def summarize(runs, references):
    summaries = []
    for variant in VARIANTS:
        selected = [r for r in runs if r["variant"] == variant]
        totals = []
        for run in selected:
            calls = [c for task in run["tasks"] for c in task["calls"]]
            coverage = [answer_coverage(t["prediction"], references[t["id"]])["coverage"] for t in run["tasks"]]
            for task, score in zip(run["tasks"], coverage):
                task["reference_coverage"] = score
            totals.append({
                "seconds": sum(t["elapsed_ms"] for t in run["tasks"]) / 1000,
                "coverage": statistics.mean(coverage),
                "failed_tasks": sum(t["error"] is not None for t in run["tasks"]),
                "input_tokens": sum(c["input_tokens"] for c in calls),
                "output_tokens": sum(c["output_tokens"] for c in calls),
                "latent_steps": sum(c["latent_steps"] for c in calls),
                "alignment_ms": sum(c["alignment_ms"] for c in calls),
                "forward_ms": sum(c["model_forward_ms"] for c in calls),
                "forward_positions": sum(c["forward_positions"] for c in calls),
                "state_peak_bytes": max(c["state_after_bytes"] for c in calls),
                "handoff_snapshot_bytes": sum(c.get("handoff", {}).get("payload_bytes", 0) for c in calls),
                "facts_reused": sum(c["facts_reused_without_encoding"] for c in calls),
                "communication_bytes": None,
            })
        values = [t["seconds"] for t in totals]
        summaries.append({"variant": variant, "repeats": len(totals),
                          "median_seconds": statistics.median(values),
                          "mean_seconds": statistics.mean(values),
                          "sample_sd_seconds": statistics.stdev(values) if len(values) > 1 else None,
                          "min_seconds": min(values), "max_seconds": max(values),
                          "coverage": statistics.mean(t["coverage"] for t in totals),
                          "totals": totals,
                          "predictions": [[t["prediction"] for t in r["tasks"]] for r in selected]})
    return summaries


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--reuse-preparation", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.repeats < 1 or args.max_tokens < 1:
        raise ValueError("重复次数及共同输出上限必须为正数。")
    locked = verify()
    question_file = BASE / "data/continuous_topiocqa_153_v1/questions.json"
    reference_file = BASE / "data/continuous_topiocqa_153_v1/references.json"
    knowledge = BASE / "data/prepared_knowledge/topiocqa_153_v2"
    calibration_file = BASE / "data/prepared_knowledge/topiocqa_119_v2/facts.json"
    archive = BASE.parents[1] / "run_result/Qwen2.5-1.5B-Instruct/validation_v4_ranked153_latent_b71781b6.json"
    replay = frozen_evidence(archive, question_file, knowledge)
    split = calibration_split(calibration_file, knowledge)
    orders = []
    randomizer = random.Random(20260927)
    for _ in range(args.repeats):
        order = list(VARIANTS)
        randomizer.shuffle(order)
        orders.append(order)
    plan = {
        "scope": "固定证据的隐状态机制对照，不是重新运行知识库检索的全流程成绩。",
        "model": MODEL, "seed": 42, "threads": 4, "max_tokens": args.max_tokens,
        "latent_steps": 2, "rank": 8, "residual_ridge": 0.01, "maximum_residual_norm": 0.25,
        "paper_ridge": 1e-5, "orders": orders, "variants": list(VARIANTS),
        "fixed_baseline_version": locked["version"], "fixed_baseline_file_count": len(locked["files"]),
        "questions_file": str(question_file), "questions_sha256": digest(question_file),
        "knowledge_manifest_sha256": digest(knowledge / "manifest.json"),
        "evidence_archive": str(archive), "evidence_archive_sha256": digest(archive),
        "calibration_source": str(calibration_file), "calibration_source_sha256": digest(calibration_file),
        "calibration_examples": [{"partition": part, "source_id": fact["id"], "document_id": fact["document_id"]}
                                 for part, fact in split],
        "calibration_target": "同一冻结模型的完整词表期望输入向量；无评测问题、无标准答案。",
        "calibration_caveat": "该自监督目标检验输入对齐，不保证下游答案正确，也不声称无信息损失。",
        "paper_implementation": "https://github.com/Gen-Verse/LatentMAS/blob/main/models.py",
        "paper_scope": "仅复现其固定对齐算子及平均幅度归一，其他角色流程保持本项目一致。",
        "metric_note": "三题顺序运行且跨题保留本组缓存；组间清空。包含最终生成，不含离线校准、模型加载及真实检索。",
        "network_transmission_measured": False,
        "experiment_code_sha256": digest(Path(__file__)),
        "reused_preparation": str(args.reuse_preparation.resolve()) if args.reuse_preparation else None,
    }
    dump(output / "plan.json", plan)
    dump(output / "frozen_evidence.json", replay)
    started = time.perf_counter()
    event("model_loading")
    native = NativeLocalChat.from_pretrained(MODEL, threads=4, seed=42, max_tokens=1024)
    bundle = native.bundle
    if bundle.device != "cpu":
        raise ValueError("此版本计时仅审核过本机处理器环境。")
    before = model_digest(bundle.model)
    setup = {"model_load_and_digest_seconds": time.perf_counter() - started}
    event("model_loaded", device=bundle.device)
    started = time.perf_counter()
    with torch.inference_mode():
        if args.reuse_preparation:
            previous_plan = json.loads((args.reuse_preparation / "plan.json").read_text(encoding="utf-8"))
            for key in ("model", "calibration_source_sha256", "calibration_examples", "rank",
                        "residual_ridge", "maximum_residual_norm", "paper_ridge"):
                if previous_plan[key] != plan[key]:
                    raise ValueError("已保存校准与当前方案不一致：" + key)
            checkpoint = torch.load(args.reuse_preparation / "alignment.pt", map_location=bundle.device,
                                    weights_only=True)
            base = LinearAlignment(checkpoint["linear_matrix"], checkpoint["target_norm"])
            adapters, rows = checkpoint["adapters"], checkpoint["calibration_rows"]
            shared = ResidualAlignment(base, {"shared": adapters["shared"]})
            role = ResidualAlignment(base, {r: adapters[r] for r in INTERNAL_ROLES})
            saved_calibration = json.loads((args.reuse_preparation / "calibration.json").read_text(encoding="utf-8"))
            calibration_report = saved_calibration["alignment_validation"]
            setup["reused_preparation_original_cost"] = saved_calibration["setup"]
            setup["load_preparation_seconds"] = time.perf_counter() - started
            event("calibration_reused_without_refitting")
        else:
            base = LinearAlignment.from_model(bundle.model)
            identity = torch.eye(base.matrix.shape[0], device=base.matrix.device)
            setup["linear_alignment_identity_relative_error"] = float((base.matrix - identity).norm() / identity.norm())
            setup["input_embedding_target_norm"] = float(base.target_norm)
            setup["linear_alignment_seconds"] = time.perf_counter() - started
            event("linear_alignment_ready", **setup)
            started = time.perf_counter()
            rows = collect_calibration(bundle, base, split)
            shared, role, adapters, calibration_report = calibrate_residuals(base, rows)
            setup["residual_calibration_seconds"] = time.perf_counter() - started
    dump(output / "calibration.json", {"setup": setup, "alignment_validation": calibration_report,
                                      "shared_adapter_parameters": sum(t.numel() for t in adapters["shared"]),
                                      "role_adapter_parameters": sum(t.numel() for r in INTERNAL_ROLES for t in adapters[r])})
    torch.save({"linear_matrix": base.matrix, "target_norm": base.target_norm, "adapters": adapters,
                "calibration_rows": rows}, output / "alignment.pt")
    alignments = {"paper_linear": base, "shared_residual": shared, "role_residual": role}
    readout = QueryReadout.load(BASE / "data/prepared_knowledge/topiocqa_153_readout_v2/readout.pt", bundle.device)
    # 五种算子均预热；正式计时不包含首次分配，预热不输入评测题目。
    warm_question = "What is known about " + split[-1][1]["subject"] + "?"
    for variant in VARIANTS:
        settings = dict(latent_steps=0 if variant == "zero_steps" else 2,
                        max_tokens=1, query_mode="calibrated", query_adapter=readout)
        warm = LatentOnlyChat(bundle, **settings) if variant in ("posterior8", "zero_steps") else ProbeChat(
            bundle, alignment=alignments[variant], **settings)
        warm.call("planner", question=warm_question, scope="warmup")
        warm.call("summarizer")
    runs = []
    for repeat, order in enumerate(orders):
        for variant in order:
            event("variant_start", variant=variant, repeat=repeat)
            runs.append(run_variant(bundle, readout, replay, variant, alignments,
                                    output / f"{repeat}-{variant}.json", repeat, max_tokens=args.max_tokens))
    # 评测答案在全部运行结束后才读取，不用于校准、选择参数或推理。
    references = {r["id"]: r["reference_answers"] for r in json.loads(reference_file.read_text(encoding="utf-8"))}
    summary = summarize(runs, references)
    after = model_digest(bundle.model)
    verify()
    new_inputs = [[c["inputs"] for t in r["tasks"] for c in t["calls"]] for r in runs]
    checks = {"base_weights_unchanged": before == after,
              "base_weights_sha256": before,
              "new_text_inputs_identical": all(v == new_inputs[0] for v in new_inputs),
              "fixed_baselines_unchanged": True,
              "no_intermediate_text": all(c["output_tokens"] == 0 for r in runs for t in r["tasks"]
                                          for c in t["calls"] if c["role"] != "summarizer"),
              "all_tasks_completed": all(t["error"] is None for r in runs for t in r["tasks"])}
    dump(output / "summary.json", {"plan": plan, "setup": setup, "checks": checks,
                                  "calibration": calibration_report, "variants": summary})
    dump(output / "runs.json", runs)
    event("finished", checks=checks, results=[{k: r[k] for k in ("variant", "median_seconds", "coverage")} for r in summary])


if __name__ == "__main__":
    main()
