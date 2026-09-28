"""Stage-local content contracts, independent of the transport method."""

import json
import re


def json_object(text):
    value = str(text or "").strip()
    if value.startswith("```") and value.endswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)[:-3].strip()
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        parsed = None
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", value):
            try:
                candidate, _ = decoder.raw_decode(value[match.start():])
            except (ValueError, TypeError):
                continue
            if isinstance(candidate, dict):
                parsed = candidate
                break
    return parsed if isinstance(parsed, dict) else None


def historical_qa(hit):
    """Read original QA, including legacy records, without summarizing it."""
    saved = hit.get("history_qa") or {}
    trace = hit.get("trace") or {}
    meta = trace.get("task_meta") or {}
    source = hit.get("source_case") or {}
    question = saved.get("question") or source.get("question") or meta.get("question")
    if not question:
        question = str(hit.get("task_topic") or "")
        if "Question:" in question:
            question = question.split("Question:", 1)[1].split("\nContext:", 1)[0].strip()
    return {
        "memory_id": hit.get("memory_id", ""),
        "group_id": saved.get("group_id") or hit.get("group_id") or source.get("group_id") or meta.get("group_id", ""),
        "turn": saved.get("turn") or hit.get("turn_id") or source.get("turn_id") or meta.get("turn_id", ""),
        "question": question,
        "answer": saved.get("answer", hit.get("answer", hit.get("summary", ""))),
        "confidence": "unknown",
        "verification_status": saved.get("verification_status", "unverified"),
    }


def memory_object(evidence, qa_hits, experience_rules):
    return {
        "schema": "memory_object_v1",
        "evidence": [{"source_id": f"evidence-{i}", "text": text}
                     for i, text in enumerate(evidence, 1)],
        "history_qa": [historical_qa(hit) for hit in qa_hits],
        "experience": [{"memory_id": rule.get("memory_id", ""),
                        "solution": rule.get("solution_pattern", ""),
                        "avoid": rule.get("failure_pattern", "")}
                       for rule in experience_rules],
    }


def decode_memory_object(delivered):
    obj = delivered if isinstance(delivered, dict) else json_object(delivered)
    if (obj and obj.get("schema") == "memory_object_v1"
            and all(isinstance(obj.get(key), list) for key in ("evidence", "history_qa", "experience"))
            and all(isinstance(e, dict) and isinstance(e.get("text"), str) for e in obj["evidence"])
            and all(isinstance(q, dict) and isinstance(q.get("question"), str)
                    and isinstance(q.get("answer"), str) for q in obj["history_qa"])
            and all(isinstance(r, dict) for r in obj["experience"])):
        return obj
    return None


def encode_answer_handoff(answer_candidate, payload, encoding):
    """Build the actual wire envelope for the Reasoner-to-Summarizer handoff."""
    return json.dumps(
        {
            "schema": "answer_handoff_v1",
            "answer_candidate": str(answer_candidate or "").strip(),
            "encoding": str(encoding or "plain"),
            "payload": str(payload or ""),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def delivered_answer(text):
    obj = json_object(text)
    if obj is not None:
        for key in ("final_answer", "answer_candidate"):
            if isinstance(obj.get(key), str):
                return obj[key].strip()
    match = re.search(
        r"(?:^|\n)(?:Answer candidate|answer_candidate|final_answer|Answer):\s*(.*?)"
        r"(?=\n(?:Reasoning summary|Limitations|Implementation notes|Run diagnostics):|\Z)",
        str(text or ""), re.S | re.I,
    )
    return match.group(1).strip() if match else None


def conversational_reference(question):
    return bool(re.search(
        r"\b(it|its|he|his|she|her|they|them|their|this|that|these|those|"
        r"above|previous|earlier|former|latter|instead|else|again|other)\b|"
        r"\bwhat about\b|\bhow about\b", str(question), re.I))


def parse_answer_review(raw):
    """只接受显式审阅动作；不把模板、解释或空响应写成最终答案。"""
    value = str(raw or "").strip().strip("`").strip()
    if value.upper().rstrip(".") == "KEEP":
        return {"decision": "keep"}
    if value.upper().rstrip(".") == "UNKNOWN":
        return {"decision": "revise", "answer": "unknown"}
    match = re.fullmatch(r"FIX:\s*(\S[^\n]*)", value, re.I)
    if match:
        return {"decision": "revise", "answer": match.group(1).strip()}
    # 兼容已有结构化客户端。
    return json_object(raw) or {}


def parse_planner_output(raw, metrics=None):
    obj = json_object(raw)
    metrics = metrics or {}
    truncated = bool(metrics.get("output_tokens") and metrics.get("max_new_tokens")
                     and metrics["output_tokens"] >= metrics["max_new_tokens"])
    valid = (
        obj
        and obj.get("query_type")
        in {"standalone", "conversational_reference", "follow_up_fact"}
        and isinstance(obj.get("steps"), list)
        and 1 <= len(obj["steps"]) <= 3
        and all(isinstance(step, str) and 3 <= len(step.split()) <= 24 for step in obj["steps"])
    )
    if valid:
        reference_target = obj.get("reference_target", "")
        return {
            "query_type": obj["query_type"],
            "reference_target": reference_target if isinstance(reference_target, str) else "",
            "steps": obj["steps"],
        }, ""
    return {"query_type": "unknown", "reference_target": "",
            "steps": ["Retrieve evidence for the current question."]}, "truncated" if truncated else "invalid_json_contract"


def decide_qa_retrieval(policy, question, qa_available):
    """Retriever owns the final gate; uncertain follow-ups recall conservatively."""
    query_type = policy.get("query_type", "unknown")
    needed = qa_available and (query_type != "standalone" or conversational_reference(question))
    return {**policy, "need_qa_memory": bool(needed), "need_memory": bool(needed),
            "decision_source": "retriever", "qa_memory_reason": (
                "qa_memory_empty" if not qa_available else "history_reference_or_uncertain_query"
                if needed else "standalone_query"),
            "qa_memory_query": question}
