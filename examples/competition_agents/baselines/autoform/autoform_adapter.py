from __future__ import annotations

import os
import time
from contextlib import contextmanager

from competition_agents.llm import get_llm_client


AUTOFORM_PRINCIPLE = (
    "You may use a more efficient communication medium than long natural language "
    "when it is clearer. Consider structured data, code-like notation, concise lists, "
    "logical expressions, or compact key-value text. Choose the communication language "
    "yourself. Preserve task-critical facts, identifiers, evidence references, "
    "constraints, and answer format. Prefer clarity, brevity, and as few words as possible."
)


def autoform_enabled() -> bool:
    """判断当前文本通信是否启用 AutoForm 任务适配复现。"""

    return os.environ.get("TEXT_COMM_STYLE", "").strip().lower() == "autoform"


def build_autoform_instruction(
    *,
    sender: str,
    receiver: str,
    action: str,
    source_message: str,
) -> str:
    """生成 AutoForm 通信提示，不在 Python 侧手工压缩或改写消息。"""

    return (
        "You are generating an inter-agent message in an AutoForm-style multi-agent system.\n"
        "The original workflow, agent roles, task, and downstream receiver are fixed.\n"
        "Only choose the most efficient communication representation for this message.\n\n"
        f"Sender: {sender}\n"
        f"Receiver: {receiver}\n"
        f"Purpose: {action}\n\n"
        "AutoForm communication principle:\n"
        f"{AUTOFORM_PRINCIPLE}\n\n"
        "Rules:\n"
        "1. Do not explain these rules.\n"
        "2. Do not add information that is not present in the source message.\n"
        "3. Do not use a fixed JSON schema unless it is naturally the clearest choice.\n"
        "4. Output only the message that should be sent to the receiver.\n\n"
        "Source message to transform:\n"
        f"{source_message}"
    )


def generate_autoform_message(
    *,
    sender: str,
    receiver: str,
    action: str,
    source_message: str,
) -> tuple[str, dict[str, object]]:
    """让模型自主生成 AutoForm 通信 payload，并返回审计信息。"""

    prompt = build_autoform_instruction(
        sender=sender,
        receiver=receiver,
        action=action,
        source_message=source_message,
    )
    started = time.perf_counter()
    with _without_hidden_capture():
        generated = get_llm_client("autoform").generate(prompt).strip()
    elapsed_ms = (time.perf_counter() - started) * 1000
    payload = generated or source_message
    audit = {
        "communication_style": "autoform",
        "autoform_prompt_chars": len(prompt),
        "generated_text_chars": len(payload),
        "transmitted_text_chars": len(payload),
        "transform_latency_ms": elapsed_ms,
        "source_text_chars": len(source_message),
    }
    return payload, audit


@contextmanager
def _without_hidden_capture():
    previous = os.environ.get("LOCAL_CAPTURE_HIDDEN")
    os.environ["LOCAL_CAPTURE_HIDDEN"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("LOCAL_CAPTURE_HIDDEN", None)
        else:
            os.environ["LOCAL_CAPTURE_HIDDEN"] = previous
