from __future__ import annotations

import os



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
