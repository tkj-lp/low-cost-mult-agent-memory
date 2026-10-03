"""协议兼容入口；不再包含独立提示、解析控制或证据充分性规则。"""
from .collaboration_transport import encode, full_evidence_packet
from .role_prompts import COMMON as FIELD_COMMON, QUESTION_ROLES as FIELD_ROLES


class InvalidSupportReference(ValueError):
    """仅供旧记录读取兼容；主流程的来源编号由程序保留。"""


class FieldCollaboration:
    def __init__(self, knowledge, client, *, max_searches=2, resources=None):
        from .unified_collaboration import UnifiedCollaboration, default_resources
        self.runtime = UnifiedCollaboration(client, resources or default_resources(knowledge),
                                            method='structured', max_searches=max_searches)
        self.memory = self.runtime.memory

    def run(self, question, *, history_scope='single'):
        return self.runtime.run(question, history_scope=history_scope)

    def close(self):
        self.runtime.close()
