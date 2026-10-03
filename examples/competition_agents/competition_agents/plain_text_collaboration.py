"""完整原文协作：全文读取、原样交接、追加式自然语言记忆。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time


from .role_prompts import COMMON as ANSWER_REQUIREMENTS, COMMON as COMMON_TEXT
from .role_prompts import FINAL_ANSWER_REQUIREMENTS, QUESTION_ROLES as TEXT_ROLES


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def _size(text):
    return len(text.encode('utf-8'))


def _section(title, text):
    return '\n\n' + title + '\n' + text


class PlainTextKnowledge:
    """只加载完整原文；不加载超图、词表、向量或段落索引。"""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('status') != 'ready' or self.manifest.get('text_retrieval') != 'full-context':
            raise ValueError('纯文本模式要求已经准备好的完整原文知识库。')
        if 'text_vectors.npy' in self.manifest.get('files', {}):
            raise ValueError('纯文本模式不接受预建的文本向量索引。')
        raw = (self.root / 'text_documents.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.manifest['files']['text_documents.json']:
            raise ValueError('完整原文知识库校验失败。')
        self.documents = json.loads(raw)
        if _digest(self.documents) != self.manifest['corpus_sha256']:
            raise ValueError('完整原文与知识库版本不一致。')
        self._validate()

    @classmethod
    def from_knowledge(cls, knowledge):
        if isinstance(knowledge, cls):
            return knowledge
        instance = cls.__new__(cls)
        instance.documents = json.loads(json.dumps(knowledge.documents))
        instance.manifest = getattr(knowledge, 'manifest', {})
        instance._validate()
        return instance

    def _validate(self):
        if not self.documents:
            raise ValueError('完整原文知识库不能为空。')
        for doc in self.documents:
            if (not isinstance(doc.get('title'), str) or not isinstance(doc.get('sentences'), list)
                    or not doc['sentences'] or not all(isinstance(s, str) for s in doc['sentences'])
                    or not ' '.join(doc['sentences']).strip()):
                raise ValueError('每份来源必须包含完整原文。')
        self.version = _digest(self.documents)
        self.text_retrieval = 'full-context'

    def read_all(self):
        # 标题标明原文来源，不生成可供模型选择的片段编号。
        # 原始仓库的来源有时由多个字符串组成；整篇读取，不建立片段索引。
        return '\n\n'.join('Original source: ' + doc['title'] + '\n' + ' '.join(doc['sentences'])
                            for doc in self.documents)


class PlainTextMemory:
    """按会话隔离的全文日志；只有会话路由，没有内容索引与相似度筛选。"""

    def __init__(self):
        self.sessions = {}

    def read_all(self, scope):
        stored = self.sessions.get(scope, {})
        return stored.get('text', ''), stored.get('corpus_version'), stored.get('turns', 0)

    def read_for_question(self, scope, question):
        return self.read_all(scope)

    def append(self, scope, text, *, corpus_version, contains_sources=False):
        previous, _, turns = self.read_all(scope)
        self.sessions[scope] = {'text': previous + text, 'corpus_version': corpus_version, 'turns': turns + 1}

    def close(self):
        pass



class PlainTextCollaboration:
    """兼容入口；完整文本传输与其他协作方式共用角色流程。"""
    def __init__(self, knowledge, client, *, memory=None, resources=None):
        from .unified_collaboration import UnifiedCollaboration, default_resources
        resources = resources or default_resources(knowledge)
        if memory is not None:
            resources = {**resources, 'memory_backend': memory}
        self.runtime = UnifiedCollaboration(client, resources, method='plain')
        self.memory = self.runtime.memory

    def run(self, question, *, history_scope='single'):
        return self.runtime.run(question, history_scope=history_scope)

    def close(self):
        self.runtime.close()
