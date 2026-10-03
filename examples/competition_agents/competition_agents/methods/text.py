"""纯文本入口：整段记忆相似度检索，完整文本交接，不加载隐状态代码。"""
import numpy as np


class SimilarityTextMemory:
    method = 'semantic similarity over complete turn text; no passage segmentation'

    def __init__(self, encode, limit=4):
        self.encode, self.limit = encode, limit
        self.sessions = {}
        self.last_retrieval = {}

    def _vector(self, text):
        value = np.asarray(self.encode(text), dtype=np.float32)
        if value.ndim != 1 or not np.isfinite(value).all() or np.linalg.norm(value) == 0:
            raise ValueError('文本相似度编码没有返回有效向量。')
        return value / np.linalg.norm(value)

    def append(self, scope, text, *, corpus_version, contains_sources=False):
        self.sessions.setdefault(scope, []).append({
            'text': text, 'vector': self._vector(text),
            'corpus_version': corpus_version if contains_sources else None})

    def read_for_question(self, scope, question):
        records = self.sessions.get(scope, [])
        if not records:
            self.last_retrieval = {'candidates': 0, 'selected': [], 'scores': []}
            return '', None, 0
        vector = self._vector(question)
        scores = [float(row['vector'] @ vector) for row in records]
        chosen = sorted(sorted(range(len(records)), key=lambda i: (-scores[i], i))[:self.limit])
        selected = [records[i] for i in chosen]
        self.last_retrieval = {'candidates': len(records), 'selected': chosen,
                               'scores': [scores[i] for i in chosen], 'unit': 'complete_turn_text'}
        return '\n\n'.join(r['text'] for r in selected), next(
            (r['corpus_version'] for r in reversed(selected) if r['corpus_version']), None), len(records)

    def read_all(self, scope):
        records = self.sessions.get(scope, [])
        return '\n\n'.join(r['text'] for r in records), next(
            (r['corpus_version'] for r in reversed(records) if r['corpus_version']), None), len(records)

    def read_units(self, scope, indices):
        """返回已选中的完整记忆单元，重查合并时不切段或再次筛选。"""
        records = self.sessions.get(scope, [])
        return [{'id': index, 'text': records[index]['text']} for index in indices]

    def close(self):
        pass


def create(client, resources):
    from competition_agents.unified_collaboration import create_collaboration
    return create_collaboration(client, resources, method='plain')
