"""独立的证据和经验读写；不决定角色是否能够回答。"""

from __future__ import annotations

import copy
import hashlib
import json
from uuid import uuid4

from .evidence_memory import SessionEvidenceMemory
from .plain_text_collaboration import PlainTextKnowledge
from .methods.text import SimilarityTextMemory
from hypergraph_memory_method.prepared_knowledge import PreparedKnowledge
from hypergraph_memory_method.discourse_reference import SourceReferenceHistory
from hypergraph_memory_method.retrieval_experience import RetrievalExperienceMemory


class CollaborationMemory:
    """两种检索后端共用的存取契约；模型答案与来源事实分别存储。"""

    def __init__(self, resources, *, graph, enabled=True):
        self.graph, self.enabled = graph, enabled
        knowledge = resources['knowledge']
        if graph:
            self.knowledge = knowledge if hasattr(knowledge, 'documents') else PreparedKnowledge(knowledge)
        else:
            self.knowledge = (PlainTextKnowledge.from_knowledge(knowledge) if hasattr(knowledge, 'documents')
                              else PlainTextKnowledge(knowledge))
        self.records, self.episodes, self.text_evidence, self.text_scopes = {}, {}, {}, {}
        self.run_id = uuid4().hex
        if graph:
            self.knowledge.prepare_focused_search(resources['embed'])
            self.backend = SessionEvidenceMemory('protocol', None, self.knowledge)
            self.references = {}
        else:
            self.backend = resources.get('memory_backend') or SimilarityTextMemory(resources['encode'])
        namespace = hashlib.sha256(json.dumps({
            'corpus': self.knowledge.manifest['corpus_sha256'], 'policy': 'role_decision_v1',
            'representation': 'graph' if graph else 'whole_text'}, sort_keys=True).encode()).hexdigest()
        self.experience = RetrievalExperienceMemory(resources.get('experience_db') or ':memory:',
                                                     namespace, policy='hint_only')

    def history(self, scope):
        return copy.deepcopy(self.episodes.get(scope, [])) if self.enabled else []

    def _reference(self, scope):
        if scope not in self.references:
            self.references[scope] = SourceReferenceHistory(self.knowledge)
        return self.references[scope]

    def recall(self, question, scope, task_kind):
        if not self.enabled:
            return {'history': [], 'evidence': [], 'text': '', 'text_units': [],
                    'query': question, 'experience': None}
        history = self.history(scope)
        text, hits, retrieval, query, diagnostics = '', [], {}, question, []
        text_units = []
        try:
            if self.graph:
                reference = self._reference(scope)
                roots, binding = reference.resolve(question, self.knowledge.relation_index.anchors(question))
                _, hits, query = self.backend.context(question, scope, roots=roots)
                retrieval = {'roots': sorted(roots), 'reference_suggestion': binding}
            else:
                text, _, turns = self.backend.read_for_question(scope, question)
                retrieval = copy.deepcopy(getattr(self.backend, 'last_retrieval', {}))
                if not retrieval:
                    retrieval = {'selected': list(range(turns)), 'unit': 'complete_turn_text'}
                selected = retrieval.get('selected', [])
                read_units = getattr(self.backend, 'read_units', None)
                if read_units is not None:
                    text_units = read_units(scope, selected)
                elif text:
                    # 外接后端没有单元接口时仍保留整份返回，不切分或丢弃原文。
                    text_units = [{'id': hashlib.sha256(text.encode('utf-8')).hexdigest(), 'text': text}]
                keys = list(dict.fromkeys(key for index in selected
                           for key in self.text_evidence.get(scope, {}).get(index, [])))
                hits = [dict(copy.deepcopy(self.records[key]), from_memory=True) for key in keys]
        except Exception as exc:
            diagnostics.append({'role': 'memory', 'kind': 'evidence_read_error', 'message': str(exc)[:500]})
        suggestion = None
        try:
            suggestion = self.experience.lookup(task_kind)
        except Exception as exc:
            diagnostics.append({'role': 'memory', 'kind': 'experience_read_error', 'message': str(exc)[:500]})
        return {'history': history, 'evidence': hits, 'text': text, 'text_units': text_units, 'query': query,
                'retrieval': retrieval, 'experience': copy.deepcopy(suggestion), 'diagnostics': diagnostics}

    def search(self, question, scope, *, expanded=False):
        """仅在角色请求后执行；查询与缓存是否命中无关。"""
        if not self.graph:
            return [dict(context_id=doc['id'], title=doc['title'], text=' '.join(doc['sentences']),
                         retrieval_backend='complete_original_text') for doc in self.knowledge.documents]
        reference = self._reference(scope)
        roots, _ = reference.resolve(question, self.knowledge.relation_index.anchors(question))
        hits = (self.knowledge.search_graph(question, 48) if expanded else
                self.knowledge.search_graph_focused(question, 24, roots=roots))
        if not hits:
            # 首查也消费角色查询；空召回才提供完整来源，并显式标记兜底。
            # 不因查询接口接通而丢弃可供模型判断的原始关系。
            hits = [dict(hit, retrieval_backend='complete_graph_sources_empty_query_fallback')
                    for hit in self.knowledge.read_graph_sources()]
        # 多跳检索返回什么就保留什么；没有答案判定或第二轮事实裁剪。
        return hits

    def observe(self, question, hits, scope):
        """取得来源即保存，后续生成或执行失败不能撤销已取得的事实。"""
        if not self.enabled:
            return 0
        added = 0
        for hit in hits:
            key = hit['context_id']
            if key not in self.records:
                self.records[key] = copy.deepcopy({k: v for k, v in hit.items()
                                                  if k not in ('from_memory', 'memory_turns')})
                added += 1
        if self.graph:
            self.backend.store_evidence(question, hits, scope)
        return added

    def commit(self, question, answer, snapshot, scope, task_kind, *, completed):
        if not self.enabled:
            return {'written': [], 'experience': None, 'experience_writes': 0}
        diagnostics = []
        hits = list(snapshot['evidence'].values())
        try:
            self.observe(question, hits, scope)
        except Exception as exc:
            diagnostics.append({'role': 'memory', 'kind': 'evidence_write_error', 'message': str(exc)[:500]})
        turn = len(self.episodes.get(scope, [])) + 1
        episode = {'question': question, 'answer': answer, 'turn': turn,
                   'verified': False, 'completed': completed,
                   'evidence_ids': [h['context_id'] for h in hits]}
        self.episodes.setdefault(scope, []).append(episode)
        written = []
        if self.graph:
            try:
                self.backend.remember(question, answer, [], scope, observed_evidence=hits)
                written = hits
                reference = self._reference(scope)
                reference.remember(question, answer, episode['evidence_ids'],
                                   self.knowledge.relation_index.anchors(question))
            except Exception as exc:
                diagnostics.append({'role': 'memory', 'kind': 'evidence_write_error', 'message': str(exc)[:500]})
        else:
            # 记忆单元是实际整轮原文，未切段、未抽取答案片段。
            known = self.text_scopes.setdefault(scope, set())
            fresh = [h for h in hits if h['context_id'] not in known]
            # 来源只在首次实际观察时存原文，不能每轮复制整库人为放大文本基线。
            original = '\n\n'.join('Original source: ' + h['title'] + '\n' + h['text'] for h in fresh)
            work = '\n\n'.join(row['role'] + ' (unverified):\n' + row['content']
                               for row in snapshot['work'])
            text = 'Question:\n' + question + '\n\n' + original + '\n\n' + work + '\n\nAnswer (unverified):\n' + answer
            # 历史记录可能已保留一次写入失败的回答；索引必须跟随实际写入的单元。
            try:
                index = self.backend.read_all(scope)[2]
                self.backend.append(scope, text, corpus_version=self.knowledge.version,
                                    contains_sources=bool(fresh))
                self.text_evidence.setdefault(scope, {})[index] = [h['context_id'] for h in fresh]
                known.update(h['context_id'] for h in fresh)
                written = {'text': text, 'committed': True}
            except Exception as exc:
                diagnostics.append({'role': 'memory', 'kind': 'evidence_write_error', 'message': str(exc)[:500]})
        relations = [[h['fact_fields']['relation']] for h in hits if h.get('fact_fields')]
        outcome = {'writes': 0}
        try:
            outcome = self.experience.commit(task_key=self.run_id + ':' + scope + ':' + str(turn),
                category=task_kind, relation_paths=relations, source_ids=episode['evidence_ids'],
                retrieved_id=(snapshot.get('experience') or {}).get('memory_id'), applied=False)
        except Exception as exc:
            diagnostics.append({'role': 'memory', 'kind': 'experience_write_error', 'message': str(exc)[:500]})
        return {'written': written, 'experience': self.experience.records.get(outcome.get('experience_id')),
                'experience_writes': outcome['writes'], 'diagnostics': diagnostics}

    def close(self):
        try:
            self.backend.close()
        finally:
            self.experience.close()
