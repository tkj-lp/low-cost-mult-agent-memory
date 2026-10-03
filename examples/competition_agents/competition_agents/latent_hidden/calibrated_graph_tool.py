"""数值查询与完整关系交接；按来源是否已驻留决定是否查库，不裁决答案充分性。"""
from __future__ import annotations

import copy
import json
import re
import time

import torch

from .graph_state_tool import GraphStateTool
from .calibrated_query import INTENTS
from hypergraph_memory_method.source_context import identity_qualifiers
from hypergraph_memory_method.discourse_reference import SourceReferenceHistory


class CalibratedGraphTool(GraphStateTool):
    def __init__(self, knowledge, bundle, teacher):
        self.teacher = teacher
        super().__init__(knowledge, bundle, terminals=24, max_facts=24)
        # 使用与离线教师一致的语义空间；不再对查询任意混合或恢复幅度。
        self.vectors = self.semantic_vectors
        self.intent_vectors = torch.as_tensor(teacher.encode([row[1] for row in INTENTS], normalize_embeddings=True),
                                              dtype=torch.float32, device=bundle.device)
        self.resident_paths = {}
        self.constrained_question = False
        self.coverage_plan = None
        self.directory_ms = 0.
        self.reference_history = SourceReferenceHistory(knowledge)
        self.reference_binding = None
        self.current_question = None

    def _encode_index(self):
        texts = []
        for key in self.keys:
            fact = self.knowledge._graph_rows[key][3]
            texts.append(' '.join([fact['relation'], fact['object'],
                                   *[q['kind'] + ': ' + q['value'] for q in fact['qualifiers']]]))
        self.semantic_vectors = torch.as_tensor(self.teacher.encode(texts, normalize_embeddings=True),
                                                dtype=torch.float32, device=self.bundle.device)
        return self.semantic_vectors

    def begin_question(self, question, scope):
        if scope != self.scope:
            self.resident_paths = {}
            self.reference_history.reset()
        super().begin_question(question, scope)
        self.current_question = question
        self.roots, self.reference_binding = self.reference_history.resolve(question, self.roots)
        # 新数值或时序条件不依据普通属性命中擅自省略补查。
        self.constrained_question = bool(re.search(r'\d|\b(before|after|between|earlier|later|except)\b', question, re.I))
        index = getattr(self.knowledge, 'relation_index', None)
        started = time.perf_counter()
        self.coverage_plan = index.plan(question, self.roots) if index is not None else None
        # 只建立可达来源编号目录，不读取未知事实正文，也不判断答案充分性。
        # 路径根据当前实体重建，不能依赖上次查询的实体集合是否完全相同。
        self.candidate_paths = dict(self._paths())
        if self.coverage_plan:
            for key in self.coverage_plan.get('unbound_source_ids', ()):
                self.candidate_paths.setdefault(key, (key,))
        self.directory_ms = (time.perf_counter() - started) * 1000 if index is not None else 0.

    def remember_answer(self, answer):
        if self.events:
            self.reference_history.remember(self.current_question, answer, self.events[-1]['source_ids'], self.roots)

    def _scores(self, query, paths):
        keys = sorted(paths)
        if not keys:
            return {}
        indices = torch.tensor([self.positions[k] for k in keys], device=self.vectors.device)
        scores = (self.vectors[indices] @ query.reshape(-1)).detach().cpu().tolist()
        return {key: score-.12*(len(paths[key])-1) for key, score in zip(keys, scores)}

    def _select(self, scores, paths):
        ranked = sorted(scores, key=lambda key: (-scores[key], len(paths[key]), key))
        selected, terminals = [], 0
        for key in ranked:
            if key in selected:
                continue
            if len(selected) >= self.max_facts:
                break
            group = list(paths[key])
            group.extend(k for k in self.action_peers.get(key, ()) if k not in group and k in scores)
            if not all(k in scores for k in group):
                continue
            # 记忆路径只能使用实际已存事实；不得从外部目录补出正文冒充命中。
            additions = [k for k in group if k not in selected]
            selected.extend(additions)
            terminals += 1
        if self.coverage_plan and self.coverage_plan['verifiable']:
            plan = {**self.coverage_plan, 'terminals': [k for k in self.coverage_plan['terminals']
                    if k in paths and all(p in scores for p in self.coverage_plan['paths'][k])]}
            for key in self.knowledge.relation_index.select(plan, scores, max_facts=self.max_facts):
                if key not in selected:
                    selected.append(key)
        return selected, ranked

    def _memory_scores(self, query, paths, category):
        return self._scores(query, paths)

    def __call__(self, query):
        if not torch.is_tensor(query) or query.shape != (1, self.vectors.shape[1]) or not torch.isfinite(query).all():
            raise ValueError('数值查询与已校准检索空间不兼容。')
        query = torch.nn.functional.normalize(query.float(), dim=-1)
        started = time.perf_counter()
        similarity = self.intent_vectors @ query.reshape(-1)
        category_index = int(similarity.argmax())
        category, alignment = INTENTS[category_index][0], float(similarity[category_index])
        category = (self.coverage_plan or {}).get('facet') or category
        # 只搜索已驻留且来源路径完整的事实，不读取知识库正文或执行全图排序。
        paths_inventory = self.candidate_paths
        memory_paths = {key: path for key, path in paths_inventory.items()
                        if all(k in self.records for k in path)}
        missing_resident = bool(set(paths_inventory) - self.records.keys())
        memory_scores = self._memory_scores(query, memory_paths, category)
        memory_selected, memory_ranked = self._select(memory_scores, memory_paths)
        best_memory = memory_scores[memory_ranked[0]] if memory_ranked else -1.
        # 仅核验来源关系驻留情况；不以相似度或目录判断答案是否正确。
        sufficient = bool(paths_inventory) and not missing_resident
        coverage = None
        if self.coverage_plan is not None:
            coverage = self.knowledge.relation_index.certificate(self.coverage_plan, self.records)
            if self.coverage_plan.get('verifiable') and not self.constrained_question:
                sufficient = sufficient or bool(coverage.get('relations_present', coverage['sufficient']))
        memory_ms = (time.perf_counter()-started)*1000
        graph_ms = 0.
        source_scope = 'graph_relations'
        if sufficient:
            selected, ranked, scores, paths = memory_selected, memory_ranked, memory_scores, memory_paths
            reason = 'requested_source_relations_resident' if coverage and coverage['sufficient'] else 'all_reachable_source_candidates_resident'
        else:
            stamp = time.perf_counter()
            paths = dict(paths_inventory)
            scores = self._scores(query, paths)
            selected, ranked = self._select(scores, paths)
            graph_ms = (time.perf_counter()-stamp)*1000
            reason = 'no_resident_evidence' if not memory_selected else 'incomplete_resident_bundle'
            # 补查增加新关系，不能丢弃记忆检索已经交出的事实。
            selected = list(dict.fromkeys([*memory_selected, *selected]))
            for key in memory_selected:
                paths.setdefault(key, memory_paths[key])
            # 与全文链路的首次完整来源读取对齐；这些事实实际送入模型后才驻留。
            complete = getattr(self.knowledge, 'read_graph_sources', None)
            if not self.records and complete:
                source_scope = 'complete_source_corpus_first_read'
                selected = [hit['context_id'] for hit in complete()]
                for key in selected:
                    paths.setdefault(key, (key,))
        stamp = time.perf_counter()
        fresh, reused, facts = [], [], []
        for key in selected:
            if key in self.records:
                reused.append(key)
            else:
                fresh.append(key)
                self.records[key] = copy.deepcopy(self.knowledge._graph_rows[key][3])
            if key in paths:
                self.resident_paths[key] = {'path': paths[key], 'roots': tuple(sorted(self.roots))}
            else:
                # 同一动作扩展出的限定、并列对象与直接主体关系一并保留来源连接。
                parent = next((p for p in selected if key in self.action_peers.get(p, ()) and p in paths), None)
                if parent is not None:
                    self.resident_paths[key] = {'path': tuple(dict.fromkeys((*paths[parent], key))),
                                                'roots': tuple(sorted(self.roots))}
            fact = self.records[key]
            relation = fact['relation']
            if fact['polarity'] == 'negative' and not relation.casefold().startswith('not '):
                relation = 'NOT ' + relation
            aliases = getattr(self.knowledge, 'graph_aliases', {})
            from hypergraph_memory_method.source_context import normalized_name
            def identity(role):
                value = fact[role]
                return aliases.get((fact['document_id'], normalized_name(value)),
                                   aliases.get(normalized_name(value), value))
            row = {'id': 'f'+str(self.positions[key]), 'subject': identity('subject'), 'relation': relation,
                   'object': identity('object'), 'polarity': fact['polarity']}
            qualifiers = identity_qualifiers(fact, getattr(self.knowledge, 'graph_aliases', {}))
            title = self.knowledge._graph_rows[key][1]
            if title:
                qualifiers.append({'kind': 'source_title', 'value': title})
            if qualifiers:
                row['qualifiers'] = qualifiers
            facts.append(row)
        read_ms = (time.perf_counter()-stamp)*1000
        audit = {'query_kind': 'calibrated_model_representation', 'query_text': None,
                 'query_bytes': query.numel()*query.element_size(), 'roots': sorted(self.roots),
                 'reference_binding': self.reference_binding,
                 'category': category, 'alignment': alignment,
                 'source_read_scope': source_scope,
                 'memory_gate_basis': 'requested_relation_membership_not_answer_correctness',
                 'reachable_source_count': len(paths_inventory),
                 'unread_reachable_source_count': len(set(paths_inventory) - set(memory_paths)),
                 'memory_best_score': best_memory, 'memory_sufficient': sufficient, 'decision_reason': reason,
                 'memory_search_ms': memory_ms, 'index_search_ms': graph_ms + self.directory_ms,
                 'knowledge_directory_ms': self.directory_ms, 'knowledge_searches': int(not sufficient),
                 'memory_coverage': coverage,
                 'returned_coverage': self.knowledge.relation_index.certificate(self.coverage_plan, selected)
                    if self.coverage_plan is not None else None,
                 'memory_and_fact_read_ms': read_ms, 'source_ids': selected, 'fresh_source_ids': fresh,
                 'memory_source_ids': reused, 'ranked_candidates': [{'id': k, 'score': scores[k]} for k in ranked],
                 'all_selected_from_memory': bool(selected) and not fresh,
                 'returned_fact_bytes': len(json.dumps(facts, ensure_ascii=False).encode()),
                 'experience_reads': 0, 'experience_applied': 0, 'experience_writes': 0,
                 'experience_status': 'not_connected; calibration and evidence cache are not experience memory'}
        self.events.append(audit)
        return {'facts': facts, 'audit': audit}
