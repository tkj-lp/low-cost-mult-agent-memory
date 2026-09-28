"""校准后的数值检索：先检索来源记忆，覆盖或置信度不足时查询完整超图。"""
from __future__ import annotations

import copy
import json
import re
import time

import torch

from .graph_state_tool import GraphStateTool
from .calibrated_query import INTENTS
from hypergraph_memory_method.source_context import identity_qualifiers


class CalibratedGraphTool(GraphStateTool):
    def __init__(self, knowledge, bundle, teacher):
        self.teacher = teacher
        super().__init__(knowledge, bundle)
        # 使用与离线教师一致的语义空间；不再对查询任意混合或恢复幅度。
        self.vectors = self.semantic_vectors
        self.intent_vectors = torch.as_tensor(teacher.encode([row[1] for row in INTENTS], normalize_embeddings=True),
                                              dtype=torch.float32, device=bundle.device)
        self.thresholds = {row[0]: 1.01 for row in INTENTS}
        self.alignment_threshold = 1.01
        self.require_topic_alignment = True
        self.resident_paths = {}
        self.constrained_question = False
        self.coverage_plan = None
        self.directory_ms = 0.

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
        super().begin_question(question, scope)
        # 新数值或时序条件不依据普通属性命中擅自省略补查。
        self.constrained_question = bool(re.search(r'\d|\b(before|after|between|earlier|later|except)\b', question, re.I))
        index = getattr(self.knowledge, 'relation_index', None)
        started = time.perf_counter()
        self.coverage_plan = index.plan(question, self.roots) if index is not None else None
        self.directory_ms = (time.perf_counter() - started) * 1000 if index is not None else 0.

    def _scores(self, query, paths):
        keys = sorted(paths)
        if not keys:
            return {}
        indices = torch.tensor([self.positions[k] for k in keys], device=self.vectors.device)
        scores = (self.vectors[indices] @ query.reshape(-1)).detach().cpu().tolist()
        return {key: score-.12*(len(paths[key])-1) for key, score in zip(keys, scores)}

    def _select(self, scores, paths):
        if self.coverage_plan and self.coverage_plan['verifiable']:
            # 缓存检索不能借用尚未驻留的正文或桥接来源。
            plan = {**self.coverage_plan, 'terminals': [k for k in self.coverage_plan['terminals']
                    if k in paths and all(p in scores for p in self.coverage_plan['paths'][k])]}
            selected = self.knowledge.relation_index.select(plan, scores, max_facts=24)
            return selected, sorted(scores, key=lambda key: (-scores[key], key))
        ranked = sorted(scores, key=lambda key: (-scores[key], len(paths[key]), key))
        selected, terminals = [], 0
        for key in ranked:
            if key in selected:
                continue
            if terminals and (scores[key] < scores[ranked[0]]-.08 or terminals >= self.terminals):
                break
            group = list(paths[key])
            group.extend(k for k in self.action_peers.get(key, ()) if k not in group)
            # 记忆路径只能使用实际已存事实；不得从外部目录补出正文冒充命中。
            additions = [k for k in group if k not in selected]
            if selected and len(selected)+len(additions) > self.max_facts:
                continue
            selected.extend(additions)
            terminals += 1
        return selected, ranked

    def calibrate_gate(self, samples, projected):
        wrong_max = {row[0]: -1. for row in INTENTS}
        alignment_errors, correct_alignments, audits = [], [], []
        for sample, query in zip(samples, projected):
            self.begin_question(sample['question'], 'gate-calibration')
            paths = self._paths()
            if not paths:
                continue
            true_category = sample['category']
            target = self.intent_vectors[[row[0] for row in INTENTS].index(true_category)]
            expected, _ = self._select(self._scores(target, paths), paths)
            predicted_scores = self._scores(query, paths)
            similarities = self.intent_vectors @ query.reshape(-1)
            category_index = int(similarities.argmax())
            category = INTENTS[category_index][0]
            alignment = float(similarities[category_index])
            (correct_alignments if category == true_category else alignment_errors).append(alignment)
            negative = [score for key, score in predicted_scores.items() if key not in expected]
            wrong_max[category] = max(wrong_max[category], max(negative, default=-1.))
            audits.append({'expected_category': true_category, 'predicted_category': category,
                           'alignment': alignment, 'teacher_relevant_source_ids': expected,
                           'best_relevant_score': max((predicted_scores[k] for k in expected if k in predicted_scores), default=-1.),
                           'best_irrelevant_score': max(negative, default=-1.)})
        self.thresholds = {k: (v+.02 if v > -1 else 1.01) for k, v in wrong_max.items()}
        self.alignment_threshold = max(.7, max(alignment_errors, default=.0)+.02)
        # 用校准集的完整/缺失关系两种记忆条件选择门控，不根据正式三题放宽门限。
        # 来源关系强匹配时，额外要求它与宽泛题型向量接近可能误拒绝有效记忆。
        policies = []
        for require_alignment in (True, False):
            positive, false_positive = 0, 0
            for audit in audits:
                allowed = not require_alignment or audit['alignment'] >= self.alignment_threshold
                threshold = self.thresholds[audit['predicted_category']]
                positive += int(allowed and audit['best_relevant_score'] >= threshold)
                false_positive += int(allowed and audit['best_irrelevant_score'] >= threshold)
            policies.append({'require_topic_alignment': require_alignment, 'accepted_complete_memory_cases': positive,
                             'accepted_missing_relation_cases': false_positive})
        selected = max(policies, key=lambda p: (-p['accepted_missing_relation_cases'], p['accepted_complete_memory_cases']))
        self.require_topic_alignment = selected['require_topic_alignment']
        self.scope, self.roots, self.records, self.resident_paths = None, set(), {}, {}
        return {'thresholds': self.thresholds, 'alignment_threshold': self.alignment_threshold,
                'require_topic_alignment': self.require_topic_alignment, 'gate_calibration_policies': policies,
                'calibration_category_correct': sum(a['expected_category']==a['predicted_category'] for a in audits),
                'samples': len(audits), 'audits': audits,
                'limits': 'thresholds selected on calibration data, not a guarantee on unseen questions'}

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
        # 只搜索已驻留且来源路径完整的事实，不读取知识库正文或执行全图排序。
        memory_paths = {key: item['path'] for key, item in self.resident_paths.items()
                        if item['roots'] == tuple(sorted(self.roots))
                        and all(k in self.records for k in item['path'])}
        memory_scores = self._memory_scores(query, memory_paths, category)
        memory_selected, memory_ranked = self._select(memory_scores, memory_paths)
        best_memory = memory_scores[memory_ranked[0]] if memory_ranked else -1.
        complete = bool(memory_selected) and all(k in self.records for k in memory_selected)
        sufficient = (complete and not self.constrained_question
                      and (not self.require_topic_alignment or alignment >= self.alignment_threshold)
                      and best_memory >= self.thresholds[category])
        coverage = None
        if self.coverage_plan is not None:
            coverage = self.knowledge.relation_index.certificate(self.coverage_plan, self.records)
            sufficient = coverage['sufficient'] and not self.constrained_question
            if sufficient:
                # 对目录中的每条必要来源逐项验证；单个最高相似度不能充当覆盖证明。
                selected_keys = self.knowledge.relation_index.select(self.coverage_plan, memory_scores, max_facts=24)
                sufficient = all(k in self.records for k in selected_keys)
                memory_selected = selected_keys if sufficient else memory_selected
        memory_ms = (time.perf_counter()-started)*1000
        graph_ms = 0.
        if sufficient:
            selected, ranked, scores, paths = memory_selected, memory_ranked, memory_scores, memory_paths
            reason = 'calibrated_relation_match_with_complete_resident_bundle'
        else:
            stamp = time.perf_counter()
            paths = self.coverage_plan['paths'] if self.coverage_plan and self.coverage_plan['verifiable'] else self._paths()
            scores = self._scores(query, paths)
            selected, ranked = self._select(scores, paths)
            graph_ms = (time.perf_counter()-stamp)*1000
            reason = ('no_resident_evidence' if not memory_selected else
                      'new_explicit_constraint' if self.constrained_question else
                      'incomplete_resident_bundle' if not complete else 'insufficient_relation_confidence')
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
            row = {'id': 'f'+str(self.positions[key]), 'subject': fact['subject'], 'relation': relation,
                   'object': fact['object'], 'polarity': fact['polarity']}
            qualifiers = identity_qualifiers(fact, getattr(self.knowledge, 'graph_aliases', {}))
            if qualifiers:
                row['qualifiers'] = qualifiers
            facts.append(row)
        read_ms = (time.perf_counter()-stamp)*1000
        audit = {'query_kind': 'calibrated_model_representation', 'query_text': None,
                 'query_bytes': query.numel()*query.element_size(), 'roots': sorted(self.roots),
                 'category': category, 'alignment': alignment, 'memory_threshold': self.thresholds[category],
                 'require_topic_alignment': self.require_topic_alignment,
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
