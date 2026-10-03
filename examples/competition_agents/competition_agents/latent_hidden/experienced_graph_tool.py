"""经验只改变检索候选计算顺序，原有充分性门限与事实输出保持不变。"""
from __future__ import annotations

import time

import torch

from .calibrated_graph_tool import CalibratedGraphTool


class ExperiencedGraphTool(CalibratedGraphTool):
    def __init__(self, knowledge, bundle, teacher, experience):
        super().__init__(knowledge, bundle, teacher)
        self.experience = experience
        self._groups_key, self._groups = None, {}
        self._experience_audit = {}
        self._finished_events = set()

    def _signature(self, path):
        return tuple(' '.join(self.records[k]['relation'].casefold().split()) for k in path)

    def _prepare_groups(self, paths):
        signatures = {key: self._signature(path) for key, path in paths.items()}
        identity = tuple(sorted((key, signatures[key], tuple(paths[key])) for key in paths))
        if identity == self._groups_key:
            return
        groups = {}
        for key, signature in signatures.items():
            groups.setdefault(signature, []).append(key)
        prepared = {}
        for signature, keys in groups.items():
            vectors = self.vectors[[self.positions[k] for k in keys]]
            center = vectors.mean(dim=0)
            radius = float((vectors-center).norm(dim=-1).max())
            prepared[signature] = (keys, center, radius)
        self._groups_key, self._groups = identity, prepared

    def _memory_scores(self, query, paths, category):
        audit = self._experience_audit
        if not paths or self.constrained_question:
            audit['experience_rejection'] = 'no_memory_or_new_constraint'
            return super()._memory_scores(query, paths, category)
        started = time.perf_counter()
        record = self.experience.lookup(category)
        audit['experience_lookup_ms'] = (time.perf_counter()-started)*1000
        if record is None:
            return super()._memory_scores(query, paths, category)
        audit['experience_reads'] = 1
        audit['experience_id'] = record['memory_id']
        self._prepare_groups(paths)
        patterns = {tuple(p) for p in record['relation_paths']}
        candidate_keys = {k for pattern, (keys, _, _) in self._groups.items()
                          if pattern in patterns for k in keys}
        if not candidate_keys or len(candidate_keys) == len(paths):
            audit['experience_rejection'] = 'no_candidate_reduction'
            return super()._memory_scores(query, paths, category)
        candidates = {key: paths[key] for key in sorted(candidate_keys)}
        scores = self._scores(query, candidates)
        top = max(scores.values())
        # 对所有未算候选给出保守上界。归一化查询下，点积不超过中心点积加半径。
        # 路径长度惩罚与原检索相同；严格低于原选择阈值才允许省略。
        upper = max(float(center @ query.reshape(-1)) + radius - .12*(len(pattern)-1)
                    for pattern, (_, center, radius) in self._groups.items() if pattern not in patterns)
        selected, _ = self._select(scores, paths)
        complete = bool(selected) and all(k in self.records for k in selected)
        # 扩大候选后，旧的最高分减固定差值不再代表选择边界。
        # 只有预算已满、目录补充也已计入，才能按最后入选分数跳过剩余计算。
        inventory = set((self.coverage_plan or {}).get('required', ()))
        cutoff = min((scores[k] for k in selected if k in scores), default=-1.)
        certified = (complete and len(selected) >= self.max_facts and inventory <= set(scores)
                     and upper < cutoff-1e-5)
        audit['experience_certificate'] = {
            'kind': 'omitted_similarity_upper_bound', 'top_score': top,
            'omitted_upper_bound': upper, 'selection_cutoff': cutoff,
            'complete_resident_bundle': complete, 'passed': certified,
            'validation_target': 'retrieval_selection_equivalence_only',
        }
        if certified:
            audit['experience_applied'] = 1
            audit['experience_candidates_skipped'] = len(paths)-len(candidates)
            return scores
        audit['experience_rejection'] = 'bound_or_evidence_not_sufficient'
        # 失败后只计算剩下的候选，不重复计算已经尝试的那部分。
        scores.update(self._scores(query, {k: p for k, p in paths.items() if k not in candidates}))
        return scores

    def __call__(self, query):
        self._experience_audit = {
            'experience_status': 'connected_retrieval_strategy', 'experience_reads': 0,
            'experience_applied': 0, 'experience_writes': 0, 'experience_created': 0,
            'experience_candidates_skipped': 0, 'experience_lookup_ms': 0.,
            'experience_write_ms': 0., 'experience_pruned': 0,
        }
        result = super().__call__(query)
        result['audit'].update(self._experience_audit)
        return result

    def finish_task(self, task_key, *, completed):
        """在任务结束后写回；任务评分和答案不进入经验模块。"""
        if not self.events or len(self.events) in self._finished_events:
            return
        self._finished_events.add(len(self.events))
        audit = self.events[-1]
        if not completed:
            audit['experience_write_status'] = 'task_incomplete'
            return
        paths = [self.resident_paths[k]['path'] for k in audit['source_ids'] if k in self.resident_paths]
        signatures = [self._signature(path) for path in paths if all(k in self.records for k in path)]
        started = time.perf_counter()
        result = self.experience.commit(
            task_key=task_key, category=audit['category'], relation_paths=signatures,
            source_ids=audit['source_ids'], retrieved_id=audit.get('experience_id'),
            applied=bool(audit['experience_applied']), rejection=audit.get('experience_rejection'),
            certificate=audit.get('experience_certificate'))
        audit['experience_write_ms'] = (time.perf_counter()-started)*1000
        audit['experience_writes'] = result['writes']
        audit['experience_created'] = result['created']
        audit['experience_pruned'] = result['pruned']
        audit['experience_write_status'] = 'committed'
        audit['experience_written_id'] = result.get('experience_id')
