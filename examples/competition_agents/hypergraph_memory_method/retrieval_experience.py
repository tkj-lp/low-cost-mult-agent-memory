"""从已完成检索提炼关系路径策略，复用现有经验存储与剪枝接口。"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .evolution import prune_memory
from .store import SQLiteHypergraphMemoryStore


class RetrievalExperienceMemory:
    """策略只建议先搜索哪些关系；证据充分性由调用者重新验证。"""

    def __init__(self, path, namespace, *, capacity=64, policy='score_bound'):
        if capacity < 1 or not namespace:
            raise ValueError('经验容量必须为正，且需要知识库及查询版本作用域。')
        if policy not in ('score_bound', 'hint_only'):
            raise ValueError('未知经验使用方式。')
        self.policy = policy
        self.strategy_version = 2 if policy == 'hint_only' else 1
        if str(path) != ':memory:':
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.store = SQLiteHypergraphMemoryStore(path)
        self.namespace, self.capacity = namespace, capacity
        self.records = {
            r['memory_id']: r for r in self.store.list(memory_types=['experience'])
            if r.get('namespace') == namespace and r.get('strategy_version') == self.strategy_version
            and r.get('lifecycle_stage') != 'deprecated'
        }

    def lookup(self, category):
        """最多尝试一条策略；跨题型只作候选提示，不能绕过运行时校验。"""
        matches = [r for r in self.records.values() if category in r['observed_categories']]
        if not matches:
            return None
        return max(matches, key=lambda r: (
            r.get('observation_order', 0),
            r['updated_at'], r['memory_id']))

    def commit(self, *, task_key, category, relation_paths, source_ids,
               retrieved_id=None, applied=False, rejection=None, certificate=None):
        """只接收检索轨迹，不接收问题答案、参考答案或评分结果。"""
        timestamp = datetime.now(timezone.utc).isoformat()
        paths = sorted({tuple(path) for path in relation_paths if path})
        if not paths or not source_ids:
            return {'writes': 0, 'created': 0, 'pruned': 0}
        digest = lambda value: hashlib.sha256(value.encode()).hexdigest()[:24]
        episode_id = 'retrieval-trace-' + digest(self.namespace + '|' + task_key)
        # 同一任务重复提交不累计经验成功次数。
        if self.store.get(episode_id):
            return {'writes': 0, 'created': 0, 'pruned': 0}
        key = 'retrieval-experience-' + digest(self.namespace + json.dumps(paths))
        changed = {}
        if retrieved_id in self.records:
            used = dict(self.records[retrieved_id])
            used['retrieval_count'] = used.get('retrieval_count', 0) + 1
            if applied:
                used['reuse_count'] = used.get('reuse_count', 0) + 1
                used['reuse_success_count'] = used.get('reuse_success_count', 0) + 1
                used['last_reused_at'] = timestamp
                used['last_validation'] = certificate
            elif rejection:
                # 适用条件不满足不等于策略本身错误，不能混算为答案失败。
                used['guard_rejections'] = used.get('guard_rejections', 0) + 1
            changed[retrieved_id] = used
        created = key not in self.records
        record = dict(changed.get(key) or self.records.get(key) or {
            'memory_id': key, 'global_memory_id': key,
            'memory_type': 'experience', 'memory_layer': 'experience',
            'source_agent': 'RetrieverAgent', 'namespace': self.namespace,
            'strategy_version': self.strategy_version, 'created_at': timestamp,
            'task_topic': '关系检索候选选择',
            'summary': '先检索过去实际采用过的关系路径；通过遗漏候选上界检查后才缩小检索范围。',
            'problem_pattern': 'relation-search-' + digest(json.dumps(paths)),
            'solution_pattern': 'prioritize-observed-relation-paths-with-score-bound',
            'trigger_conditions': ['same-knowledge-and-query-version', 'resident-source-paths',
                                   'no-new-explicit-constraint', 'omitted-score-bound-certified'],
            'failure_pattern': 'fall-back-to-complete-search-if-conditions-fail',
            'relation_paths': [list(p) for p in paths],
            'observed_categories': [], 'source_memory_ids': [], 'observations': 0,
            'confidence': .5, 'lifecycle_stage': 'episodic',
            'validation_target': 'retrieval_selection_equivalence_only',
            'answer_correctness_verified': False,
        })
        if self.policy == 'hint_only':
            record.update(summary='曾实际检索到的关系路径；仅供角色规划，不判断答案充分性。',
                          solution_pattern='suggest-observed-relation-paths',
                          trigger_conditions=['same-knowledge-version', 'role-decides-applicability'],
                          failure_pattern='role-requested-knowledge-lookup',
                          validation_target='observed_retrieval_trace_only')
        record['observed_categories'] = sorted(set(record['observed_categories']) | {category})
        record['source_memory_ids'] = list(dict.fromkeys([*record['source_memory_ids'], episode_id]))[-8:]
        record['observations'] += 1
        # 快速连续写入可能获得相同系统时间；用持久化顺序保持“最近实际采用”的含义。
        record['observation_order'] = 1 + max((r.get('observation_order', 0) for r in self.records.values()), default=0)
        record['updated_at'] = timestamp
        changed[key] = record
        episode = {
            'memory_id': episode_id, 'global_memory_id': episode_id,
            'memory_type': 'evidence', 'memory_layer': 'episodic',
            'source_agent': 'RetrieverAgent', 'namespace': self.namespace,
            'task_topic': '已完成任务的检索轨迹', 'summary': '来源关系路径与采用记录；不含答案。',
            'knowledge_refs': list(source_ids),
            'retrieval_trace': {'category': category, 'relation_paths': [list(p) for p in paths],
                                'retrieved_experience': retrieved_id, 'applied': applied,
                                'rejection': rejection, 'certificate': certificate},
        }
        next_records = {**self.records, **changed}
        kept = prune_memory(list(next_records.values()), max_records=self.capacity, aggressive=True)
        kept_ids = {r['memory_id'] for r in kept}
        removed = set(next_records) - kept_ids
        with self.store.transaction():
            self.store.upsert(episode)
            for memory_id, item in changed.items():
                if memory_id in kept_ids:
                    next_records[memory_id] = self.store.upsert(item)
            for memory_id in removed:
                self.store.delete(memory_id)
            # 来源轨迹只保留仍被经验引用的记录，容量和每条来源数量都有上限。
            referenced = {s for r in kept for s in r['source_memory_ids']}
            for old in self.store.list(memory_types=['evidence']):
                if old.get('namespace') == self.namespace and old['memory_id'] not in referenced:
                    self.store.delete(old['memory_id'])
        self.records = {k: v for k, v in next_records.items() if k in kept_ids}
        return {'writes': len(changed), 'created': int(created and key in kept_ids),
                'pruned': len(removed), 'experience_id': key, 'trace_id': episode_id}

    def close(self):
        self.store.close()
