"""协议方案专用的会话证据记忆与来源编号规范。"""

from __future__ import annotations

import copy

import re

import sqlite3

import numpy as np

from hypergraph_memory_method.prepared_knowledge import digest, entity_key, incidence_key, PreparedKnowledge

from hypergraph_memory_method.text import indexed_tokens

def source_label(value):
    """只消除来源编号后附加的说明；仍须校验编号存在，不采用说明中的事实。"""
    if isinstance(value, str) and re.fullmatch(r'\[s\d+\]', value.strip()):
        return value.strip()[1:-1]
    return re.sub(r'^(s\d+)(?:\s+|:\s*).*$', r'\1', value) if isinstance(value, str) else value

class SessionEvidenceMemory:
    """文本语义记忆与实体关系记忆；只存来源事实，答案另作未验证对话记录。"""
    def __init__(self, mode, embed, knowledge=None):
        self.mode, self.embed, self.knowledge = mode, embed, knowledge
        self.document_vectors = {} if knowledge is None else {doc['id']: vector for doc, vector in zip(knowledge.documents, knowledge.text_vectors)}
        self.records, self.vectors, self.episodes = {}, {}, []
        self.neighbors, self.record_nodes = {}, {}
        self.scopes = {}
        self.db = sqlite3.connect(':memory:')
        self.db.executescript('''CREATE VIRTUAL TABLE facts USING fts5(id UNINDEXED, terms);
            CREATE TABLE incidence(entity TEXT, fact_id TEXT, role TEXT, UNIQUE(entity,fact_id,role));
            CREATE INDEX entity_lookup ON incidence(entity);
            CREATE TABLE task_facts(task TEXT, fact_id TEXT, UNIQUE(task,fact_id));
            CREATE TABLE session_facts(scope TEXT, fact_id TEXT, UNIQUE(scope,fact_id));''')

    def recall(self, query, limit=6, *, scope=None, roots=None):
        eligible = self.scopes.get(scope, set()) if scope is not None else set(self.records)
        if not eligible:
            return []
        if self.mode == 'text':
            vector = np.asarray(self.embed(query).vector, dtype=np.float32)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            ranked = sorted(((float(vector @ self.vectors[k]), k) for k in eligible), reverse=True)
            ids = [key for score, key in ranked[:limit] if score >= .35]
        else:
            terms = sorted(set(indexed_tokens(query)) - {'what', 'which', 'where', 'who', 'when', 'how'})
            index = getattr(self.knowledge, 'relation_index', None)
            roots = roots or (index.anchors(query) if index is not None else set())
            if not terms and not roots:
                return []
            match = ' OR '.join('"' + t + '"' for t in terms)
            scope_filter = ' AND id IN (SELECT fact_id FROM session_facts WHERE scope=?)' if scope is not None else ''
            params = [match, *([scope] if scope is not None else [])]
            ids = ([r[0] for r in self.db.execute('SELECT id FROM facts WHERE facts MATCH ?' + scope_filter + ' ORDER BY bm25(facts)', params)]
                   if terms and not roots else [])
            chains = [(key,) for key in ids]
            candidates = {key: (0, self.records[key]['title'], self.records[key]['text'], self.records[key]['fact_fields']) for key in ids}
            frontier = [(key, (key,)) for key in ids]
            if roots:
                frontier = []
                for root in sorted(roots):
                    for key in sorted(self.neighbors.get(root, ())):
                        if key in eligible:
                            frontier.append((key, (key,)))
                # 精确实体入口不受关键词命中顺序限制。
                ids, chains, candidates = [], [], {}
            seen = set()
            scheduled = {key for key, _ in frontier}
            for key, path in frontier:
                if key in seen:
                    continue
                seen.add(key)
                hit = self.records[key]
                candidates[key] = (0, hit['title'], hit['text'], hit['fact_fields'])
                chains.append(path)
                if len(path) >= 3:
                    continue
                nodes = self.record_nodes.get(key, set())
                for node in sorted(nodes):
                    if not node.startswith('entity:'):
                        continue
                    if hasattr(self.knowledge, '_graph_named_nodes') and node not in self.knowledge._graph_named_nodes:
                        continue
                    for neighbor in sorted(self.neighbors.get(node, ())):
                        if neighbor == key or neighbor not in eligible:
                            continue
                        hit = self.records[neighbor]
                        candidates[neighbor] = (0, hit['title'], hit['text'], hit['fact_fields'])
                        if neighbor not in scheduled:
                            scheduled.add(neighbor)
                            frontier.append((neighbor, (*path, neighbor)))
            ids = []
            for chain in sorted(chains, key=lambda c: (-PreparedKnowledge._chain_score(c, candidates, set(terms)), len(c), c)):
                new = [key for key in chain if key not in ids]
                ids.extend(new)
        return [dict(copy.deepcopy(self.records[k]), from_memory=True,
                     memory_turns=[e['turn'] for e in self.episodes if e['scope'] == scope
                                   and k in e.get('observed_evidence_ids', e['evidence_ids'])]) for k in ids]

    def store_evidence(self, question, evidence, history_scope):
        """保存实际取得的来源事实，与是否用于本题答案、生成是否成功无关。"""
        new_count = 0
        for hit in evidence:
            key = hit['context_id']
            if key not in self.scopes.get(history_scope, set()):
                new_count += 1
            self.scopes.setdefault(history_scope, set()).add(key)
            self.db.execute('INSERT OR IGNORE INTO session_facts VALUES (?,?)', (history_scope, key))
            self.db.execute('INSERT OR IGNORE INTO task_facts VALUES (?,?)', (digest(question), key))
            if key in self.records:
                continue
            # 召回标记属于一次读取，不属于来源事实；调用方也不能改写已存证据。
            self.records[key] = copy.deepcopy({k: v for k, v in hit.items()
                                              if k not in ('from_memory', 'memory_turns')})
            if self.mode == 'text':
                # 同一离线文档直接使用完整分窗向量，避免再次编码且被静默截断。
                vector = self.document_vectors.get(key)
                if vector is None:
                    vector = np.asarray(self.embed(hit['title'] + ': ' + hit['text']).vector, dtype=np.float32)
                self.vectors[key] = vector / max(float(np.linalg.norm(vector)), 1e-12)
            else:
                self.db.execute('INSERT INTO facts VALUES (?,?)', (key, ' '.join(indexed_tokens(hit['title'] + ' ' + hit['text']))))
                fact = hit['fact_fields']
                nodes = {incidence_key(fact[role], role, fact, getattr(self.knowledge, 'graph_aliases', {}))
                         for role in ('subject', 'object') if fact.get(role)}
                nodes.update(link[1] for link in getattr(self.knowledge, '_graph_links', {}).get(fact.get('id', key), ()))
                self.record_nodes[key] = nodes
                for node in nodes:
                    self.neighbors.setdefault(node, set()).add(key)
                for role in ('subject', 'object', 'relation'):
                    if fact[role]:
                        self.db.execute('INSERT OR IGNORE INTO incidence VALUES (?,?,?)', (incidence_key(fact[role], role, {**fact, 'id': key}, getattr(self.knowledge, 'graph_aliases', {})), key, role))
                for qualifier in fact['qualifiers']:
                    self.db.execute('INSERT OR IGNORE INTO incidence VALUES (?,?,?)', (incidence_key(qualifier['value'], qualifier['kind'], {**fact, 'id': key}), key, qualifier['kind']))
        self.db.commit()
        return new_count

    def remember(self, question, answer, evidence, history_scope, *, resolved_question=None,
                 observed_evidence=None):
        # 对话保留本轮使用的支持关系；事实库额外保留已见但本题未采用的来源。
        observed = {hit['context_id']: hit for hit in (observed_evidence or [])}
        observed.update({hit['context_id']: hit for hit in evidence})
        new_count = self.store_evidence(question, observed.values(), history_scope)
        turn = 1 + sum(e['scope'] == history_scope for e in self.episodes)
        self.episodes.append({'question': question, 'resolved_question': resolved_question or question,
                              'answer': answer, 'scope': history_scope, 'verified': False,
                              'turn': turn, 'evidence_ids': [hit['context_id'] for hit in evidence],
                              'observed_evidence_ids': list(observed)})
        return new_count

    def history(self, scope):
        return [{k: e[k] for k in ('turn', 'question', 'resolved_question', 'answer', 'verified')}
                for e in self.episodes if e['scope'] == scope][-2:]

    def context(self, question, scope, *, roots=None):
        history = self.history(scope)
        # 协议按当前问题找关系；历史单独提供给规划角色承接实体，
        # 不把上一轮长答案混进排序，挤掉本轮已存但尚未用过的事实。
        query = (question if self.mode != 'text' else
                 question + '\n' + '\n'.join(e['resolved_question'] + ' ' + e['answer'] for e in history))
        hits = self.recall(query, scope=scope, roots=roots)
        # 最近两轮实际使用的来源帮助承接指代；其余已知事实按本题需要检索。
        selected = {h['context_id']: h for h in hits}
        if not roots and not hits:
            # 指代尚未绑定时提供前轮来源；已绑定时以完整多跳召回为准，
            # 不把历史观察过的整库无条件反复追加到每个问题。
            for episode in [e for e in self.episodes if e['scope'] == scope][-2:]:
                for key in episode.get('observed_evidence_ids', episode['evidence_ids']):
                    selected.setdefault(key, dict(copy.deepcopy(self.records[key]), from_memory=True,
                                                  memory_turns=[episode['turn']]))
        return history, list(selected.values()), query

    def close(self):
        self.db.close()
