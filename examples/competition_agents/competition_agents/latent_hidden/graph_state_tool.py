"""用已计算的模型表示查询固定超图；不将隐向量解码成检索文字。"""
from __future__ import annotations

import copy
import json
import time
from collections import defaultdict

from hypergraph_memory_method.source_context import normalized_name
from hypergraph_memory_method.relation_coverage import global_anchor, PRONOUNS


class GraphStateTool:
    """原型数值检索器；基础模型权重不变，表示相似度尚未经过校准。"""

    def __init__(self, knowledge, bundle, *, terminals=3, max_facts=8):
        self.knowledge, self.bundle = knowledge, bundle
        self.terminals, self.max_facts = terminals, max_facts
        self.keys = sorted(knowledge._graph_rows)
        self.positions = {key: i for i, key in enumerate(self.keys)}
        self.names = defaultdict(set)
        self.peers = defaultdict(list)
        for key in self.keys:
            fact = knowledge._graph_rows[key][3]
            for part in ('subject', 'object'):
                node = knowledge._graph_nodes[key][part]
                if node in knowledge._graph_named_nodes and global_anchor(node, fact[part]):
                    self.names[normalized_name(fact[part])].add(node)
            # 保留同一主体同一动作的地点、并列对象及否定，不扩大系词属性。
            words = [w for w in normalized_name(fact['relation']).split() if w not in {
                'not', 'never', 'in', 'at', 'on', 'with', 'to', 'from', 'her', 'his', 'the', 'a', 'an'}]
            if words and words[0] not in {'is', 'was', 'are', 'were', 'has', 'have', 'had'}:
                self.peers[(fact['document_id'], knowledge._graph_nodes[key]['subject'], words[0])].append(key)
        self.action_peers = {key: tuple(group) for group in self.peers.values() for key in group}
        for node, forms in knowledge._graph_owned_names.items():
            for form in forms:
                if node.startswith('entity:') and normalized_name(form).split() and normalized_name(form).split()[0] not in PRONOUNS:
                    self.names[form.strip()].add(node)
        for alias, canonical in getattr(knowledge, 'alias_index', {}).items():
            node = 'entity:' + normalized_name(canonical)
            if node in knowledge._graph_neighbors:
                self.names[normalized_name(alias)].add(node)
        for name, nodes in getattr(getattr(knowledge, 'relation_index', None), 'names', {}).items():
            self.names[name].update(nodes)
        self.scope, self.roots = None, set()
        self.records = {}
        self.events = []
        self.vectors = self._encode_index()
        self.center = self.vectors.mean(dim=0, keepdim=True)
        self.vectors = bundle.torch.nn.functional.normalize(self.vectors - self.center, dim=-1)

    def _encode_index(self):
        vectors = []
        torch, model, tokenizer = self.bundle.torch, self.bundle.model, self.bundle.tokenizer
        with torch.inference_mode():
            for key in self.keys:
                fact = self.knowledge._graph_rows[key][3]
                text = ' '.join([fact['subject'], 'NOT' if fact['polarity'] == 'negative' else '',
                                 fact['relation'], fact['object'],
                                 *[q['kind'] + ': ' + q['value'] for q in fact['qualifiers']]])
                rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                                          tokenize=False, add_generation_prompt=False)
                ids = tokenizer(rendered, return_tensors='pt', add_special_tokens=False)['input_ids'].to(self.bundle.device)
                states = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
                vectors.append(states.mean(dim=1).squeeze(0).float().detach())
        return torch.stack(vectors)

    def begin_question(self, question, scope):
        """只从外部新问题绑定显式实体；无显式实体时沿用会话焦点，不生成改写问题。"""
        if scope != self.scope:
            self.scope, self.roots, self.records = scope, set(), {}
        words = normalized_name(question).split()
        matches = []
        for start in range(len(words)):
            for end in range(start + 1, min(start + 12, len(words)) + 1):
                name = ' '.join(words[start:end])
                if name in self.names:
                    matches.append((start, end, self.names[name]))
        matches = [m for m in matches if not any(
            other[0] <= m[0] and other[1] >= m[1] and other[1]-other[0] > m[1]-m[0]
            for other in matches)]
        if matches:
            self.roots = set().union(*(m[2] for m in matches))
        relation_index = getattr(self.knowledge, 'relation_index', None)
        if relation_index is not None:
            self.roots = relation_index.anchors(question) or self.roots

    def _paths(self):
        library = self.knowledge
        if not self.roots:
            return {key: (key,) for key in self.keys}
        paths, visited, frontier = {}, set(self.roots), [(n, ()) for n in sorted(self.roots)]
        for _ in range(3):
            next_frontier = []
            for node, prefix in frontier:
                for key, *_rest in library._graph_neighbors.get(node, ()):
                    if key in prefix:
                        continue
                    path = (*prefix, key)
                    if key not in paths or (len(path), path) < (len(paths[key]), paths[key]):
                        paths[key] = path
                    for _, neighbor, _, _ in library._graph_links[key]:
                        if neighbor in library._graph_named_nodes and neighbor not in visited:
                            next_frontier.append((neighbor, path))
            frontier = []
            for node, path in sorted(next_frontier, key=lambda p: (len(p[1]), p[1], p[0])):
                if node not in visited:
                    visited.add(node)
                    frontier.append((node, path))
        return paths

    def __call__(self, query):
        started = time.perf_counter()
        torch = self.bundle.torch
        if not torch.is_tensor(query) or query.shape != (1, self.vectors.shape[1]):
            raise ValueError('检索只接受一个模型表示，不接受文字查询。')
        if not torch.isfinite(query).all():
            raise ValueError('数值查询包含无效值。')
        paths = self._paths()
        # 查询先恢复与事实池化向量相近的幅度，再消除公共方向。
        direction = torch.nn.functional.normalize(query.float(), dim=-1)
        scale = self.center.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        vector = torch.nn.functional.normalize(direction * scale - self.center, dim=-1)
        scores_tensor = (self.vectors @ vector.squeeze(0)).detach().cpu()
        scores = {key: float(scores_tensor[self.positions[key]]) - .12*(len(path)-1)
                  for key, path in paths.items()}
        ranked = sorted(paths, key=lambda key: (-scores[key], len(paths[key]), key))
        selected, taken = [], 0
        for key in ranked:
            if key in selected:
                continue
            bundle = list(paths[key])
            bundle.extend(peer for peer in self.action_peers.get(key, ()) if peer not in bundle)
            additions = [item for item in bundle if item not in selected]
            if selected and len(selected) + len(additions) > self.max_facts:
                continue
            selected.extend(additions)
            taken += 1
            if taken >= self.terminals or len(selected) >= self.max_facts:
                break
        index_ms = (time.perf_counter() - started) * 1000
        memory_started = time.perf_counter()
        reused, fresh, facts = [], [], []
        for key in selected:
            if key in self.records:
                reused.append(key)
                fact = self.records[key]
            else:
                fresh.append(key)
                fact = copy.deepcopy(self.knowledge._graph_rows[key][3])
                self.records[key] = fact
            row = {'id': 'f' + str(self.positions[key]), 'subject': fact['subject'],
                   'relation': fact['relation'], 'object': fact['object'], 'polarity': fact['polarity']}
            if fact['qualifiers']:
                row['qualifiers'] = fact['qualifiers']
            facts.append(row)
        memory_ms = (time.perf_counter() - memory_started) * 1000
        audit = {'query_kind': 'continuous_model_representation',
                 'query_bytes': query.numel() * query.element_size(),
                 'query_text': None, 'roots': sorted(self.roots),
                 'index_search_ms': index_ms, 'memory_and_fact_read_ms': memory_ms,
                 'source_ids': selected, 'fresh_source_ids': fresh, 'memory_source_ids': reused,
                 'ranked_candidates': [{'id': key, 'score': scores[key]} for key in ranked],
                 'returned_fact_bytes': len(json.dumps(facts, ensure_ascii=False).encode()),
                 'all_selected_from_memory': bool(selected) and not fresh,
                 'scope': 'index searched even for resident facts; no claim of memory-only retrieval'}
        self.events.append(audit)
        return {'facts': facts, 'audit': audit}
