"""实体范围内的语义排序；保留连接路径，不以整篇共现补满候选。"""
from __future__ import annotations

import copy
from collections import defaultdict

import numpy as np

from .source_context import normalized_name


class FocusedGraphSearch:
    def __init__(self, knowledge, embed):
        self.knowledge, self.embed = knowledge, embed
        self.keys = sorted(knowledge._graph_rows)
        self.positions = {key: i for i, key in enumerate(self.keys)}
        self.names = defaultdict(set)
        self.action_peers = {}
        action_groups = defaultdict(list)
        for key in self.keys:
            fact = knowledge._graph_rows[key][3]
            # 同一动作的地点、对象、肯否定一起保留，不能靠一个相似度分差拆散。
            # 系词仍按单条属性处理，避免颜色查询展开该主体的全部描述。
            relation_words = normalized_name(fact['relation']).split()
            content_words = [word for word in relation_words if word not in {
                'in', 'at', 'on', 'to', 'from', 'with', 'by', 'of', 'for', 'as',
                'into', 'during', 'before', 'after', 'not', 'never', 'no', 'the', 'a', 'an'}]
            if content_words and content_words not in ([v] for v in ('is', 'was', 'are', 'were', 'has', 'have', 'had')):
                signature = (fact['document_id'], knowledge._graph_nodes[key]['subject'], tuple(content_words))
                action_groups[signature].append(key)
            for role in ('subject', 'object'):
                node = knowledge._graph_nodes[key][role]
                if node in knowledge._graph_named_nodes:
                    self.names[normalized_name(fact[role])].add(node)
        self.action_peers = {key: tuple(group) for group in action_groups.values() for key in group}
        for node, forms in knowledge._graph_owned_names.items():
            for form in forms:
                self.names[form.strip()].add(node)
        for alias, canonical in getattr(knowledge, 'alias_index', {}).items():
            node = 'entity:' + normalized_name(canonical)
            if node in knowledge._graph_neighbors:
                self.names[normalized_name(alias)].add(node)
        vectors = []
        providers = set()
        for key in self.keys:
            fact = knowledge._graph_rows[key][3]
            # 主体通过实体节点定位；排序向量重点表达关系、值和限定。
            text = ' '.join([fact['relation'], fact['object'],
                             *[q['kind'] + ': ' + q['value'] for q in fact['qualifiers']]])
            result = embed(text)
            providers.add((result.provider, result.model))
            vectors.append(result.vector)
        if len(providers) != 1:
            raise ValueError('图事实排序向量必须使用同一个编码器。')
        self.provider, self.model = providers.pop()
        self.vectors = np.asarray(vectors, dtype=np.float32)
        if self.vectors.ndim != 2 or not np.isfinite(self.vectors).all():
            raise ValueError('图事实排序向量无效。')
        self.vectors /= np.maximum(np.linalg.norm(self.vectors, axis=1, keepdims=True), 1e-12)

    def anchors(self, query):
        if getattr(self.knowledge, 'relation_index', None):
            roots = self.knowledge.relation_index.anchors(query)
            if roots:
                return roots, query
        words = normalized_name(query).split()
        matches = []
        for start in range(len(words)):
            for end in range(start + 1, min(start + 12, len(words)) + 1):
                name = ' '.join(words[start:end])
                if name in self.names:
                    matches.append((start, end, self.names[name]))
        # 群体全名优先于其中的所属者；避免问姐妹颜色却锚定到小猫本身。
        matches = [m for m in matches if not any(
            other[0] <= m[0] and other[1] >= m[1] and other[1] - other[0] > m[1] - m[0]
            for other in matches)]
        nodes = set().union(*(m[2] for m in matches)) if matches else set()
        covered = {i for start, end, _ in matches for i in range(start, end)}
        intent = ' '.join(word for i, word in enumerate(words) if i not in covered) or query
        return nodes, intent

    def search(self, query, max_facts=8, *, constraints=None, roots=None):
        if max_facts <= 0:
            return []
        library = self.knowledge
        if roots is None:
            roots, intent = self.anchors(query)
        else:
            roots, intent = set(roots), query
            if not roots <= library._graph_named_nodes:
                raise ValueError('指代绑定包含未知实体节点。')
        if not roots:
            return library.search_graph(query, max_facts, constraints=constraints)
        encoded = self.embed(intent)
        if (encoded.provider, encoded.model) != (self.provider, self.model):
            raise ValueError('查询和图事实排序向量不属于同一编码器。')
        vector = np.asarray(encoded.vector, dtype=np.float32)
        if vector.shape != (self.vectors.shape[1],) or not np.isfinite(vector).all():
            raise ValueError('图查询向量无效。')
        vector /= max(float(np.linalg.norm(vector)), 1e-12)

        def eligible(key):
            values = library._graph_conditions[key]
            return all(' '.join(value.casefold().split()) in values.get(kind.casefold(), ())
                       for kind, value in (constraints or {}).items())

        # 只遍历实体关联；每条候选保留从问题实体走到它的路径。
        paths, frontier = {}, [(node, ()) for node in sorted(roots)]
        visited = set(roots)
        for depth in range(3):
            following = []
            for node, prefix in frontier:
                for row in library._graph_neighbors.get(node, ()):
                    key = row[0]
                    if key in prefix or not eligible(key):
                        continue
                    path = (*prefix, key)
                    if key not in paths or (len(path), path) < (len(paths[key]), paths[key]):
                        paths[key] = path
                    if depth == 2:
                        continue
                    for _, neighbor, _, _ in library._graph_links[key]:
                        if neighbor not in library._graph_named_nodes or neighbor in visited:
                            continue
                        following.append((neighbor, path))
            following.sort(key=lambda entry: (len(entry[1]), entry[1], entry[0]))
            frontier = []
            for node, path in following:
                if node not in visited:
                    frontier.append((node, path))
                    visited.add(node)
            if not frontier:
                break
        if not paths:
            return []
        # 等相似度的邻居属性不能挤入直接主体的候选带；跨跳证据需要更强的关系匹配。
        scores = {key: float(self.vectors[self.positions[key]] @ vector) - .12 * (len(path) - 1)
                  for key, path in paths.items()}
        ranked = sorted(paths, key=lambda key: (-scores[key], len(paths[key]), key))
        best = scores[ranked[0]]
        selected, used_paths = [], {}
        terminals = 0
        for key in ranked:
            # 有明显相关性落差时停止；不为凑足预算继续追加无关事实。
            if terminals and (scores[key] < best - .10 or terminals >= 3):
                break
            bundle = list(paths[key])
            for peer in self.action_peers.get(key, ()):
                if peer not in bundle and eligible(peer):
                    bundle.append(peer)
            additions = [item for item in bundle if item not in selected]
            if len(selected) + len(additions) > max_facts:
                # 首组允许超出软预算，不能截断关系链或并列值，也不能重新排序丢失它。
                if selected:
                    continue
            for item in additions:
                selected.append(item)
                used_paths[item] = list(paths[key])
            terminals += 1
        plan = library.relation_index.plan(query, roots)
        if plan['verifiable']:
            selected = library.relation_index.select(plan, scores, max_facts=max(24, max_facts))
            used_paths = {key: plan['paths'].get(key, (key,)) for key in selected}
        return [library._hit(key, library._graph_rows[key][1], library._graph_rows[key][2],
                             scores.get(key, scores[ranked[0]]), 'focused_hypergraph',
                             fact_fields=copy.deepcopy(library._graph_rows[key][3]),
                             bridge_source_context_ids=[item for item in used_paths[key] if item != key],
                             matched_hyperedges=[key]) for key in selected]
