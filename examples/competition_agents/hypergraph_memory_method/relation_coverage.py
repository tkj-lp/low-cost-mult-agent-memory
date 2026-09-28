"""来源关系目录、完整路径选择和记忆覆盖核验；不读取评测答案。"""
from collections import defaultdict
import re

from .source_context import normalized_name


def fact_facets(fact):
    relation = normalized_name(fact['relation'])
    value = normalized_name(fact['object'])
    text = relation + ' ' + value
    if any(q['kind'] == 'subject_owner' for q in fact.get('qualifiers', [])):
        text = normalized_name(fact['subject']) + ' ' + text
    found = set()
    patterns = {
        'creator': r'\b(writ\w*|wrote|author\w*|compos\w*|music by|lyrics by|book by|score by|creat\w*)\b',
        'performer': r'\b(singer|sang|sung|singing|vocals|vocalist|perform\w*|starred|starring|portray\w*)\b',
        'narrative': r'\b(story|stories|plot|protagonist|tells|centres|centers|depicts|recounts)\b',
        'appearance': r'\b(appearance|features|coat|fur|tail|neck|limbs|back|slender|small|tall|height|length|weigh\w*|colou?r\w*)\b',
        'color': r'\b(colou?r\w*|white|black|brown|yellow\w*|red|blue|green|rufous)\b',
        'location': r'\b(locat\w*|liv\w* in|resid\w*|found in|occurs in|originat\w*|from|born in|situat\w*)\b',
        'time': r'\b(date|time|born|released|premier\w*|founded|formed|appointed|since|until|year)\b',
        'quantity': r'\b(number|count|total|population|quantity)\b',
        'owner': r'\b(own\w*|belong\w*|possess\w*)\b',
        'family': r'\b(mother|father|sister|brother|parent|child|siblings|family|spouse|wife|husband)\b',
        'companions': r'\b(alone|together|lived with|share\w*|companions)\b',
        'cause': r'\b(cause\w*|because|reason|due to|result\w* from)\b',
        'speech': r'\b(said|says|statement|told|announce\w*)\b',
    }
    for facet, pattern in patterns.items():
        if re.search(pattern, text):
            found.add(facet)
    if re.search(r'^(is a|is an|was a|was an|type of|kind of)|\b(identity|known as|occupation|full name|real name)\b', relation):
        found.add('identity')
    if relation in {'is', 'was', 'are', 'were'} and not re.search(r'\b(grazer|browser|herbivore|nocturnal|diurnal)\b', value):
        found.add('identity')
    if any(q['kind'].casefold() in {'time', 'date', 'start', 'end', 'since', 'until', 'year'} for q in fact.get('qualifiers', [])):
        found.add('time')
    return found or {'action'}


def requested_facet(question):
    text = normalized_name(question)
    rules = [
        ('appearance', r'look like|appearance|physical|how does .+ look'),
        ('color', r'\b(colou?r|hue)\b'),
        ('time', r'\b(when|what year|what date|what time|which year)\b'),
        ('location', r'\b(where|which place|what place)\b'),
        ('creator', r'\b(wrote|written|writer|author|composed|composer|lyrics|created|creator)\b'),
        ('narrative', r'\b(story|plot|protagonist|convey|depict)\b'),
        ('performer', r'\b(singer|sang|sings|vocals|vocalist|performer|portrays|played the role)\b'),
        ('quantity', r'\b(how many|how much|what number)\b'),
        ('owner', r'\b(owns|owner|belong)\b'),
        ('family', r'\b(mother|father|sister|brother|parent|siblings|family|wife|husband)\b'),
        ('companions', r'\b(alone|with anyone|with whom|share a home)\b'),
        ('cause', r'\b(why|reason|cause)\b'),
        ('speech', r'\b(say|said|statement)\b'),
        ('identity', r'\b(what is|who is|what kind|what type|who was)\b'),
    ]
    return next((facet for facet, pattern in rules if re.search(pattern, text)), None)


class RelationCoverageIndex:
    """初始化建立有限知识库的关系目录；在线只读目录，不以相似度冒充完整性。"""

    def __init__(self, knowledge):
        self.knowledge = knowledge
        self.names = defaultdict(set)
        self.facets, self.signatures, self.tokens = {}, {}, {}
        for key, row in knowledge._graph_rows.items():
            fact = row[3]
            self.facets[key] = fact_facets(fact)
            self.tokens[key] = set(normalized_name(fact['relation'] + ' ' + fact['object']).split())
            if any(q['kind'] == 'subject_owner' for q in fact['qualifiers']):
                self.tokens[key].update(normalized_name(fact['subject']).split())
            subject = knowledge._graph_nodes[key]['subject']
            obj = knowledge._graph_nodes[key]['object']
            self.signatures[key] = (subject, normalized_name(fact['relation']), obj, fact['polarity'],
                                    tuple(sorted((q['kind'], normalized_name(q['value'])) for q in fact['qualifiers'])))
            for role in ('subject', 'object'):
                node = knowledge._graph_nodes[key][role]
                if node in knowledge._graph_named_nodes and not node.startswith('entity:owned:'):
                    self.names[normalized_name(fact[role])].add(node)
                    base = re.sub(r'\s*\([^)]*\)\s*$', '', fact[role]).strip()
                    self.names[normalized_name(base)].add(node)
                    self.names[normalized_name(node.removeprefix('entity:'))].add(node)
            for role, node, value, via in knowledge._graph_links[key]:
                if via == 'explicit_owner':
                    self.names[normalized_name(value)].add(node)
        self.root_paths = {node: self._walk(node) for node in knowledge._graph_named_nodes}
        for node, forms in knowledge._graph_owned_names.items():
            for form in forms:
                self.names[normalized_name(form)].add(node)
        self.unbound = defaultdict(list)
        for key, row in knowledge._graph_rows.items():
            fact = row[3]
            node = knowledge._graph_nodes[key]['subject']
            if node in knowledge._graph_named_nodes or any(q['kind'] == 'subject_owner' for q in fact['qualifiers']):
                continue
            topic = row[1].split('[SEP]', 1)[0].strip()
            for root in self.names.get(normalized_name(topic), ()):
                self.unbound[root].append(key)

    def _walk(self, root):
        library = self.knowledge
        paths, seen, frontier = {}, {root}, [(root, ())]
        for _ in range(3):
            following = []
            for node, prefix in frontier:
                for key, *_ in library._graph_neighbors.get(node, ()):
                    if key in prefix:
                        continue
                    path = (*prefix, key)
                    if key not in paths or (len(path), path) < (len(paths[key]), paths[key]):
                        paths[key] = path
                    for _, other, _, _ in library._graph_links[key]:
                        if other in library._graph_named_nodes and other not in seen:
                            following.append((other, path))
            frontier = []
            for node, path in sorted(following, key=lambda p: (len(p[1]), p[1], p[0])):
                if node not in seen:
                    seen.add(node)
                    frontier.append((node, path))
        return paths

    def anchors(self, question):
        words = normalized_name(question).split()
        hits = [(i, j, self.names[' '.join(words[i:j])]) for i in range(len(words))
                for j in range(i + 1, min(len(words), i + 12) + 1) if ' '.join(words[i:j]) in self.names]
        hits = [h for h in hits if not any(g[0] <= h[0] and g[1] >= h[1] and g[1]-g[0] > h[1]-h[0] for g in hits)]
        roots = set().union(*(h[2] for h in hits)) if hits else set()
        # 同名作品明确提及类型时，使用对应节点，不能把电影和音乐剧当作同一实体。
        kinds = set(words) & {'film', 'musical', 'album', 'song', 'book', 'novel', 'play'}
        typed = {n for n in roots if any(n.endswith('(' + k + ')') for k in kinds)}
        return typed or roots

    def plan(self, question, roots=None):
        roots = set(roots) if roots is not None else self.anchors(question)
        facet = requested_facet(question)
        paths = {}
        for node in sorted(roots):
            for key, path in self.root_paths.get(node, {}).items():
                if key not in paths or (len(path), path) < (len(paths[key]), paths[key]):
                    paths[key] = path
        candidates = [key for key in paths if facet in self.facets[key]] if facet else []
        # 明确的关系关键词须被覆盖；联合请求保留每一个关系，不能只保留最高分项。
        words = set(normalized_name(question).split())
        specifics = words & {'music', 'lyrics', 'book', 'singer', 'vocals', 'story', 'plot', 'date', 'born'}
        matching = [key for key in candidates if self.tokens[key] & specifics]
        if matching:
            candidates = matching
        if candidates:
            depth = min(len(paths[k]) for k in candidates)
            candidates = [k for k in candidates if len(paths[k]) == depth]
        chosen, seen = [], set()
        for key in sorted(candidates):
            signature = self.signatures[key]
            if signature not in seen:
                chosen.append(key)
                seen.add(signature)
        required = list(dict.fromkeys(k for key in chosen for k in paths[key]))
        unbound = sorted({k for root in roots for k in self.unbound.get(root, ())
                          if facet in self.facets[k] and (not specifics or self.tokens[k] & specifics)})
        return {'facet': facet, 'roots': sorted(roots), 'paths': paths, 'terminals': chosen,
                'required': required, 'verifiable': bool(required), 'unbound_source_ids': unbound,
                'basis': 'source_relation_inventory; not a proof of extraction completeness'}

    def certificate(self, plan, keys):
        keys = set(keys)
        signatures = {self.signatures[k] for k in keys if k in self.signatures}
        missing = [k for k in plan['required'] if self.signatures[k] not in signatures]
        return {'sufficient': bool(plan['verifiable'] and not missing and not plan.get('unbound_source_ids')), 'facet': plan['facet'],
                'required_count': len(plan['required']), 'missing_source_ids': missing,
                'unbound_source_ids': plan.get('unbound_source_ids', []),
                'basis': plan['basis']}

    def select(self, plan, scores, max_facts=24):
        """关系已明确时选择完整的去重来源束；软预算不得切断一条关系链。"""
        selected = []
        for terminal in sorted(plan['terminals'], key=lambda k: (-scores.get(k, -1.), k)):
            additions = [k for k in plan['paths'][terminal] if k not in selected]
            if selected and len(selected) + len(additions) > max_facts:
                continue
            selected.extend(additions)
        return selected
