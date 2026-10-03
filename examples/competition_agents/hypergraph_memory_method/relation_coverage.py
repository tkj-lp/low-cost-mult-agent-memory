"""来源关系目录、完整路径选择和记忆覆盖核验；不读取评测答案。"""
from collections import defaultdict
import re

from .source_context import normalized_name
from .text import indexed_tokens, question_content_tokens

PRONOUNS = {'it', 'its', 'he', 'his', 'him', 'she', 'her', 'they', 'them', 'their',
            'this', 'that', 'these', 'those', 'we', 'our', 'i', 'you'}


def global_anchor(node, name):
    return node.startswith('entity:') and not node.startswith('entity:owned:') and normalized_name(name) not in PRONOUNS


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
        'time': r'\b(date|time|born|released|premier\w*|debut|began|started|founded|formed|appointed|since|until|year)\b',
        'debut': r'\b(debut|first album|first work|first book)\b',
        'reception': r'\b(received|reviews?|critics?|acclaim\w*|reception|box office|sales|won|awarded)\b',
        'publisher': r'\b(released|published|label|publisher|record company)\b',
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
    if any(q['kind'].casefold() in {'count', 'number', 'quantity'} for q in fact.get('qualifiers', [])):
        found.add('quantity')
    if re.search(r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
                 r'thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|'
                 r'thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|\d+)\b',
                 normalized_name(fact['subject']) + ' ' + value):
        found.add('quantity')
    return found or {'action'}


def requested_facet(question):
    text = normalized_name(question)
    rules = [
        ('appearance', r'look like|appearance|physical|how does .+ look'),
        ('color', r'\b(colou?r|hue)\b'),
        ('debut', r'\b(which|what)\b.*\b(debut|first work|first album|first book)\b'),
        ('reception', r'\b(response|reception|reviews?|received|successful)\b'),
        ('publisher', r'\b(who|which company|which label)\b.*\b(released|published)\b'),
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
        ('identity', r'\b(what is|who is|what kind|what type|who was|tell me about|what do you know about)\b'),
    ]
    return next((facet for facet, pattern in rules if re.search(pattern, text)), None)


class RelationCoverageIndex:
    """初始化建立有限知识库的关系目录；在线只读目录，不以相似度冒充完整性。"""

    def __init__(self, knowledge):
        self.knowledge = knowledge
        self.names = defaultdict(set)
        self.facets, self.signatures, self.tokens = {}, {}, {}
        self.quantity_tokens = {}
        self.relation_tokens = {}
        for key, row in knowledge._graph_rows.items():
            fact = row[3]
            self.facets[key] = fact_facets(fact)
            self.tokens[key] = set(normalized_name(fact['relation'] + ' ' + fact['object']).split())
            self.quantity_tokens[key] = set(indexed_tokens(fact['subject'] + ' ' + fact['relation'] + ' ' + fact['object']))
            self.relation_tokens[key] = question_content_tokens(fact['relation'])
            if any(q['kind'] == 'subject_owner' for q in fact['qualifiers']):
                self.tokens[key].update(normalized_name(fact['subject']).split())
            subject = knowledge._graph_nodes[key]['subject']
            obj = knowledge._graph_nodes[key]['object']
            self.signatures[key] = (subject, normalized_name(fact['relation']), obj, fact['polarity'],
                                    tuple(sorted((q['kind'], normalized_name(q['value'])) for q in fact['qualifiers'])))
            for role in ('subject', 'object'):
                node = knowledge._graph_nodes[key][role]
                if node in knowledge._graph_named_nodes and global_anchor(node, fact[role]):
                    self.names[normalized_name(fact[role])].add(node)
                    base = re.sub(r'\s*\([^)]*\)\s*$', '', fact[role]).strip()
                    self.names[normalized_name(base)].add(node)
                    self.names[normalized_name(node.removeprefix('entity:'))].add(node)
            for role, node, value, via in knowledge._graph_links[key]:
                if via == 'explicit_owner' and global_anchor(node, value):
                    self.names[normalized_name(value)].add(node)
        self.root_paths = {node: self._walk(node) for node in knowledge._graph_named_nodes}
        for node, forms in knowledge._graph_owned_names.items():
            for form in forms:
                if node.startswith('entity:') and normalized_name(form).split() and normalized_name(form).split()[0] not in PRONOUNS:
                    self.names[normalized_name(form)].add(node)
        self.document_roots = {doc['id']: self.anchors(doc['title'].split('[SEP]', 1)[0])
                               for doc in getattr(knowledge, 'documents', [])}
        self.unbound = defaultdict(list)
        for key, row in knowledge._graph_rows.items():
            fact = row[3]
            node = knowledge._graph_nodes[key]['subject']
            if (node in knowledge._graph_named_nodes and global_anchor(node, fact['subject'])) or any(q['kind'] == 'subject_owner' for q in fact['qualifiers']):
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
        if facet == 'identity':
            # 开放式人物或作品介绍保留入口实体的直接关系；不能用同文档中别人的职业替代。
            candidates = [key for key, path in paths.items() if len(path) == 1]
        # 明确的关系关键词须被覆盖；联合请求保留每一个关系，不能只保留最高分项。
        words = set(normalized_name(question).split())
        specifics = words & {'music', 'lyrics', 'book', 'singer', 'vocals', 'story', 'plot', 'date', 'born'}
        quantity_terms = set()
        if facet == 'quantity':
            # 数量落在主语或宾语都可检索；用本题数的对象收敛，
            # 不把同主题的销量、年份、奖项等所有数字都当作所求数量。
            match = re.search(r'\bhow (?:many|much)\s+(.+?)(?=\s+(?:is|are|was|were|do|does|did|'
                              r'has|have|had|can|could|will|would|from|in|of)\b|$)', normalized_name(question))
            if match:
                quantity_terms = set(indexed_tokens(match.group(1)))
        unbound = sorted({k for root in roots for k in self.unbound.get(root, ())
                          if facet in self.facets[k] and (not specifics or self.tokens[k] & specifics)})
        if quantity_terms:
            matching_quantity = {k for k in [*candidates, *unbound]
                                 if self.quantity_tokens[k] & quantity_terms}
            if matching_quantity:
                # 同一对象可能有发行数、获奖数等不同动作，优先本题明确问到的动作。
                requested_relation = question_content_tokens(question) - quantity_terms - {'many', 'much'}
                overlaps = {k: len(self.relation_tokens[k] & requested_relation) for k in matching_quantity}
                best = max(overlaps.values())
                if best:
                    matching_quantity = {k for k in matching_quantity if overlaps[k] == best}
                candidates = [k for k in candidates if k in matching_quantity]
                unbound = [k for k in unbound if k in matching_quantity]
        matching = [key for key in candidates if self.tokens[key] & specifics]
        if matching:
            candidates = matching
        if facet == 'time':
            query_text = normalized_name(question)
            event_pattern = (r'\b(career|began|start\w*|form\w*|join\w*|work\w*|record\w*|debut)\b'
                             if re.search(r'\bcareer\b', query_text) else
                             r'\b(releas\w*|premier\w*|open\w*|tryout\w*|preview\w*|debut)\b'
                             if re.search(r'\b(releas\w*|premier\w*|open\w*)\b', query_text) else None)
            if event_pattern:
                event_candidates = [k for k in candidates if re.search(event_pattern, ' '.join(self.tokens[k]))]
                if event_candidates:
                    candidates = event_candidates
                unbound = [k for k in unbound if re.search(event_pattern, ' '.join(self.tokens[k]))]
            def dated(key):
                fact = self.knowledge._graph_rows[key][3]
                return (any(q['kind'].casefold() in {'time', 'date', 'start', 'end', 'since', 'until', 'year'}
                            for q in fact['qualifiers']) or
                        bool(re.search(r'\b\d{4}\b', fact['relation'] + ' ' + fact['object'])))
            if any(dated(k) for k in candidates):
                candidates = [k for k in candidates if dated(k)]
            unbound = [k for k in unbound if dated(k)]
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
        # 来源关联是检索通路，不是实体等价：返回同文档的未消解关系供下游判断，
        # 同时保留未消解标记，不能一边报告缺关系一边永远不返回这些来源。
        for key in unbound:
            paths.setdefault(key, (key,))
            if key not in chosen:
                chosen.append(key)
            if key not in required:
                required.append(key)
        return {'facet': facet, 'roots': sorted(roots), 'paths': paths, 'terminals': chosen,
                'required': required, 'verifiable': bool(required), 'unbound_source_ids': unbound,
                'source_association_ids': unbound,
                'basis': 'source_relation_inventory; not a proof of extraction completeness'}

    def certificate(self, plan, keys):
        keys = set(keys)
        signatures = {self.signatures[k] for k in keys if k in self.signatures}
        missing = [k for k in plan['required'] if self.signatures[k] not in signatures]
        return {'sufficient': bool(plan['verifiable'] and not missing and not plan.get('unbound_source_ids')),
                'relations_present': bool(plan['verifiable'] and not missing), 'facet': plan['facet'],
                'required_count': len(plan['required']), 'missing_source_ids': missing,
                'unbound_source_ids': plan.get('unbound_source_ids', []),
                'basis': plan['basis']}

    def select(self, plan, scores, max_facts=24):
        """目录补充完整关系束；候选预算不能删除已经定位的并列关系。"""
        selected = []
        for terminal in sorted(plan['terminals'], key=lambda k: (-scores.get(k, -1.), k)):
            additions = [k for k in plan['paths'][terminal] if k not in selected]
            selected.extend(additions)
        return selected
