"""同源离线知识库；在线只读，不建图、不抽取、不回退到任务上下文。"""
from __future__ import annotations

import hashlib
import copy
import json
import os
import re
import sqlite3
import math
from collections import Counter, defaultdict
from contextlib import closing
from functools import lru_cache
from pathlib import Path

from .text import indexed_tokens, question_content_tokens
from .source_context import derive_source_aliases, normalized_name, source_alias_index

SCHEMA = 'prepared-knowledge-v3'


@lru_cache(maxsize=1)
def _english_stemmer():
    import snowballstemmer
    return snowballstemmer.stemmer('english')


@lru_cache(maxsize=32768)
def _search_stem(term):
    return _english_stemmer().stemWord(term) if term.isascii() and term.isalpha() else term


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def public_documents(samples):
    """白名单导入原始来源；不读取问题、答案或支撑事实标签。"""
    documents = {}
    for sample in samples:
        for source in sample.get('context', []):
            title, sentences = source.get('title'), source.get('sentences')
            if not isinstance(title, str) or not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
                raise ValueError('来源必须包含标题与原始句子。')
            if not sentences or not any(s.strip() for s in sentences):
                raise ValueError('空来源不能建库。')
            content = {'title': title, 'sentences': sentences}
            key = digest(content)
            documents[key] = {'id': key, **content}
    if not documents:
        raise ValueError('知识库没有原始来源。')
    return sorted(documents.values(), key=lambda d: d['id'])


def validate_extraction(document, result):
    """检查结构、来源定位与覆盖；不将这些检查冒充语义正确性证明。"""
    facts = result.get('facts') if isinstance(result, dict) else None
    if not isinstance(facts, list) or not facts:
        raise ValueError('抽取结果缺少事实。')
    covered = set()
    validated = []
    source = ' '.join([document['title'], *document['sentences']]).casefold()
    for item in facts:
        if not isinstance(item, dict):
            raise ValueError('事实格式错误。')
        for key in ('subject', 'relation'):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError('实体、关系和对象不能为空。')
        if not isinstance(item.get('object', ''), str):
            raise ValueError('对象必须为字符串；不及物事件允许空对象。')
        refs = item.get('sentences')
        if not isinstance(refs, list) or not refs or any(type(i) is not int or not 0 <= i < len(document['sentences']) for i in refs):
            raise ValueError('事实引用超出来源范围。')
        qualifiers = item.get('qualifiers', [])
        if not isinstance(qualifiers, list) or any(not isinstance(q, dict) or set(q) != {'kind', 'value'} or not all(isinstance(v, str) and v.strip() for v in q.values()) for q in qualifiers):
            raise ValueError('限定条件格式错误。')
        polarity = item.get('polarity')
        if polarity not in ('positive', 'negative', 'uncertain'):
            raise ValueError('事实必须显式声明肯定、否定或不确定性。')
        # 实体和对象必须有原文锚点；关系允许归一，不能借此引入新实体。
        for value in [item['subject'], item.get('object', ''), *(q['value'] for q in qualifiers)]:
            if not value:
                continue
            if ' '.join(value.casefold().split()) not in ' '.join(source.split()):
                raise ValueError('抽取值缺少原文锚点：' + value[:70])
        cited = ' '.join(' '.join([document['title'], *[document['sentences'][i] for i in refs]]).casefold().split())
        for value in [item.get('object', ''), *(q['value'] for q in qualifiers)]:
            if value and ' '.join(value.casefold().split()) not in cited:
                raise ValueError('对象或限定值不在引用句中；跨句指代须包含先行词来源：' + value[:70])
        fact = {k: item[k] for k in ('subject', 'relation', 'polarity')}
        fact['object'] = item.get('object', '')
        fact.update(qualifiers=qualifiers, sentences=sorted(set(refs)), document_id=document['id'])
        fact['id'] = digest(fact)
        validated.append(fact)
        covered.update(refs)
    missing = set(range(len(document['sentences']))) - covered
    if missing:
        raise ValueError('存在未覆盖的原文句子：' + str(sorted(missing)))
    return list({f['id']: f for f in validated}.values())


def render_fact(fact):
    # 已带语法否定的谓词不可再次加 NOT。负面事件不在此处猜测修正。
    prefix = {'positive': '', 'negative': 'NOT ', 'uncertain': 'UNCERTAIN '}[fact['polarity']]
    if fact['polarity'] == 'negative' and re.search(r'\b(?:not|never|no|neither|without|devoid)\b', fact['relation'], re.I):
        prefix = ''
    return ' | '.join([
        fact['subject'], prefix + fact['relation'], fact['object'],
        *(q['kind'] + '=' + q['value'] for q in fact['qualifiers']),
    ])


def entity_key(value, scope=''):
    normalized = ' '.join(value.casefold().split())
    local = normalized in {'he', 'she', 'it', 'they', 'them', 'his', 'her', 'their', 'this', 'that', 'we', 'its'}
    local = local or bool(re.match(r'^(?:the|this|that|these|those|their|his|her|its) [a-z]', value))
    local = local or bool(re.match(r'^(?:The|This|That|These|Those|Their|His|Her|Its) [a-z]', value))
    return ('source:' + scope + ':' if scope and local else '') + normalized


def incidence_key(value, role, fact, aliases=None):
    """关系、限定值和参与实体分属不同命名空间，避免按相同字符错误连边。"""
    if role == 'relation':
        return 'predicate:' + ' '.join(value.casefold().split())
    if role not in ('subject', 'object'):
        return 'qualifier:' + role.casefold() + ':' + ' '.join(value.casefold().split())
    owners = {q['value'] for q in fact.get('qualifiers', []) if q['kind'] == role + '_owner'}
    if len(owners) == 1:
        # 只有来源明确绑定了所属实体，才连接“她的姐妹”等关系性指称。
        # 限于同篇文档，防止不同故事的同名人物和未解析代词被合并。
        owner = next(iter(owners))
        mention = ' '.join(value.casefold().split())
        owner_prefix = ' '.join(owner.casefold().split()) + "'s "
        if mention.startswith(owner_prefix):
            mention = mention[len(owner_prefix):]
        else:
            mention = re.sub(r'^(?:her|his|its|their)\s+', '', mention)
        return 'entity:owned:' + fact['document_id'] + ':' + entity_key(owner) + ':' + mention
    # 未解析指称只属于当前断言，不将同篇中不同的“他们”自动合并。
    canonical = (aliases or {}).get((fact['document_id'], normalized_name(value)),
                                   (aliases or {}).get(normalized_name(value), value))
    named = re.match(r'^(?i:the|this)\s+((?i:band|film|musical|album|song|book|novel|play))\s+([A-Z].*)$', canonical)
    if named:
        canonical = named[2]
        if named[1].lower() != 'band' and '(' not in canonical:
            canonical += ' (' + named[1].lower() + ')'
    types = {q['value'].casefold() for q in fact.get('qualifiers', []) if q['kind'] == role + '_type'}
    if len(types) == 1 and '(' not in canonical:
        kind = next(iter(types))
        if kind in {'film', 'musical', 'album', 'song', 'book', 'novel', 'play'}:
            canonical += ' (' + kind + ')'
    key = entity_key(canonical, fact['document_id'])
    if key.startswith('source:'):
        return 'local:' + fact['id'] + ':' + key
    return 'entity:' + key


def write_graph(path, documents, facts):
    aliases = source_alias_index(derive_source_aliases(documents))
    connection = sqlite3.connect(path)
    try:
        connection.executescript('''
            CREATE TABLE documents(id TEXT PRIMARY KEY, title TEXT, sentences TEXT);
            CREATE TABLE facts(id TEXT PRIMARY KEY, document_id TEXT, title TEXT, body TEXT, payload TEXT);
            CREATE INDEX facts_document ON facts(document_id);
            CREATE VIRTUAL TABLE search USING fts5(id UNINDEXED, terms);
            CREATE TABLE incidence(entity TEXT, fact_id TEXT, role TEXT, PRIMARY KEY(entity, fact_id, role));
            CREATE INDEX incidence_fact ON incidence(fact_id);
            CREATE INDEX incidence_role_entity ON incidence(role,entity,fact_id);
            CREATE TABLE fact_sources(fact_id TEXT, document_id TEXT, sentence_index INTEGER,
                PRIMARY KEY(fact_id,document_id,sentence_index));
            CREATE INDEX source_lookup ON fact_sources(document_id,sentence_index);
            CREATE TABLE constraints(fact_id TEXT, kind TEXT, value TEXT,
                PRIMARY KEY(fact_id,kind,value));
            CREATE INDEX constraint_lookup ON constraints(kind,value,fact_id);
        ''')
        titles = {d['id']: d['title'] for d in documents}
        connection.executemany('INSERT INTO documents VALUES (?,?,?)',
            [(d['id'], d['title'], json.dumps(d['sentences'], ensure_ascii=False)) for d in documents])
        for fact in facts:
            title, body = titles[fact['document_id']], render_fact(fact)
            connection.execute('INSERT INTO facts VALUES (?,?,?,?,?)', (fact['id'], fact['document_id'], title, body, json.dumps(fact, ensure_ascii=False)))
            connection.execute('INSERT INTO search VALUES (?,?)', (fact['id'], ' '.join(indexed_tokens(title + ' ' + body))))
            for role, value in [('subject', fact['subject']), ('object', fact['object']), ('relation', fact['relation']), *[(q['kind'], q['value']) for q in fact['qualifiers']]]:
                if value:
                    connection.execute('INSERT OR IGNORE INTO incidence VALUES (?,?,?)', (incidence_key(value, role, fact, aliases), fact['id'], role))
            connection.executemany('INSERT INTO fact_sources VALUES (?,?,?)',
                [(fact['id'], fact['document_id'], index) for index in fact['sentences']])
            connection.executemany('INSERT OR IGNORE INTO constraints VALUES (?,?,?)',
                [(fact['id'], q['kind'].casefold(), ' '.join(q['value'].casefold().split())) for q in fact['qualifiers']])
        connection.commit()
        connection.execute('ANALYZE')
    finally:
        connection.close()


class PreparedKnowledge:
    def __init__(self, root):
        import numpy as np
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('schema') not in (SCHEMA, 'prepared-knowledge-v2') or self.manifest.get('status') != 'ready':
            raise ValueError('统一知识库尚未构建完成，禁止在线回退。')
        for filename, checksum in self.manifest['files'].items():
            path = (self.root / filename).resolve()
            if path.parent != self.root or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError('知识库产物校验失败：' + filename)
        self.documents = json.loads((self.root / 'text_documents.json').read_text(encoding='utf-8'))
        self.source_aliases = derive_source_aliases(self.documents)
        self.document_frequencies = Counter(term for doc in self.documents
            for term in set(indexed_tokens(doc['title'] + ' ' + ' '.join(doc['sentences']))))
        self.alias_index = ({normalized_name(row['name']): row['alias'] for row in self.source_aliases if not row.get('scoped')}
                            if self.manifest['schema'] == SCHEMA else {})
        self.graph_aliases = source_alias_index(self.source_aliases)
        if digest(self.documents) != self.manifest['corpus_sha256']:
            raise ValueError('文本库与超图来源版本不一致。')
        self.text_retrieval = self.manifest.get('text_retrieval', 'semantic')
        if self.text_retrieval == 'full-context':
            if 'text_vectors.npy' in self.manifest['files']:
                raise ValueError('完整原文模式不得发布文本向量索引。')
            self.text_vectors = np.empty((0, 0), dtype=np.float32)
        elif self.text_retrieval == 'semantic':
            self.text_vectors = np.load(self.root / 'text_vectors.npy', allow_pickle=False)
            if self.text_vectors.shape[0] != len(self.documents) or not np.isfinite(self.text_vectors).all():
                raise ValueError('文本索引损坏。')
        else:
            raise ValueError('未知的文本读取方式。')
        self.graph_path = self.root / 'hypergraph.sqlite'
        self._prepare_graph_runtime()

    def read_graph_sources(self):
        """完整来源首次读入；事实实际进入角色后才可记作会话记忆。"""
        rows = ([row for doc in self.documents for row in self._graph_documents[doc['id']]]
                if self.documents else list(self._graph_rows.values()))
        return [{'context_id': key, 'title': title, 'text': body,
                 'kind': 'prepared_knowledge', 'memory_layer': 'knowledge',
                 'retrieval_backend': 'complete_graph_sources',
                 'fact_fields': fact, 'metadata': {'prepared_knowledge': True}}
                for key, title, body, fact in rows]

    def _prepare_graph_runtime(self):
        """启动时加载静态事实与关联；不缓存问题、答案或查询结果。"""
        rows, documents, neighbors = {}, defaultdict(list), defaultdict(list)
        links = defaultdict(list)
        features, nodes, names, conditions = {}, {}, {}, {}
        aliases = getattr(self, 'graph_aliases', {})
        with closing(sqlite3.connect(self.graph_path.as_uri() + '?mode=ro', uri=True)) as db:
            for key, title, body, payload in db.execute('SELECT id,title,body,payload FROM facts ORDER BY id'):
                fact = json.loads(payload)
                rows[key] = (key, title, body, fact)
                documents[fact['document_id']].append(rows[key])
                features[key] = (frozenset(map(_search_stem, indexed_tokens(body))),
                                 frozenset(map(_search_stem, indexed_tokens(fact['relation']))),
                                 frozenset(map(_search_stem, indexed_tokens(fact['subject'] + ' ' + fact['object']))))
                names[key], nodes[key], conditions[key] = [], {}, defaultdict(set)
                for role in ('subject', 'object'):
                    value = fact[role]
                    nodes[key][role] = incidence_key(value, role, fact, aliases)
                    canonical = aliases.get(normalized_name(value), value)
                    if (len(canonical) >= 3 and any(c.isupper() for c in value)
                            and not incidence_key(value, role, fact).startswith('local:')):
                        names[key].append(' ' + normalized_name(canonical) + ' ')
                for qualifier in fact['qualifiers']:
                    conditions[key][qualifier['kind'].casefold()].add(' '.join(qualifier['value'].casefold().split()))
            for entity, key, role in db.execute("SELECT entity,fact_id,role FROM incidence WHERE role IN ('subject','object') ORDER BY fact_id,role"):
                neighbors[entity].append((*rows[key], role))
                links[key].append((role, entity, rows[key][3][role], 'entity'))
            titles = [row[0] for row in db.execute('SELECT title FROM documents')]
        # 字段中提到已知实体可作为遍历入口，但不把整段字段等同于该实体。
        # 大小写一致或明确全大写的名称才连接，避免把普通类别词连到同名作品。
        catalog = {}
        for title in titles:
            name = title.split('[SEP]', 1)[0].strip()
            node = 'entity:' + entity_key(name)
            if node not in neighbors:
                base_name = re.sub(r'\s*\([^)]*\)\s*$', '', name).strip()
                if 'entity:' + entity_key(base_name) in neighbors:
                    name, node = base_name, 'entity:' + entity_key(base_name)
            if len(name) >= 3 and node in neighbors:
                catalog[name] = (node, re.compile(r'(?<!\w)' + re.escape(name) + r'(?!\w)', re.I))
        base_candidates = defaultdict(set)
        for name, (node, _) in list(catalog.items()):
            base_candidates[re.sub(r'\s*\([^)]*\)\s*$', '', name).strip()].add(node)
        for name, matching_nodes in base_candidates.items():
            if len(matching_nodes) == 1 and name not in catalog:
                catalog[name] = (next(iter(matching_nodes)), re.compile(r'(?<!\w)' + re.escape(name) + r'(?!\w)', re.I))
        named_nodes = {node for node, _ in catalog.values()}
        owned_names = defaultdict(set)
        connectors = {'of', 'the', 'and', 'for', 'de', 'van', 'von', 'der'}
        for key, row in rows.items():
            fact = row[3]
            # 所属关系来自抽取中有原文锚点的限定；不是凭同篇共现连接实体。
            for role in ('subject', 'object'):
                owners = {q['value'] for q in fact['qualifiers'] if q['kind'] == role + '_owner'}
                if len(owners) == 1:
                    owner = next(iter(owners))
                    node = incidence_key(owner, role, {**fact, 'qualifiers': []}, aliases)
                    if node.startswith('entity:'):
                        neighbors[node].append((*row, role + '_owner'))
                        links[key].append((role + '_owner', node, owner, 'explicit_owner'))
                        named_nodes.add(node)
            for role in ('subject', 'object'):
                value = fact[role]
                words = re.findall(r'\w+', value)
                if (words and len(words) <= 8 and any(c.isalpha() for c in value)
                        and all(word[0].isupper() or word.isdigit() or word.casefold() in connectors for word in words)):
                    named_nodes.add(nodes[key][role])
                if role == 'object' and re.search(r'\b(?:name|alias|known as)\b', fact['relation'], re.I):
                    named_nodes.add(nodes[key][role])
            # 普通类别中出现地名大写，不能使整段类别获得精确姓名的奖励。
            names[key] = [' ' + normalized_name(aliases.get(normalized_name(fact[role]), fact[role])) + ' '
                          for role in ('subject', 'object')
                          if nodes[key][role] in named_nodes and nodes[key][role].startswith('entity:')
                          and len(fact[role]) >= 3]
            for role in ('subject', 'object'):
                node = nodes[key][role]
                if not node.startswith('entity:owned:'):
                    continue
                owner = next(q['value'] for q in fact['qualifiers'] if q['kind'] == role + '_owner')
                description = node.rsplit(':', 1)[-1]
                forms = {' ' + normalized_name(owner + "'s " + description) + ' ',
                         ' ' + normalized_name(description + ' of ' + owner) + ' '}
                if role == 'object':
                    for qualifier in fact['qualifiers']:
                        if qualifier['kind'] == 'count':
                            forms.add(' ' + normalized_name(owner + "'s " + qualifier['value'] + ' ' + description) + ' ')
                names[key].extend(sorted(forms))
                owned_names[node].update(forms)
                named_nodes.add(node)
        for key in rows:
            for role in ('subject', 'object'):
                names[key].extend(sorted(owned_names.get(nodes[key][role], ())))
        scoped_mentions = defaultdict(list)
        for alias in getattr(self, 'source_aliases', []):
            if not alias.get('scoped'):
                continue
            node = 'entity:' + entity_key(alias['alias'])
            if node in neighbors:
                scoped_mentions[alias['document_id']].append((alias['name'],
                    (node, re.compile(r'(?<!\w)' + re.escape(alias['name']) + r'(?!\w)'))))
        for key, row in rows.items():
            fact = row[3]
            for role in ('subject', 'object'):
                value = fact[role]
                # 显式作品类型优先于当前文章的简称：改编自某电影不能连回同名音乐剧。
                typed_mentions = {}
                for base, matching_nodes in base_candidates.items():
                    for candidate in matching_nodes:
                        kind = re.search(r'\((film|musical|album|song|book|novel|play)\)$', candidate)
                        if kind and re.search(r'\b' + kind[1] + r'\s+' + re.escape(base) + r'(?!\w)|'
                                              + re.escape(base) + r'\s*\(' + kind[1] + r'\)', value, re.I):
                            typed_mentions[normalized_name(base)] = candidate
                            if candidate != nodes[key][role]:
                                neighbors[candidate].append((*row, role + '_mention'))
                                links[key].append((role, candidate, base, 'typed_entity_mention'))
                for name, (node, pattern) in [*catalog.items(), *scoped_mentions[fact['document_id']]]:
                    if nodes[key][role] == node:
                        continue
                    base = normalized_name(re.sub(r'\s*\([^)]*\)\s*$', '', name))
                    if base in typed_mentions and typed_mentions[base] != node:
                        continue
                    match = pattern.search(value)
                    if match and (match.group() == name or match.group().isupper()):
                        neighbors[node].append((*row, role + '_mention'))
                        links[key].append((role, node, name, 'entity_mention'))
        for entries in neighbors.values():
            entries.sort(key=lambda row: (row[0], row[-1]))
        self._graph_rows, self._graph_documents, self._graph_neighbors = rows, documents, neighbors
        self._graph_features, self._graph_nodes, self._graph_names = features, nodes, names
        self._graph_conditions = conditions
        self._graph_links = links
        self._graph_named_nodes = named_nodes
        self._graph_owned_names = owned_names
        self._graph_vocabulary = defaultdict(set)
        assertion_groups = defaultdict(list)
        for _, title, body, _ in rows.values():
            for term in indexed_tokens(title + ' ' + body):
                self._graph_vocabulary[_search_stem(term)].add(term)
        for key, row in rows.items():
            fact = row[3]
            signature = (fact['document_id'], nodes[key]['subject'], fact['relation'].casefold(), fact['polarity'])
            assertion_groups[signature].append(key)
        self._graph_assertion_peers = {key: tuple(sorted(group)) for group in assertion_groups.values() for key in group}
        from .relation_coverage import RelationCoverageIndex
        self.relation_index = RelationCoverageIndex(self)

    def prepare_focused_search(self, embed):
        """启动时准备图事实排序向量；不编码任务或参考答案。"""
        from .focused_retrieval import FocusedGraphSearch
        self.focused_index = FocusedGraphSearch(self, embed)

    def search_graph_focused(self, query, top_k=8, *, constraints=None, roots=None):
        if not hasattr(self, 'focused_index'):
            raise ValueError('需要在在线计时之前准备图事实排序索引。')
        return self.focused_index.search(query, top_k, constraints=constraints, roots=roots)

    def search_text(self, query, embed, top_k=4):
        import numpy as np
        if top_k <= 0:
            return []
        if self.text_retrieval == 'full-context':
            # 文本侧无排名、无切片、无索引过滤；全部原始文档交由在线角色读取。
            # top_k 不得隐式截断全库，query 也不用于提前选出答案段落。
            return [self._hit(doc['id'], doc['title'], ' '.join(doc['sentences']),
                              0., 'prepared_text_full_context') for doc in self.documents]
        encoded = embed(query)
        if encoded.model != self.manifest['embedding_model'] or encoded.provider != self.manifest['embedding_provider']:
            raise ValueError('查询向量与离线文本索引的模型不一致。')
        vector = np.asarray(encoded.vector, dtype=np.float32)
        if vector.shape != (self.text_vectors.shape[1],) or not np.isfinite(vector).all() or not np.linalg.norm(vector):
            raise ValueError('查询向量无效。')
        scores = self.text_vectors @ (vector / np.linalg.norm(vector))
        chosen = np.argsort(-scores, kind='stable')[:top_k]
        return [self._hit(self.documents[i]['id'], self.documents[i]['title'], ' '.join(self.documents[i]['sentences']), float(scores[i]), 'prepared_text') for i in chosen]

    @staticmethod
    def _hit(identifier, title, body, score, backend, **extra):
        return {'context_id': identifier, 'title': title, 'text': body, 'kind': 'prepared_knowledge',
                'section': '', 'memory_layer': 'knowledge', 'context_score': score, 'retrieval_score': score,
                'retrieval_backend': backend, 'matched_nodes': [], 'matched_hyperedges': [],
                'reasoning_path': [], 'bridge_source_context_ids': [], 'metadata': {'prepared_knowledge': True}, **extra}

    def search_graph(self, query, top_k=4, *, constraints=None):
        """倒排候选与有界实体邻接；在线不接触原文抽取流程。"""
        terms = sorted(set(indexed_tokens(query)) & (question_content_tokens(query) | {t for t in indexed_tokens(query) if t.isdigit() or re.search('[\u3400-\u9fff]', t)}))
        if not terms or top_k <= 0:
            return []
        if not hasattr(self, '_graph_rows'):
            self._prepare_graph_runtime()
        # sqlite 的事务上下文不负责关闭连接，必须显式关闭，避免 Windows 文件占用。
        with closing(sqlite3.connect(self.graph_path.as_uri() + '?mode=ro', uri=True)) as db:
            lookup_terms = set(terms)
            for term in terms:
                lookup_terms.update(self._graph_vocabulary.get(_search_stem(term), ()))
            parameters = [' OR '.join('"' + t + '"' for t in sorted(lookup_terms))]
            filters = ''
            for kind, value in sorted((constraints or {}).items()):
                filters += ' AND EXISTS (SELECT 1 FROM constraints c WHERE c.fact_id=f.id AND c.kind=? AND c.value=?)'
                parameters += [kind.casefold(), ' '.join(value.casefold().split())]
            seed_ids = {row[0] for row in db.execute('SELECT f.id FROM search JOIN facts f ON f.id=search.id WHERE search MATCH ?' + filters + ' ORDER BY bm25(search), f.id LIMIT 48', parameters)}
            # 精确实体通过角色索引直接进入候选，不能先被通用词的前48名截掉。
            words = normalized_name(query).split()
            nodes = set()
            normalized_query = ' ' + normalized_name(query) + ' '
            nodes.update(node for node, forms in self._graph_owned_names.items()
                         if any(form in normalized_query for form in forms))
            for start in range(len(words)):
                for length in range(1, min(6, len(words) - start) + 1):
                    name = ' '.join(words[start:start + length])
                    nodes.add('entity:' + name)
                    canonical = getattr(self, 'alias_index', {}).get(name)
                    if canonical:
                        nodes.add('entity:' + entity_key(canonical))
            if nodes:
                for node in nodes:
                    for key, *_ in self._graph_neighbors.get(node, ()):
                        if all(' '.join(value.casefold().split()) in self._graph_conditions[key].get(kind.casefold(), ())
                               for kind, value in (constraints or {}).items()):
                            seed_ids.add(key)
            terms = sorted(set(map(_search_stem, terms)))
            query_set = set(terms)
            frequencies = getattr(self, 'document_frequencies', {})
            corpus_size = len(getattr(self, 'documents', [])) or 1
            weights = {term: 1 + math.log((corpus_size + 1) / (frequencies.get(term, 0) + 1)) for term in query_set}
            normalized_query = ' ' + normalized_name(query) + ' '
            candidates = {}
            paths = {}
            features, chain_scores, anchors = {}, {}, {}
            term_bits = {term: 1 << index for index, term in enumerate(terms)}
            bit_weights = {term_bits[term]: weights[term] for term in terms}
            weight_cache = {0: 0.}
            def mask_weight(mask):
                if mask not in weight_cache:
                    remaining, total = mask, 0.
                    while remaining:
                        bit = remaining & -remaining
                        total += bit_weights[bit]
                        remaining ^= bit
                    weight_cache[mask] = total
                return weight_cache[mask]
            def score_chain(chain):
                if chain not in chain_scores:
                    body_mask, relation_mask, anchor = 0, 0, 0
                    for key in chain:
                        body_mask |= features[key][0]
                        relation_mask |= features[key][1]
                        anchor = max(anchor, anchors[key])
                    chain_scores[chain] = (3 * mask_weight(body_mask) + 2 * mask_weight(relation_mask)
                                           + 12 * anchor - .25 * (len(chain) - 1))
                return chain_scores[chain]
            def add(identifier, title, body, fact):
                body_terms, relation_terms, endpoints = self._graph_features[identifier]
                anchors[identifier] = max((len(name.split()) for name in self._graph_names[identifier]
                                           if name in normalized_query), default=0)
                score = sum(weights[t] for t in query_set & body_terms) + 2 * sum(weights[t] for t in query_set & endpoints) + 12 * anchors[identifier]
                candidates[identifier] = (score, title, body, fact)
                features[identifier] = (sum(term_bits[t] for t in query_set & body_terms),
                                        sum(term_bits[t] for t in query_set & relation_terms))
            for identifier in sorted(seed_ids):
                _, title, body, fact = self._graph_rows[identifier]
                add(identifier, title, body, fact)
                paths[(identifier,)] = []
            # 同篇的重复实体不能占满探索入口；篇内事实会在下一步一并展开。
            frontier, frontier_documents = [], set()
            for identifier in sorted(candidates, key=lambda k: (-candidates[k][0], k)):
                document_id = candidates[identifier][3]['document_id']
                if document_id in frontier_documents:
                    continue
                frontier.append((identifier,))
                frontier_documents.add(document_id)
                if len(frontier) == 4:
                    break
            # 保留同一来源中的叙事上下文；这是来源关联，不把共现当成实体等价。
            for (identifier,) in frontier:
                document_id = candidates[identifier][3]['document_id']
                for key, title, body, item in self._graph_documents[document_id][:96]:
                    if key == identifier:
                        continue
                    if constraints and any(q['kind'].casefold() in {k.casefold() for k in constraints} and
                        all(q['value'].casefold() != v.casefold() for k, v in constraints.items() if k.casefold() == q['kind'].casefold()) for q in item['qualifiers']):
                        continue
                    if key not in candidates:
                        add(key, title, body, item)
                    paths[(identifier, key)] = [{'from': identifier, 'to': key, 'document_id': document_id, 'via': 'source_context'}]
            neighbor_cache = {}
            for _ in range(2):
                expanded = []
                for chain in frontier:
                    identifier = chain[-1]
                    fact = candidates[identifier][3]
                    # 到达新来源后只补充与问题相关的少量上下文；保留路径上的桥。
                    if len(chain) > 1:
                        contextual = []
                        for key, title, body, item in self._graph_documents[fact['document_id']][:96]:
                            if key in chain:
                                continue
                            values = self._graph_conditions[key]
                            if any(kind.casefold() in values and ' '.join(value.casefold().split()) not in values[kind.casefold()]
                                   for kind, value in (constraints or {}).items()):
                                continue
                            if key not in candidates:
                                add(key, title, body, item)
                            if features[key][0]:
                                contextual.append(key)
                        for key in sorted(contextual, key=lambda k: (-score_chain((*chain, k)), k))[:6]:
                            next_chain = (*chain, key)
                            paths[next_chain] = [*paths[chain], {'from': identifier, 'to': key,
                                'document_id': fact['document_id'], 'via': 'source_context'}]
                            expanded.append(next_chain)
                    for role, node, value, via in self._graph_links[identifier]:
                        if not value:
                            continue
                        if not node.startswith('entity:') or node not in self._graph_named_nodes:
                            continue
                        if node not in neighbor_cache:
                            neighbor_cache[node] = self._graph_neighbors.get(node, ())[:128]
                        for key, title, body, neighbor_fact, next_role in neighbor_cache[node]:
                            if key in chain:
                                continue
                            values = self._graph_conditions[key]
                            if any(kind.casefold() in values and ' '.join(value.casefold().split()) not in values[kind.casefold()]
                                   for kind, value in (constraints or {}).items()):
                                continue
                            if key not in candidates:
                                add(key, title, body, neighbor_fact)
                            next_chain = (*chain, key)
                            paths[next_chain] = [*paths[chain], {'from': identifier, 'to': key, 'entity': value,
                                'from_role': role, 'to_role': next_role,
                                **({'via': 'entity_mention'} if via == 'entity_mention' or next_role.endswith('_mention') else {})}]
                            expanded.append(next_chain)
                frontier = sorted(set(expanded), key=lambda p: (-score_chain(p), p))[:16]
            # 按新增查询覆盖选择完整路径；已经选中的高分种子不能让每条
            # 无关邻边都再次领取全额分数，挤掉其他来源中的必要条件。
            # 同一主体、同一关系的并列对象共同构成回答；预算允许时作为一组保留，
            # 避免只取“与母亲同住”却丢掉同关系的“与五位姐妹同住”。不跨来源合并。
            expanded_paths = {}
            for chain, steps in paths.items():
                expanded, extra_steps = list(chain), []
                for key in chain:
                    peers = self._graph_assertion_peers[key]
                    if len(peers) > top_k:
                        continue
                    for peer in peers:
                        if peer in expanded:
                            continue
                        values = self._graph_conditions[peer]
                        if any(' '.join(value.casefold().split()) not in values.get(kind.casefold(), ())
                               for kind, value in (constraints or {}).items()):
                            continue
                        if peer not in candidates:
                            _, title, body, fact = self._graph_rows[peer]
                            add(peer, title, body, fact)
                        expanded.append(peer)
                        extra_steps.append({'from': key, 'to': peer, 'via': 'same_subject_relation'})
                if len(expanded) <= top_k:
                    expanded_paths[tuple(expanded)] = [*steps, *extra_steps]
                else:
                    expanded_paths[chain] = steps
            paths = expanded_paths
            selected, bridges, selected_paths = [], {}, {}
            covered, related, anchored = 0, 0, 0
            path_features = {}
            for chain in paths:
                body_terms, relation_terms = 0, 0
                for key in chain:
                    body_terms |= features[key][0]
                    relation_terms |= features[key][1]
                path_features[chain] = (body_terms, relation_terms, max(anchors[key] for key in chain))
            while len(selected) < top_k:
                choices = []
                for chain, (body_terms, relation_terms, anchor) in path_features.items():
                    new = [key for key in chain if key not in selected]
                    if not new or len(selected) + len(new) > top_k:
                        continue
                    gain = (3 * mask_weight(body_terms & ~covered)
                            + 2 * mask_weight(relation_terms & ~related)
                            + 12 * max(0, anchor - anchored))
                    # 少量保留新增事实的相关度，允许补充同一关系的不同答案值。
                    # 已选高分种子不能再次给无关邻边贡献残余分数。
                    choices.append((gain + .1 * score_chain(tuple(new)), -len(new), chain))
                if not choices:
                    break
                chain = min(choices, key=lambda item: (-item[0], -item[1], item[2]))[2]
                new = [key for key in chain if key not in selected]
                selected.extend(new)
                body_terms, relation_terms, anchor = path_features[chain]
                covered |= body_terms
                related |= relation_terms
                anchored = max(anchored, anchor)
                for index, key in enumerate(chain):
                    if key in new:
                        bridges[key] = list(chain[:index])
                        selected_paths[key] = paths[chain][:index]
            return [self._hit(key, candidates[key][1], candidates[key][2], candidates[key][0], 'prepared_hypergraph',
                              fact_fields=copy.deepcopy(candidates[key][3]), bridge_source_context_ids=bridges.get(key, []),
                              reasoning_path=selected_paths.get(key, []),
                              matched_hyperedges=[key]) for key in selected]

    @staticmethod
    def _chain_score(chain, candidates, query_set, features=None, weights=None, anchors=None):
        covered, relations = set(), set()
        for key in chain:
            fact = candidates[key][3]
            body_terms, relation_terms = features[key] if features is not None else (indexed_tokens(candidates[key][2]), indexed_tokens(fact['relation']))
            covered.update(body_terms)
            relations.update(relation_terms)
        # 只奖励新的查询覆盖，重复主体不能靠多放几条相同事实累积分数。
        weights = weights or {term: 1 for term in query_set}
        return (3 * sum(weights[t] for t in query_set & covered)
                + 2 * sum(weights[t] for t in query_set & relations)
                + 12 * max(((anchors or {}).get(key, 0) for key in chain), default=0)
                - .25 * (len(chain) - 1))

    def sources(self, fact_id):
        """取回该断言明确引用的原始句子，用于审查；检索正文仍使用字段。"""
        with closing(sqlite3.connect(self.graph_path.as_uri() + '?mode=ro', uri=True)) as db:
            rows = db.execute('SELECT d.id,d.title,d.sentences,s.sentence_index FROM fact_sources s JOIN documents d ON d.id=s.document_id WHERE s.fact_id=? ORDER BY d.id,s.sentence_index', (fact_id,)).fetchall()
        return [{'document_id': doc_id, 'title': title, 'sentence_index': index,
                 'text': json.loads(sentences)[index]} for doc_id, title, sentences, index in rows]


@lru_cache(maxsize=2)
def load_prepared(root):
    return PreparedKnowledge(root)


def configured_knowledge():
    root = os.environ.get('PREPARED_KNOWLEDGE_DIR', '').strip()
    return load_prepared(str(Path(root).resolve())) if root else None


def search_prepared_knowledge(query, *, mode, top_k=4, embed=None):
    """面向任意角色的统一入口；协议调用者不能借缺库回退原文。"""
    library = configured_knowledge()
    if library is None:
        raise ValueError('需要先配置预建知识库，不能在线临时建立。')
    if mode == 'protocol':
        return library.search_graph(query, top_k)
    if mode == 'text' and embed is not None:
        return library.search_text(query, embed, top_k)
    raise ValueError('知识模式必须为文本或协议；文本模式需要查询编码器。')
