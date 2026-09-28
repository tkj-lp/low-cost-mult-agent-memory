"""来源标题和可追溯姓名对应；只读取文档，不访问问题或答案。"""
import re


def normalized_name(value):
    return ' '.join(re.findall(r'\w+', value.casefold()))


def derive_source_aliases(documents):
    """仅接受标题姓名在首句定义全名中按顺序完整出现的保守规则。"""
    rows = []
    for document in documents:
        topic = document['title'].split('[SEP]', 1)[0].strip()
        opening = document['sentences'][0].strip()
        base_topic = re.sub(r'\s*\([^)]*\)\s*$', '', topic).strip()
        primary = re.match(r'^(?:the\s+)?' + re.escape(base_topic) + r'(?=\W|$)', opening, re.I)
        if primary:
            if normalized_name(primary.group()) != normalized_name(topic):
                rows.append({'alias': topic, 'name': primary.group(), 'document_id': document['id'],
                             'sentences': [0], 'scoped': True,
                             'basis': 'disambiguated article title and introductory subject', 'quote': opening})
            definition = re.match(r'^(?:the\s+)?' + re.escape(base_topic) + r'\s+is\s+(?:a|an)\s+([^.;]+)', opening, re.I)
            if definition:
                prefix = re.split(r'\b(?:with|found|which|that|directed|written|from)\b', definition.group(1), 1)[0]
                noun = re.findall(r'[A-Za-z]+', prefix)[-1:]
                if noun:
                    for determiner in ('the ', 'this '):
                        descriptor = determiner + noun[0]
                        if re.search(r'\b' + re.escape(descriptor) + r'\b', opening, re.I):
                            rows.append({'alias': topic, 'name': descriptor, 'document_id': document['id'],
                                         'sentences': [0], 'scoped': True,
                                         'basis': 'introductory type and scoped definite reference', 'quote': opening})
        # 人物文章的标题与首句主定义保留同一来源中的姓名对应；不跨文档按姓氏合并。
        person = re.match(r'^([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+){1,5})\s*\(', opening)
        if (person and len(topic.split()) >= 2 and person.group(1).split()[-1] == topic.split()[-1]
                and not set(normalized_name(topic).split()) <= set(normalized_name(person.group(1)).split())):
            rows.append({'alias': topic, 'name': person.group(1), 'document_id': document['id'],
                         'sentences': [0], 'scoped': True,
                         'basis': 'person article title and primary full-name definition', 'quote': opening})
        # 标题与开篇主语逐字对应时，冠词不是新实体。只在当前来源内绑定。
        named = re.match(r'^(?:the\s+)?' + re.escape(topic) + r'(?=\W|$)', opening, re.I)
        if named:
            mention = named.group()
            if normalized_name(mention) != normalized_name(topic):
                rows.append({'alias': topic, 'name': mention, 'document_id': document['id'],
                             'sentences': [0], 'scoped': True,
                             'basis': 'document topic equals introductory subject after determiner removal', 'quote': opening})
            definition = re.match(r'^(?:the\s+)?' + re.escape(topic) +
                                  r'.{0,60}?\bis\s+(?:a|an)\s+([^.;]+)', opening, re.I)
            if definition:
                category = re.split(r'\b(?:found|which|that|with|in)\b', definition.group(1), 1)[0].strip()
                noun = re.findall(r'[A-Za-z]+', category)[-1:]
                if noun:
                    descriptor = 'This ' + noun[0]
                    for sentence in document['sentences']:
                        if re.search(r'\b' + re.escape(descriptor) + r'\b', sentence, re.I):
                            rows.append({'alias': topic, 'name': descriptor, 'document_id': document['id'],
                                         'sentences': [0], 'scoped': True,
                                         'basis': 'introductory type and matching demonstrative description', 'quote': sentence})
        title = re.sub(r'\s*\([^)]*\)\s*$', '', document['title']).strip()
        first = document['sentences'][0].strip()
        match = re.match(r'^([^,(]{2,90}?)(?:\s*\(|\s+(?:is|was|are|were)\b)', first)
        if not match:
            continue
        name = match.group(1).strip(' "')
        left, right = normalized_name(title).split(), normalized_name(name).split()
        if left == right or len(left) < 2 or len(right) > 8 or len(right) <= len(left):
            continue
        remaining = iter(right)
        if all(any(word == candidate for candidate in remaining) for word in left):
            rows.append({'alias': title, 'name': name, 'document_id': document['id'],
                         'sentences': [0], 'basis': 'document title and introductory subject', 'quote': first})
    # 同一名字对应不同标题时不自动建立身份等价。
    targets = {}
    for row in rows:
        key = (row.get('document_id') if row.get('scoped') else None, normalized_name(row['name']))
        targets.setdefault(key, set()).add(normalized_name(row['alias']))
    return [row for row in rows if len(targets[(row.get('document_id') if row.get('scoped') else None, normalized_name(row['name']))]) == 1]


def bind_source_parts(document, result):
    """从明确的主语—附属部分句式补所属限定，不依据题目或整篇共现猜连边。"""
    topic = document['title'].split('[SEP]', 1)[0].strip()
    base = re.sub(r'\s*\([^)]*\)\s*$', '', topic).strip()
    source = ' '.join(document['sentences'])
    for fact in result.get('facts', []):
        subject = fact.get('subject', '')
        if not subject or any(q['kind'] == 'subject_owner' for q in fact.get('qualifiers', [])):
            continue
        # 例如作品介绍中的“作品 ... with music and lyrics by ... and book by ...”。
        attached = re.search(r'^(?:the\s+)?' + re.escape(base) + r'\s+is\s+[^.]{0,100}\bwith\s+[^.]{0,140}?\b'
                             + re.escape(subject) + r'\s+by\b', source, re.I)
        if attached:
            fact.setdefault('qualifiers', []).append({'kind': 'subject_owner', 'value': topic})
    return result


def source_alias_index(rows):
    return {(row['document_id'], normalized_name(row['name'])) if row.get('scoped') else normalized_name(row['name']): row['alias']
            for row in rows}


def identity_qualifiers(fact, aliases):
    """把图中已验证的姓名对应随事实交接，避免下游只收到失去连接的别名。"""
    values = list(fact.get('qualifiers', []))
    for role in ('subject', 'object'):
        canonical = aliases.get((fact['document_id'], normalized_name(fact[role])),
                                aliases.get(normalized_name(fact[role])))
        if canonical and normalized_name(canonical) != normalized_name(fact[role]):
            item = {'kind': role + '_identity', 'value': canonical}
            if item not in values:
                values.append(item)
    return values


def evidence_packet(evidence, aliases=()):
    """展开真实事实载荷；短来源标签仅去重，不替代正文计量。"""
    sources, facts, document_keys = {}, [], {}
    alias_map = source_alias_index(aliases)
    for source_id, hit in evidence.items():
        fact = hit['fact_fields']
        doc = fact['document_id']
        if doc not in document_keys:
            document_keys[doc] = 'd' + str(len(document_keys) + 1)
            sources[document_keys[doc]] = hit['title']
        relation = fact['relation']
        if fact['polarity'] == 'negative' and not re.search(r'\b(?:not|never|no|without|devoid)\b', relation, re.I):
            relation = 'NOT ' + relation
        elif fact['polarity'] == 'uncertain':
            relation = 'UNCERTAIN ' + relation
        values, expressed = {}, set()
        for role in ('subject', 'object'):
            value = fact[role]
            owners = {q['value'] for q in fact['qualifiers'] if q['kind'] == role + '_owner'}
            if len(owners) == 1 and re.match(r'^(?:her|his|their|its)\s+', value, re.I):
                noun = re.sub(r'^(?:her|his|their|its)\s+', '', value, flags=re.I)
                counts = {q['value'] for q in fact['qualifiers'] if q['kind'] == 'count'} if role == 'object' else set()
                if len(counts) == 1:
                    noun = next(iter(counts)) + ' ' + noun
                    expressed.add('count')
                value = next(iter(owners)) + "'s " + noun
                expressed.add(role + '_owner')
            values[role] = value
        # 所属实体和数量直接进入其对应字段，不让接收方重新解析指代或重复读取限定。
        qualifiers = '; '.join(q['kind'] + '=' + q['value'] for q in identity_qualifiers(fact, alias_map) if q['kind'] not in expressed)
        facts.append([source_id, values['subject'], relation, values['object'], qualifiers, document_keys[doc]])
    packet = {'sources': sources, 'facts': facts}
    labels = {normalized_name(row[1]) for row in facts} | {normalized_name(row[3]) for row in facts}
    used_aliases = [[a['alias'], a['name'], document_keys[a['document_id']]] for a in aliases
                   if a['document_id'] in document_keys and
                   (normalized_name(a['alias']) in labels or normalized_name(a['name']) in labels)]
    if used_aliases:
        packet['aliases'] = used_aliases
    return packet
