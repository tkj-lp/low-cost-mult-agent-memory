"""用实际历史和来源连接选择指代入口；不把未证实的历史回答写成知识。"""
import re

from .source_context import normalized_name

WORK_TYPES = {'film', 'musical', 'album', 'song', 'book', 'novel', 'play', 'series'}
PERSON_ROLES = {'person', 'singer', 'songwriter', 'composer', 'author', 'writer', 'actor',
                'actress', 'director', 'scientist', 'politician', 'president', 'poet', 'artist', 'activist'}


def definition_types(description):
    """只取定义的主体类别，不把后续动作或所属对象的类别赋给主体。"""
    head = re.split(r'\b(?:to|from|by|with|of|in|on|at|who|which|that|whose|for)\b', description, 1, flags=re.I)[0]
    kinds = set(normalized_name(head).split()) & (WORK_TYPES | PERSON_ROLES)
    return kinds | ({'person'} if kinds & PERSON_ROLES else set())


class SourceReferenceHistory:
    def __init__(self, knowledge):
        self.knowledge = knowledge
        self.index = getattr(knowledge, 'relation_index', None)
        self.types, self.turns = {}, []
        if self.index is None:
            return
        for doc in getattr(knowledge, 'documents', []):
            title = doc['title'].split('[SEP]', 1)[0].strip()
            # 类型只由来源的标题和定义句建立，正文提及别的作品不改变本文主题类型。
            opening = ' '.join(doc['sentences'][:1])
            first = re.split(r'[.!?](?:\s|$)', opening, 1)[0]
            definition = re.search(r'\b(?:is|was)\s+(?:a|an|the)\s+([^.;]+)', first, re.I)
            kinds = definition_types(definition.group(1)) if definition else set()
            title_type = re.search(r'\((\w+)\)$', title)
            if title_type and title_type[1].lower() in WORK_TYPES:
                kinds.add(title_type[1].lower())
            if kinds & PERSON_ROLES or re.search(r'\(born\b', opening, re.I):
                kinds.add('person')
            for node in self.index.anchors(title):
                self.types.setdefault(node, set()).update(kinds)
        for key, row in knowledge._graph_rows.items():
            fact = row[3]
            if normalized_name(fact['relation']) not in {'is', 'was', 'is a', 'is an', 'was a', 'was an', 'occupation'}:
                continue
            kinds = definition_types(fact['object'])
            if kinds:
                node = knowledge._graph_nodes[key]['subject']
                self.types.setdefault(node, set()).update(kinds | ({'person'} if kinds & PERSON_ROLES else set()))

    def reset(self):
        self.turns = []

    def remember(self, question, answer, source_ids, roots):
        if self.index is None or not source_ids:
            return
        supported = {link[1] for key in source_ids for link in self.knowledge._graph_links.get(key, ())
                     if link[1].startswith('entity:')}
        for key in source_ids:
            supported.update(n for n in self.knowledge._graph_nodes.get(key, {}).values() if n.startswith('entity:'))
        # 关系值和局部描述仍保存在原图中，但不是可作为检索起点的命名实体。
        supported.intersection_update(self.knowledge._graph_named_nodes)
        self.turns.append({'question': question, 'answer': answer, 'source_ids': list(source_ids),
                           'supported': supported, 'question_nodes': set(roots) & supported,
                           'answer_nodes': self.index.anchors(answer) & supported})

    def resolve(self, question, fallback):
        if self.index is None:
            return set(fallback), None
        fallback = set(fallback) & self.knowledge._graph_named_nodes
        if not self.turns:
            return set(fallback), None
        explicit = self.index.anchors(question)
        if explicit:
            return explicit, None
        words = set(normalized_name(question).split())
        requested = words & (WORK_TYPES | PERSON_ROLES)
        if not requested and words & {'she', 'her', 'he', 'his', 'him'}:
            requested = {'person'}
        neuter = bool(words & {'it', 'its', 'this', 'that'}) and not requested
        for index in range(len(self.turns)-1, -1, -1):
            turn = self.turns[index]
            # 人物身份优先承接已证实回答；无生命对象先承接上一问题的主题。
            pools = ([turn['question_nodes'], turn['answer_nodes'], turn['supported']] if neuter else
                     [turn['answer_nodes'], turn['question_nodes'], turn['supported']])
            for pool in pools:
                nodes = {n for n in pool if (bool(self.types.get(n, set()) & requested) if requested else
                         bool(self.types.get(n, set()) & WORK_TYPES) if neuter else True)}
                if not nodes:
                    continue
                return nodes, {'previous_turn': index+1, 'source_ids': turn['source_ids'],
                               'entities': sorted(nodes), 'requested_types': sorted(requested),
                               'basis': 'unverified_history_with_observed_source_entities',
                               'suggestion_only': True}
        return set(fallback), None
