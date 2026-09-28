"""超图检索与原生字段交接；下游只接收任务及本阶段必要事实。"""
from __future__ import annotations

import json
import math
import re
import time

from .plain_text_collaboration import COMMON_TEXT, TEXT_ROLES
from hypergraph_memory_method.source_context import evidence_packet


PROTOCOL_PARSER_VERSION = 'field-types-v7'


class InvalidSupportReference(ValueError):
    """候选答案的来源引用无效；不能把生成的错误关系当成原始证据。"""


# 与文本链路共用来源、历史答案和完整性要求，只改变信息表示与交接方式。
FIELD_COMMON = COMMON_TEXT.split('Write ordinary natural language', 1)[0]
INPUT_SCHEMA = ('Input evidence rows are [id,subject,relation,value,qualifiers,document]; '
                'sources gives document titles and aliases gives identity links.')
OUTPUT_SCHEMA = ('Output fact rows are [id,subject,relation,value] with an optional fifth string for needed qualifiers. '
                 'Use supplied ids for provenance. Extract needed fields, not prose sentences; '
                 'keep relation links, ownership, counts, negation and source-title context when needed to answer.')
FIELD_ROLES = {
    'planner': ('Resolve references using the recent conversation and source memory. Do not answer. '
                'Generate a JSON object with task (the actual question with references resolved), '
                'lookup (a boolean: whether knowledge retrieval is needed), and reuse (a list of supplied memory ids needed by the retriever). '
                'Select the ids of facts that actually provide the requested attribute or relation. '
                'lookup=false requires nonempty reuse covering the whole task. '
                'If the entity is known but the requested attribute is absent, use lookup=true. '
                'With no usable facts, use reuse=[] and lookup=true.'),
    'retriever': ('Locate the requested information in the supplied fields. Do not give the final answer. '
                  'Generate a JSON object with facts: a list of extracted fact rows needed by the reasoner, including intermediate links. '
                  'If evidence is missing, add missing (a list of absent attributes or links) and query (a focused lookup). '
                  'Otherwise omit missing and query; do not recheck supplied evidence.'),
    'reasoner': ('Derive the answer from the received fields. Generate a JSON object with answer (the actual candidate) '
                 'and support (the source ids of the received rows supporting it). Do not regenerate fact rows. '
                 'The runtime forwards the selected received rows unchanged. Correct unsupported earlier interpretations. '
                 'If evidence is missing, add missing (a list of absent attributes or links) and query (a focused lookup).'),
    'summarizer': TEXT_ROLES['summarizer'],
}


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def size(value):
    return len(value.encode('utf-8'))


class FieldCollaboration:
    def __init__(self, knowledge, client, *, max_searches=2):
        from .evidence_memory import SessionEvidenceMemory
        self.knowledge, self.client, self.max_searches = knowledge, client, max_searches
        self.memory = SessionEvidenceMemory('protocol', None, knowledge)
        self.observed = {}

    def _call(self, role, payload):
        schema = INPUT_SCHEMA if role in ('planner', 'retriever') else ''
        if role == 'retriever':
            schema += '\n' + OUTPUT_SCHEMA
        elif role in ('reasoner', 'summarizer'):
            schema = 'Received facts are [source id,subject,relation,value,optional qualifier].'
        if role != 'summarizer':
            schema += '\nWrite compact single-line JSON. Omit empty optional fields.'
        response = self.client.call(role, [
            {'role': 'system', 'content': FIELD_COMMON + '\n' + FIELD_ROLES[role] + '\n' + schema},
            {'role': 'user', 'content': encode(payload)},
        ])
        content = response.get('content')
        if not isinstance(content, str) or not content.strip():
            raise ValueError('角色没有生成交接内容：' + role)
        if self.client.calls[-1].get('finish_reason') == 'length':
            raise ValueError('字段生成达到上限，不能把截断内容当作完整交接。')
        self.outputs[role] = content
        if role == 'summarizer':
            return content.strip()
        # 共用读取方法的其他运行器也可以独立初始化每题审计。
        if 'protocol_parse_ms' not in self.metrics:
            self.protocol_parsing = []
            self.metrics.update(protocol_parse_ms=0., field_normalization_ms=0.)
        started = time.perf_counter_ns()
        audit = {'role': role, 'raw_content': content, 'normalizations': [],
                 'normalization_ms': 0.}
        try:
            fenced = re.fullmatch(r'\s*```(?:json)?\s*(\{.*\}|\[.*\])\s*```\s*', content, re.S | re.I)
            value = json.loads(fenced.group(1) if fenced else content)
            conversion_started = time.perf_counter_ns()
            try:
                try:
                    value = self._normalize_fields(role, value, audit['normalizations'], payload)
                except InvalidSupportReference as exc:
                    self._reject_reference(role, value.get('support'), exc)
                    # 不改答案候选；丢弃错误引用，由运行器转交已经收到的必要原始字段。
                    value = {**value, 'support': []}
            finally:
                audit['normalization_ms'] = (time.perf_counter_ns() - conversion_started) / 1e6
            audit['parsed_value'] = value
            return value
        finally:
            audit['parse_ms'] = (time.perf_counter_ns() - started) / 1e6
            self.protocol_parsing.append(audit)
            self.metrics['protocol_parse_ms'] += audit['parse_ms']
            self.metrics['field_normalization_ms'] += audit['normalization_ms']

    def _normalize_fields(self, role, value, changes, payload=None):
        """统一字段表示；不修补缺失事实、不猜测来源、不改变检索决定。"""
        from .evidence_memory import source_label
        if role == 'retriever' and isinstance(value, list):
            changes.append({'field': 'root', 'before': value, 'after': {'facts': value}})
            value = {'facts': value}
        if not isinstance(value, dict):
            raise ValueError('协议输出必须是字段对象。')
        if role == 'planner':
            if 'task' not in value and isinstance(value.get('question'), str):
                value['task'] = value.pop('question')
                changes.append({'field': 'task', 'alias': 'question', 'after': value['task']})
            original = value.get('lookup')
            if isinstance(original, str) and original in ('true', 'false'):
                value['lookup'] = original == 'true'
                changes.append({'field': 'lookup', 'before': original, 'after': value['lookup']})
        if role == 'retriever' and 'facts' not in value and isinstance(value.get('missing'), list) and value['missing']:
            value['facts'] = []
            changes.append({'field': 'facts', 'before': None, 'after': []})
        if role == 'reasoner' and isinstance(value.get('support'), list):
            support = []
            for index, item in enumerate(value['support']):
                if isinstance(item, list) and len(item) in (4, 5):
                    # 文档号只有在事实内容完全一致且唯一匹配时才能转为事实号。
                    # 改写关系或颠倒主客体仍拒绝；下游始终使用收到的原行。
                    matches = [row for row in (payload or {}).get('facts', [])
                               if item[1:] == row[1:len(item)] and
                               (source_label(item[0]) == row[0] or
                                (isinstance(item[0], str) and len(row) == 5 and
                                 'document=' + item[0] in row[4].split('; ')))]
                    if len(matches) != 1:
                        raise InvalidSupportReference('支持事实整行与已收到的字段不一致。')
                    label = matches[0][0]
                    changes.append({'field': 'support.' + str(index), 'before': item, 'after': label})
                    item = label
                support.append(item)
            value = {**value, 'support': support}
        if role != 'retriever' or not isinstance(value.get('facts'), list):
            return value
        rows, packet = [], None
        for index, original in enumerate(value['facts']):
            row = original
            if isinstance(row, dict):
                row = dict(row)
                if 'source_title' in row and 'document' not in row:
                    row['document'] = row.pop('source_title')
                    changes.append({'field': 'facts.' + str(index) + '.document', 'alias': 'source_title'})
                required = {'id', 'subject', 'relation', 'value'}
                if not required <= row.keys() or row.keys() - required - {'qualifiers', 'document'}:
                    raise ValueError('命名事实字段不完整或包含无法归一化的字段。')
                named = row
                row = [named[k] for k in ('id', 'subject', 'relation', 'value')]
                if 'qualifiers' in named or 'document' in named:
                    row.append(named.get('qualifiers'))
                if 'document' in named:
                    row.append(named['document'])
            if isinstance(row, list) and len(row) in (4, 5, 6):
                row = list(row)
                if isinstance(row[0], str):
                    row[0] = source_label(row[0])
                if (isinstance(row[0], str) and re.fullmatch(r'[1-9][0-9]*', row[0])
                        and 's' + row[0] in self.evidence):
                    row[0] = 's' + row[0]
                if type(row[3]) in (bool, int) or (type(row[3]) is float and math.isfinite(row[3])):
                    row[3] = encode(row[3])
                if len(row) >= 5 and (row[4] is None or row[4] == []):
                    row[4] = ''
                elif len(row) >= 5 and isinstance(row[4], list) and all(isinstance(v, str) for v in row[4]):
                    row[4] = '; '.join(row[4])
                if len(row) == 6:
                    packet = self._packet() if packet is None else packet
                    documents = {r[0]: r[5] for r in packet['facts']}
                    if (not isinstance(row[0], str) or not isinstance(row[5], str)
                            or documents.get(source_label(row[0])) != row[5]):
                        raise ValueError('交接文档与所引用来源不一致。')
                    if not isinstance(row[4], str):
                        raise ValueError('事实限定必须为文本。')
                    # 六列输入式记录转换成五列交接式，仍保留文档标题，不丢掉来源语义。
                    row[4] = '; '.join(filter(None, [row[4], 'document=' + row[5],
                                                   'title=' + packet['sources'][row[5]]]))
                    row = row[:5]
            if row != original:
                changes.append({'field': 'facts.' + str(index), 'before': original, 'after': row})
            rows.append(row)
        return {**value, 'facts': rows}

    def _facts(self, rows, allowed):
        from .evidence_memory import source_label
        if not isinstance(rows, list):
            raise ValueError('事实字段必须为列表。')
        result = []
        for row in rows:
            if (not isinstance(row, list) or len(row) not in (4, 5)
                    or not all(isinstance(v, str) for v in row) or not all(row[:4])):
                self._reject_reference('retriever', row, ValueError('事实必须包含来源、实体、关系、值以及可选限定。'))
                continue
            row = [source_label(row[0]), *row[1:]]
            if row[0] not in allowed:
                self._reject_reference('retriever', row, ValueError('交接引用了未提供的来源。'))
                continue
            if row not in result:
                result.append(row)
        return result

    def _packet(self):
        return evidence_packet(self.evidence, getattr(self.knowledge, 'source_aliases', []))

    def _reject_reference(self, role, rejected, error):
        """仅在来源关系已齐备时隔离坏引用；未知题型或缺证据仍返回失败。"""
        coverage = self._coverage()
        if not coverage or not coverage['sufficient']:
            raise error
        self.protocol_rejections.append({'role': role, 'rejected': rejected, 'reason': str(error),
                                         'action': 'discard_invalid_reference_keep_received_source_fields',
                                         'coverage': coverage})
        self.metrics['protocol_reference_recoveries'] += 1

    def _support(self, ids, received):
        from .evidence_memory import source_label
        if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
            raise InvalidSupportReference('支持事实必须选择已收到的来源编号。')
        chosen = set(source_label(value) for value in ids)
        if chosen - {row[0] for row in received}:
            raise InvalidSupportReference('交接引用了未提供的来源。')
        # 复制的是已压缩的交接字段，不从知识库重新展开原始记录。
        return [list(row) for row in received if row[0] in chosen]

    def _send(self, role, destination, fields):
        content = encode(fields)
        message = {'from': role, 'to': destination, 'content': content}
        self.messages.append({**message, 'content_bytes': size(content), 'wire_bytes': size(encode(message))})
        return fields

    def _search(self, query):
        started = time.perf_counter()
        focused = getattr(self.knowledge, 'search_graph_focused', None)
        # 首查按实体与所问关系收敛；明确缺口补查时再扩大范围。
        options = {'roots': self.reference_roots} if getattr(self, 'reference_roots', None) else {}
        hits = focused(query, 8, **options) if focused and not self.searches else self.knowledge.search_graph(query, 24)
        self.metrics['knowledge_search_ms'] += (time.perf_counter() - started) * 1000
        self.metrics['knowledge_searches'] += 1
        ids = []
        for hit in hits:
            # 写回范围独立于消息筛选：未用于当前答案的来源事实也不能丢失。
            self.observed.setdefault(hit['context_id'], hit)
            next_id = max((int(key[1:]) for key in self.evidence), default=0) + 1
            label = next((key for key, prior in self.evidence.items()
                          if prior['context_id'] == hit['context_id']), 's' + str(next_id))
            self.evidence.setdefault(label, hit)
            ids.append(label)
        packet = evidence_packet({key: self.evidence[key] for key in ids}, getattr(self.knowledge, 'source_aliases', []))
        self.searches.append({'query': query, 'source_ids': ids, 'backend': 'hypergraph',
                              'tool_result_bytes': size(encode(packet)), 'result': packet})

    def _gap(self, decision):
        missing = decision.get('missing', [])
        if isinstance(missing, str):
            missing = [missing] if missing else []
        if not isinstance(missing, list) or not all(isinstance(v, str) and v.strip() for v in missing):
            raise ValueError('证据缺口必须是属性或关系名称。')
        query = decision.get('query')
        # 可选的空查询有三种常见写法，均表示不补查，不应因此中断正确答案。
        if query is None or query == '' or query == []:
            query = None
        if query is not None and (not isinstance(query, str) or not query.strip()):
            raise ValueError('补查查询必须为非空文本。')
        if query and not missing:
            raise ValueError('没有证据缺口，不得申请补查。')
        if missing and query is None and not self.searches:
            query = self.task + ' Missing: ' + '; '.join(missing)
        seen = {' '.join(s['query'].casefold().split()) for s in self.searches}
        stop = ('complete' if not missing else 'no_followup' if not query else
                'budget_exhausted' if len(self.searches) >= self.max_searches else
                'duplicate_query' if ' '.join(query.casefold().split()) in seen else None)
        return missing, query, stop

    def _coverage(self, evidence=None):
        index = getattr(self.knowledge, 'relation_index', None)
        if index is None or not getattr(self, 'coverage_plan', None):
            return None
        started = time.perf_counter()
        result = index.certificate(self.coverage_plan, [hit['context_id'] for hit in
                                   (self.evidence if evidence is None else evidence).values()])
        self.metrics['coverage_check_ms'] = self.metrics.get('coverage_check_ms', 0.) + (time.perf_counter()-started)*1000
        return result

    def _bind_followup(self, history, scope):
        """只用实际历史和其来源承接省略主语；未被来源支持的历史答案不能成为绑定。"""
        self.reference_roots = None
        self.reference_binding = None
        index = getattr(self.knowledge, 'relation_index', None)
        if index is None or index.anchors(self.task) or not history:
            return
        episodes = [e for e in self.memory.episodes if e['scope'] == scope]
        if not episodes:
            return
        previous = episodes[-1]
        supported_nodes = set()
        for key in previous['evidence_ids']:
            supported_nodes.update(link[1] for link in self.knowledge._graph_links.get(key, ()))
        answer_nodes = index.anchors(previous['answer']) & supported_nodes
        question_nodes = index.anchors(previous['resolved_question']) & supported_nodes
        from hypergraph_memory_method.relation_coverage import requested_facet
        candidates = ((answer_nodes, previous['answer']), (question_nodes, previous['resolved_question']))
        if requested_facet(self.task) not in {'identity', 'performer'}:
            candidates = tuple(reversed(candidates))
        for nodes, text in candidates:
            if not nodes or not index.plan(self.task, nodes)['verifiable']:
                continue
            names = sorted({node.removeprefix('entity:') for node in nodes
                            if node.startswith('entity:') and not node.startswith('entity:owned:')})
            if not names:
                continue
            self.reference_binding = {'previous_turn': previous['turn'], 'source_ids': previous['evidence_ids'],
                                      'entities': names, 'basis': 'actual_history_and_supporting_source_nodes'}
            # 指代约束只限定图的入口；拼进问题会把“谁是歌手”变成询问已知人的职业。
            self.reference_roots = set(nodes)
            self.metrics['memory_reference_bindings_used'] += 1
            return

    def _retain_required_fields(self, facts):
        """保留已取得的必要路径和限定；不会凭空补事实，也不重新读取原文。"""
        index = getattr(self.knowledge, 'relation_index', None)
        if index is None or not self.coverage_plan['verifiable']:
            return facts
        supplied = {row[0]: row for row in self._packet()['facts']}
        ids = {row[0] for row in facts}
        for label, hit in self.evidence.items():
            if hit['context_id'] not in self.coverage_plan['required']:
                continue
            original = supplied[label]
            if label not in ids:
                facts.append(original[:5] if original[4] else original[:4])
                ids.add(label)
                self.metrics['required_fields_restored'] += 1
            elif original[4]:
                row = next(r for r in facts if r[0] == label)
                if len(row) == 4:
                    row.append(original[4])
                elif original[4] not in row[4]:
                    row[4] += '; ' + original[4]
        return facts

    def _retrieve(self, query, incoming):
        while True:
            if query:
                self._search(query)
            # 已传入的记忆字段与新证据合并成一个候选区，不在同次调用中重复放两份。
            payload = {'task': self.task, 'evidence': self._packet()}
            if incoming.get('missing'):
                payload['missing'] = incoming['missing']
            decision = self._call('retriever', payload)
            facts = self._facts(decision.get('facts'), self.evidence)
            facts = self._retain_required_fields(facts)
            coverage = self._coverage()
            if coverage is not None and not coverage['sufficient'] and not decision.get('missing'):
                decision = {**decision, 'missing': [self.coverage_plan['facet'] or 'requested relation'],
                            'query': self.task}
            missing, following, stop = self._gap(decision)
            if not facts and not missing:
                raise ValueError('空证据不能声明任务证据充分。')
            self.retrieval_decisions.append({**decision, 'coverage': coverage,
                                            'stop_reason': stop, 'search_number': len(self.searches)})
            if stop:
                return {'task': self.task, 'facts': facts, **({'missing': missing} if missing else {})}
            query = following

    def run(self, question, *, history_scope='single'):
        from .evidence_memory import source_label
        self.messages, self.outputs, self.searches, self.evidence = [], {}, [], {}
        self.observed = {}
        self.retrieval_decisions, self.reasoning_decisions = [], []
        self.protocol_parsing = []
        self.protocol_rejections = []
        self.coverage_plan = None
        self.reference_binding = None
        self.reference_roots = None
        self.metrics = {'memory_search_ms': 0., 'memory_write_ms': 0., 'knowledge_search_ms': 0.,
                        'knowledge_searches': 0, 'memory_bindings_used': 0, 'memory_evidence_used': 0,
                        'memory_facts_observed': 0, 'memory_facts_added': 0,
                        'memory_only_tasks': 0, 'protocol_parse_ms': 0., 'field_normalization_ms': 0.,
                        'knowledge_directory_ms': 0., 'required_fields_restored': 0,
                        'memory_reference_bindings_used': 0, 'protocol_reference_recoveries': 0}
        self.task = question
        start_call = len(self.client.calls)
        started = time.perf_counter()
        prediction, error, plan, memory_read, support = 'unknown', None, {}, {}, []
        planning_control = {}
        try:
            recall_started = time.perf_counter()
            history, hits, recall_query = self.memory.context(question, history_scope)
            self.metrics['memory_search_ms'] += (time.perf_counter() - recall_started) * 1000
            self.evidence = {'s' + str(i): hit for i, hit in enumerate(hits, 1)}
            memory_read = {'history': history, 'evidence': dict(self.evidence), 'query': recall_query}
            plan = self._call('planner', {
                'question': question, 'recent': [[row['resolved_question'], row['answer']] for row in history],
                'memory_evidence': self._packet(),
            })
            if not isinstance(plan.get('task'), str) or not plan['task'].strip():
                raise ValueError('规划必须包含独立任务。')
            if 'lookup' in plan and not isinstance(plan['lookup'], bool):
                raise ValueError('是否查询知识库必须为布尔值。')
            reuse = plan.get('reuse', [])
            if not isinstance(reuse, list) or not all(isinstance(k, str) for k in reuse):
                raise ValueError('记忆选择必须为来源编号列表。')
            reuse = list(dict.fromkeys(source_label(k) for k in reuse))
            if any(key not in self.evidence for key in reuse):
                raise ValueError('规划引用了未提供的记忆。')
            self.task = plan['task']
            self.evidence = {key: self.evidence[key] for key in reuse}
            relation_index = getattr(self.knowledge, 'relation_index', None)
            if relation_index is not None:
                directory_started = time.perf_counter()
                self._bind_followup(history, history_scope)
                self.coverage_plan = relation_index.plan(self.task, self.reference_roots)
                if self.coverage_plan['verifiable']:
                    required = set(self.coverage_plan['required'])
                    self.evidence = {label: hit for label, hit in self.evidence.items() if hit['context_id'] in required}
                directory_ms = (time.perf_counter() - directory_started) * 1000
                self.metrics['knowledge_directory_ms'] += directory_ms
                self.metrics['knowledge_search_ms'] += directory_ms
                # 目录只给来源编号；正文只能从本会话真实存下的记忆补齐。
                recall_started = time.perf_counter()
                for key in self.coverage_plan['required']:
                    if key in self.memory.scopes.get(history_scope, set()) and not any(
                            h['context_id'] == key for h in self.evidence.values()):
                        label = 's' + str(max((int(k[1:]) for k in memory_read['evidence']), default=0) + 1)
                        self.evidence[label] = {**self.memory.records[key], 'from_memory': True}
                        memory_read['evidence'][label] = self.evidence[label]
                self.metrics['memory_search_ms'] += (time.perf_counter()-recall_started)*1000
            self.metrics['memory_bindings_used'] = len(self.evidence)
            # 模型可能省略否定开关；有已校验的记忆选择则先交检索角色判断缺口。
            # 这不代表证据已被判定充分：检索与推理仍可针对缺口申请查库。
            lookup = plan.get('lookup', not self.evidence) or not self.evidence
            planning_control = {'lookup': lookup, 'defaulted_lookup': 'lookup' not in plan}
            coverage = self._coverage()
            if coverage is not None:
                lookup = not coverage['sufficient']
                planning_control.update(lookup=lookup, model_lookup=plan.get('lookup'), coverage=coverage)
            incoming = {'task': self.task, 'lookup': lookup}
            if self.evidence:
                incoming['evidence'] = self._packet()
            incoming = self._send('planner', 'retriever', incoming)
            retrieved = self._retrieve(self.task if lookup else None, incoming)
            incoming = self._send('retriever', 'reasoner', retrieved)
            for _ in range(self.max_searches + 1):
                decision = self._call('reasoner', incoming)
                try:
                    facts = self._support(decision.get('support', []), incoming['facts'])
                except InvalidSupportReference as exc:
                    self._reject_reference('reasoner', decision.get('support'), exc)
                    facts = []
                facts = self._retain_required_fields(facts)
                answer = decision.get('answer')
                if not ((isinstance(answer, str) and answer.strip()) or type(answer) in (bool, int)
                        or (type(answer) is float and math.isfinite(answer))):
                    raise ValueError('推理结果缺少有效答案候选。')
                missing, query, stop = self._gap(decision)
                self.reasoning_decisions.append({**decision, 'stop_reason': stop})
                outgoing = {'task': self.task, 'answer': decision['answer'], 'facts': facts}
                if missing:
                    outgoing['missing'] = missing
                if stop:
                    support = [self.evidence[row[0]] for row in facts]
                    incoming = self._send('reasoner', 'summarizer', outgoing)
                    break
                recovery = self._send('reasoner', 'retriever', outgoing)
                incoming = self._send('retriever', 'reasoner', self._retrieve(query, recovery))
            else:
                raise RuntimeError('推理补查预算耗尽。')
            prediction = self._call('summarizer', incoming)
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)[:300]

        write_started = time.perf_counter()
        used = list({hit['context_id']: hit for hit in support}.values()) if not error else []
        retained = dict(self.observed)
        retained.update({hit['context_id']: hit for hit in used})
        retained = list(retained.values())
        self.metrics['memory_evidence_used'] = sum(bool(h.get('from_memory')) for h in used)
        self.metrics['memory_facts_observed'] = len(self.observed)
        self.metrics['memory_only_tasks'] = int(bool(used) and not error and not self.searches
                                              and all(hit.get('from_memory') for hit in used))
        if not error:
            added = self.memory.remember(question, prediction, used, history_scope,
                                         resolved_question=self.task, observed_evidence=retained)
        else:
            # 下游格式或接口失败不使已取得的来源失效，但失败答案不写为事实或成功对话。
            added = self.memory.store_evidence(question, retained, history_scope)
        self.metrics['memory_facts_added'] = added
        self.metrics['memory_write_ms'] = (time.perf_counter() - write_started) * 1000
        elapsed = (time.perf_counter() - started) * 1000
        calls = self.client.calls[start_call:]
        role_metrics = {}
        for role in FIELD_ROLES:
            selected = [c for c in calls if c['role'] == role]
            for key in ('input_tokens', 'output_tokens', 'elapsed_ms'):
                role_metrics[role + '_' + key] = sum(c[key] for c in selected)
            role_metrics[role + '_calls'] = len(selected)
        return {'question': question, 'prediction': prediction, 'error': error,
                'implementation': 'native-field-collaboration-v4', 'history_scope': history_scope,
                'protocol_parser_version': PROTOCOL_PARSER_VERSION, 'protocol_parsing': self.protocol_parsing,
                'protocol_rejections': self.protocol_rejections,
                'resolved_question': self.task, 'planning_decision': plan,
                'planning_control': planning_control,
                'memory_read': memory_read, 'memory_written': retained,
                'memory_support_used': used,
                'memory_state': {'fact_count': len(self.memory.scopes.get(history_scope, set())),
                                 'conversation_committed': not error},
                'retrieval_decisions': self.retrieval_decisions, 'reasoning_decisions': self.reasoning_decisions,
                'evidence_coverage': self._coverage(),
                'reference_binding': self.reference_binding,
                'source_evidence': self.evidence, 'calls': calls, 'messages': self.messages,
                'outputs': self.outputs, 'searches': self.searches, 'metrics': {
                    **self.metrics, **role_metrics, 'elapsed_ms': elapsed,
                    'input_tokens': sum(c['input_tokens'] for c in calls),
                    'output_tokens': sum(c['output_tokens'] for c in calls),
                    'model_service_ms': sum(c['elapsed_ms'] for c in calls),
                    'communication_bytes': sum(m['wire_bytes'] for m in self.messages),
                    'message_content_bytes': sum(m['content_bytes'] for m in self.messages),
                    'model_api_request_bytes': sum(c['request_bytes'] for c in calls),
                    'model_api_response_bytes': sum(c['response_bytes'] for c in calls),
                    'tool_result_bytes': sum(s['tool_result_bytes'] for s in self.searches),
                }}
