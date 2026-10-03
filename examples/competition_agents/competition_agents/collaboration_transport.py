"""表示与交接；不判断证据充分性，不筛选模型已经召回的关系。"""

from __future__ import annotations

import copy
import json
import time

from .collaboration_fields import has_reference_plan, projected_work


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def full_evidence_packet(evidence, aliases=()):
    """完整关系与来源逐项对应；别名仅传身份和定位，审计引用留在知识库。"""
    sources, facts = {}, []
    for key, hit in evidence.items():
        fact = hit.get('fact_fields')
        if not fact:
            facts.append([key, hit.get('title', ''), 'original_text', hit.get('text', ''),
                          'positive', [], key, []])
            sources[key] = hit.get('title', '')
            continue
        document = fact['document_id']
        sources[document] = hit['title']
        facts.append([key, fact['subject'], fact['relation'], fact['object'],
                      fact['polarity'], copy.deepcopy(fact['qualifiers']), document,
                      list(fact.get('sentences', []))])
    return {'columns': ['id', 'subject', 'relation', 'object', 'polarity', 'qualifiers', 'document', 'source_sentences'],
            'sources': sources, 'facts': facts,
            'aliases': [{key: copy.deepcopy(a[key]) for key in (
                'alias', 'name', 'document_id', 'sentences', 'scoped') if key in a}
                for a in aliases if a['document_id'] in sources]}


def projected_history(history, identifiers):
    """仅替换历史的证据指针，问题、答案和其他字段原样保留。"""
    return [{**copy.deepcopy(row), 'evidence_ids': [identifiers.get(key, key)
             for key in row.get('evidence_ids', [])]} if 'evidence_ids' in row else copy.deepcopy(row)
            for row in history]


class CollaborationTransport:
    """任务和原始证据完整传递；压缩对照只压缩角色工作说明。"""

    def __init__(self, method, client, knowledge):
        self.method, self.client, self.knowledge = method, client, knowledge
        self.messages = []
        self.identifiers = {}
        if method == 'llmlingua':
            from .methods.llmlingua import prepare_compressor
            prepare_compressor()

    def packet(self, snapshot):
        packet = full_evidence_packet(snapshot['evidence'], getattr(self.knowledge, 'source_aliases', []))
        for row in snapshot['history']:
            for key in row.get('evidence_ids', []):
                self._identifier(key)
        packet['sources'] = {self._identifier(key): title for key, title in packet['sources'].items()}
        for row in packet['facts']:
            row[0], row[6] = self._identifier(row[0]), self._identifier(row[6])
        for alias in packet['aliases']:
            alias['document_id'] = self._identifier(alias['document_id'])
        return packet

    def _identifier(self, original):
        # 原编号在审计、存储和索引中不变。短号是稳定的一一映射，不是删掉来源。
        if original not in self.identifiers:
            self.identifiers[original] = 'E' + str(len(self.identifiers) + 1)
        return self.identifiers[original]

    def history(self, snapshot):
        for row in snapshot['history']:
            for key in row.get('evidence_ids', []):
                self._identifier(key)
        return projected_history(snapshot['history'], self.identifiers)

    def render(self, snapshot, receiver=None):
        if self.method in ('structured', 'latent'):
            packet = self.packet(snapshot)
            payload = {'task': snapshot['task'], 'evidence': packet}
            if receiver is None:
                # 兼容审计入口；正式调用必须传接收角色。
                payload.update(conversation=self.history(snapshot), agent_work=snapshot['work'])
            else:
                payload['role_work'] = projected_work(snapshot['work'], receiver)
                if receiver == 'planner' or not has_reference_plan(snapshot['work']):
                    # 历史用于规划指代；未知格式保留完整回退，不猜测删掉哪些答案。
                    payload['conversation'] = [
                        {key: copy.deepcopy(row[key]) for key in ('question', 'answer', 'turn', 'verified') if key in row}
                        for row in snapshot['history']]
                    if receiver != 'planner' and snapshot['history']:
                        payload['history_fallback'] = 'reference_plan_unavailable'
            if snapshot.get('experience'):
                payload['retrieval_experience'] = snapshot['experience']['relation_paths']
            if snapshot.get('execution') is not None:
                payload['actual_execution'] = snapshot['execution']
            if snapshot['diagnostics']:
                payload['workflow_status'] = snapshot['diagnostics']
            return encode(payload)
        text = 'CURRENT TASK:\n' + encode(snapshot['task'])
        text += '\n\nConversation (answers unverified):\n' + encode(self.history(snapshot))
        if snapshot.get('remembered_text'):
            text += '\n\nRemembered complete original turn text:\n' + snapshot['remembered_text']
        # 本轮所有原文原样保留，不预先定位、裁剪答案段落。
        for hit in snapshot['evidence'].values():
            if not hit.get('from_memory'):
                text += '\n\nOriginal source: ' + hit['title'] + '\n' + hit['text']
        for row in snapshot['work']:
            text += '\n\n' + row['role'] + ' work (unverified, action=' + str(row.get('action')) + '):\n' + row['content']
            if row.get('lookup_query'):
                text += '\nRequested knowledge query:\n' + row['lookup_query']
        if snapshot.get('execution') is not None:
            text += '\n\nActual execution:\n' + encode(snapshot['execution'])
        if snapshot['diagnostics']:
            text += '\n\nWorkflow status:\n' + encode(snapshot['diagnostics'])
        return text

    def handoff(self, sender, receiver, snapshot):
        if self.method == 'latent':
            return
        conversion = {'converted': False, 'conversion_ms': 0.}
        if self.method in ('autoform', 'llmlingua') and snapshot['work']:
            work = snapshot['work'][-1]
            # 对照方法可能有损，因此不让它们改写任务和原始关系证据。
            started, begin = time.perf_counter(), len(self.client.calls)
            instruction = None
            try:
                if not work['content'].strip():
                    conversion['skipped_empty_work'] = True
                else:
                    if self.method == 'autoform':
                        from baselines.autoform.autoform_adapter import build_autoform_instruction
                        instruction = build_autoform_instruction(sender=sender, receiver=receiver,
                            action='handoff', source_message=work['content'])
                        response = self.client.call('autoform', [{'role': 'user', 'content': instruction}])
                        converted = response.get('content')
                    else:
                        from baselines.llmlingua2.llmlingua2_adapter import compress_message
                        result = compress_message(work['content'], sender=sender, receiver=receiver, action='handoff')
                        if result.provider != 'llmlingua2':
                            raise RuntimeError('官方消息压缩模型不可用。')
                        converted = result.payload
                    if not isinstance(converted, str) or not converted.strip():
                        raise ValueError('转换器没有返回可交接的内容。')
                    snapshot['work'][-1] = {**work, 'content': converted}
                    conversion['converted'] = True
            except Exception as exc:
                conversion['fallback_to_original_work'] = True
                conversion['error'] = str(exc)[:500]
                snapshot['diagnostics'].append({'role': sender, 'kind': 'transport_error',
                                                'message': conversion['error']})
                if instruction is not None and len(self.client.calls) == begin:
                    self.client.calls.append({'role': 'autoform', 'model': self.client.model,
                        'messages': [{'role': 'user', 'content': instruction}],
                        'input_tokens': None, 'output_tokens': None,
                        'elapsed_ms': (time.perf_counter()-started)*1000, 'finish_reason': 'error',
                        'error': conversion['error'], 'response': {'content': ''}})
            finally:
                conversion['conversion_ms'] = (time.perf_counter()-started)*1000
        content = self.render(snapshot, receiver=receiver)
        envelope = {'from': sender, 'to': receiver, 'content': content}
        self.messages.append({**envelope, **conversion, 'kind': 'agent_handoff',
                              'content_bytes': len(content.encode('utf-8')),
                              'wire_bytes': len(encode(envelope).encode('utf-8'))})
