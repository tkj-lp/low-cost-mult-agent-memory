"""固定隐状态入口：已校准查询、事实与计算复用、独立经验库。"""
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4

from competition_agents.latent_hidden.calibrated_query import QueryReadout
from competition_agents.latent_hidden.experienced_graph_tool import ExperiencedGraphTool
from competition_agents.llm.latent_only_chat import LatentOnlyChat
from hypergraph_memory_method.prepared_knowledge import PreparedKnowledge
from hypergraph_memory_method.retrieval_experience import RetrievalExperienceMemory


class LatentCollaboration:
    def __init__(self, client, resources):
        import torch
        cache = Path(resources['calibration'])
        manifest = json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
        knowledge = PreparedKnowledge(resources['knowledge'])
        if client.model != manifest['config']['model'] or knowledge.manifest['files'] != manifest['config']['knowledge_files']:
            raise ValueError('本地模型或知识库与离线隐状态映射不匹配。')
        base = Path(__file__).resolve().parents[2]
        for name in ('competition_agents/llm/latent_only_chat.py',
                     'competition_agents/latent_hidden/calibrated_query.py',
                     'competition_agents/latent_hidden/posterior_alignment.py'):
            expected = manifest['config']['implementation_hashes'][name]
            if hashlib.sha256((base/name).read_bytes()).hexdigest() != expected:
                raise ValueError('隐状态特征代码与已有校准不匹配：' + name)
        readout = QueryReadout.load(cache/'readout.pt', client.bundle.device)
        validation = torch.load(cache/'validation_vectors.pt', map_location=client.bundle.device, weights_only=True)
        identity = json.dumps({'knowledge': knowledge.manifest['files'], 'model': client.model,
                              'readout': hashlib.sha256((cache/'readout.pt').read_bytes()).hexdigest(),
                              'method': 'locked-latent-experience-v1'}, sort_keys=True)
        self.memory = RetrievalExperienceMemory(resources['experience_db'], hashlib.sha256(identity.encode()).hexdigest())
        self.tool = ExperiencedGraphTool(knowledge, client.bundle, resources['encoder'], self.memory)
        self.gate = self.tool.calibrate_gate(manifest['validation_samples'], validation)
        self.tool.teacher = None
        self.client = LatentOnlyChat(client.bundle, latent_steps=manifest['config']['latent_steps'],
                                    max_tokens=client.max_tokens, query_mode='calibrated', query_adapter=readout,
                                    event_sink=resources.get('latent_event_sink'))
        self.run_id = uuid4().hex

    def run(self, question, *, history_scope='single'):
        started, begin, event_begin = time.perf_counter(), len(self.client.calls), len(self.tool.events)
        prediction, error = '', None
        try:
            self.tool.begin_question(question, history_scope)
            self.client.call('planner', question=question, scope=history_scope)
            self.client.call('retriever', tool=self.tool)
            self.client.call('reasoner')
            prediction = self.client.call('summarizer')
            if self.client.calls[-1]['finish_reason'] != 'stop':
                raise ValueError('最终输出达到长度上限。')
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        if len(self.tool.events) > event_begin:
            self.tool.finish_task(self.run_id + ':' + str(begin), completed=error is None)
        calls = self.client.calls[begin:]
        audit = self.tool.events[-1] if len(self.tool.events) > event_begin else {}
        def facts(keys):
            return [{**self.tool.records[key], 'context_id': key,
                     'text': ' | '.join(str(self.tool.records[key].get(name, '')) for name in ('subject', 'relation', 'object'))}
                    for key in keys if key in self.tool.records]
        metrics = {
            'elapsed_ms': (time.perf_counter()-started)*1000,
            'input_tokens': sum(c['input_tokens'] for c in calls),
            'output_tokens': sum(c['output_tokens'] for c in calls),
            'model_service_ms': sum(c['elapsed_ms'] for c in calls),
            'communication_bytes': None, 'message_content_bytes': 0,
            'handoff_payload_bytes': sum(c.get('handoff', {}).get('payload_bytes', 0) for c in calls),
            'handoff_measurement': 'state_snapshot',
            'state_peak_bytes': max((c['state_after_bytes'] for c in calls), default=0),
            'state_reuse_count': sum(c['role'] in ('retriever', 'reasoner', 'summarizer') for c in calls),
            'forward_positions': sum(c['forward_positions'] for c in calls),
            'facts_encoded': sum(c['facts_encoded'] for c in calls),
            'facts_reused_without_encoding': sum(c['facts_reused_without_encoding'] for c in calls),
            'knowledge_search_ms': audit.get('index_search_ms', 0),
            'knowledge_searches': audit.get('knowledge_searches', 0),
            'memory_search_ms': audit.get('memory_search_ms', 0),
            'memory_write_ms': audit.get('experience_write_ms', 0),
            **{key: audit.get(key, 0) for key in ('experience_reads', 'experience_applied', 'experience_writes')},
        }
        return {'question': question, 'prediction': prediction, 'error': error,
                'history_scope': history_scope, 'implementation': 'locked-latent-experience-v1',
                'calls': calls, 'messages': [], 'metrics': metrics, 'retrieval_audit': audit,
                'memory_read': facts(audit.get('memory_source_ids', [])),
                'memory_written': facts(audit.get('fresh_source_ids', [])),
                'experience_written': self.memory.records.get(audit.get('experience_written_id')),
                'outputs': {'summarizer': prediction},
                'communication_note': '同进程共享计算缓存；未测跨机器状态传输，缓存峰值不作为传输量。'}


def create(client, resources):
    return LatentCollaboration(client, resources)
