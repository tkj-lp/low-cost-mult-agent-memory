"""只有总结生成词元；内部角色接续缓存，工具直接接收数值表示。"""
from __future__ import annotations

import hashlib
import json
import time

from competition_agents.latent_hidden.posterior_alignment import align_posterior_hidden
from .experimental_state_chat import cache_bytes


COMMON = (
    'Use original source facts to answer the current question. Previous answers may be wrong. '
    'Preserve identities, relations, negation, numbers and qualifications. '
    'Source records contain subject, relation, object, polarity and qualifiers. '
    'All roles continue the same computation; earlier source facts remain available. '
    'Only the final role writes an answer. Internal roles work without writing words.')
ROLES = {
    'planner': 'Resolve references in the current question. Identify the requested relation and constraints internally.',
    'retriever': 'Use the inherited question and constraints to locate the required source relations. Retain evidence internally.',
    'reasoner': 'Derive the answer from original source facts in the current context. Preserve negation and ownership internally.',
    'summarizer': 'Answer the current question using the original source facts. Return only the complete final answer. '
                  'Return unknown if the sources are insufficient.',
}


class LatentOnlyChat:
    """不把上游结果序列化回输入；事实跨题复用仅限同一驻留会话。"""

    def __init__(self, bundle, *, latent_steps=2, max_tokens=1024, event_sink=None,
                 query_mode='legacy', query_adapter=None):
        if latent_steps < 0:
            raise ValueError('潜在步骤必须非负。')
        self.bundle = bundle
        self.latent_steps, self.max_tokens = latent_steps, max_tokens
        self.event_sink = event_sink
        if query_mode not in ('legacy', 'features', 'calibrated'):
            raise ValueError('未知查询表示方式。')
        self.query_mode, self.query_adapter = query_mode, query_adapter
        self.calls = []
        self.scope = None
        self.reset()

    def reset(self):
        self.past = self.last_hidden = self.question_vector = None
        self.question_content = self.question_embedding = None
        self.encoded_facts = {}
        self.seen_ids = []

    def state_bytes(self):
        return cache_bytes(self.past)

    def handoff_size(self):
        """接续所需状态的逻辑快照大小；不宣称发生了网络传输。"""
        tensors = {name: int(value.numel() * value.element_size())
                   for name in ('last_hidden', 'question_vector', 'question_content', 'question_embedding')
                   if (value := getattr(self, name)) is not None}
        metadata = {'scope': self.scope, 'encoded_facts': self.encoded_facts,
                    'seen_ids': self.seen_ids}
        metadata_bytes = len(json.dumps(metadata, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
        cache = self.state_bytes()
        return {'cache_bytes': cache, 'hidden_tensor_bytes': sum(tensors.values()),
                'tensor_components': tensors, 'metadata_bytes': metadata_bytes,
                'payload_bytes': cache + sum(tensors.values()) + metadata_bytes,
                'measurement': '完整接续状态逻辑快照；同进程驻留，未测跨进程序列化及网络传输。'}

    def _forward(self, *, ids=None, embeds=None):
        torch, model = self.bundle.torch, self.bundle.model
        values = ids if ids is not None else embeds
        count = int(values.shape[1])
        old = self.past.get_seq_length() if self.past is not None else 0
        kwargs = {'past_key_values': self.past, 'use_cache': True, 'return_dict': True,
                  'attention_mask': torch.ones((1, old + count), dtype=torch.long, device=values.device),
                  'cache_position': torch.arange(old, old + count, device=values.device)}
        kwargs['input_ids' if ids is not None else 'inputs_embeds'] = values
        started = time.perf_counter()
        output = model.model(**kwargs)
        self.metrics['model_forward_ms'] += (time.perf_counter() - started) * 1000
        self.metrics['forward_positions'] += count
        self.past = output.past_key_values
        self.last_hidden = output.last_hidden_state[:, -1:, :]
        if ids is not None:
            self.seen_ids.extend(ids[0].tolist())
        return output.last_hidden_state

    def _append(self, text, category):
        ids = self.bundle.tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'].to(self.bundle.device)
        self.metrics['input_tokens'] += int(ids.shape[1])
        self.inputs.append({'category': category, 'text': text, 'tokens': int(ids.shape[1])})
        return self._forward(ids=ids)

    def _message(self, role, content, *, open_assistant=False):
        return self.bundle.tokenizer.apply_chat_template(
            [{'role': role, 'content': content}], tokenize=False, add_generation_prompt=open_assistant)

    def _question(self, question):
        rendered = self._message('user', question)
        if self.query_mode == 'legacy':
            self.question_vector = self._append(rendered, 'external_question').mean(dim=1).detach()
            return
        # 同一次分词中保留正文位置，模板词元不混入查询表示。
        encoded = self.bundle.tokenizer(rendered, return_tensors='pt', add_special_tokens=False,
                                        return_offsets_mapping=True)
        ids = encoded['input_ids'].to(self.bundle.device)
        start = rendered.index(question)
        end = start + len(question)
        offsets = encoded['offset_mapping'][0].tolist()
        positions = [i for i, (left, right) in enumerate(offsets) if right > start and left < end and right > left]
        if not positions:
            raise ValueError('未能定位新问题正文，不得静默混用模板表示。')
        self.metrics['input_tokens'] += int(ids.shape[1])
        self.inputs.append({'category': 'external_question', 'text': rendered, 'tokens': int(ids.shape[1]),
                            'content_positions': positions})
        states = self._forward(ids=ids)
        self.question_content = states[:, positions, :].mean(dim=1).detach()
        self.question_embedding = self.bundle.model.get_input_embeddings()(ids[:, positions]).mean(dim=1).detach()

    def _query(self):
        torch = self.bundle.torch
        norm = torch.nn.functional.normalize
        current = self.last_hidden[:, -1, :].float()
        if self.query_mode == 'legacy':
            return norm(.75 * norm(self.question_vector.float(), dim=-1)
                        + .25 * norm(current, dim=-1), dim=-1).detach()
        features = torch.cat([norm(self.question_embedding.float(), dim=-1),
                              norm(self.question_content.float(), dim=-1), norm(current, dim=-1)], dim=-1)
        if self.query_mode == 'features':
            return features.detach()
        if self.query_adapter is None:
            raise ValueError('校准查询缺少已拟合的小映射。')
        return self.query_adapter(features).detach()

    def _work(self):
        for _ in range(self.latent_steps):
            start = time.perf_counter()
            vector, _ = align_posterior_hidden(self.bundle.model, self.last_hidden)
            self.metrics['alignment_ms'] += (time.perf_counter() - start) * 1000
            self.metrics['latent_vector_bytes'] += vector.numel() * vector.element_size()
            self._forward(embeds=vector)
            self.metrics['latent_steps'] += 1

    def add_facts(self, facts):
        """仅接受外部来源事实；相同版本已编码时不再追加任何文本。"""
        added, reused = [], []
        for fact in facts:
            body = json.dumps(fact, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
            signature = hashlib.sha256(body.encode()).hexdigest()
            key = fact['id']
            if self.encoded_facts.get(key) == signature:
                reused.append(key)
                continue
            added.append(fact)
            self.encoded_facts[key] = signature
        if added:
            self._append(self._message('user', 'New original source facts:\n' + json.dumps(
                added, ensure_ascii=False, separators=(',', ':'))), 'external_facts')
        self.metrics['facts_encoded'] += len(added)
        self.metrics['facts_reused_without_encoding'] += len(reused)
        return {'encoded': [f['id'] for f in added], 'reused': reused}

    def _decode_final(self):
        torch, model, tokenizer = self.bundle.torch, self.bundle.model, self.bundle.tokenizer
        eos = model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        penalty = float(model.generation_config.repetition_penalty or 1.)
        generated = []
        for _ in range(self.max_tokens):
            logits = model.lm_head(self.last_hidden[:, -1, :]).squeeze(0).float()
            if penalty != 1. and self.seen_ids:
                positions = torch.tensor(sorted(set(self.seen_ids)), device=logits.device)
                scores = logits[positions]
                logits[positions] = torch.where(scores < 0, scores * penalty, scores / penalty)
            token = int(logits.argmax().item())
            generated.append(token)
            self._forward(ids=torch.tensor([[token]], device=self.bundle.device))
            if token in eos_ids:
                break
        self.metrics['output_tokens'] = len(generated)
        self.metrics['finish_reason'] = 'stop' if generated and generated[-1] in eos_ids else 'length'
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def call(self, role, *, question=None, scope=None, tool=None):
        if role not in ROLES:
            raise ValueError('未知角色。')
        if role == 'planner':
            if not isinstance(question, str) or not question.strip():
                raise ValueError('新任务必须有外部问题。')
            if self.scope != scope:
                self.reset()
                self.scope = scope
        elif question is not None:
            raise ValueError('已处理的问题不得重新输入后续角色。')
        if role != 'planner' and self.past is None:
            raise ValueError('后续角色必须接续已有计算。')
        self.inputs = []
        self.metrics = {key: 0 for key in ('input_tokens', 'output_tokens', 'forward_positions',
                        'model_forward_ms', 'alignment_ms', 'latent_steps', 'latent_vector_bytes',
                        'facts_encoded', 'facts_reused_without_encoding')}
        before = self.state_bytes()
        old_positions = self.past.get_seq_length() if self.past is not None else 0
        started = time.perf_counter()
        if self.event_sink:
            self.event_sink({'role': role, 'event': 'start'})
        answer, tool_result = None, None
        layers = [0] * len(self.bundle.model.model.layers)
        hooks = []
        for index, layer in enumerate(self.bundle.model.model.layers):
            def count(_module, args, kwargs, index=index):
                states = kwargs.get('hidden_states', args[0] if args else None)
                layers[index] += int(states.shape[1])
            hooks.append(layer.register_forward_pre_hook(count, with_kwargs=True))
        try:
            with self.bundle.torch.inference_mode():
                if self.past is None:
                    self._append(self._message('system', COMMON), 'instruction')
                if role == 'planner':
                    self._question(question)
                self._append(self._message('system', ROLES[role], open_assistant=True), 'instruction')
                if role == 'summarizer':
                    answer = self._decode_final()
                else:
                    self._work()
                    self._append(self.bundle.tokenizer.eos_token, 'boundary')
                    if tool is not None:
                        # 已编码问题和继承的当前角色表示合成数值查询；不生成检索词或字段文字。
                        query = self._query()
                        tool_result = tool(query.detach())
                        self.add_facts(tool_result['facts'])
        finally:
            for hook in hooks:
                hook.remove()
        record = {'role': role, **self.metrics, 'elapsed_ms': (time.perf_counter() - started) * 1000,
                  'inputs': self.inputs, 'raw_response_content': answer or '',
                  'cached_prefix_positions': old_positions, 'layer_positions': layers,
                  'state_before_bytes': before, 'state_after_bytes': self.state_bytes(),
                  'tool': tool_result['audit'] if tool_result else None}
        self.calls.append(record)
        if role != 'summarizer':
            record['handoff'] = self.handoff_size()
        if self.event_sink:
            self.event_sink({'role': role, 'event': 'done', 'seconds': record['elapsed_ms']/1000,
                             'output_tokens': record['output_tokens'], 'answer': answer,
                             'handoff': record.get('handoff')})
        return answer
