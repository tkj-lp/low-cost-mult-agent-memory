"""用预备知识及通用问法校准检索读出；不读取评测任务或答案。"""
from __future__ import annotations

import copy
from collections import Counter
import hashlib
import json
import time

import torch

from competition_agents.llm.latent_only_chat import LatentOnlyChat


# 通用属性目录同时覆盖当前知识库没有答案的属性，不为三题选择题型。
INTENTS = [
    ('color', 'color of something', 'What is the color of {name}?', 'Which color does {name} have?'),
    ('location', 'where someone lives place residence location', 'Where does {name} live?', 'In which place does {name} reside?'),
    ('companions', 'live alone or share a home with others', 'Does {name} live with anyone?', 'Does {name} share a home with others?'),
    ('time', 'date time when an event occurred', 'When did this happen to {name}?', 'At what time did this occur for {name}?'),
    ('identity', 'identity type kind of person or thing', 'What kind of thing is {name}?', 'What is the identity of {name}?'),
    ('quantity', 'number count how many', 'How many are associated with {name}?', 'What is the number associated with {name}?'),
    ('owner', 'ownership belonging owner', 'Who owns {name}?', 'To whom does {name} belong?'),
    ('family', 'family mother father sisters siblings relationship', 'Who are the family members of {name}?', 'Who is related to {name}?'),
    ('cause', 'cause reason why', 'Why did this happen to {name}?', 'What caused this for {name}?'),
    ('action', 'action activity did', 'What did {name} do?', 'Which action did {name} take?'),
    ('appearance', 'appearance description physical characteristics', 'How is the appearance of {name} described?', 'What does {name} look like?'),
    ('speech', 'words said statement', 'What did {name} say?', 'Which words did {name} speak?'),
]


def calibration_samples(knowledge, max_source_samples=64):
    counts = Counter(row[3]['subject'] for row in knowledge._graph_rows.values())
    names = [name for name, _ in sorted(counts.items(), key=lambda p: (-p[1], p[0]))[:4]]
    train, validation = [], []
    for name in names:
        for category, target, first, second in INTENTS:
            train.append({'question': first.format(name=name), 'target': target, 'category': category,
                          'entity': name, 'kind': 'general_attribute'})
            if name in names[:2]:
                validation.append({'question': second.format(name=name), 'target': target, 'category': category,
                                   'entity': name, 'kind': 'held_out_wording'})
    # 按关系轮转取固定数量的来源样本；不让文档变长导致校准成本无上限增长。
    # 排序仅依据来源关系与标识，不读取评测题目或答案。
    groups = {}
    for key, row in sorted(knowledge._graph_rows.items()):
        groups.setdefault(row[3]['relation'].casefold(), []).append((key, row))
    selected = []
    while groups and len(selected) < max_source_samples:
        for relation in sorted(list(groups)):
            selected.append(groups[relation].pop(0))
            if not groups[relation]:
                del groups[relation]
            if len(selected) >= max_source_samples:
                break
    for key, row in selected:
        fact = row[3]
        target = ' '.join([fact['relation'], fact['object'], *[q['value'] for q in fact['qualifiers']]])
        train.append({'question': 'Find the source statement about ' + fact['subject'] + ': ' + target + '.',
                      'target': target, 'category': None, 'entity': fact['subject'], 'kind': 'source_description',
                      'source_id': key})
    return train, validation


class QueryReadout:
    def __init__(self, mean, weight, bias, report):
        self.mean, self.weight, self.bias, self.report = mean, weight, bias, report

    def __call__(self, features):
        if features.shape[-1] != self.mean.shape[-1]:
            raise ValueError('查询表示维度与校准版本不一致。')
        return torch.nn.functional.normalize((features.float()-self.mean) @ self.weight + self.bias, dim=-1)

    def save(self, path):
        torch.save({'mean': self.mean.cpu(), 'weight': self.weight.cpu(), 'bias': self.bias.cpu(),
                    'report': self.report}, path)

    @classmethod
    def load(cls, path, device):
        data = torch.load(path, map_location=device, weights_only=True)
        return cls(data['mean'], data['weight'], data['bias'], data['report'])


def fit_ridge_readout(x, targets, validation_x, validation_targets):
    mean, bias = x.mean(dim=0, keepdim=True), targets.mean(dim=0, keepdim=True)
    centered, y = x-mean, targets-bias
    trials = []
    best = None
    for strength in (.01, .1, 1., 10.):
        gram = centered @ centered.T
        solution = torch.linalg.solve(gram + strength*torch.eye(len(x), device=x.device), y)
        weight = centered.T @ solution
        predicted = torch.nn.functional.normalize((validation_x-mean) @ weight + bias, dim=-1)
        score = float((predicted*validation_targets).sum(dim=-1).mean())
        trials.append({'strength': strength, 'held_out_mean_cosine': score})
        if best is None or score > best[0]:
            best = (score, strength, weight)
    report = {'regularization': best[1], 'trials': trials,
              'train_samples': len(x), 'validation_samples': len(validation_x),
              'trainable_parameters': int(best[2].numel()+bias.numel()),
              'base_model_training': False}
    return QueryReadout(mean, best[2], bias, report)


def calibrate(knowledge, bundle, teacher, *, latent_steps=2, progress=None):
    train, validation = calibration_samples(knowledge)
    samples = train+validation
    outputs = []
    started = time.perf_counter()
    # 上文来源不同于当前问题；让校准同时看到冷启动和有来源缓存的情形。
    prefix_client = LatentOnlyChat(bundle, latent_steps=latent_steps, query_mode='features')
    prefix_client.call('planner', question='Prepare to consult the original sources.', scope='calibration')
    seed_facts = []
    for i, (_key, row) in enumerate(sorted(knowledge._graph_rows.items())[:8]):
        fact = row[3]
        seed_facts.append({'id': 'seed'+str(i), 'subject': fact['subject'], 'relation': fact['relation'],
                           'object': fact['object'], 'polarity': fact['polarity']})
    prefix_client.call('retriever', tool=lambda _q: {'facts': seed_facts, 'audit': {}})
    for index, sample in enumerate(samples):
        client = LatentOnlyChat(bundle, latent_steps=latent_steps, query_mode='features')
        if index % 2:
            client.scope = 'calibration'
            client.past = copy.deepcopy(prefix_client.past)
            client.last_hidden = prefix_client.last_hidden.clone()
            client.seen_ids = list(prefix_client.seen_ids)
        client.call('planner', question=sample['question'], scope='calibration')
        captured = []
        def capture(query):
            captured.append(query.clone())
            return {'facts': [], 'audit': {}}
        client.call('retriever', tool=capture)
        outputs.append(captured[0].squeeze(0))
        if progress and (index % 8 == 0 or index+1 == len(samples)):
            progress({'phase': 'calibration', 'completed': index+1, 'total': len(samples),
                      'seconds': time.perf_counter()-started})
    features = torch.stack(outputs)
    targets = torch.as_tensor(teacher.encode([s['target'] for s in samples], normalize_embeddings=True),
                              dtype=torch.float32, device=bundle.device)
    with torch.inference_mode(False):
        x = features[:len(train)].clone()
        vx = features[len(train):].clone()
        readout = fit_ridge_readout(x, targets[:len(train)].clone(), vx, targets[len(train):].clone())
    readout.report.update({'feature_parts': ['question_input_embedding', 'question_content_last_layer', 'current_role_last_layer'],
                           'corpus_sha256': knowledge.manifest['corpus_sha256'],
                           'sample_sha256': hashlib.sha256(json.dumps(samples,sort_keys=True).encode()).hexdigest(),
                           'teacher': 'sentence-transformers/all-MiniLM-L6-v2',
                           'samples': samples, 'calibration_seconds': time.perf_counter()-started})
    return readout, validation, readout(features[len(train):]).detach()
