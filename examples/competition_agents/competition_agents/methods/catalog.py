"""前后端共用的方法清单；不加载任何模型或协作实现。"""
import importlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
CATALOG = json.loads((BASE/'config/methods.json').read_text(encoding='utf-8'))
METHODS = {item['value']: item for item in CATALOG['methods']}
DATASETS = {item['value']: item for item in CATALOG['datasets']}


def validate(config):
    method = METHODS.get(config.get('collaborationMode'))
    if method is None:
        raise ValueError('只支持纯文本、协议字段、隐状态、AutoForm 和 LLMLingua-2。')
    if config.get('retrievalMode') != method['retrieval'] or config.get('memoryMode') != method['memory']:
        raise ValueError('协作方案与记忆检索固定搭配，当前选择不匹配。')
    if config.get('backend') not in method['backends']:
        raise ValueError('隐状态需要本地模型的内部表示，不能使用在线文字接口。')
    if config.get('dataset') not in DATASETS:
        raise ValueError('所选数据集没有绑定已准备好的知识库。')
    dataset = DATASETS[config['dataset']]
    for key, default in (('maxTokens', 1024), ('maxSearches', 2)):
        if type(config.get(key, default)) is not int or config.get(key, default) < 1:
            raise ValueError('生成和检索预算必须为正整数。')
    if type(config.get('memoryEnabled', True)) is not bool:
        raise ValueError('记忆开关必须为布尔值。')
    steps = config.get('latentSteps', CATALOG.get('defaultLatentSteps', 40))
    if type(steps) is not int or steps < 1:
        raise ValueError('隐式推理步数必须为正整数。')
    if config.get('latentMemoryPolicy', CATALOG.get('defaultLatentMemoryPolicy', 'release_transients')) not in ('full', 'release_transients'):
        raise ValueError('未知隐状态生命周期策略。')
    repeat = config.get('outputRepeatNgramSize', 0)
    if type(repeat) is not int or (repeat != 0 and repeat < 2):
        raise ValueError('输出重复保护长度应为零或至少二。')
    if repeat and config['backend'] != 'local':
        raise ValueError('输出重复保护仅支持本地后端。')
    if dataset.get('methods') and method['value'] not in dataset['methods']:
        raise ValueError('所选数据集尚未支持该协作方式。')
    limit = config.get('taskCount')
    if type(limit) is not int or not 1 <= limit <= DATASETS[config['dataset']]['taskCount']:
        raise ValueError(f"该数据集支持一至{dataset['taskCount']}题，必须从第一题按顺序执行。")
    return method


def create_runtime(config, client, resources):
    method = validate(config)
    if DATASETS[config['dataset']]['datasetType'] == 'codeact':
        if method['value'] == 'latent':
            from .latent_codeact import LatentCodeActCollaboration
            return LatentCodeActCollaboration(client, resources)
        from .codeact import CodeActCollaboration
        return CodeActCollaboration(client, resources, structured=method['value'] == 'structured', method=method['value'])
    module, function = method['entry'].split(':')
    return getattr(importlib.import_module(module), function)(client, resources)
