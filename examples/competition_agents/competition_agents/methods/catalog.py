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
    if method.get('models') and config.get('localModel') not in method['models']:
        raise ValueError('当前隐状态映射只校准了本地千问十五亿参数模型；其他模型需要先离线校准。')
    if config.get('dataset') not in DATASETS:
        raise ValueError('所选数据集没有绑定已准备好的知识库。')
    dataset = DATASETS[config['dataset']]
    if dataset.get('methods') and method['value'] not in dataset['methods']:
        raise ValueError('代码执行目前支持纯文本和协议字段；隐状态的代码工具边界尚未适配。')
    limit = config.get('taskCount')
    if type(limit) is not int or not 1 <= limit <= DATASETS[config['dataset']]['taskCount']:
        raise ValueError(f"该数据集支持一至{dataset['taskCount']}题，必须从第一题按顺序执行。")
    return method


def create_runtime(config, client, resources):
    method = validate(config)
    if DATASETS[config['dataset']]['datasetType'] == 'codeact':
        from .codeact import CodeActCollaboration
        return CodeActCollaboration(client, resources, structured=method['value'] == 'structured')
    module, function = method['entry'].split(':')
    return getattr(importlib.import_module(module), function)(client, resources)
