"""各模型共用的工具决定格式；语法约束不代替证据充分性判断。"""

from __future__ import annotations

import json
from functools import lru_cache


def allowed_actions(role, allow_reuse):
    if role == 'planner':
        # 规划只交付计划；真正的工具调用由检索和推理角色提出。
        return ('deliver',)
    return ('deliver', 'lookup', 'reuse') if allow_reuse else ('deliver', 'lookup')


def work_schema(task_kind, role):
    if task_kind == 'codeact':
        return {'type': 'string'}
    # 工作字段不是再次抄写整份证据的容器。各方式采用相同界限，
    # 在生成过程中合法闭合字段，避免长叙述用光预算后留下半截对象。
    # 来源事实、记忆输入和最终回答不经过这些工作字段上限。
    def strings(count, length):
        return {'type': 'array', 'maxItems': count,
                'items': {'type': 'string', 'maxLength': length}}
    properties = {'entities': strings(8, 64), 'target': {'type': 'string', 'maxLength': 128},
                  'constraints': strings(4, 64),
                  'candidate': {'type': 'string', 'maxLength': 512},
                  'basis': strings(3, 128), 'missing': strings(3, 128)}
    required = {'planner': ['entities', 'target', 'constraints'],
                'retriever': ['basis'], 'reasoner': ['candidate', 'basis']}.get(role, [])
    # 每个角色的字段全部显式定义，也满足在线严格工具接口的 required 约定。
    return {'type': 'object', 'properties': {key: properties[key] for key in required}, 'required': required,
            'additionalProperties': False}


def decision_branch(action, task_kind, role):
    return {'type': 'object', 'properties': {
        'action': {'type': 'string', 'enum': [action]},
        'query': {'type': 'string', 'minLength': 1} if action == 'lookup' else {'type': 'null'},
        'work': work_schema(task_kind, role) if action == 'deliver' else {'type': 'null'},
    }, 'required': ['action', 'query', 'work'], 'additionalProperties': False}


def decision_schema(role, task_kind='continuous', allow_reuse=False):
    return {'anyOf': [decision_branch(a, task_kind, role) for a in allowed_actions(role, allow_reuse)]}


def tool_decision_schema(role='retriever', allow_reuse=False):
    """沿用共用动作语义，只移除已保存在内部状态中的工作正文。"""
    branches = []
    for action in allowed_actions(role, allow_reuse):
        branch = decision_branch(action, 'continuous', role)
        del branch['properties']['work']
        branch['required'].remove('work')
        branches.append(branch)
    return {'anyOf': branches}


def parse_tool_decision(text, *, role='retriever', allow_reuse=False):
    from jsonschema import Draft202012Validator
    try:
        value = json.loads(text)
        if list(Draft202012Validator(tool_decision_schema(role, allow_reuse)).iter_errors(value)):
            raise ValueError('外部工具参数不符合类型。')
        if value['query'] is not None and not value['query'].strip():
            raise ValueError('外部检索查询不能为空。')
        return {'action': value['action'], 'lookup_query': value['query'],
                'content': '', 'decision_error': None}
    except (ValueError, TypeError) as exc:
        return {'action': None, 'lookup_query': None, 'content': '', 'decision_error': str(exc)}


@lru_cache(maxsize=24)
def _validator(role, task_kind, allow_reuse):
    from jsonschema import Draft202012Validator
    return Draft202012Validator(decision_schema(role, task_kind, allow_reuse))


def parse_decision(text, *, role, task_kind='continuous', allow_reuse=False):
    """只执行完整且有效的工具参数；不把自然语言或截断对象猜成动作。"""
    try:
        value = json.loads(text)
        errors = list(_validator(role, task_kind, allow_reuse).iter_errors(value))
        if errors:
            raise ValueError('返回内容不符合角色工具参数类型。')
        if value['action'] == 'lookup' and not value['query'].strip():
            raise ValueError('检索查询不能为空白。')
    except (ValueError, TypeError) as exc:
        return {'action': None, 'lookup_query': None, 'content': text,
                'decision_error': str(exc)}
    work = value['work']
    return {'action': value['action'], 'lookup_query': value['query'],
            'content': work if isinstance(work, str) else (
                json.dumps(work, ensure_ascii=False, separators=(',', ':')) if work is not None else ''),
            'decision_error': None}


def parse_hidden_header(tokenizer, token_ids, *, role, allow_reuse):
    """字段顺序由语法保证；只解码动作和外部查询，不展开中间工作字段。"""
    import re
    decoder = json.JSONDecoder()
    for stop in range(16, len(token_ids) + 16, 16):
        text = tokenizer.decode(token_ids[:stop], skip_special_tokens=True)
        match = re.match(r'\s*\{\s*"action"\s*:\s*', text)
        if not match:
            continue
        try:
            action, offset = decoder.raw_decode(text, match.end())
            match = re.match(r'\s*,\s*"query"\s*:\s*', text[offset:])
            if not match:
                continue
            query, _ = decoder.raw_decode(text, offset + match.end())
        except ValueError:
            continue
        if action not in allowed_actions(role, allow_reuse):
            break
        if action == 'lookup' and (not isinstance(query, str) or not query.strip()):
            break
        if action != 'lookup' and query is not None:
            break
        return {'action': action, 'lookup_query': query, 'content': '', 'decision_error': None}
    return {'action': None, 'lookup_query': None, 'content': '',
            'decision_error': '未取得完整的动作和查询参数。'}


def prepare_decision_tokenizer(bundle):
    """分词器词表索引与模型一同准备，不在每个任务里重建。"""
    from lmformatenforcer import TokenEnforcerTokenizerData
    tokenizer = bundle.tokenizer
    data = getattr(bundle, '_decision_tokenizer_data', None)
    if data is None:
        # 独立适配库的核心接口，避免其旧版 transformers 导入路径。
        # 词首空格的处理遵循 lm-format-enforcer 的公开分词器适配方式。
        zero = tokenizer.encode('0', add_special_tokens=False)[-1]
        prefix = tokenizer.decode([zero], clean_up_tokenization_spaces=False)
        special = set(tokenizer.all_special_ids)
        tokens = []
        for index in range(len(tokenizer)):
            if index in special:
                continue
            after = tokenizer.decode([zero, index], clean_up_tokenization_spaces=False)[len(prefix):]
            alone = tokenizer.decode([index], clean_up_tokenization_spaces=False)
            tokens.append((index, after, len(after) > len(alone)))
        data = TokenEnforcerTokenizerData(tokens,
            lambda ids: tokenizer.decode(ids, clean_up_tokenization_spaces=False).rstrip('\ufffd'),
            tokenizer.eos_token_id, False, len(tokenizer))
        bundle._decision_tokenizer_data = data
    return data


class DecisionGrammar:
    """使用现成的 JSON 语法解析器，通过官方生成接口约束合法续写。"""

    def __init__(self, bundle, input_count, *, role, task_kind, allow_reuse, tool_only=False):
        from lmformatenforcer import JsonSchemaParser, TokenEnforcer
        self.input_count = input_count
        data = prepare_decision_tokenizer(bundle)
        parser = JsonSchemaParser(tool_decision_schema(role, allow_reuse) if tool_only else
                                  decision_schema(role, task_kind, allow_reuse))
        self.enforcer = TokenEnforcer(data, parser)
        parser.config.force_json_field_order = True
        parser.config.max_json_array_length = 1024
        self.reason = None

    def allowed(self, batch_id, ids):
        # 只保存本次生成前缀，不在每个词元的语法状态里复制完整长上下文。
        return self.enforcer.get_allowed_tokens(ids[self.input_count:].tolist()).allowed_tokens

    def __call__(self, sequences, scores, **kwargs):
        values = sequences[0, self.input_count:].tolist()
        self.enforcer.get_allowed_tokens(values)
        state = self.enforcer.prefix_states[tuple(values)]
        if state.parser.can_end():
            self.reason = 'complete_decision_object'
            return True
        return False
