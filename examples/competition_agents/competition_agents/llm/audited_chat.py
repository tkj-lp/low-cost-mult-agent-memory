"""记录真实提示、工具调用、服务端词元与网络耗时的兼容接口客户端。"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from hypergraph_memory_method.prepared_knowledge import digest


class AuditedChat:
    def __init__(self, model, *, max_tokens=1024, seed=42):
        self.model, self.max_tokens, self.seed = model, max_tokens, seed
        self.endpoint = os.environ['LLM_BASE_URL'].rstrip('/')
        self.key = os.environ['LLM_API_KEY']
        self.calls = []

    def call(self, role, messages, *, tools=None, tool_choice=None, role_control=False,
             shared_roles=False):
        from .response_content import final_content, model_messages, normalize_lookup_query
        effective = model_messages(self.model, messages, shared_roles=shared_roles)
        payload = {'model': self.model, 'messages': effective, 'temperature': 0,
                   'max_tokens': self.max_tokens, 'seed': self.seed, 'enable_thinking': False}
        if role_control:
            from .decision_contract import allowed_actions, decision_branch
            if tools is not None:
                raise ValueError('角色控制接口不能与另一组工具定义混用。')
            tools = []
            allow_reuse = role_control.get('allow_reuse', False) if isinstance(role_control, dict) else True
            task_kind = role_control.get('task_kind', 'continuous') if isinstance(role_control, dict) else 'continuous'
            descriptions = {'deliver': 'Hand off new role work using available evidence.',
                            'lookup': 'Request knowledge because a needed fact or relation is missing.',
                            'reuse': 'Adopt preceding work unchanged after checking the evidence.'}
            for name in allowed_actions(role, allow_reuse):
                schema = decision_branch(name, task_kind, role)
                properties = {k: v for k, v in schema['properties'].items() if k != 'action'}
                tools.append({'type': 'function', 'function': {
                    'name': name, 'description': descriptions[name], 'strict': True,
                    'parameters': {'type': 'object', 'properties': properties,
                                   'required': ['query', 'work'], 'additionalProperties': False}}})
            tool_choice = 'required'
        if tools:
            payload['tools'] = tools
            payload['parallel_tool_calls'] = False
            if tool_choice:
                payload['tool_choice'] = tool_choice
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        started = time.perf_counter()
        for attempt in range(3):
            request = urllib.request.Request(self.endpoint + '/chat/completions', data=data,
                headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.key})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    wire = response.read()
                    body = json.loads(wire)
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise RuntimeError(f'模型服务请求失败，状态码 {exc.code}') from exc
                time.sleep(2 ** attempt)
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        message = body['choices'][0]['message']
        usage = body.get('usage') or {}
        measured = lambda value: value if type(value) is int and value >= 0 else None
        input_tokens, output_tokens = measured(usage.get('prompt_tokens')), measured(usage.get('completion_tokens'))
        # 统计缺失不能删掉成功生成的内容；未知用量不写成零，也不伪造估计。
        self.calls.append({'role': role, 'model': body.get('model', self.model),
            'messages': json.loads(json.dumps(messages)), 'effective_messages': json.loads(json.dumps(effective)),
            'response': message, 'prompt_sha256': digest(payload),
            'input_tokens': input_tokens, 'output_tokens': output_tokens,
            'token_usage_complete': input_tokens is not None and output_tokens is not None,
            'cached_input_tokens': measured((usage.get('prompt_tokens_details') or {}).get('cached_tokens')),
            'reasoning_tokens': measured((usage.get('completion_tokens_details') or {}).get('reasoning_tokens')),
            'elapsed_ms': (time.perf_counter() - started) * 1000, 'attempts': attempt + 1,
            'request_bytes': len(data), 'response_bytes': len(wire),
            'finish_reason': body['choices'][0].get('finish_reason'), 'usage': usage,
            'shared_roles': shared_roles})
        # 服务端附加推理内容不回填后续调用；语义输出和工具调用原样交接。
        result = {k: message[k] for k in ('role', 'content', 'tool_calls') if k in message}
        result['finish_reason'] = body['choices'][0].get('finish_reason')
        if role_control:
            from .decision_contract import parse_decision
            tool_calls = message.get('tool_calls') or []
            decision = None
            if len(tool_calls) == 1:
                function = tool_calls[0].get('function') or {}
                try:
                    raw = function.get('arguments', '')
                    arguments = json.loads(raw) if isinstance(raw, str) else raw
                    if not isinstance(arguments, dict) or 'action' in arguments:
                        raise ValueError('工具参数不是约定对象。')
                    decision = parse_decision(json.dumps({'action': function.get('name'), **arguments}),
                                              role=role, task_kind=task_kind, allow_reuse=allow_reuse)
                except (ValueError, TypeError):
                    pass
            if decision is None or decision['decision_error']:
                # 保留原返回，但绝不把非法参数或多个工具请求猜成一次有效检索。
                decision = {'action': None, 'lookup_query': None,
                            'content': json.dumps(message, ensure_ascii=False),
                            'decision_error': '服务未返回一个完整且有效的角色工具决定。'}
            elif result.get('content'):
                # 未按工具约定附带的工作仍保留，不把它拼进查询。
                decision['content'] = str(result['content']) + '\n' + decision['content']
            result.update(decision)
            self.calls[-1]['role_control'] = True
            self.calls[-1]['raw_response_content'] = json.dumps(message, ensure_ascii=False)
        else:
            self.calls[-1]['raw_response_content'] = message.get('content')
        if not role_control:
            result['content'] = final_content(result.get('content'))
        if role_control:
            self.calls[-1]['decision_response'] = dict(result)
        return result
