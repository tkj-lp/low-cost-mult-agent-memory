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

    def call(self, role, messages, *, tools=None, tool_choice=None):
        payload = {'model': self.model, 'messages': messages, 'temperature': 0,
                   'max_tokens': self.max_tokens, 'seed': self.seed, 'enable_thinking': False}
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
        usage = body.get('usage', {})
        if 'prompt_tokens' not in usage or 'completion_tokens' not in usage:
            raise ValueError('服务没有返回实际词元用量，不能伪造估算值。')
        self.calls.append({'role': role, 'model': body.get('model', self.model),
            'messages': json.loads(json.dumps(messages)), 'response': message, 'prompt_sha256': digest(payload),
            'input_tokens': usage['prompt_tokens'], 'output_tokens': usage['completion_tokens'],
            'cached_input_tokens': usage.get('prompt_tokens_details', {}).get('cached_tokens', 0),
            'reasoning_tokens': usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0),
            'elapsed_ms': (time.perf_counter() - started) * 1000, 'attempts': attempt + 1,
            'request_bytes': len(data), 'response_bytes': len(wire),
            'finish_reason': body['choices'][0].get('finish_reason'), 'usage': usage})
        # 服务端附加推理内容不回填后续调用；语义输出和工具调用原样交接。
        return {k: message[k] for k in ('role', 'content', 'tool_calls') if k in message}
