"""本地官方生成接口；共用工具格式可选，模型权重和生成实现保持不变。"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace


def text_decoder(bundle):
    """访问官方语言解码器；不替换模型或改写权重。"""
    decoder = bundle.model.model
    return getattr(decoder, 'language_model', decoder)


def local_model_source(name):
    """按部署清单定位固定模型，模型身份仍使用官方仓库名称。"""
    manifest_path = os.environ.get('COMPETITION_MODEL_MANIFEST')
    if not manifest_path:
        return name
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    for entry in manifest.values():
        if entry.get('repo') == name:
            if not entry.get('complete') or not Path(entry['path']).is_dir():
                raise ValueError('模型下载尚未完成：' + name)
            return entry['path']
    return name


def state_message(bundle, messages, *, open_assistant, continuation=False, template_kwargs=None):
    """从官方模板提取增量角色片段，渲染占位内容不会进入模型。"""
    tokenizer = bundle.tokenizer
    options = dict(tokenize=False, **(getattr(bundle, 'template_kwargs', {})
                                    if template_kwargs is None else template_kwargs))
    if continuation:
        # 独立渲染一条用户消息会使一些模板再加默认系统提示。
        # 使用合法前缀提取后续消息片段，定位标记不进入实际模型输入。
        # 最后停在用户消息：某些模板会按最后一个用户位置改写此前助手的思考区。
        # 若锚点停在助手，追加用户消息便会改变前缀，无法安全续接。
        anchor = [{'role': 'user', 'content': '__COMPETITION_PRIOR_USER__'},
                  {'role': 'assistant', 'content': '__COMPETITION_PRIOR_ASSISTANT__'},
                  {'role': 'user', 'content': '__COMPETITION_PRIOR_BOUNDARY__'}]
        prefix = tokenizer.apply_chat_template(anchor, add_generation_prompt=False, **options)
        complete = tokenizer.apply_chat_template([*anchor, *messages],
                                                 add_generation_prompt=open_assistant, **options)
        if not prefix or not complete.startswith(prefix):
            raise ValueError('官方模板不能安全分离后续消息片段。')
        return complete[len(prefix):]
    if not getattr(bundle, 'state_template_requires_user', False) or messages[0]['role'] != 'system':
        return tokenizer.apply_chat_template(messages, add_generation_prompt=open_assistant, **options)
    # 部分官方模板要求完整请求中有用户轮次，不能直接渲染单个系统片段。
    # 用合法完整模板分离角色边界，不重输已编码的问题，不手写模型特殊词元。
    marker = '__COMPETITION_TEMPLATE_RENDER_ONLY__'
    user = [{'role': 'user', 'content': marker}]
    suffix = tokenizer.apply_chat_template(user, add_generation_prompt=False, **options)
    complete = tokenizer.apply_chat_template([*messages, *user], add_generation_prompt=False, **options)
    bos = getattr(tokenizer, 'bos_token', None)
    if bos and suffix.startswith(bos):
        suffix = suffix[len(bos):]
    if not suffix or not complete.endswith(suffix):
        raise ValueError('官方模板无法安全分离增量角色片段。')
    fragment = complete[:-len(suffix)]
    if open_assistant:
        opened = tokenizer.apply_chat_template(user, add_generation_prompt=True, **options)
        if bos and opened.startswith(bos):
            opened = opened[len(bos):]
        if not opened.startswith(suffix):
            raise ValueError('官方模板的生成前缀与消息片段不一致。')
        fragment += opened[len(suffix):]
    return fragment


class NativeLocalChat:
    """使用模型自带模板；普通回答返回文本，角色调用解析共用工具参数。"""

    def __init__(self, bundle, *, max_tokens=1024, seed=42, output_repeat_ngram_size=0):
        self.bundle = bundle
        self.model = bundle.model_name
        self.max_tokens = max_tokens
        self.seed = seed
        self.output_repeat_ngram_size = output_repeat_ngram_size
        self.bundle.output_repeat_ngram_size = output_repeat_ngram_size
        self.calls = []

    @classmethod
    def from_pretrained(cls, name, *, max_tokens=1024, seed=42, threads=4, output_repeat_ngram_size=0):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        torch.set_num_threads(threads)
        torch.manual_seed(seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        source = local_model_source(name)
        # 使用部署目录随模型发布的官方类，兼容 MiMo 等含 auto_map 的模型。
        # 只读本地已部署文件，不另行下载或替换为相近架构。
        config = AutoConfig.from_pretrained(source, local_files_only=True, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=True)
        loader = AutoModelForImageTextToText if config.model_type == 'qwen3_5' else AutoModelForCausalLM
        model = loader.from_pretrained(
            source, config=config, local_files_only=True, trust_remote_code=True,
            dtype=torch.float32 if device == 'cpu' else 'auto',
            device_map={'': device}, low_cpu_mem_usage=True)
        model.eval()
        bundle = SimpleNamespace(model=model, tokenizer=tokenizer, torch=torch,
                                 model_name=name, device=device, model_source=source,
                                 state_template_requires_user=config.model_type == 'qwen3_5',
                                 template_kwargs={'enable_thinking': False})
        from .decision_contract import prepare_decision_tokenizer
        prepare_decision_tokenizer(bundle)
        return cls(bundle, max_tokens=max_tokens, seed=seed, output_repeat_ngram_size=output_repeat_ngram_size)

    def call(self, role, messages, *, tools=None, tool_choice=None, role_control=False,
             shared_roles=False):
        if tools is not None or tool_choice is not None:
            raise ValueError('此标准文本接口不实现工具格式转换。')
        bundle = self.bundle
        started = time.perf_counter()
        original = copy.deepcopy(messages)
        from .response_content import final_content, model_messages
        effective = model_messages(self.model, copy.deepcopy(messages), shared_roles=shared_roles)
        text = bundle.tokenizer.apply_chat_template(
            effective, tokenize=False, add_generation_prompt=True,
            **getattr(bundle, 'template_kwargs', {}))
        inputs = bundle.tokenizer(text, return_tensors='pt', add_special_tokens=False)
        inputs = {key: value.to(bundle.device) for key, value in inputs.items()}
        input_count = int(inputs['input_ids'].shape[1])
        # 与原在线请求的零温度、输出上限对应；其他设置由模型发布配置提供。
        # 可选输出去重仅检查本次新词元；不传连续向量或上一调用缓存，不改写模型方法。
        from .generation import standard_generate
        result = standard_generate(bundle, inputs['input_ids'], max_tokens=self.max_tokens,
                                   control=bool(role_control), allow_reuse=(
                                       role_control.get('allow_reuse', False) if isinstance(role_control, dict) else True),
                                   role=role, task_kind=(role_control.get('task_kind', 'continuous')
                                       if isinstance(role_control, dict) else 'continuous'))
        ids = result['output_token_ids']
        content = result['content'] if role_control else final_content(
            result['content'], thinking_prefix='<think>' in text[-64:])
        elapsed_ms = (time.perf_counter() - started) * 1000
        finish = result['finish_reason']
        response = {'role': 'assistant', 'content': content, 'action': result['action'],
                    'lookup_query': result['lookup_query'], 'finish_reason': finish,
                    'decision_error': result['decision_error']}
        raw_content = bundle.tokenizer.decode(ids, skip_special_tokens=True)
        request = {'model': self.model, 'messages': original, 'temperature': 0,
                   'max_tokens': self.max_tokens, 'seed': self.seed}
        data = json.dumps(request, ensure_ascii=False).encode('utf-8')
        self.calls.append({
            'role': role, 'model': self.model, 'backend': 'native_standard_generate',
            'messages': original, 'effective_messages': copy.deepcopy(effective),
            'effective_input_text': text, 'response': copy.deepcopy(response),
            'raw_response_content': raw_content, 'input_tokens': input_count,
            'output_tokens': len(ids), 'cached_input_tokens': 0, 'reasoning_tokens': 0,
            'elapsed_ms': elapsed_ms, 'attempts': 1, 'finish_reason': finish,
            'request_bytes': len(data), 'response_bytes': len(raw_content.encode('utf-8')),
            'byte_scope': 'serialized logical request and raw text response; no network transfer',
            'prompt_sha256': hashlib.sha256(data).hexdigest(),
            'generation_overrides': {'do_sample': False, 'max_new_tokens': result['effective_output_budget'],
                                      'eos_token_id': result['effective_eos_token_ids']},
            'requested_output_budget': self.max_tokens,
            'effective_eos_token_ids': result['effective_eos_token_ids'],
            'control_stop_reason': result['control_stop_reason'],
            'generation_call_ms': result['generation_call_ms'],
            'generation_prefill_forward_ms': result['generation_prefill_forward_ms'],
            'generation_decode_forward_ms': result['generation_decode_forward_ms'],
            'generation_forward_calls': result['generation_forward_calls'],
            'forward_timing_kind': result['forward_timing_kind'],
            'template_kwargs': dict(getattr(bundle, 'template_kwargs', {})),
            'model_source': getattr(bundle, 'model_source', self.model),
            'normalizations': ['shared_role_user_messages'] if shared_roles else [],
            'cross_call_cache': False, 'shared_roles': shared_roles,
            'output_repeat_guard': result['output_repeat_guard'], 'role_control': role_control,
            'grammar_setup_ms': result['grammar_setup_ms'],
            'context_limit': result['context_limit'],
            'effective_output_budget': result['effective_output_budget'],
            'context_budget_reduced': result['context_budget_reduced'],
        })
        return response
