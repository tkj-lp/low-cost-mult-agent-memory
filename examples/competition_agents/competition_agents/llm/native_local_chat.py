"""标准本地文本调用；不依赖任何协作表示、格式约束或隐状态实验。"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from types import SimpleNamespace


class NativeLocalChat:
    """仅将原始消息送入模型自带模板与标准生成接口，原样返回文本。"""

    def __init__(self, bundle, *, max_tokens=1024, seed=42):
        self.bundle = bundle
        self.model = bundle.model_name
        self.max_tokens = max_tokens
        self.seed = seed
        self.calls = []

    @classmethod
    def from_pretrained(cls, name, *, max_tokens=1024, seed=42, threads=4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(threads)
        torch.manual_seed(seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            name, local_files_only=True, dtype=torch.float32 if device == 'cpu' else 'auto',
            low_cpu_mem_usage=True)
        model.to(device)
        model.eval()
        bundle = SimpleNamespace(model=model, tokenizer=tokenizer, torch=torch,
                                 model_name=name, device=device)
        return cls(bundle, max_tokens=max_tokens, seed=seed)

    def call(self, role, messages, *, tools=None, tool_choice=None):
        if tools is not None or tool_choice is not None:
            raise ValueError('此标准文本接口不实现工具格式转换。')
        bundle = self.bundle
        started = time.perf_counter()
        original = copy.deepcopy(messages)
        text = bundle.tokenizer.apply_chat_template(
            copy.deepcopy(messages), tokenize=False, add_generation_prompt=True)
        inputs = bundle.tokenizer(text, return_tensors='pt', add_special_tokens=False)
        inputs = {key: value.to(bundle.device) for key, value in inputs.items()}
        input_count = int(inputs['input_ids'].shape[1])
        # 与原在线请求的零温度、输出上限对应；其他设置由模型发布配置提供。
        # 不传入自定义处理器、连续向量或上一调用的缓存，也不改写模型方法。
        with bundle.torch.inference_mode():
            output = bundle.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.max_tokens)
        generated = output[0, input_count:]
        ids = generated.tolist()
        content = bundle.tokenizer.decode(generated, skip_special_tokens=True)
        elapsed_ms = (time.perf_counter() - started) * 1000
        eos = bundle.model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        finish = 'stop' if ids and ids[-1] in eos_ids else (
            'length' if len(ids) >= self.max_tokens else 'stop')
        response = {'role': 'assistant', 'content': content}
        request = {'model': self.model, 'messages': original, 'temperature': 0,
                   'max_tokens': self.max_tokens, 'seed': self.seed}
        data = json.dumps(request, ensure_ascii=False).encode('utf-8')
        self.calls.append({
            'role': role, 'model': self.model, 'backend': 'native_standard_generate',
            'messages': original, 'effective_messages': copy.deepcopy(original),
            'effective_input_text': text, 'response': copy.deepcopy(response),
            'raw_response_content': content, 'input_tokens': input_count,
            'output_tokens': len(ids), 'cached_input_tokens': 0, 'reasoning_tokens': 0,
            'elapsed_ms': elapsed_ms, 'attempts': 1, 'finish_reason': finish,
            'request_bytes': len(data), 'response_bytes': len(content.encode('utf-8')),
            'byte_scope': 'serialized logical request and raw text response; no network transfer',
            'prompt_sha256': hashlib.sha256(data).hexdigest(),
            'generation_overrides': {'do_sample': False, 'max_new_tokens': self.max_tokens},
            'normalizations': [], 'cross_call_cache': False,
        })
        return response
