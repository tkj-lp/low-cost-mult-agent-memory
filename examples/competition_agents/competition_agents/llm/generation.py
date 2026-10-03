"""标准生成与极小工具动作接口；全部本地协作方式使用同一实现。"""

from __future__ import annotations

import time

from .forward_timing import ForwardTiming
from .context_budget import generation_budget


def generation_eos_token_ids(bundle):
    """同时识别模型结束词元和官方分词器的消息结束词元。"""
    configured = bundle.model.generation_config.eos_token_id
    ids = {configured} if isinstance(configured, int) else set(configured or [])
    message_end = getattr(bundle.tokenizer, 'eos_token_id', None)
    if isinstance(message_end, int):
        ids.add(message_end)
    return sorted(ids)


def standard_generate(bundle, input_ids, *, max_tokens, past=None, control=False,
                      prompt_tokens=None, decode_content=True, allow_reuse=True,
                      role='retriever', task_kind='continuous', reserve_tokens=0,
                      tool_only=False):
    """官方生成负责选词；中间调用仅约束工具语法，最终回答自由生成。"""
    if max_tokens < 1:
        raise ValueError('生成预算必须为正。')
    input_count = int(input_ids.shape[1])
    requested_tokens = max_tokens
    max_tokens, limit = generation_budget(bundle, input_count, max_tokens, reserve=reserve_tokens)
    options = {'do_sample': False, 'max_new_tokens': max_tokens, 'use_cache': True,
               'return_dict_in_generate': True}
    # 模型文件缺少 generation_config 时，框架可能只从文字配置取得 endoftext，
    # 忽略官方聊天模板的 im_end，导致已结束的回答继续生成下一轮会话。
    # 所有方式共用这个结束契约；不截答案、不改变权重或模型发布配置。
    eos_ids = generation_eos_token_ids(bundle)
    if eos_ids:
        options['eos_token_id'] = eos_ids
    pad_id = getattr(bundle.tokenizer, 'pad_token_id', None)
    if isinstance(pad_id, int):
        options['pad_token_id'] = pad_id
    guard = None
    processors = []
    guard_size = getattr(bundle, 'output_repeat_ngram_size', 0)
    if guard_size and (not control or task_kind == 'continuous' or tool_only):
        from .output_repeat_guard import OutputRepeatGuard
        guard = OutputRepeatGuard(guard_size, input_count, preserve_grammar=control)
        processors.append(guard)
    action_stop = None
    grammar_started = time.perf_counter()
    if control:
        from .decision_contract import DecisionGrammar
        action_stop = DecisionGrammar(bundle, input_count, role=role, task_kind=task_kind,
                                      allow_reuse=allow_reuse, tool_only=tool_only)
        options['prefix_allowed_tokens_fn'] = action_stop.allowed
        options['stopping_criteria'] = [action_stop]
    grammar_setup_ms = (time.perf_counter() - grammar_started) * 1000 if control else 0.
    if processors:
        options['logits_processor'] = processors
    if past is not None:
        options['past_key_values'] = past
    cached_count = int(past.get_seq_length()) if past is not None else 0
    torch = bundle.torch
    generation_started = time.perf_counter()
    timer = ForwardTiming(torch, input_ids.device)
    handles = []
    if hasattr(bundle.model, 'register_forward_pre_hook'):
        handles = [bundle.model.register_forward_pre_hook(timer.before),
                   bundle.model.register_forward_hook(timer.after)]
    try:
        with torch.inference_mode():
            output = bundle.model.generate(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids), **options)
    finally:
        for handle in handles:
            handle.remove()
    generated = output.sequences[0, input_count:]
    ids = generated.tolist()
    forward_times = timer.milliseconds()
    generation_ms = (time.perf_counter() - generation_started) * 1000
    text = bundle.tokenizer.decode(generated, skip_special_tokens=True) if decode_content else ''
    decision = {'action': None, 'lookup_query': None, 'content': text, 'decision_error': None}
    if control:
        from .decision_contract import parse_decision, parse_hidden_header
        if not action_stop.reason:
            # 预算截断不能被认成完整工具请求；原始工作仍保留给下游和审计。
            decision['decision_error'] = '工具参数未完整生成，未执行不完整请求。'
        elif tool_only:
            from .decision_contract import parse_tool_decision
            decision = parse_tool_decision(bundle.tokenizer.decode(generated, skip_special_tokens=True),
                                           role=role, allow_reuse=allow_reuse)
        elif decode_content:
            decision = parse_decision(text, role=role, task_kind=task_kind, allow_reuse=allow_reuse)
        else:
            decision = parse_hidden_header(bundle.tokenizer, ids, role=role, allow_reuse=allow_reuse)
    finish = 'stop' if action_stop and action_stop.reason or ids and ids[-1] in eos_ids else (
        'length' if len(ids) >= max_tokens else 'stop')
    return {**decision, 'finish_reason': finish,
            'output_token_ids': ids, 'output_tokens': len(ids),
            'past_key_values': getattr(output, 'past_key_values', None),
            'output_repeat_guard': guard.report() if guard else None,
            'effective_eos_token_ids': eos_ids,
            'control_stop_reason': action_stop.reason if action_stop else None,
            'grammar_setup_ms': grammar_setup_ms,
            'context_limit': limit, 'requested_output_tokens': requested_tokens,
            'effective_output_budget': max_tokens,
            'context_budget_reduced': max_tokens < requested_tokens,
            'generation_call_ms': generation_ms,
            'generation_prefill_forward_ms': forward_times[0] if forward_times else None,
            'generation_decode_forward_ms': sum(forward_times[1:]) if forward_times else None,
            'generation_forward_calls': len(forward_times),
            'forward_timing_kind': 'cuda_events' if timer.cuda else 'cpu_wall',
            'cached_input_tokens': cached_count,
            'input_tokens': input_count if prompt_tokens is None else prompt_tokens}
