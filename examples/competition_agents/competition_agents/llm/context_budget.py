"""显存回收之外的独立上下文容量检查，不裁剪输入或伪造模型能力。"""


class ContextCapacityError(ValueError):
    pass


def context_limit(bundle):
    config = getattr(bundle.model, 'config', None)
    text = getattr(config, 'text_config', config)
    candidates = [getattr(text, name, None) for name in
                  ('max_position_embeddings', 'n_positions', 'max_sequence_length', 'seq_length')]
    candidates.append(getattr(bundle.tokenizer, 'model_max_length', None))
    limits = [value for value in candidates if isinstance(value, int) and 0 < value < 10**9]
    return min(limits) if limits else None


def generation_budget(bundle, input_count, requested, *, reserve=0):
    limit = context_limit(bundle)
    available = limit - input_count - reserve if limit is not None else requested
    if available < 1:
        raise ContextCapacityError(
            f'当前完整输入有{input_count}个词元，模型上下文上限为{limit}，'
            f'还需保留{reserve}个消息结束位置；未删除证据，也未调用超限模型。')
    return min(requested, available), limit
