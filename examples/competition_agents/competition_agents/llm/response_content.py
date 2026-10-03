"""分离角色工作与工具参数；原始生成和计算成本仍完整保留。"""


def normalize_lookup_query(value):
    """检索工具只接受查询字符串；不裁剪实体、关系或条件。"""
    if not isinstance(value, str):
        return None
    return ' '.join(value.split()) or None


def split_lookup_output(text):
    """本地补查输出的首个非空行是查询，后续内容仍是角色工作。"""
    query, _, work = text.lstrip(' \t\r\n').partition('\n')
    return normalize_lookup_query(query), work.lstrip('\r\n')


def decode_lookup_query(tokenizer, token_ids):
    """只解码工具查询前缀，不把缓存中的中间工作转成文字交接。"""
    # 这是已生成词元的格式转换，不是另一次模型调用或手写生成循环。
    # 分块覆盖完整词元，避免在字节分词边界逐个解码而破坏实体名称。
    for stop in range(32, len(token_ids) + 32, 32):
        prefix = tokenizer.decode(token_ids[:stop], skip_special_tokens=True)
        value = prefix.lstrip(' \t\r\n')
        if '\n' in value or stop >= len(token_ids):
            return split_lookup_output(value)[0]
    return None


def final_content(text, *, thinking_prefix=False):
    value = str(text or '')
    # 部分模板在生成前已经写入开始标记，因此只要求出现结束标记。
    if '</think>' in value and (value.lstrip().startswith('<think>') or thinking_prefix):
        return value.split('</think>', 1)[-1].strip()
    return value


def model_messages(model, messages, *, shared_roles=False):
    """适配消息角色，不增加、删除或重写指令内容。"""
    if shared_roles:
        from ..role_prompts import split_role_prompt
        instructions = [m['content'] for m in messages if m['role'] == 'system']
        if len(instructions) == 1:
            common, current = split_role_prompt(instructions[0])
            if common:
                # 固定公共规则是系统消息；本步职责放在数据之后，不能让诊断取代任务。
                data = '\n\n'.join(m['content'] for m in messages if m['role'] != 'system')
                return [{'role': 'system', 'content': common},
                        {'role': 'user', 'content': data + '\n\n' + current}]
        # 非共用角色的旧入口逐字保留，不推测它的职责结构。
        return [{**message, 'role': 'user' if message['role'] == 'system' else message['role']}
                for message in messages]
    if not model.startswith('deepseek-ai/DeepSeek-R1-'):
        return messages
    instructions = [m['content'] for m in messages if m['role'] == 'system']
    if not instructions:
        return messages
    result = [dict(m) for m in messages if m['role'] != 'system']
    for message in result:
        if message['role'] == 'user':
            message['content'] = '\n\n'.join([*instructions, message['content']])
            break
    else:
        result.insert(0, {'role': 'user', 'content': '\n\n'.join(instructions)})
    return result
