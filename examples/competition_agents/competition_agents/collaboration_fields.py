"""角色产物的无模型解析和投影；原始输出与完整来源另外保留。"""

from __future__ import annotations

import copy
import json


def work_fields(text):
    """只接受完整字段对象，不从长段文字猜测或截取所谓重要信息。"""
    text = str(text or '').strip()
    if not text:
        return {}, None
    value = text
    if value.startswith('```') and value.endswith('```'):
        _, _, value = value.partition('\n')
        value = value.rsplit('```', 1)[0].strip()
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return {}, text
    if not isinstance(parsed, dict):
        return {}, text
    # 不删未知字段、不按问题筛选值，不把错误类型当作中止任务的理由。
    return parsed, None


def projected_work(work, receiver):
    """交接字段的当前值及显式修正；已处理说明不在每次交接中反复拼接。"""
    roles, unparsed = {}, []
    for row in work:
        role = row['role']
        if role == 'summarizer':
            continue
        fields, fallback = work_fields(row.get('content'))
        if fallback:
            item = {'role': role, 'unparsed_work': fallback}
            if item not in unparsed:
                unparsed.append(item)
        if fields:
            target = roles.setdefault(role, {})
            for key, value in fields.items():
                if key not in target or target[key] == value:
                    target[key] = copy.deepcopy(value)
                else:
                    # 旧值可能仍含限定或另一解释；不能静默覆盖，保留显式更新序列。
                    prior = target[key]
                    if isinstance(prior, dict) and set(prior) == {'current', 'previous'}:
                        previous = prior['previous'] + [prior['current']]
                    else:
                        previous = [prior]
                    target[key] = {'current': copy.deepcopy(value), 'previous': previous}
        if row.get('lookup_query'):
            roles.setdefault(role, {})['requested_lookup'] = row['lookup_query']
    payload = {'fields_by_role': roles}
    if unparsed:
        # 格式未遵循时完整回退，不能虚报字段压缩成功或丢失新信息。
        payload['unparsed_work'] = unparsed
    return payload


def has_reference_plan(work):
    """只有规划明确提供实体与目标时，后续角色才不再携带整段历史。"""
    for row in reversed(work):
        if row['role'] == 'planner':
            fields, fallback = work_fields(row.get('content'))
            return bool(fields.get('entities') and fields.get('target') and fallback is None)
    return False


def can_reuse_work(role, snapshot, task_kind):
    """限定动作适用范围，不替模型决定已有证据是否充分。"""
    if role in ('planner', 'summarizer') or not snapshot['evidence']:
        return False
    if role == 'retriever':
        return bool(snapshot['work'])
    if task_kind == 'codeact':
        # 执行器仍需要真实程序；不能用空的沿用决定替代程序输入。
        return False
    return any(row['role'] == 'reasoner' and row.get('action') == 'deliver' for row in snapshot['work'])
