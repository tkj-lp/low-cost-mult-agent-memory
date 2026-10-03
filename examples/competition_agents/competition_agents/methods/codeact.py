"""独立代码任务流程：四角色、真实执行；标准答案和参考程序不进入模型。"""
import json
import ast
import math
import re


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def generated_code(content):
    # 优先读取完整程序块，不能把程序中的结果字典错认成传输包装。
    fenced = re.search(r'```python\s*\n(.*?)```', content, re.S)
    if fenced:
        code = fenced.group(1)
    else:
        raw = content.strip()
        if raw.startswith('```json') and raw.endswith('```'):
            raw = raw[len('```json'):-len('```')].strip()
        try:
            obj = json.loads(raw)
        except (ValueError, TypeError):
            obj = None
        code = obj['code'] if isinstance(obj, dict) and isinstance(obj.get('code'), str) else content
    tree = ast.parse(code)
    if not any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == 'result' for node in ast.walk(tree)):
        raise ValueError('需要生成给 result 赋值的可执行程序，直接给出结果不能代替代码。')
    return code


def evaluate_execution(execution, expected):
    def equal(actual, reference):
        if isinstance(reference, dict):
            return isinstance(actual, dict) and all(k in actual and equal(actual[k], v) for k, v in reference.items())
        if isinstance(reference, list):
            return isinstance(actual, list) and len(actual) == len(reference) and all(equal(a, b) for a, b in zip(actual, reference))
        if isinstance(reference, (float, int)) and not isinstance(reference, bool):
            return isinstance(actual, (float, int)) and not isinstance(actual, bool) and math.isclose(actual, reference, rel_tol=1e-5, abs_tol=1e-6)
        return actual == reference
    result = execution.get('result', {})
    ok = execution.get('sandbox_status') == 'ok'
    if not ok:
        score = 0.
    elif isinstance(expected, dict):
        score = (sum(k in result and equal(result[k], v) for k, v in expected.items()) / len(expected)
                 if expected and isinstance(result, dict) else float(equal(result, expected)))
    elif isinstance(expected, list):
        score = (sum(equal(a, b) for a, b in zip(result, expected)) / len(expected)
                 if expected and isinstance(result, list) else float(equal(result, expected)))
    else:
        score = float(equal(result, expected))
    return {'score': score, 'correct': ok and equal(result, expected), 'metric': 'execution_value_coverage'}



class CodeActCollaboration:
    """兼容入口；所有代码任务使用统一四角色与真实执行流程。"""
    def __init__(self, client, resources, structured=False, method=None):
        from competition_agents.unified_collaboration import UnifiedCollaboration
        self.runtime = UnifiedCollaboration(client, {**resources, 'task_kind': 'codeact'},
            method=method or ('structured' if structured else 'plain'),
            max_searches=resources.get('max_searches', 2))
        self.memory = self.runtime.memory

    def run_task(self, task, *, history_scope):
        return self.runtime.run_task(task, history_scope=history_scope)

    def close(self):
        self.runtime.close()
