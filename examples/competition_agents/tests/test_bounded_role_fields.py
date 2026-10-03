"""中间工作字段的生成上限；不改来源输入和最终回答。"""
import json

import torch

from competition_agents.llm.decision_contract import work_schema
from competition_agents.llm.generation import standard_generate
from runtime_fixtures import cache_bundle


def test_long_role_narrative_closes_to_valid_fields_before_common_output_budget():
    bundle = cache_bundle()
    a = bundle.tokenizer.encode('a', add_special_tokens=False)[0]
    close = bundle.tokenizer.encode('}', add_special_tokens=False)[0]
    def score(module, args, output):
        # 始终偏好续写长字符串；只有语法达到字段上限才允许它合法收尾。
        output.logits[:, -1, :] = -20
        output.logits[:, -1, a] = 100
        output.logits[:, -1, close] = 80
    hook = bundle.model.register_forward_hook(score)
    try:
        result = standard_generate(bundle, torch.tensor([[3, 4]]), max_tokens=1024,
                                   role='reasoner', control=True)
    finally:
        hook.remove()
    assert result['decision_error'] is None
    assert result['finish_reason'] == 'stop'
    assert result['output_tokens'] < 1024
    assert result['action'] == 'deliver'
    fields = json.loads(result['content'])
    assert len(fields['candidate']) <= 512
    assert len(fields['basis']) <= 3
    assert all(len(s) <= 128 for s in fields['basis'])


def test_code_programs_are_not_limited_by_question_answer_work_fields():
    assert work_schema('codeact', 'reasoner') == {'type': 'string'}
