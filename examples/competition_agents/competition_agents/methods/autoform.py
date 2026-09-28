"""保留原自主选择交接格式的提示与转换方式，客户端按本次运行注入。"""
from baselines.autoform.autoform_adapter import build_autoform_instruction
from competition_agents.plain_text_collaboration import PlainTextCollaboration, PlainTextKnowledge, COMMON_TEXT
from .text import SimilarityTextMemory


class AutoFormCollaboration(PlainTextCollaboration):
    common = COMMON_TEXT.split('The runtime forwards', 1)[0]

    def _handoff(self, role, destination, context):
        prompt = build_autoform_instruction(sender=role, receiver=destination,
                                            action='handoff', source_message=context)
        response = self.client.call('autoform', [{'role': 'user', 'content': prompt}])
        if self.client.calls[-1].get('finish_reason') == 'length':
            raise ValueError('自主格式转换输出截断，不能当作完整消息。')
        return response.get('content', '').strip() or context


def create(client, resources):
    return AutoFormCollaboration(PlainTextKnowledge(resources['knowledge']), client,
                                 memory=SimilarityTextMemory(resources['encode']))
