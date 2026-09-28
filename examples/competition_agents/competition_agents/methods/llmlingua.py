"""保留官方消息压缩接口；不把规则兜底记为官方压缩模型。"""
from baselines.llmlingua2.llmlingua2_adapter import compress_message, _get_official_compressor
from competition_agents.plain_text_collaboration import PlainTextCollaboration, PlainTextKnowledge, COMMON_TEXT
from .text import SimilarityTextMemory


class LLMLinguaCollaboration(PlainTextCollaboration):
    common = COMMON_TEXT.split('The runtime forwards', 1)[0]

    def _handoff(self, role, destination, context):
        result = compress_message(context, sender=role, receiver=destination, action='handoff')
        if result.provider != 'llmlingua2':
            raise RuntimeError('官方压缩模型不可用，本次不使用规则压缩替代实验。')
        return result.payload


def create(client, resources):
    import os
    os.environ['LLMLINGUA2_ENABLE_OFFICIAL'] = '1'
    os.environ['LLMLINGUA2_STRICT'] = '1'
    os.environ.pop('LLMLINGUA2_DISABLE_OFFICIAL', None)
    os.environ.setdefault('LLMLINGUA2_DEVICE_MAP', 'cpu')
    if _get_official_compressor() is None:
        raise RuntimeError('请先准备官方 LLMLingua-2 依赖和压缩模型。')
    return LLMLinguaCollaboration(PlainTextKnowledge(resources['knowledge']), client,
                                  memory=SimilarityTextMemory(resources['encode']))
