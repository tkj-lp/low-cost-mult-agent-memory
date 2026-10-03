"""保留官方消息压缩接口；不把规则兜底记为官方压缩模型。"""
from baselines.llmlingua2.llmlingua2_adapter import compress_message, _get_official_compressor


def prepare_compressor():
    import os
    os.environ['LLMLINGUA2_ENABLE_OFFICIAL'] = '1'
    os.environ['LLMLINGUA2_STRICT'] = '1'
    os.environ.pop('LLMLINGUA2_DISABLE_OFFICIAL', None)
    os.environ.setdefault('LLMLINGUA2_DEVICE_MAP', 'cpu')
    if _get_official_compressor() is None:
        raise RuntimeError('请先准备官方 LLMLingua-2 依赖和压缩模型。')


def create(client, resources):
    from competition_agents.unified_collaboration import create_collaboration
    return create_collaboration(client, resources, method='llmlingua')
