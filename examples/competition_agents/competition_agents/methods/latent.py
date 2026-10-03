"""隐状态入口：共用角色流程，驻留完整缓存，标准最终生成。"""
from competition_agents.unified_collaboration import UnifiedCollaboration


class LatentCollaboration(UnifiedCollaboration):
    def __init__(self, client, resources):
        super().__init__(client, resources, method='latent',
                         max_searches=resources.get('max_searches', 2))


def create(client, resources):
    return LatentCollaboration(client, resources)
