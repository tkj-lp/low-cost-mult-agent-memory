"""代码隐状态入口；执行器和补查逻辑与其他方式完全共用。"""
from .latent import LatentCollaboration


class LatentCodeActCollaboration(LatentCollaboration):
    def __init__(self, client, resources):
        super().__init__(client, {**resources, 'task_kind': 'codeact'})
