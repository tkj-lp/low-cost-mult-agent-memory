"""协议入口：共用角色流程，完整关系字段交接。"""


def create(client, resources):
    from competition_agents.unified_collaboration import create_collaboration
    return create_collaboration(client, resources, method='structured')
