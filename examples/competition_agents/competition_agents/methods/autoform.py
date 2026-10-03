"""自主格式入口；转换插件与四角色提示分离。"""


def create(client, resources):
    from competition_agents.unified_collaboration import create_collaboration
    return create_collaboration(client, resources, method='autoform')
