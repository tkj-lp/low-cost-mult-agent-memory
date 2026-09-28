"""协议入口：独立文字调用、模型生成字段、超图关系跳跃。"""
from competition_agents.field_collaboration import FieldCollaboration
from hypergraph_memory_method.prepared_knowledge import PreparedKnowledge


def create(client, resources):
    knowledge = PreparedKnowledge(resources['knowledge'])
    knowledge.prepare_focused_search(resources['embed'])
    return FieldCollaboration(knowledge, client, max_searches=2)
