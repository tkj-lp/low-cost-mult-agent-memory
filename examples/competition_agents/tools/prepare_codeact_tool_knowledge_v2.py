"""离线发布执行接口第二版；只读工具说明，不读取题目或标准答案。"""
import json
from pathlib import Path
from types import SimpleNamespace

from hypergraph_memory_method.prepared_knowledge import public_documents
from tools.build_prepared_knowledge import build


def main():
    base = Path(__file__).resolve().parents[1]
    source = json.loads((base/'data/codeact_tool_sources/knowledge_sources.json').read_text(encoding='utf-8'))
    paragraph = source[0]['context'][0]['sentences'][0]
    paragraph = paragraph.replace('a JSON-compatible dictionary', 'a JSON-compatible value')
    paragraph = paragraph.replace('The task declares output_fields as the required result keys.',
        'The task declares output_schema as the result type and output_fields as the required result keys for objects. '
        'For arrays, result contains the requested values in task order.')
    paragraph += ' Printed output is captured in a bounded log and does not replace result.'
    source[0]['context'][0]['sentences'] = [paragraph]
    target = base/'data/codeact_tool_sources/knowledge_sources_v2.json'
    target.write_text(json.dumps(source, indent=2), encoding='utf-8')
    facts = json.loads((base/'data/prepared_knowledge/codeact_tools_v1/facts.json').read_text(encoding='utf-8'))
    for fact in facts:
        fact.pop('id'); fact.pop('document_id')
        if fact['object'] == 'a JSON-compatible dictionary':
            fact['object'] = 'a JSON-compatible value'
            fact['qualifiers'] = [{'kind': 'destination', 'value': 'result'}]
        if fact['object'] == 'output_fields':
            fact['qualifiers'] = [{'kind': 'attribute', 'value': 'as the required result keys for objects'}]
    for subject, relation, obj in [
        ('output_schema', 'declares', 'the result type'),
        ('result', 'contains for arrays', 'the requested values in task order'),
        ('Printed output', 'is captured in', 'a bounded log'),
        ('Printed output', 'does not replace', 'result'),
    ]:
        facts.append(dict(subject=subject, relation=relation, object=obj, polarity='positive', qualifiers=[], sentences=[0]))
    extracted = base/'data/codeact_tool_sources/extracted_facts_v2.json'
    extracted.write_text(json.dumps({public_documents(source)[0]['id']: facts}, indent=2), encoding='utf-8')
    build(SimpleNamespace(dataset=str(target), output=str(base/'data/prepared_knowledge/codeact_tools_v2'),
                          env_file=str(base/'data/codeact_tool_sources/no_credentials.env'), model='manual-import', provider='configured', workers=1,
                          facts_file=str(extracted), corrections_file='', text_retrieval='full-context', embedding_model=''))


if __name__ == '__main__':
    main()
