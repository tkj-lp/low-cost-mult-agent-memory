"""固定完整来源与最终提示，只改变缓存回收；不读取评测答案。"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from competition_agents.llm.latent_only_chat import LatentOnlyChat
from competition_agents.llm.native_local_chat import NativeLocalChat
from competition_agents.plain_text_collaboration import FINAL_ANSWER_REQUIREMENTS
from hypergraph_memory_method.prepared_knowledge import PreparedKnowledge
from hypergraph_memory_method.source_context import identity_qualifiers, normalized_name


def source_facts(knowledge):
    positions = {key: index for index, key in enumerate(sorted(knowledge._graph_rows))}
    aliases = getattr(knowledge, 'graph_aliases', {})
    rows = []
    for hit in knowledge.read_graph_sources():
        fact = hit['fact_fields']
        def identity(role):
            value = fact[role]
            return aliases.get((fact['document_id'], normalized_name(value)),
                               aliases.get(normalized_name(value), value))
        relation = fact['relation']
        if fact['polarity'] == 'negative' and not relation.casefold().startswith('not '):
            relation = 'NOT ' + relation
        qualifiers = identity_qualifiers(fact, aliases)
        if hit['title']:
            qualifiers.append({'kind': 'source_title', 'value': hit['title']})
        rows.append({'id': 'f' + str(positions[hit['context_id']]),
                     'subject': identity('subject'), 'relation': relation, 'object': identity('object'),
                     'polarity': fact['polarity'], 'qualifiers': qualifiers})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen3.5-9B')
    parser.add_argument('--knowledge', default='data/prepared_knowledge/topiocqa_153_v4')
    parser.add_argument('--questions', default='data/continuous_topiocqa_153_10turns_v1/questions.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    questions_raw = Path(args.questions).read_bytes()
    questions = json.loads(questions_raw)
    if len(questions) != 10:
        raise ValueError('机制核验固定为原顺序完整十题。')
    knowledge = PreparedKnowledge(args.knowledge)
    facts = source_facts(knowledge)
    plan = {'scope': '固定完整来源的机制消融；不包含在线检索，不能混作完整流程成绩',
            'model': args.model, 'policies': ['full', 'consumer_lifetime'],
            'question_sha256': hashlib.sha256(questions_raw).hexdigest(),
            'source_fact_sha256': hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest(),
            'source_fact_count': len(facts), 'corpus_sha256': knowledge.manifest['corpus_sha256'],
            'final_instruction_sha256': hashlib.sha256(FINAL_ANSWER_REQUIREMENTS.encode()).hexdigest(),
            'max_tokens': 3072, 'output_repeat_ngram_size': 32, 'base_weights_modified': False,
            'reference_answers_read': False}
    (args.output/'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    native = NativeLocalChat.from_pretrained(args.model, max_tokens=1, output_repeat_ngram_size=32)
    runs = {}
    for policy in plan['policies']:
        native.call('warmup', [{'role': 'user', 'content': 'Reply ready.'}])
        chat = LatentOnlyChat(native.bundle, latent_steps=0, feedback_mode='cache_only',
                              query_mode='features', cache_policy=policy, max_tokens=3072)
        rows = []
        for task in questions:
            started, begin = time.perf_counter(), len(chat.calls)
            chat.call('planner', question=task['question'], scope=task['group_id'])
            chat.call('retriever', tool=lambda _query: {'facts': facts, 'audit': {'fixed_complete_sources': True}})
            chat.call('reasoner')
            answer = chat.call('summarizer')
            calls = chat.calls[begin:]
            row = {'task_id': task['id'], 'question': task['question'], 'answer': answer,
                   'seconds': time.perf_counter()-started, 'calls': calls,
                   'state_peak_bytes': max(c['cache_peak_bytes_including_compaction'] for c in calls),
                   'removed_positions': sum(c['lifetime']['removed_positions'] for c in calls),
                   'output_tokens': calls[-1]['output_tokens'], 'finish_reason': calls[-1]['finish_reason']}
            rows.append(row)
            print(json.dumps({'policy': policy, 'task': task['id'], 'seconds': row['seconds'],
                              'peak_mib': row['state_peak_bytes']/1048576,
                              'removed': row['removed_positions'], 'output_tokens': row['output_tokens']}, ensure_ascii=False), flush=True)
            (args.output/(policy+'.json')).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
        runs[policy] = {'seconds': sum(r['seconds'] for r in rows),
                        'state_peak_bytes': max(r['state_peak_bytes'] for r in rows),
                        'output_tokens': sum(r['output_tokens'] for r in rows),
                        'removed_positions': sum(r['removed_positions'] for r in rows),
                        'finished_tasks': sum(r['finish_reason'] == 'stop' for r in rows),
                        'retained_state_bytes': chat.state_bytes()}
        del chat
        native.bundle.torch.cuda.empty_cache()
    (args.output/'summary.json').write_text(json.dumps({'plan': plan, 'runs': runs}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
