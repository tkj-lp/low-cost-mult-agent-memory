"""固定同一组三题，顺序测试；在线两组、本地三组分别比较，不混算模型差异。"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from competition_agents.methods.catalog import CATALOG, METHODS
from competition_agents.methods.runner import run
from tools.build_prepared_knowledge import read_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--backend', choices=['both', 'local', 'openai-compatible'], default='both')
    parser.add_argument('--method', choices=['plain', 'structured', 'latent'])
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    read_env(Path('.env.local'))
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', local_files_only=True, device='cpu')
    encode = lambda text: encoder.encode(text, normalize_embeddings=True).tolist()
    resources = {'encoder': encoder, 'encode': encode,
                 'embed': lambda text: SimpleNamespace(vector=encode(text), provider='local', model='sentence-transformers/all-MiniLM-L6-v2')}
    summary = []
    backends = ['openai-compatible', 'local'] if args.backend == 'both' else [args.backend]
    for backend in backends:
        if backend == 'local':
            from competition_agents.llm.native_local_chat import NativeLocalChat
            client = NativeLocalChat.from_pretrained(CATALOG['defaultLocalModel'], max_tokens=1024, seed=42, threads=4)
        else:
            from competition_agents.llm.audited_chat import AuditedChat
            client = AuditedChat(CATALOG['defaultRemoteModel'], max_tokens=1024, seed=42)
        for method_name in (['plain', 'structured', 'latent'] if backend == 'local' else ['plain', 'structured']):
            if args.method and method_name != args.method:
                continue
            name = backend + '-' + method_name
            print(json.dumps({'phase': 'start', 'run': name, 'questions': 3}), flush=True)
            method = METHODS[method_name]
            config = {'collaborationMode': method_name, 'retrievalMode': method['retrieval'], 'memoryMode': method['memory'],
                      'backend': backend, 'localModel': CATALOG['defaultLocalModel'],
                      'remoteModel': CATALOG['defaultRemoteModel'], 'dataset': CATALOG['defaultDataset'], 'taskCount': 3}
            result = run(config, output/name, output/(name+'.events.jsonl'), client=client, resources=resources)
            actual = result['runs'][0]
            row = {'run': name, 'model': result['model'], 'complete': result['full_workflow_timing_valid'],
                   'coverage': actual['answer_accuracy'], 'metrics': actual['metrics'],
                   'tasks': [{k: t.get(k) for k in ('task_id', 'question', 'prediction', 'error', 'answer_accuracy',
                               'evidence_coverage', 'planning_control', 'metrics')}
                             for t in actual['task_results']]}
            summary.append(row)
            (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({'phase': 'done', 'run': name, 'coverage': row['coverage'],
                              'seconds': row['metrics']['elapsed_seconds'], 'complete': row['complete']}, ensure_ascii=False), flush=True)
    print('所有指定三题测试已保存。', flush=True)


if __name__ == '__main__':
    main()
