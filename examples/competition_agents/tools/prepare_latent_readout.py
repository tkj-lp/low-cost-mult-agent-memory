"""从预备知识生成校准样本并拟合小映射，不读取问答标签或修改基础模型。"""
import argparse
import hashlib
import json
from pathlib import Path

from competition_agents.methods.catalog import BASE, CATALOG, DATASETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='必须使用新的校准目录。')
    parser.add_argument('--model', default=CATALOG['defaultLocalModel'])
    parser.add_argument('--knowledge', default=str(BASE/DATASETS[CATALOG['defaultDataset']]['knowledge']))
    parser.add_argument('--latent-steps', type=int, default=0)
    parser.add_argument('--feedback-mode', choices=['cache_only', 'linear', 'posterior8'], default='cache_only')
    parser.add_argument('--cache-policy', choices=['full', 'consumer_lifetime'], default='consumer_lifetime')
    parser.add_argument('--task-kind', choices=['continuous', 'codeact'], default='continuous')
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer
    import torch
    from competition_agents.llm.native_local_chat import NativeLocalChat
    from competition_agents.latent_hidden.calibrated_query import calibrate
    from hypergraph_memory_method.prepared_knowledge import PreparedKnowledge
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    knowledge = PreparedKnowledge(args.knowledge)
    client = NativeLocalChat.from_pretrained(args.model)
    teacher = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', local_files_only=True, device='cpu')
    readout, samples, vectors = calibrate(knowledge, client.bundle, teacher, latent_steps=args.latent_steps,
                                         task_kind=args.task_kind,
                                         feedback_mode=args.feedback_mode, cache_policy=args.cache_policy,
                                         progress=lambda event: print(json.dumps(event), flush=True))
    readout.save(output/'readout.pt')
    torch.save(vectors.cpu(), output/'validation_vectors.pt')
    dependencies = ['competition_agents/llm/latent_only_chat.py',
                    'competition_agents/plain_text_collaboration.py',
                    'competition_agents/llm/response_content.py',
                    'competition_agents/latent_hidden/calibrated_query.py',
                    'competition_agents/latent_hidden/posterior_alignment.py']
    dependencies.append('competition_agents/latent_hidden/state_lifecycle.py')
    dependencies.append('competition_agents/llm/output_repeat_guard.py')
    manifest = {'config': {'model': args.model, 'latent_steps': args.latent_steps, 'task_kind': args.task_kind,
                          'feedback_mode': args.feedback_mode, 'cache_policy': args.cache_policy,
                          'knowledge_files': knowledge.manifest['files'],
                          'implementation_hashes': {name: hashlib.sha256((BASE/name).read_bytes()).hexdigest()
                                                    for name in dependencies}},
                'validation_samples': samples, 'calibration_report': readout.report}
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
