"""同一输入核验原版隐式推理与释放临时输出版本；不将其当作准确率测试。"""
import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from competition_agents.llm.native_local_chat import NativeLocalChat
from competition_agents.llm.latentmas_chat import RecordedCore
from competition_agents.llm.experimental_state_chat import cache_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    client = NativeLocalChat.from_pretrained(args.model)
    b, torch = client.bundle, client.bundle.torch
    base = Path(__file__).resolve().parents[1]
    docs = json.loads((base/'data/prepared_knowledge/topiocqa_153_v4/text_documents.json').read_text())
    # 所有原始知识作为数值核验输入，不读取题目参考答案。
    prompt = json.dumps(docs, ensure_ascii=False)
    ids = b.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')['input_ids'].to(b.device)
    models = {policy: RecordedCore(b, policy) for policy in ('full','release_transients')}
    results = []
    reference = None
    for policy, model in models.items():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            past = model.generate_latent_batch(ids, latent_steps=4)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            states = {(i, name): getattr(layer, name).detach().cpu().clone()
                      for i,layer in enumerate(past.layers)
                      for name in ('keys','values','conv_states','recurrent_states')
                      if getattr(layer,name,None) is not None}
            equal = reference is None or all(torch.equal(value,reference[key]) for key,value in states.items())
            if reference is None:
                reference = states
            row = {'policy':policy,'input_positions':ids.shape[1],'latent_steps':4,
                   'layer_count':len(past.layers),'cache_bytes':cache_bytes(past),
                   'cuda_baseline_bytes':baseline,'cuda_peak_bytes':peak,
                   'cuda_peak_extra_bytes':peak-baseline,'cache_identical_to_full':equal}
            results.append(row)
            print(json.dumps(row),flush=True)
        del past, states
        model.latent_inputs=[]
        model.pre_aligned=None
    Path(args.output).write_text(json.dumps(results,indent=2))
    assert results[-1]['cache_identical_to_full'], '优化后缓存与完整上游计算不一致。'


if __name__ == '__main__':
    main()
