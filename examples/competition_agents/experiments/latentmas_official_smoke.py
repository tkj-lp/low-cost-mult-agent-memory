"""官方模型、官方提示、官方方法的独立小样本检查；不替换项目数据集。"""
import argparse
import gc
import hashlib
import json
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from competition_agents.llm.native_local_chat import NativeLocalChat
from competition_agents.llm.latentmas_chat import StateForward
from competition_agents.vendor.latentmas import models as core_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=3)
    args = parser.parse_args()
    source = Path(args.upstream)
    sys.path.insert(0, str(source))
    # 官方方法的类型引用使用已记录的新版缓存兼容模块。
    sys.modules['models'] = core_module
    from methods.text_mas import TextMASMethod
    from methods.latent_mas import LatentMASMethod
    rows = json.loads((source/'data/medqa.json').read_text())[:args.limit]
    items = [{'question':r['query'], 'solution':r['answer'],
              'gold':next(option[0].lower() for option in r['options'] if option[3:] == r['answer'])}
             for r in rows]
    client = NativeLocalChat.from_pretrained(args.model_path, max_tokens=2048)
    b, torch = client.bundle, client.bundle.torch
    settings = SimpleNamespace(device='cuda', device2='cuda', latent_only=False,
        sequential_info_only=False, max_new_tokens=2048, task='medqa', prompt='sequential',
        model_name='Qwen/Qwen3-4B', think=False, latent_space_realign=True,
        text_mas_context_length=-1)
    wrapper = core_module.ModelWrapper.__new__(core_module.ModelWrapper)
    wrapper.model, wrapper.tokenizer, wrapper.device = b.model, b.tokenizer, b.device
    wrapper.args, wrapper.use_vllm, wrapper.vllm_engine = settings, False, None
    wrapper.model_name = settings.model_name
    wrapper._latent_realign_matrices = {}
    wrapper.pre_aligned = None
    wrapper.latent_space_realign = True
    wrapper._ensure_latent_realign_matrix(b.model, b.device, settings)
    logs = []

    def generated(self, input_ids, attention_mask=None, *, max_new_tokens=256,
                  temperature=.7, top_p=.95, past_key_values=None):
        # 新版生成接口用逻辑长度定位新增后缀，旧状态对应的占位编号不会重新输入模型。
        start = time.perf_counter()
        count = core_module._past_length(past_key_values)
        logical = input_ids
        if count:
            logical = torch.cat([torch.full((1,count),b.tokenizer.pad_token_id,
                                           device=b.device,dtype=torch.long),input_ids],dim=1)
        with torch.inference_mode():
            result = b.model.generate(input_ids=logical, attention_mask=torch.ones_like(logical),
                past_key_values=past_key_values, do_sample=True, temperature=temperature,
                top_p=top_p, max_new_tokens=max_new_tokens, pad_token_id=b.tokenizer.pad_token_id,
                return_dict_in_generate=True, output_scores=False)
        new = result.sequences[:,logical.shape[1]:]
        torch.cuda.synchronize()
        logs.append({'kind':'text','input_tokens':input_ids.shape[1], 'cached_positions':count,
                     'output_tokens':new.shape[1],'seconds':time.perf_counter()-start})
        return [b.tokenizer.decode(row,skip_special_tokens=True).strip() for row in new], result.past_key_values

    wrapper.generate_text_batch = types.MethodType(generated, wrapper)
    original_latent = wrapper.generate_latent_batch
    def latent(self, input_ids, attention_mask=None, *, latent_steps, past_key_values=None):
        start=time.perf_counter()
        old=core_module._past_length(past_key_values)
        with torch.inference_mode():
            past=original_latent(input_ids,attention_mask,latent_steps=latent_steps,past_key_values=past_key_values)
        torch.cuda.synchronize()
        logs.append({'kind':'latent','input_tokens':input_ids.shape[1],'latent_steps':latent_steps,
                     'cached_positions':old,'output_tokens':0,'seconds':time.perf_counter()-start})
        return past
    wrapper.generate_latent_batch=types.MethodType(latent,wrapper)
    result = {'upstream_commit':'9a9e4d331eb11430bd9e64754c6b252b06d73031',
              'model':'Qwen/Qwen3-4B','dataset':'upstream MedQA first '+str(args.limit),
              'temperature':.6,'top_p':.95,'max_new_tokens':2048,'latent_steps':40,
              'scope':'独立上游链路检查，非本项目连续任务成绩，非论文完整九项基准复现。',
              'prompt_sha256':hashlib.sha256((source/'prompts.py').read_bytes()).hexdigest(),
              'runs':[]}
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    for mode in ('text_mas','latent_mas','latent_mas_release_transients'):
        if mode == 'latent_mas_release_transients':
            wrapper.model=StateForward(b)
            wrapper._latent_realign_matrices[id(wrapper.model)] = wrapper._latent_realign_matrices[id(b.model)]
        else:
            wrapper.model=b.model
        method = (TextMASMethod(wrapper,max_new_tokens_each=2048,temperature=.6,top_p=.95,args=settings)
                  if mode == 'text_mas' else LatentMASMethod(wrapper,latent_steps=40,
                       judger_max_new_tokens=2048,temperature=.6,top_p=.95,args=settings))
        run={'mode':mode,'tasks':[]}
        result['runs'].append(run)
        for index,item in enumerate(items):
            torch.manual_seed(42+index)
            logs.clear()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            baseline=torch.cuda.memory_allocated()
            start=time.perf_counter()
            answer=method.run_item(item)
            torch.cuda.synchronize()
            answer.update(seconds=time.perf_counter()-start,calls=logs[:],
                          cuda_peak_bytes=torch.cuda.max_memory_allocated(),cuda_baseline_bytes=baseline)
            run['tasks'].append(answer)
            run['seconds']=sum(x['seconds'] for x in run['tasks'])
            run['correct']=sum(x['correct'] for x in run['tasks'])
            run['output_tokens']=sum(c['output_tokens'] for x in run['tasks'] for c in x['calls'])
            out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
            print(json.dumps({'mode':mode,'task':index+1,'prediction':answer['prediction'],
                              'correct':answer['correct'],'seconds':answer['seconds']},ensure_ascii=False),flush=True)
    assert [x['raw_prediction'] for x in result['runs'][1]['tasks']] == [x['raw_prediction'] for x in result['runs'][2]['tasks']], '释放临时状态后答案发生改变。'
    result['finished']=True
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('OFFICIAL_SMOKE_FINISHED',flush=True)


if __name__=='__main__':
    main()
