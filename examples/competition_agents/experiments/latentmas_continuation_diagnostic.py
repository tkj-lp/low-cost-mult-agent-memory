"""固定同一份实际召回证据，分离缓存接续错误与连续表示问题；不计作任务成绩。"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from competition_agents.llm.native_local_chat import NativeLocalChat, state_message
from competition_agents.llm.latentmas_chat import RecordedCore
from competition_agents.llm.generation import standard_generate
from competition_agents.role_prompts import role_duty_prompt, split_role_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--source-tasks', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--modes', nargs='+', default=['cache_only', 'upstream_latent_default',
                        'upstream_latent_think_flag'], choices=['cache_only', 'upstream_latent_default',
                        'upstream_latent_think_flag', 'upstream_full_requests'])
    args = parser.parse_args()
    row = json.loads(Path(args.source_tasks).read_text(encoding='utf8'))[0]
    question = row['question']
    # 使用首次实际查库返回的完整字段；不读取标准答案、模型候选或最终输出。
    evidence = next(p['text'] for c in row['calls'] for p in c['inputs']
                    if p['category'] == 'external_facts')
    native = NativeLocalChat.from_pretrained(args.model, max_tokens=256, output_repeat_ngram_size=32)
    b, torch = native.bundle, native.bundle.torch
    core = RecordedCore(b, 'full')
    results = {'scope': '仅定位缓存数值与连续表示问题，非端到端成绩；固定首题完整召回字段。',
               'model': args.model, 'question': question, 'max_tokens': 256,
               'source_tasks': args.source_tasks, 'runs': []}
    out = Path(args.output)

    def tokens(text):
        return b.tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'].to(b.device)

    def message(role, text, opened=False, default=False, continued=True):
        return state_message(b, [{'role': role, 'content': text}], open_assistant=opened,
                             continuation=continued, template_kwargs={} if default else None)

    with torch.inference_mode():
        for mode in args.modes:
            past, ids, segments, prefixes, vectors = None, [], [], [], []
            common, _ = split_role_prompt(role_duty_prompt('continuous', 'planner'))
            initial = message('system', common, continued=False)
            initial += message('user', json.dumps({'current_task': {'question': question}}, separators=(',', ':')))
            initial += evidence
            prefix = tokens(initial)
            past = core.generate_latent_batch(prefix, latent_steps=0)
            ids.extend(prefix[0].tolist())
            segments.append(b.model.get_input_embeddings()(prefix))
            for role in ('planner', 'retriever', 'reasoner'):
                _, duty = split_role_prompt(role_duty_prompt('continuous', role))
                duty += '\n\nCurrent task to complete:\n' + json.dumps({'question': question}, separators=(',', ':'))
                text = (b.tokenizer.apply_chat_template([{'role': 'system', 'content': common},
                        {'role': 'user', 'content': duty}], tokenize=False, add_generation_prompt=True)
                        if mode == 'upstream_full_requests' else message('user', duty, opened=True, default=True))
                # 上游自带的 think 开关；保持其追加方式，不搜索模型专属提示或参数。
                if mode == 'upstream_latent_think_flag':
                    text += '<think>'
                prefixes.append(text)
                request = tokens(text)
                core.latent_inputs = []
                steps = 0 if mode == 'cache_only' else 40
                past = core.generate_latent_batch(request, latent_steps=steps, past_key_values=past)
                ids.extend(request[0].tolist())
                segments.append(b.model.get_input_embeddings()(request))
                if steps:
                    values = torch.stack(core.latent_inputs, dim=1)
                    segments.append(values)
                    vectors.append(values)
                    ids.extend([b.tokenizer.pad_token_id or 0] * steps)
            _, duty = split_role_prompt(role_duty_prompt('continuous', 'summarizer'))
            duty += '\n\nCurrent task to complete:\n' + json.dumps({'question': question}, separators=(',', ':'))
            final_text = (b.tokenizer.apply_chat_template([{'role': 'system', 'content': common},
                          {'role': 'user', 'content': duty}], tokenize=False, add_generation_prompt=True)
                          if mode == 'upstream_full_requests' else message('user', duty, opened=True))
            suffix = tokens(final_text)
            before_final = copy.deepcopy(past)
            logical = torch.tensor([ids + suffix[0].tolist()], device=b.device)
            cached = b.model(input_ids=suffix, past_key_values=past,
                             attention_mask=torch.ones_like(logical), use_cache=True, return_dict=True)
            cached_scores = cached.logits[:, -1, :].float().clone()
            full = b.model(inputs_embeds=torch.cat([*segments, b.model.get_input_embeddings()(suffix)], dim=1),
                           attention_mask=torch.ones_like(logical), use_cache=False, return_dict=True)
            replay_scores = full.logits[:, -1, :].float()
            score_check = {
                'all_finite': bool(torch.isfinite(cached_scores).all() and torch.isfinite(replay_scores).all()),
                'max_abs_difference': float((cached_scores-replay_scores).abs().max()),
                'mean_abs_difference': float((cached_scores-replay_scores).abs().mean()),
                'cosine_similarity': float(torch.nn.functional.cosine_similarity(cached_scores,replay_scores).item()),
                'cached_next_token': int(cached_scores.argmax()),
                'replay_next_token': int(replay_scores.argmax()),
            }
            del cached, full, cached_scores, replay_scores, past
            torch.cuda.synchronize()
            started = time.perf_counter()
            answer = standard_generate(b, logical, max_tokens=256, past=before_final, role='summarizer')
            torch.cuda.synchronize()
            result = {'mode': mode, 'cache_numerical_check': score_check,
                      'role_inputs': prefixes, 'final_request': final_text,
                      'answer': answer['content'], 'output_tokens': answer['output_tokens'],
                      'finish_reason': answer['finish_reason'],
                      'generation_seconds': time.perf_counter()-started}
            if mode == 'cache_only':
                reference = standard_generate(b, logical, max_tokens=256, role='summarizer')
                result['cache_vs_fresh_token_identical'] = answer['output_token_ids'] == reference['output_token_ids']
                result['fresh_answer'] = reference['content']
                del reference
            if vectors:
                v = torch.cat(vectors, dim=1).float()
                result['latent_adjacent_cosine_mean'] = float(torch.nn.functional.cosine_similarity(v[:,1:,:],v[:,:-1,:],dim=-1).mean())
            del answer, before_final, segments
            core.latent_inputs, core.pre_aligned = [], None
            results['runs'].append(result)
            out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf8')
            print(json.dumps({k:v for k,v in result.items() if k not in ('role_inputs','final_request')},ensure_ascii=False),flush=True)
    results['finished'] = True
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
