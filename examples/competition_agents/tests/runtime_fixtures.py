"""不下载权重的真实生成和缓存接口夹具；不作为准确率或速度实验。"""

from types import SimpleNamespace
import json
import string

import torch


def character_tokenizer():
    from tokenizers import Tokenizer, models, decoders
    from transformers import PreTrainedTokenizerFast
    chars = list(dict.fromkeys(string.printable + '测试关系中文'))
    vocab = {'[PAD]': 0, '[UNK]': 1, '[EOS]': 2}
    vocab.update({c: i + 3 for i, c in enumerate(chars)})
    backend = Tokenizer(models.BPE(vocab, [], unk_token='[UNK]'))
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='[EOS]',
                                       pad_token='[PAD]', unk_token='[UNK]', model_max_length=32768)
    tokenizer.chat_template = (
        "{% for m in messages %}{{ '<' + m['role'] + '>\\n' + m['content'] + '\\n[end]\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<assistant>\\n' }}{% endif %}")
    return tokenizer


def cache_bundle(architecture='attention'):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    tokenizer = character_tokenizer()
    options = dict(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                   num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                   eos_token_id=2, pad_token_id=0, max_position_embeddings=32768)
    if architecture in ('hybrid', 'multimodal'):
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
        config = Qwen3_5TextConfig(**options, head_dim=8, linear_key_head_dim=8,
            linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=2,
            layer_types=['linear_attention', 'full_attention'], partial_rotary_factor=1.0)
        if architecture == 'multimodal':
            from transformers import Qwen3_5Config, Qwen3_5VisionConfig, Qwen3_5ForConditionalGeneration
            vision = Qwen3_5VisionConfig(depth=1, hidden_size=16, intermediate_size=32,
                num_heads=2, patch_size=2, spatial_merge_size=1, temporal_patch_size=1,
                out_hidden_size=16, num_position_embeddings=16)
            config = Qwen3_5Config(text_config=config, vision_config=vision,
                image_token_id=len(tokenizer), video_token_id=len(tokenizer)+1,
                vision_start_token_id=len(tokenizer)+2, vision_end_token_id=len(tokenizer)+3)
            model = Qwen3_5ForConditionalGeneration(config).eval()
        else:
            model = Qwen3_5ForCausalLM(config).eval()
    else:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        model = Qwen2ForCausalLM(Qwen2Config(**options)).eval()
    return SimpleNamespace(model=model, tokenizer=tokenizer, torch=torch,
                           model_name='tiny-' + architecture, device='cpu', template_kwargs={})


def scripted_logits(bundle, body, *, tempting_number=False):
    """给真实官方生成接口可复现的分数，单独核验语法约束及缓存行为。"""
    text = json.dumps(body, separators=(',', ':'), ensure_ascii=False) if isinstance(body, dict) else body
    ids = bundle.tokenizer.encode(text, add_special_tokens=False) + [bundle.tokenizer.eos_token_id]
    position = 0

    def score(module, args, output):
        nonlocal position
        if not hasattr(output, 'logits'):
            return
        output.logits[:, -1, :] = -20
        output.logits[:, -1, ids[min(position, len(ids)-1)]] = 20
        if tempting_number and position == 0:
            output.logits[:, -1, bundle.tokenizer.encode('1', add_special_tokens=False)[0]] = 100
        position += 1
    return bundle.model.register_forward_hook(score)
