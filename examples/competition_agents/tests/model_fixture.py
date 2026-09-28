from types import SimpleNamespace

def tiny_bundle():
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(31)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=256, eos_token_id=2, pad_token_id=0)).eval()
    return SimpleNamespace(model=model, torch=torch, tokenizer=None, model_name='tiny', device='cpu')
