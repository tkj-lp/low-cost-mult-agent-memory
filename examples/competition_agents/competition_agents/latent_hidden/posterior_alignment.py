"""末层状态经模型自身输出分布映射为连续输入，不解码或重新分词。"""
from __future__ import annotations


def align_posterior_hidden(model, hidden, *, top_k=8, temperature=1.0):
    if top_k < 2 or temperature <= 0:
        raise ValueError('连续映射至少需要两个分量，温度必须为正数。')
    logits = model.get_output_embeddings()(hidden).float()
    scores, indices = logits.topk(min(top_k, logits.shape[-1]), dim=-1)
    weights = (scores / temperature).softmax(dim=-1)
    embeddings = model.get_input_embeddings()(indices).float()
    mixed = (weights.unsqueeze(-1) * embeddings).sum(dim=-2)
    # 将连续混合的幅度保持在这些候选输入向量的加权幅度附近。
    target_norm = (weights * embeddings.norm(dim=-1)).sum(dim=-1, keepdim=True)
    mixed = mixed * target_norm / mixed.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    info = {'method': 'posterior_embedding_mixture', 'top_k': top_k, 'temperature': temperature,
            'max_component_weight': float(weights.max().item()),
            'entropy': float(-(weights * weights.clamp_min(1e-12).log()).sum(dim=-1).mean().item())}
    return mixed.to(dtype=model.get_input_embeddings().weight.dtype), info
