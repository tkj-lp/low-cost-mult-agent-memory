"""缓存字节计量；保留路径以兼容已校准特征的源码校验。"""

def cache_bytes(past):
    total, seen = 0, set()
    for layer in getattr(past, 'layers', []):
        for name in ('keys', 'values', 'conv_states', 'recurrent_states'):
            tensor = getattr(layer, name, None)
            if tensor is not None and id(tensor) not in seen:
                seen.add(id(tensor))
                total += tensor.numel() * tensor.element_size()
    return total
