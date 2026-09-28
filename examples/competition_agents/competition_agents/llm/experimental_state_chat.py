"""缓存字节计量；保留路径以兼容已校准特征的源码校验。"""

def cache_bytes(past):
    return sum(t.numel() * t.element_size() for layer in getattr(past, 'layers', [])
               for t in (getattr(layer, 'keys', None), getattr(layer, 'values', None)) if t is not None)
