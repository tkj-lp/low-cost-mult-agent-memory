"""完整各层状态的期限管理；不量化、不合并或重编码保留的键值。"""
from __future__ import annotations

from dataclasses import dataclass
import copy


class TaskStateLifecycle:
    """任务内完整接续；任务提交后恢复独立保存的公共前缀，兼容递归缓存。"""

    def __init__(self):
        self.prefix_ids = ()
        self.prefix_cache = None

    def capture_prefix(self, past, ids):
        if self.prefix_cache is not None:
            return
        # 混合注意力的递归状态不能通过裁剪长度恢复；保留真实公共前缀状态。
        self.prefix_cache = copy.deepcopy(past)
        self.prefix_ids = tuple(ids)

    def restore_prefix(self):
        return copy.deepcopy(self.prefix_cache), list(self.prefix_ids)


@dataclass
class StateBlock:
    length: int
    kind: str
    turn: int
    expires_after: int | None
    retain_through_turn: int | None = None


class StateLifecycle:
    def __init__(self):
        self.blocks = []
        self.next_position = 0
        self.turn = 0

    def append(self, count, kind, role):
        roles = ('planner', 'retriever', 'reasoner', 'summarizer')
        # 来源、问题与外部结果跨题保留；回答保留至下一题结束。
        # 回收旧回答的注意力条目不等于从递归状态中删除历史语义。
        expiry = min(roles.index(role) + 1, 3) if kind == 'role_instruction' else (
            3 if kind in ('latent', 'boundary') else None)
        retention = self.turn + 1 if kind == 'answer' else None
        block = StateBlock(count, kind, self.turn, expiry, retention)
        if self.blocks and (self.blocks[-1].kind, self.blocks[-1].turn,
                            self.blocks[-1].expires_after, self.blocks[-1].retain_through_turn) == (kind, self.turn, expiry, retention):
            self.blocks[-1].length += count
        else:
            self.blocks.append(block)
        self.next_position += count

    def reclaim(self, past, role, torch):
        from transformers.cache_utils import DynamicLayer, LinearAttentionLayer
        from competition_agents.llm.experimental_state_chat import cache_bytes

        stage = ('planner', 'retriever', 'reasoner', 'summarizer').index(role)
        keep, retained, removed, offset = [], [], 0, 0
        removed_by_kind = {}
        for block in self.blocks:
            expired = (block.expires_after is not None
                       and (block.turn < self.turn or block.turn == self.turn and block.expires_after <= stage))
            expired |= block.retain_through_turn is not None and self.turn > block.retain_through_turn
            if expired:
                removed += block.length
                removed_by_kind[block.kind] = removed_by_kind.get(block.kind, 0) + block.length
            else:
                retained.append(block)
                keep.extend(range(offset, offset + block.length))
            offset += block.length
        if offset != past.get_seq_length():
            raise ValueError('缓存分段与实际存储长度不一致。')
        before = peak = cache_bytes(past)
        if not removed:
            return {'removed_positions': 0, 'before_bytes': before, 'after_bytes': before, 'peak_bytes': peak}
        if not keep:
            raise ValueError('不得清空仍在执行的会话状态。')
        # 只裁完整注意力层；线性注意力的固定大小递归状态原样保留。
        # 滑动窗口、量化和未知布局在任何张量变更之前拒绝裁剪。
        attention_layers = [layer for layer in past.layers if type(layer) is DynamicLayer]
        if (not attention_layers
                or any(type(layer) not in (DynamicLayer, LinearAttentionLayer) for layer in past.layers)
                or any(layer.get_seq_length() != offset for layer in attention_layers)):
            return {'removed_positions': 0, 'before_bytes': before, 'after_bytes': before,
                    'peak_bytes': peak, 'skipped': 'unsupported_cache_layout_keep_full_state'}
        for layer in attention_layers:
            indices = torch.tensor(keep, dtype=torch.long, device=layer.keys.device)
            keys = layer.keys.index_select(-2, indices).contiguous()
            values = layer.values.index_select(-2, indices).contiguous()
            peak = max(peak, cache_bytes(past) + keys.numel()*keys.element_size() + values.numel()*values.element_size())
            layer.keys, layer.values = keys, values
        self.blocks = retained
        return {'removed_positions': removed, 'removed_by_kind': removed_by_kind,
                'before_bytes': before, 'after_bytes': cache_bytes(past), 'peak_bytes': peak,
                'compacted_attention_layers': len(attention_layers),
                'preserved_recurrent_layers': len(past.layers) - len(attention_layers),
                'answer_retention': 'current_and_previous_turn',
                'recurrent_history_erased': False}


class LinearStateAlignment:
    """一次求解同维映射；不筛选词元，不声明语义无损。"""
    def __init__(self, model, torch):
        source = model.get_output_embeddings().weight.detach().float()
        target = model.get_input_embeddings().weight.detach().float()
        gram = source.T @ source
        gram.diagonal().add_(1e-5)
        self.matrix = torch.linalg.solve(gram, source.T @ target)
        self.target_norm = target.norm(dim=-1).mean()
        if not torch.isfinite(self.matrix).all():
            raise ValueError('状态对齐矩阵包含非有限值。')

    def __call__(self, hidden):
        aligned = hidden.float() @ self.matrix
        return (aligned * self.target_norm / aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)).to(hidden.dtype)
