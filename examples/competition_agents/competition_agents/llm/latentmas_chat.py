"""接入 LatentMAS 官方隐式推理核心；工具与最终输出仍使用官方生成接口。"""
from __future__ import annotations

from types import SimpleNamespace

from .continuous_state_chat import ContinuousStateChat
from .context_budget import ContextCapacityError, context_limit
from .forward_timing import ForwardTiming
from .native_local_chat import state_message, text_decoder
from ..vendor.latentmas.models import ModelWrapper


class LanguageForward:
    """纯文本隐式推理直接调用原语言解码器，避开多模态外壳的位置推断。"""

    def __init__(self, bundle):
        self.bundle = bundle

    def get_input_embeddings(self):
        return self.bundle.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.bundle.model.get_output_embeddings()

    def __call__(self, **kwargs):
        output = text_decoder(self.bundle)(**kwargs)
        # 完整策略仍保留词表分数和全部隐藏输出，不能混用临时输出释放策略。
        return SimpleNamespace(past_key_values=output.past_key_values,
                               hidden_states=output.hidden_states,
                               logits=self.get_output_embeddings()(output.last_hidden_state))


class StateForward(LanguageForward):
    """保留完整各层缓存，只释放未用于隐式推理的词表分数和中间层输出。"""

    def __call__(self, **kwargs):
        kwargs.pop('output_hidden_states', None)
        output = text_decoder(self.bundle)(**kwargs, output_hidden_states=False)
        inputs = kwargs.get('inputs_embeds')
        if inputs is None:
            # 上游只把初始嵌入用于诊断记录；无须再分配整段嵌入。
            inputs = self.get_input_embeddings()(kwargs['input_ids'][:, -1:])
        return SimpleNamespace(past_key_values=output.past_key_values,
                               hidden_states=(inputs, output.last_hidden_state))


class RecordedCore(ModelWrapper):
    """直接继承上游算法；额外记录真实连续输入，用于异常后的无损恢复。"""

    def __init__(self, bundle, memory_policy):
        if memory_policy == 'release_transients':
            self.model = StateForward(bundle)
        elif hasattr(bundle.model.model, 'language_model'):
            # 官方生成仍使用原模型；这里只选择纯文本计算的入口，不改权重或上游算法。
            self.model = LanguageForward(bundle)
        else:
            self.model = bundle.model
        self.device = bundle.device
        self.args = SimpleNamespace(latent_space_realign=True)
        self.latent_space_realign = True
        self._latent_realign_matrices = {}
        self.pre_aligned = None
        self.latent_inputs = []
        self._ensure_latent_realign_matrix(self.model, bundle.device, self.args)

    def _apply_latent_realignment(self, hidden, model):
        aligned = super()._apply_latent_realignment(hidden, model)
        self.latent_inputs.append(aligned.detach().clone())
        return aligned


class LatentMASChat(ContinuousStateChat):
    """角色内按上游生成连续状态，角色间接续完整缓存，提交记忆后按题回收。"""

    upstream_commit = '9a9e4d331eb11430bd9e64754c6b252b06d73031'
    feedback_mode = 'upstream_ridge_alignment_and_norm_then_autoregressive_latent_steps'
    state_mechanism = 'upstream_latent_reasoning_plus_complete_layer_cache_handoff'

    def __init__(self, native, *, event_sink=None, latent_steps=40, memory_policy='full'):
        if type(latent_steps) is not int or latent_steps < 1:
            raise ValueError('正式隐状态协作必须执行正数内部推理步数。')
        if memory_policy not in ('full', 'release_transients'):
            raise ValueError('未知状态生命周期策略。')
        self.memory_policy = memory_policy
        self.latent_step_budget = latent_steps
        super().__init__(native, event_sink=event_sink)
        # 对齐矩阵只从冻结模型权重求解一次，属于模型准备，不在每题重复构建。
        self.core = RecordedCore(self.bundle, memory_policy)

    def reset(self):
        super().reset()
        self.replay_segments = []
        self.replaying = False

    def _forward(self, *, ids=None, embeds=None):
        super()._forward(ids=ids, embeds=embeds)
        if not self.replaying:
            self.replay_segments.append(('ids' if ids is not None else 'embeds',
                                         (ids if ids is not None else embeds).detach().clone()))

    def _ensure_state(self):
        if not self.ids or self.past is not None and self.past.get_seq_length() == len(self.ids):
            return
        original = self.ids[:]
        self.past, self.last_hidden, self.ids = None, None, []
        self.replaying = True
        try:
            for kind, values in self.replay_segments:
                self._forward(**{kind: values})
            if self.ids != original:
                raise ValueError('状态恢复没有覆盖全部真实文字与连续向量位置。')
        except Exception:
            self.past, self.last_hidden, self.ids = None, None, original
            raise
        finally:
            self.replaying = False
        self.metrics['cache_recovery_positions'] += len(original)

    def _state_size(self):
        result = super()._state_size()
        replay = sum(t.numel() * t.element_size() for _, t in getattr(self, 'replay_segments', []))
        result['recovery_storage_bytes'] = replay
        result['payload_bytes'] += replay
        return result

    def _internal_role(self, current):
        # 上游以模型默认聊天模板的生成前缀开始隐式推理，完整本步提示一起前向。
        # 不能继承外部短回答使用的 enable_thinking=False，否则会提前关闭思考区。
        text = state_message(self.bundle, [{'role': 'user', 'content': current}],
                             open_assistant=True, continuation=bool(self.ids), template_kwargs={})
        prefix = self._tokens(text)
        size = int(prefix.shape[1])
        old = len(self.ids)
        limit = context_limit(self.bundle)
        if limit is not None and old + size + self.latent_step_budget + len(self._ending_ids()) > limit:
            raise ContextCapacityError('完整角色输入与内部推理超过模型上下文上限；未裁剪在用缓存。')
        self.inputs.append({'category': 'latent_role_request', 'text': text, 'tokens': size})
        self.metrics['input_tokens'] += size
        self.core.latent_inputs = []
        timer = ForwardTiming(self.bundle.torch, prefix.device)
        timer.before()
        with self.bundle.torch.inference_mode():
            past = self.core.generate_latent_batch(prefix, latent_steps=self.latent_step_budget,
                                                   past_key_values=self.past)
        timer.after()
        self.metrics['latent_forward_ms'] += sum(timer.milliseconds())
        if past.get_seq_length() != old + size + self.latent_step_budget:
            raise ValueError('上游隐式推理的缓存长度与实际步骤不一致。')
        self.past = past
        self.ids.extend(prefix[0].tolist())
        self.ids.extend([self.bundle.tokenizer.pad_token_id or 0] * self.latent_step_budget)
        self.replay_segments.append(('ids', prefix.detach().clone()))
        self.replay_segments.append(('embeds', self.bundle.torch.stack(self.core.latent_inputs, dim=1)))
        self.core.latent_inputs = []
        self.core.pre_aligned = None
        self.metrics['latent_steps'] += self.latent_step_budget
        self.metrics['forward_positions'] += size + self.latent_step_budget
        self.peak = max(self.peak, self._state_size()['payload_bytes'])
        # 与上游一致，直接将全部各层缓存接给下一请求，不擅自插入助手结束词元。

    def _generate(self, *args, **kwargs):
        before, offset = len(self.ids), len(self.replay_segments)
        result = super()._generate(*args, **kwargs)
        # 生成接口已经处理的词元也登记一次；不能只登记最后补算的词元。
        del self.replay_segments[offset:]
        self.replay_segments.append(('ids', self.bundle.torch.tensor([self.ids[before:]], device=self.bundle.device)))
        return result

    def finish_task(self, *, persisted):
        result = super().finish_task(persisted=persisted)
        if result['released']:
            self.replay_segments = [('ids', self.bundle.torch.tensor([self.ids], device=self.bundle.device))] if self.ids else []
            self.core.latent_inputs = []
            self.core.pre_aligned = None
            result['after_bytes'] = self._state_size()['payload_bytes']
            self.peak = result['after_bytes']
        return result

    def _call_role(self, *args, **kwargs):
        cuda = str(self.bundle.device).startswith('cuda')
        if cuda:
            self.bundle.torch.cuda.synchronize(self.bundle.device)
            allocated = self.bundle.torch.cuda.memory_allocated(self.bundle.device)
            self.bundle.torch.cuda.reset_peak_memory_stats(self.bundle.device)
        result = super()._call_role(*args, **kwargs)
        if cuda:
            self.calls[-1].update(cuda_allocated_before_bytes=allocated,
                cuda_peak_allocated_bytes=self.bundle.torch.cuda.max_memory_allocated(self.bundle.device))
        return result
