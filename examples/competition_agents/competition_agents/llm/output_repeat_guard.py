"""统一的本次输出去重保护；不检查提示、来源或此前角色的输出。"""


class OutputRepeatGuard:
    def __init__(self, ngram_size, prompt_length=0, *, preserve_grammar=False):
        if ngram_size < 2:
            raise ValueError('输出重复保护的片段长度至少为2。')
        self.ngram_size, self.prompt_length = ngram_size, prompt_length
        self.followers = {}
        self.blocked_steps = self.blocked_top1 = 0
        self.preserve_grammar = preserve_grammar
        self.grammar_fallbacks = 0

    def apply_output(self, generated, scores):
        size = self.ngram_size
        if len(generated) >= size:
            self.followers.setdefault(tuple(generated[-size:-1]), set()).add(generated[-1])
        banned = self.followers.get(tuple(generated[-(size-1):]), ()) if len(generated) >= size-1 else ()
        if banned:
            if self.preserve_grammar:
                # 语法掩码已经应用，不能删掉唯一可用的闭合符或转义词元。
                finite = scores[0].isfinite()
                allowed_count = int(finite.sum().item())
                banned_allowed = sum(bool(finite[index]) for index in banned)
                if allowed_count and banned_allowed == allowed_count:
                    self.grammar_fallbacks += 1
                    return scores
            self.blocked_steps += 1
            self.blocked_top1 += int(int(scores[0].argmax().item()) in banned)
            scores[:, list(banned)] = float('-inf')
        return scores

    def __call__(self, input_ids, scores):
        if input_ids.shape[0] != 1:
            raise ValueError('当前输出重复保护只支持单请求生成。')
        return self.apply_output(input_ids[0, self.prompt_length:].tolist(), scores)

    def report(self):
        return {'ngram_size': self.ngram_size, 'scope': 'current_generated_output_only',
                'blocked_steps': self.blocked_steps, 'blocked_top1': self.blocked_top1,
                'preserve_grammar': self.preserve_grammar, 'grammar_fallbacks': self.grammar_fallbacks}
