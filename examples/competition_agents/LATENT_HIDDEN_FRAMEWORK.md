# Hidden Latent 实验框架

本文档说明比赛中如何解释和推进 hidden latent。这里的 hidden latent 指：

```text
本地 LLM 前向推理 -> 读取 outputs.hidden_states[-1] -> 存入共享 latent memory -> Agent 消息只传 state_ref
```

它不同于本地 embedding：

```text
文本 -> sentence-transformers -> 384 维句向量 -> state_ref
```

## 为什么不能只传一个 hidden 向量就让下游模型继续推理

`outputs.hidden_states[-1]` 是模型最后一层的内部表示，形状通常是：

```text
[batch_size, sequence_length, hidden_size]
```

它适合做：

- 非文本状态传递证明。
- 状态相似度计算。
- 状态引用和低文本开销通信。
- 后续超图记忆的 latent 节点。

但它不是 `generate()` 可以直接接收的 `past_key_values`。如果要让下游 Agent 真正“接着上游状态继续生成”，后续应研究 KV cache / prefix adapter：

```text
past_key_values -> 下游模型继续生成
hidden_state -> adapter/projector -> prefix embedding
```

当前比赛原型先实现可运行、可统计、可展示的 hidden-state reference transfer。

## 比赛可用的四段流程

正式 hidden latent 样例脚本采用：

```text
PlannerAgent -> RetrieverAgent -> ReasonerAgent -> SummarizerAgent
```

每一段的职责：

- Planner：根据 question 构造计划输入，提取 planner hidden state。
- Retriever：从数据集 context/supporting_facts 取证据，提取 evidence hidden state。
- Reasoner：读取 planner/evidence 的 state_ref，结合证据生成 answer hidden state。
- Summarizer：接收 answer state_ref，输出最终答案和指标。

重要说明：

```text
当前 Reasoner 仍保留 evidence 文本用于可复现实验答案抽取。
hidden state 的作用是非文本状态传递、引用通信和开销统计。
不要把当前版本描述成“完全不需要文本就能推理”。
```

## 运行脚本

```powershell
cd D:\qq\low-cost-memory-integrate
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH="libs/langgraph;examples/competition_agents"
$env:LOCAL_HIDDEN_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
python examples\competition_agents\run_hidden_latent_samples.py
```

第一次运行前需要：

```powershell
pip install torch transformers
```

## 输出指标

脚本输出：

```text
hidden_state_refs
hidden_bytes
text_chars
messages
planner_evidence_similarity
elapsed_ms
exact_match
```

其中：

- `hidden_state_refs`：传递了多少个 hidden state 引用。
- `hidden_bytes`：完整 hidden tensor 按 float32 估算的数据规模。
- `text_chars`：结构化消息本身的字符开销。
- `planner_evidence_similarity`：planner hidden 和 evidence hidden 的 pooled 向量相似度。
- `exact_match`：当前用确定性 evidence extractor 计算，用于控制变量。

## 和其他模块的接口关系

页面同学仍然读取：

```text
messages
metrics
latent_states
```

hidden state 只是在 `latent_states` 中新增一种类型：

```json
{
  "content_type": "latent.llm_hidden_state",
  "type": "latent_hidden_state",
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "layer": "last",
  "shape": [1, 33, 896],
  "pooled_shape": [896],
  "checksum": "..."
}
```

超图同学不需要改接口。后续如果要把 hidden state 接入超图，只需要把 `state_id` 放进 memory 的 `state_refs`。

## 后续增强方向

优先级从低风险到高风险：

1. 把 `run_hidden_latent_samples.py` 的结果接到页面。
2. 把 hidden state 元数据写入 memory 的 `state_refs`。
3. 增加 hidden state 与 memory embedding 的相似度分析。
4. 尝试 KV cache / `past_key_values` 传递实验。
5. 研究 adapter，把 hidden state 转成下游模型可消费的 prefix embedding。

