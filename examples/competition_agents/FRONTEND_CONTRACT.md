# 页面同学工作文档

本文档说明前端控制台应该如何选择实验、运行命令、读取输出字段。页面只展示正式实验模式；mock/quick 仅用于开发者在终端自测流程是否跑通，不放到页面选项里。

## 1. 页面选择模型

页面建议保留两层选择：

```text
experiment_mode: baseline_text / protocol_api / protocol_api_embedding / latent_hidden
dataset: competition_10 / hotpot_multihop / hotpot_continuous / hidden_hotpot_5 / hidden_continuous
```

### experiment_mode

| experiment_mode | 页面展示名 | 含义 |
| --- | --- | --- |
| `baseline_text` | 纯文本协作基线 | 真实 API 生成文字，Agent 间用自然语言长文本通信。用于作为其他协议方式的统一基线。 |
| `protocol_api` | 结构化协议协作 | 真实 API 生成文字，Agent 间用结构化协议包通信，embedding 使用轻量 hash，占位且稳定。 |
| `protocol_api_embedding` | 协议 + API embedding | 真实 API 生成文字，Agent 间用结构化协议通信，并用 OpenAI-compatible embedding 生成 `state_ref`。 |
| `latent_hidden` | Hidden state 引用传递 | 本地 Qwen 读取最后一层 hidden state，消息中只传 `hidden_state_ref`。 |

### dataset

| dataset | 页面展示名 | 含义 |
| --- | --- | --- |
| `competition_10` | 10 轮主流程 | 内置 10 个比赛主流程任务，适合看通信开销、记忆复用和总耗时。 |
| `hotpot_multihop` | HotpotQA 多跳问答 | 抽样多跳问答，适合看 `gold` 和 `exact_match`。 |
| `hotpot_continuous` | 连续关联任务 | 从 HotpotQA 构造的连续任务，适合看跨轮记忆复用。 |
| `hidden_hotpot_5` | Hidden Hotpot 样例 | 5 条 hidden-state reference transfer 样例。 |
| `hidden_continuous` | Hidden 连续任务 | 连续任务上的 hidden-state reference transfer + memory reuse。 |

## 2. 可用组合

| experiment_mode | 可用 dataset |
| --- | --- |
| `baseline_text` | `competition_10`, `hotpot_multihop`, `hotpot_continuous` |
| `protocol_api` | `competition_10`, `hotpot_multihop`, `hotpot_continuous` |
| `protocol_api_embedding` | `competition_10`, `hotpot_continuous` |
| `latent_hidden` | `hidden_hotpot_5`, `hidden_continuous` |

不建议组合：

```text
latent_hidden + competition_10
latent_hidden + hotpot_multihop
baseline_text + hidden_*
protocol_api + hidden_*
protocol_api_embedding + hidden_*
```

原因：`latent_hidden` 当前使用独立入口 `run_hidden_latent_samples.py`，验证的是本地 LLM hidden state 引用传递，不接入 `run_demo.py` 的 10 轮主流程。

## 3. 命令映射

正式页面命令优先使用 `--experiment-mode`。`run_demo.py` 在设置该参数时会自动读取：

```text
examples/competition_agents/.env.local
```

并把 Planner / Reasoner / Summarizer 设置为 `openai_compatible`。因此页面不要再使用 mock/quick 作为正式展示模式。

### baseline_text

```powershell
python examples\competition_agents\run_demo.py `
  --experiment-mode baseline_text `
  --dataset-type builtin
```

```powershell
python examples\competition_agents\run_demo.py `
  --experiment-mode baseline_text `
  --dataset-type multihop `
  --dataset-file examples\competition_agents\data\hotpot_multihop_sample.json `
  --limit 5
```

```powershell
python examples\competition_agents\run_demo.py `
  --experiment-mode baseline_text `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json `
  --limit 1
```

### protocol_api

```powershell
python examples\competition_agents\run_demo.py `
  --experiment-mode protocol_api `
  --dataset-type builtin
```

```powershell
python examples\competition_agents\run_demo.py `
  --experiment-mode protocol_api `
  --dataset-type multihop `
  --dataset-file examples\competition_agents\data\hotpot_multihop_sample.json `
  --limit 5
```

```powershell
python examples\competition_agents\run_demo.py `
  --experiment-mode protocol_api `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json `
  --limit 1 `
  --show-memory `
  --show-latent `
  --show-state
```

### protocol_api_embedding

```powershell
python examples\competition_agents\run_demo.py `
  --experiment-mode protocol_api_embedding `
  --dataset-type builtin
```

```powershell
python examples\competition_agents\run_demo.py `
  --experiment-mode protocol_api_embedding `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json `
  --limit 1 `
  --show-memory `
  --show-latent `
  --show-state
```

### latent_hidden

```powershell
python examples\competition_agents\run_hidden_latent_samples.py
```

```powershell
python examples\competition_agents\run_hidden_latent_samples.py `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json `
  --limit 1
```

## 4. 开发自测命令

以下命令只用于开发者确认流程能不能跑通，不进入页面正式选项：

```powershell
$env:LLM_PROVIDER="mock"
$env:LLM_PLANNER_PROVIDER="mock"
$env:LLM_REASONER_PROVIDER="mock"
$env:LLM_SUMMARIZER_PROVIDER="mock"
$env:EMBEDDING_PROVIDER="hash"

python examples\competition_agents\run_demo.py --dataset-type builtin --modes text,protocol
```

## 5. 页面主要模块

建议页面包含：

- Agent 通信图：读取 `messages`。
- 指标卡片：读取 `metrics`、`memory_hits`、`elapsed_ms`。
- 基线对比图：用 `baseline_text` 作为参照，对比 `protocol_api` 和 `protocol_api_embedding`。
- 记忆命中面板：读取 `memory_hits`。
- latent 状态面板：读取 `latent_states` 和 `state_refs`。
- 问答结果表：读取 `final_answer`、`gold`、`exact_match`。
- hidden 实验面板：读取 `hidden_state_refs`、`hidden_bytes`、`pooled_similarity`、`memory_hits`。

## 6. 稳定输出字段

页面优先依赖这些稳定字段：

```text
run_result_json.schema_version
run_result_json.experiment_mode
run_result_json.dataset_type
run_result_json.dataset_file
run_result_json.limit
run_result_json.task_count
run_result_json.runs[]
run_result_json.runs[].experiment_mode
run_result_json.runs[].dataset_type
run_result_json.runs[].dataset_file
run_result_json.runs[].task_count
run_result_json.runs[].metrics
messages[].from
messages[].to
messages[].action
messages[].char_cost
metrics.message_count
metrics.text_chars
metrics.latent_transfer_count
metrics.latent_bytes
metrics.memory_hits
metrics.memory_hit_rate
metrics.elapsed_ms
metrics.exact_match
latent_states
state_refs
```

不要强依赖 `payload` 内部结构。`payload` 未来可能继续压缩，但 `from`、`to`、`action`、`char_cost`、`metrics`、`memory_hits`、`latent_states` 会保持稳定。

正式运行脚本会在最后输出一个机器可读区块。`run_demo.py` 和 `run_hidden_latent_samples.py` 都遵循这个格式：

```text
=== run_result_json ===
{"schema_version":1,"experiment_mode":"protocol_api","dataset_type":"continuous","dataset_file":"examples/competition_agents/data/hotpot_continuous_tasks.json","limit":1,"task_count":3,"runs":[...]}
```

前端建议优先解析 `=== run_result_json ===` 后面的 JSON 行，不要从 `summary:` 或 `hidden latent summary` 文本里反向解析指标。

## 7. messages 字段

每条 Agent 消息格式：

```json
{
  "from": "planner",
  "to": "retriever",
  "action": "retrieve",
  "payload": "...",
  "char_cost": 305
}
```

页面用途：

- `from`：通信图起点。
- `to`：通信图终点。
- `action`：边标签。
- `char_cost`：边权重或 tooltip。
- `payload`：详情面板可选展示。

## 8. metrics 字段

单任务指标格式：

```json
{
  "message_count": 3,
  "text_chars": 3283,
  "latent_transfer_count": 3,
  "latent_bytes": 4608
}
```

页面用途：

- `message_count`：消息数量。
- `text_chars`：文本通信开销。
- `latent_transfer_count`：非文本状态传递次数。
- `latent_bytes`：非文本状态规模。

脚本还会额外输出：

```json
{
  "memory_hits": 3,
  "memory_hit_rate": 1.0,
  "elapsed_ms": 8642.5,
  "exact_match": 0.67,
  "task_count": 3
}
```

汇总结果里这些字段已经合并到 `runs[].metrics`，同时为了兼容旧页面，也保留了 `runs[].memory_hits`、`runs[].memory_hit_rate`、`runs[].elapsed_ms`、`runs[].exact_match` 顶层字段。新页面优先读取 `runs[].metrics`。

hidden 实验会额外输出：

```json
{
  "hidden_state_refs": 9,
  "hidden_bytes": 8368640
}
```

因此页面画“字符数对比”时仍读 `metrics.text_chars`；画 hidden 状态规模时读 `metrics.hidden_state_refs` 和 `metrics.hidden_bytes`。

## 9. 收益统计计算

右下角“收益统计”由页面计算，不需要后端直接给百分比。页面逻辑：

1. 每次运行结束后，把 `run_result_json.runs[0]` 保存到实验记录列表。
2. 当前运行不是 `baseline_text` 时，在实验记录里查找同一 `dataset_type`、同一 `dataset_file`、同一 `limit` 下最近一次 `baseline_text`。
3. 找到 baseline 后，用当前运行结果和 baseline 计算收益。
4. 没有找到 baseline 时，收益统计显示 `请先运行 baseline_text`。
5. 当前运行本身是 `baseline_text` 时，显示 `基线已记录`，不要和自己比较。

推荐展示 4 个指标：

```text
字符节省率 = (baseline.metrics.text_chars - current.metrics.text_chars) / baseline.metrics.text_chars
耗时降低率 = (baseline.metrics.elapsed_ms - current.metrics.elapsed_ms) / baseline.metrics.elapsed_ms
memory hit rate = current.metrics.memory_hit_rate
exact match = current.metrics.exact_match
```

建议显示文案：

```text
字符节省：42%，较 baseline_text
耗时降低：31%，较 baseline_text
memory hit rate：0.83，较 baseline_text +0.50
exact match：0.67，较 baseline_text +0.00
```

如果分母为 0 或 `exact_match` 为 `null`，页面显示 `N/A`。`latent_hidden` 也可以和同数据集的 `baseline_text` 做文本开销、耗时对比；但 hidden 的 `hidden_state_refs`、`hidden_bytes` 属于机制展示指标，不参与字符节省率公式。

## 10. memory_hits 字段

`memory_hits` 是 `list[dict]`，每条表示一次命中的历史记忆。

基础字段：

```json
{
  "memory_id": "mem-0001",
  "source_agent": "summarizer",
  "created_at": "2026-05-25T...",
  "task_topic": "任务主题",
  "summary": "摘要",
  "tags": ["langgraph", "protocol"]
}
```

超图检索解释字段：

```json
{
  "retrieval_score": 75.5,
  "hypergraph_score": 65.75,
  "embedding_score": 0.59,
  "matched_nodes": ["artifact:evidence", "state:evidence"],
  "matched_hyperedges": ["hyperedge:mem-0001:evidence"],
  "reasoning_path": []
}
```

页面可以展示记忆卡片、命中数量、命中分数、节点/超边和推理路径。

## 11. latent_states 字段

embedding 状态示例：

```json
{
  "state-0001": {
    "state_id": "state-0001",
    "type": "latent_state_ref",
    "content_type": "latent.embedding",
    "producer": "retriever",
    "shape": [384],
    "embedding_provider": "openai_compatible",
    "embedding_model": "text-embedding-v4",
    "checksum": "..."
  }
}
```

hidden state 示例：

```json
{
  "hidden-0001": {
    "content_type": "latent.llm_hidden_state",
    "type": "latent_hidden_state",
    "producer": "planner",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "layer": "last",
    "shape": [1, 44, 896],
    "pooled_shape": [896],
    "checksum": "..."
  }
}
```

页面只展示元数据，不展示完整向量或完整 tensor。

## 12. hidden_continuous 展示建议

`hidden_continuous` 最适合展示 hidden-state reference transfer。核心趋势：

```text
memory_hits: 0 -> 1 -> 2
```

每个 case 的链路固定为：

```text
planner -> retriever action=retrieve refs=hidden-0001
retriever -> reasoner action=reason refs=hidden-0001,hidden-0002
reasoner -> summarizer action=summarize refs=hidden-0003
```

顶部指标卡建议展示：

```text
samples
exact_match
messages
text_chars
hidden_state_refs
hidden_bytes
memory_hits
elapsed_ms
```

每轮任务表建议展示：

```text
group_id
turn_id
question
final_answer
gold
exact_match
memory_hits
text_chars
hidden_state_refs
hidden_bytes
elapsed_ms
```

说明文案建议保留：

```text
当前 exact_match 由确定性 evidence extractor 验证，用于隔离 hidden-state 传递机制的开销；完整 LLM 问答质量由 API/HotpotQA 主流程实验验证。
```
