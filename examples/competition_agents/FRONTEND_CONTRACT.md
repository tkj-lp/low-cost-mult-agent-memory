# 页面同学工作文档

本文档说明网页控制台需要读取哪些字段，以及如何把运行方式和数据集做成可选择的实验入口。页面同学不需要理解 Agent 内部实现，只需要按稳定输出字段做可视化。

## 1. 页面推荐做两个选择框

```text
运行方式：quick / api / api_embedding / latent_hidden
数据集：competition_10 / hotpot_multihop / hotpot_continuous / hidden_hotpot_5 / hidden_continuous
```

含义：

| 运行方式 | 页面展示说明 |
| --- | --- |
| `quick` | mock 模型快速跑通流程，适合调试页面 |
| `api` | Planner / Reasoner / Summarizer 调用真实 API，展示真实生成效果 |
| `api_embedding` | 真实 API 生成文字，本地 embedding 生成向量状态并通过 `state_ref` 传递 |
| `latent_hidden` | 本地 Qwen 直接读取最后一层 hidden state，消息只传 `hidden_state_ref` |

| 数据集 | 页面展示说明 |
| --- | --- |
| `competition_10` | 10 轮比赛主流程，重点展示通信开销、记忆复用、总耗时 |
| `hotpot_multihop` | HotpotQA 抽样多跳问答，重点展示 `gold` 和 `exact_match` |
| `hotpot_continuous` | 从 HotpotQA 构造的连续关联任务，重点展示跨轮记忆命中 |
| `hidden_hotpot_5` | 5 条 hidden-state reference transfer 实验，重点展示 hidden refs 和 hidden bytes |
| `hidden_continuous` | 连续关联任务的 hidden-state reference transfer 实验，重点展示 hidden refs、hidden bytes 和 memory hits |

## 2. 推荐命令映射

| 运行方式 | 数据集 | 命令 |
| --- | --- | --- |
| `quick` | `competition_10` | `python examples\competition_agents\run_demo.py --dataset-type builtin` |
| `quick` | `hotpot_continuous` | `python examples\competition_agents\run_demo.py --dataset-type continuous --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json --limit 1 --modes protocol` |
| `api` | `hotpot_multihop` | `python examples\competition_agents\run_demo.py --dataset-type multihop --dataset-file examples\competition_agents\data\hotpot_multihop_sample.json --limit 5` |
| `api_embedding` | `competition_10` | `python examples\competition_agents\run_demo.py --dataset-type builtin` |
| `api_embedding` | `hotpot_continuous` | `python examples\competition_agents\run_demo.py --dataset-type continuous --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json --limit 5` |
| `latent_hidden` | `hidden_hotpot_5` | `python examples\competition_agents\run_hidden_latent_samples.py` |
| `latent_hidden` | `hidden_continuous` | `python examples\competition_agents\run_hidden_latent_samples.py --dataset-type continuous --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json --limit 1` |

前端第一版不建议组合 `latent_hidden + competition_10`，因为 hidden state 目前是独立实验入口，不接入 10 轮主流程。

## 3. 主要展示模块

建议页面包含：

- Agent 通信图：读取 `messages`。
- 指标卡片：读取 `metrics`、`memory_hits`、`elapsed_ms`。
- text/protocol 对比图：读取 comparison 输出。
- 记忆命中面板：读取 `memory_hits`。
- latent 状态面板：读取 `latent_states`。
- 问答结果表：读取 `final_answer`、`gold`、`exact_match`。

## 4. messages 字段

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

页面稳定依赖字段：

- `from`：通信图起点。
- `to`：通信图终点。
- `action`：边标签。
- `char_cost`：边权重或 tooltip。
- `payload`：详情面板可选展示，但不要依赖内部结构。

## 5. metrics 字段

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

- `message_count`：消息数卡片。
- `text_chars`：文本通信开销。
- `latent_transfer_count`：非文本状态传递次数。
- `latent_bytes`：非文本状态数据规模。

脚本还会额外输出：

```text
memory_hits
memory_hit_rate
elapsed_ms
exact_match
```

## 6. memory_hits 字段

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

超图同学后续可能补充：

```json
{
  "retrieval_score": 3.25,
  "hypergraph_score": 2.5,
  "embedding_score": 0.75,
  "matched_nodes": ["tag:protocol"],
  "matched_hyperedges": ["hyperedge:mem-0001"],
  "reasoning_path": []
}
```

页面可以展示记忆卡片、命中数量、命中解释路径。

## 7. latent_states 字段

embedding 状态示例：

```json
{
  "state-0001": {
    "state_id": "state-0001",
    "type": "latent_state_ref",
    "content_type": "latent.embedding",
    "producer": "retriever",
    "shape": [384],
    "embedding_provider": "local",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
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

## 8. 不要强依赖的内容

不要强依赖 `payload` 内部结构。protocol payload 后续会继续压缩，但下面字段保持稳定：

```text
messages[].from
messages[].to
messages[].action
messages[].char_cost
metrics.message_count
metrics.text_chars
metrics.latent_transfer_count
metrics.latent_bytes
memory_hits
latent_states
```

## 9. hidden_continuous 页面展示方案

`hidden_continuous` 是 `latent_hidden` 运行方式下最适合展示的连续任务实验。它验证的是：

```text
本地 LLM 最后一层 hidden state 被写入共享 latent memory。
Agent 消息中只传 hidden-0001 这类 state_ref。
连续任务后续 turn 可以复用前序任务记忆。
```

推荐运行命令：

```powershell
python examples\competition_agents\run_hidden_latent_samples.py `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json
```

### 9.1 总览指标卡片

页面顶部建议展示这些汇总指标：

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

一次完整 5 组连续任务的示例结果：

```text
samples = 15
exact_match = 100%
messages = 45
text_chars = 8490
hidden_state_refs = 45
hidden_bytes = 39703552
memory_hits = 15
elapsed_ms = 152110.17
```

页面展示建议：

```text
样例数：15
准确率：100%
消息数：45
文本通信开销：8,490 chars
Hidden 引用数：45
Hidden 状态规模：39.70 MB
记忆命中：15
总耗时：152.11 s
```

注意：`hidden_bytes` 表示共享 latent memory 中保存的完整 hidden tensor 规模，不是 Agent 文本消息开销。

### 9.2 连续任务组视图

页面可以把每个连续任务 group 做成一个折叠卡片：

```text
chain-001 Kiss and Tell -> Shirley Temple
chain-002 The Hork-Bajir Chronicles -> Animorphs
chain-003 Big Stone Gap -> Adriana Trigiani
chain-004 2014 S/S -> Winner
chain-005 Eenasul Fateh -> Management consulting
```

每个 group 里面展示 3 个 turn：

```text
turn-1：原始多跳问答
turn-2：复用上一轮证据链
turn-3：总结证据链并沉淀记忆
```

每个 group 建议展示一条记忆命中趋势：

```text
memory_hits: 0 -> 1 -> 2
```

这条趋势是“连续任务记忆复用”的核心展示点。

### 9.3 Agent 通信链路图

每个 case 的链路固定为：

```text
Planner -> Retriever -> Reasoner -> Summarizer
```

页面可以画成横向流程图：

```text
Planner
  hidden-0001
    ↓ action=retrieve
Retriever
  hidden-0002
    ↓ action=reason
Reasoner
  hidden-0003
    ↓ action=summarize
Summarizer
```

边上展示：

```text
action
refs
```

例如：

```text
planner -> retriever action=retrieve refs=hidden-0001
retriever -> reasoner action=reason refs=hidden-0001,hidden-0002
reasoner -> summarizer action=summarize refs=hidden-0003
```

### 9.4 Hidden State 元数据表

每个 case 都会产生 3 个 hidden state。页面不要展示完整 tensor，只展示元数据：

```text
state_id
producer
content_type
model
layer
shape
pooled_shape
checksum
```

表格示例：

| state_id | producer | content_type | layer | shape | pooled_shape | checksum |
| --- | --- | --- | --- | --- | --- | --- |
| hidden-0001 | planner | latent.llm_hidden_state | last | [1, 51, 896] | [896] | a5e447... |
| hidden-0002 | retriever | latent.llm_hidden_state | last | [1, 188, 896] | [896] | 0bf0e7... |
| hidden-0003 | reasoner | latent.llm_hidden_state | last | [1, 218, 896] | [896] | c9986d... |

### 9.5 每轮任务结果表

每个 turn 建议展示：

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

表格示例：

| group | turn | answer | exact_match | memory_hits | hidden_refs | hidden_bytes |
| --- | --- | --- | --- | --- | --- | --- |
| chain-001 | turn-1 | Chief of Protocol | True | 0 | 3 | 1637888 |
| chain-001 | turn-2 | Chief of Protocol | True | 1 | 3 | 2343936 |
| chain-001 | turn-3 | Evidence chain supports: Chief of Protocol | True | 2 | 3 | 4386816 |

### 9.6 推荐图表

建议至少做三张图：

```text
1. memory_hits 折线图
   x轴：turn-1 / turn-2 / turn-3
   y轴：memory_hits
   每个 group 一条线

2. text_chars 柱状图
   展示每个 turn 的文本通信开销。

3. hidden_bytes 柱状图
   展示每个 turn 在共享 latent memory 中保存的非文本状态规模。
```

图表解释口径：

```text
text_chars 越小，说明 Agent 间文本通信越低开销。
hidden_state_refs 表示消息中传递的是状态引用。
hidden_bytes 表示非文本状态本体存放在共享 latent memory 中，而不是直接塞进 Agent 消息。
memory_hits 从 0 增加到 2，说明连续任务后续轮次复用了前序记忆。
```

### 9.7 页面说明文案

可以直接放在页面上的说明：

```text
本实验在 5 组连续任务、15 个 turn 中完成 hidden-state reference transfer。
系统共传递 45 个 hidden state 引用，消息文本开销为 8,490 chars，
共享 latent memory 中保存的 hidden state 总规模为 39.70 MB。
连续任务记忆命中从每组 turn-1 的 0 增加到 turn-3 的 2，总命中 15 次。
```

也建议保留 caveat：

```text
当前 exact_match 由确定性 evidence extractor 验证，用于隔离 hidden-state 传递机制的开销；
完整 LLM 问答质量由 API/HotpotQA 主流程实验验证。
```
