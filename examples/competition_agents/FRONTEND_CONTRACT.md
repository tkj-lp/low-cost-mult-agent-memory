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
