# 页面同学工作文档

本文档说明页面控制台需要读取哪些字段，以及每个字段适合展示什么。页面同学不需要理解 Agent 内部实现，只需要读取运行脚本输出或后续服务接口返回的 state 数据。

## 页面展示目标

页面建议展示五类信息：

- Agent 通信链路。
- text/protocol 通信开销对比。
- latent state 非文本状态传递。
- shared memory / hypergraph memory 命中情况。
- 单任务和总实验指标。

## 推荐页面模块

建议页面包含：

- 运行配置选择：`quick`、`api`、`api_embedding`、`latent_hidden`。
- 数据集选择：`competition_10`、`hotpot_5`、`hidden_demo`。
- Agent 通信图：读取 `messages`。
- 指标卡片：读取 `metrics`。
- 记忆命中面板：读取 `memory_hits`。
- latent 状态面板：读取 `latent_states`。
- text/protocol 对比图：读取 summary / comparison 输出。

## messages 字段

每条 Agent 消息格式：

```json
{
  "from": "planner",
  "to": "retriever",
  "action": "retrieve",
  "payload": "... 或结构化协议包 ...",
  "char_cost": 305
}
```

页面用法：

- `from`：通信图起点。
- `to`：通信图终点。
- `action`：边标签。
- `char_cost`：边权重或 tooltip。
- `payload`：详情面板展示。

## metrics 字段

单任务指标格式：

```json
{
  "message_count": 3,
  "text_chars": 3283,
  "latent_transfer_count": 3,
  "latent_bytes": 4608
}
```

页面用法：

- `message_count`：消息数卡片。
- `text_chars`：文本通信开销。
- `latent_transfer_count`：非文本状态传递次数。
- `latent_bytes`：非文本状态数据规模。

脚本还会额外输出：

```text
memory_hits
elapsed_ms
exact_match
```

这些可以作为任务表格列。

## memory_hits 字段

`memory_hits` 是 `list[dict]`，每条是命中的历史记忆。

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

页面用法：

- 记忆命中数量：`len(memory_hits)`。
- 命中卡片：展示 `memory_id`、`summary`、`tags`。
- 可解释路径：展示 `matched_nodes`、`matched_hyperedges`、`reasoning_path`。

## latent_states 字段

`latent_states` 是一个 dict：

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

本地 embedding 类型：

```json
{
  "content_type": "latent.embedding",
  "shape": [384],
  "embedding_provider": "local",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

本地 LLM hidden state 类型：

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

页面用法：

- 按 `content_type` 分类展示。
- embedding 状态显示 provider、model、shape。
- hidden state 显示 model、layer、shape、pooled_shape。
- 不要展示完整向量或 tensor，只展示元数据。

## 运行方式和数据集选择

页面可以做两个选择框：

```text
运行方式：quick / api / api_embedding / latent_hidden
数据集：competition_10 / hotpot_5 / hidden_demo
```

推荐组合：

| 运行方式 | 数据集 | 脚本 |
| --- | --- | --- |
| `quick` | `competition_10` | `run_demo.py` |
| `quick` | `hotpot_5` | `run_hotpot_samples.py` |
| `api` | `competition_10` | `run_demo.py` |
| `api` | `hotpot_5` | `run_hotpot_samples.py` |
| `api_embedding` | `competition_10` | `run_demo.py` |
| `api_embedding` | `hotpot_5` | `run_hotpot_samples.py` |
| `latent_hidden` | `hidden_demo` | `run_latent_hidden_demo.py` |

暂不支持：

| 运行方式 | 数据集 | 原因 |
| --- | --- | --- |
| `latent_hidden` | `competition_10` | hidden state 还没有接进 10 轮主流程 |
| `latent_hidden` | `hotpot_5` | hidden state 还没有接进多跳问答主流程 |

## 页面不要依赖的内容

不要依赖完整 `payload` 内部结构，因为 protocol 字段后续可能优化。

稳定字段是：

- `messages[].from`
- `messages[].to`
- `messages[].action`
- `messages[].char_cost`
- `metrics.message_count`
- `metrics.text_chars`
- `metrics.latent_transfer_count`
- `metrics.latent_bytes`
- `memory_hits`
- `latent_states`

