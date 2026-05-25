# 协议 0.2 与非文本通信说明

这份文档说明当前多 Agent 原型中新加入的通信协议、能力发现、CKM 认知差距和非文本 latent state 接口。超图记忆模块后续可以基于这些字段接入，不需要修改 LangGraph 流程本身。

## 能力发现

完整能力表由 `competition_agents/capabilities.py` 定义，并在每个任务初始状态里写入 `state["capability_registry"]`。

当前 Agent 能力如下：

- `planner`：`task_decomposition`、`capability_routing`、`cognitive_gap_detection`
- `retriever`：`memory_recall`、`evidence_search`、`latent_state_write`
- `executor`：`tool_execution`、`codeact_simulation`、`latent_state_read`
- `summarizer`：`summary_generation`、`memory_write`、`cognitive_gap_closure`

协议消息中只发送 `sender_capability_ref`，例如 `capability://retriever`。网页端或记忆模块可以用这个引用回查 `capability_registry`，避免每条消息重复发送完整能力列表。

## 协议包

协议包由 `competition_agents/protocol.py` 中的 `make_packet()` 创建。协议版本为 `0.2`。

示例：

```json
{
  "protocol_version": "0.2",
  "msg_id": "msg-xxxxxxxx",
  "from": "retriever",
  "to": "executor",
  "action": "execute",
  "content_type": "protocol.state_ref",
  "schema_ref": "competition.protocol.state_ref.v0.2",
  "params": {
    "tool": "codeact_stats",
    "evidence_count": 2
  },
  "result": {
    "evidence_refs": ["state-0001"],
    "memory_hits": ["mem-0001"]
  },
  "state_refs": ["state-0001"],
  "memory_refs": ["mem-0001"],
  "capability_required": ["tool_execution"],
  "sender_capability_ref": "capability://retriever",
  "receiver_capability": "tool_execution",
  "confidence": 0.78
}
```

`state_refs` 用于引用非文本中间状态，`memory_refs` 用于引用历史记忆。后续超图记忆模块可以把这两个字段映射成超图节点或超边。

## CKM 认知差距

每个 Agent 会生成一条轻量认知差距记录，并追加到 `state["cognitive_gaps"]`。

格式如下：

```json
{
  "agent": "retriever",
  "known": ["current_task", "local_evidence_templates", "memory_hit_count"],
  "unknown": ["hypergraph_expansion", "true_embedding_similarity"],
  "needed": ["latent_state_write", "tool_execution"],
  "gap_type": "partial_context",
  "confidence": 0.78
}
```

这些字段用于表达 Agent 当前已经知道什么、还缺什么、需要哪些能力继续推进。网页端可以把它展示成“已知 / 未知 / 需要能力 / 置信度”，超图记忆模块可以用它增强任务、证据和记忆之间的关系建模。

## Latent State

非文本状态由 `state.py` 中的 `put_latent_state()` 创建。当前仍使用确定性的 `hash-v1` embedding 占位，但 schema 已经升级为 LatentMAS-inspired 的轻量状态引用。

示例：

```json
{
  "state_id": "state-0001",
  "type": "latent_state_ref",
  "content_type": "latent.embedding",
  "producer": "retriever",
  "consumers": ["executor", "summarizer"],
  "topic": "current task",
  "summary": "compressed evidence representation",
  "embedding_provider": "hash-v1",
  "shape": [16],
  "vector": [0.1, -0.2],
  "scope": "task",
  "ttl": 3,
  "checksum": "xxxxxxxxxxxxxxxx"
}
```

这里没有声称已经实现真实 KV cache 传递，而是用 `state_ref`、embedding、摘要、校验值和消费方列表模拟“隐藏状态引用传递”。后续接真实 embedding 或超图记忆时，优先替换 `embedding_provider` 和 `vector` 来源，保持 `state_id`、`producer`、`consumers`、`checksum` 等字段稳定。
# 当前实现状态

当前协议版本仍是 `0.2`。HotpotQA-style 主实验中，`protocol` 模式的实际链路是：

```text
PlannerAgent -> ReasonerAgent -> SummarizerAgent
```

协议包会携带 `protocol_version`、`action`、`params`、`result`、`capability_required`、`sender_capability_ref`、`receiver_capability`、`cognitive_gap` 等字段。当前非文本协作传输通过 `state_refs` 实现：

```text
plan_ref: PlannerAgent 写入，ReasonerAgent 读取
answer_ref: ReasonerAgent 写入，SummarizerAgent 读取
```

`latent_states` 的 embedding provider 不再固定为早期的 `hash-v1` 占位值，而是由环境变量 `EMBEDDING_PROVIDER` 决定：

```text
hash
openai_compatible
local
```

当前 Qwen / DashScope `text-embedding-v4` 输出 1024 维向量；本地 `sentence-transformers/all-MiniLM-L6-v2` 输出 384 维向量。当前实现没有声称传递真实 LLM hidden state 或 KV cache，而是实现 embedding-based latent state reference。
