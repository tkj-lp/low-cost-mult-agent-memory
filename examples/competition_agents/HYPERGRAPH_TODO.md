# 超图记忆对接任务说明

这份文档给负责超图记忆的同学看，目标是把超图记忆模块和当前多智能体框架稳定接起来，同时避免改坏 Agent 主流程。

## 建议目录结构

超图记忆代码建议单独放一个子包，避免和 Agent 逻辑混在一起：

```text
examples/competition_agents/competition_agents/hypergraph/
  __init__.py
  schema.py
  index.py
  search.py
  scoring.py
```

建议职责：

- `schema.py`：定义超图节点、超边、记忆视图字段。
- `index.py`：构建内存版超图索引，例如 task、agent、memory、tag、keyword、evidence、state_ref 节点。
- `search.py`：对外提供检索函数。
- `scoring.py`：融合关键词、标签、超图路径、embedding 相似度的打分逻辑。

如果当前已经实现了 `hypergraph_memory.py`，可以先继续使用；等功能稳定后再拆成上面的子包。

## 不建议改动的接口

请尽量不要直接改 Agent 文件。Agent 侧只依赖 `memory.py` 暴露的两个函数：

```python
add_memory(memory, source_agent, task_topic, summary, tags) -> list[dict]
search_memory(memory, query, top_k) -> list[dict]
```

也就是说，超图逻辑最好放在 `memory.py` 后面：

```text
SummarizerAgent -> add_memory() -> 超图记忆写入
RetrieverAgent -> search_memory() -> 超图记忆检索
```

这样 Agent、网页、评测脚本都不用频繁改。

## 记忆记录需要保留的基础字段

每条记忆至少保留这些字段：

```json
{
  "memory_id": "mem-0001",
  "source_agent": "summarizer",
  "created_at": "2026-05-25T...",
  "task_topic": "任务主题",
  "summary": "摘要",
  "tags": ["langgraph", "multi-agent"],
  "embedding": [0.1, -0.2],
  "embedding_provider": "local",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

可以额外增加超图字段，但不要删除上面的基础字段。

## 推荐增加的超图字段

可以给记忆记录增加这些可选字段：

```json
{
  "entities": ["Alan Turing", "Bletchley Park"],
  "supporting_facts": ["..."],
  "evidence": ["..."],
  "state_refs": ["state-0001"],
  "reasoning_path": [],
  "hypergraph": {
    "memory_node": "memory:mem-0001",
    "task_node": "task:...",
    "agent_node": "agent:summarizer",
    "tag_nodes": ["tag:multi-agent"],
    "keyword_nodes": ["keyword:protocol"],
    "event_hyperedge": {
      "hyperedge_id": "hyperedge:mem-0001",
      "type": "memory_event",
      "nodes": []
    }
  }
}
```

## 检索返回结果建议

`search_memory()` 返回的每条结果除了原始记忆字段，建议补充这些解释字段：

```json
{
  "retrieval_score": 3.25,
  "hypergraph_score": 2.5,
  "embedding_score": 0.75,
  "matched_nodes": ["tag:protocol", "keyword:memory"],
  "matched_hyperedges": ["hyperedge:mem-0001"],
  "reasoning_path": [
    {
      "query_node": "keyword:protocol",
      "hyperedge_id": "hyperedge:mem-0001",
      "memory_id": "mem-0001"
    }
  ]
}
```

这些字段可以直接给网页同学展示，也方便实验报告解释“为什么命中这条记忆”。

## embedding 相似度要做什么

当前智能体部分已经能生成本地 latent embedding：

```text
provider=local
model=sentence-transformers/all-MiniLM-L6-v2
shape=[384]
```

超图记忆下一步建议把 embedding 用到检索打分里：

1. `add_memory()` 写入记忆时保存 `embedding`、`embedding_provider`、`embedding_model`。
2. `search_memory()` 收到 query 后调用 `embed_text(query)` 生成 query embedding。
3. 对 query embedding 和 memory embedding 计算 cosine similarity。
4. 将语义相似度和超图命中分数融合：

```text
final_score = 0.5 * hypergraph_score + 0.3 * embedding_score + 0.2 * tag_score
```

权重可以调整，但需要在实验报告里说明。

## 和 latent state 的关系

`latent_states` 是当前任务内的非文本状态传递，主要由 `put_latent_state()` 生成。

超图记忆是跨任务长期记忆，主要由 `add_memory()` 和 `search_memory()` 管理。

两者建议这样关联：

```text
latent_states[state_id] -> 当前任务证据向量
memory[memory_id].state_refs -> 指向相关 state_id
hypergraph evidence/state_ref node -> 连接 memory 和 evidence
```

这样可以在报告里说明：系统既支持任务内 latent state 传递，也支持跨任务超图记忆复用。

## 验证标准

超图模块完成后，至少要能证明：

- 连续任务中 `memory_hits` 大于 0。
- 返回结果包含 `matched_nodes`、`matched_hyperedges`、`reasoning_path`。
- 同一组任务在第二次运行时能复用之前记忆。
- 本地 embedding 模式下，记忆记录里能看到 `embedding_provider=local`。
- 网页同学可以直接读取 `memory_hits` 展示检索路径。

推荐运行：

```powershell
cd D:\qq\low-cost-memory-integrate
$env:PYTHONPATH="libs/langgraph;examples/competition_agents"
$env:EMBEDDING_PROVIDER="local"
$env:LOCAL_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
python examples\competition_agents\run_demo.py
```

输出里重点看：

```text
memory_hits=...
latent_transfers=...
latent_bytes=...
```

如果有 Hotpot 样例，也可以运行：

```powershell
python examples\competition_agents\run_hotpot_samples.py
```

## 新增任务：记忆生命周期管理

超图记忆需要支持同一个 `MemoryUnit` 下的两类记忆，不要拆成两套存储：

```text
episodic memory：连续任务中的短期记忆，scope=当前 group_id，ttl=session
experience memory：任务结束后蒸馏出的长期经验，scope=global，ttl=persistent
```

### 需要新增或兼容的字段

每条记忆记录建议补充：

```json
{
  "memory_type": "episodic",
  "scope": "chain-001",
  "status": "active",
  "ttl": "session",
  "strategy": "...",
  "experience": "...",
  "source_memory_ids": []
}
```

字段要求：

- `memory_type` 只能是 `episodic` 或 `experience`。
- `scope` 对短期记忆是连续任务组 id，例如 `chain-001`；对长期经验是 `global`。
- `status` 默认是 `active`；任务结束后可改成 `archived`。
- `source_memory_ids` 用于记录长期经验是从哪些短期记忆蒸馏来的。

### 需要实现的能力

第一阶段先做兼容字段和检索过滤：

```python
search_memory(memory, query, top_k, scope=None, memory_type=None, include_global=True)
```

如果暂时不改函数签名，也可以在内部默认执行：

```text
连续任务：优先查当前 scope 的 episodic，再查 global 的 experience
普通任务：优先查 global 的 experience
```

第二阶段再补生命周期函数：

```python
archive_scope(memory, scope) -> list[dict]
distill_experience(memory, scope) -> list[dict]
```

`archive_scope()` 负责把当前 scope 下的 episodic memory 从 `active` 改成 `archived`，不要直接删除，方便页面回放和实验复现。

`distill_experience()` 负责从一个 scope 下的多条 episodic memory 里生成一条或多条 `memory_type=experience`、`scope=global` 的长期经验记忆。

### 超图建模建议

新增节点：

```text
MemoryType: episodic / experience
Scope: chain-001 / global
Status: active / archived
Strategy: 策略片段
Experience: 经验片段
```

新增超边：

```text
scope_membership: Memory + Scope + MemoryType
experience_distillation: Experience + source episodic memories
```

### 验证标准

完成后至少能证明：

- 连续任务第 2、3 轮能命中同一 `scope` 下的 episodic memory。
- 任务结束后 episodic memory 可以被标记为 `archived`。
- 可以从 archived episodic memory 生成 `scope=global` 的 experience memory。
- 新任务可以命中 `memory_type=experience` 的长期经验。
- `memory_hits` 中保留 `memory_type/scope/status`，页面可以直接展示。
