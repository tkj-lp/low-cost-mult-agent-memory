# 超图记忆同学工作文档

本文档说明超图记忆模块需要对接的稳定接口。目标是让超图同学可以独立实现超图记忆和 embedding 相似度融合检索，而不需要改 Agent 主流程。

## 工作边界

超图同学主要负责：

- 记忆单元的超图组织。
- 任务、Agent、工具、证据、记忆、标签等节点建模。
- 超边构建和可解释检索路径。
- 关键词、标签、超图路径和 embedding 相似度融合打分。
- 返回 `memory_hits`，供 Agent 和页面复用。

不建议直接修改：

- `agents/planner/agent.py`
- `agents/retriever/agent.py`
- `agents/reasoner/agent.py`
- `agents/summarizer/agent.py`
- `competition_agents/graph.py`

如果确实需要新增字段，优先在 `memory.py` 返回的记忆记录里做兼容扩展。

## 稳定入口

Agent 侧只依赖 `competition_agents/memory.py` 中的两个函数：

```python
add_memory(memory, source_agent, task_topic, summary, tags) -> list[dict]
search_memory(memory, query, top_k) -> list[dict]
```

推荐结构：

```text
SummarizerAgent -> add_memory() -> 超图记忆写入
RetrieverAgent -> search_memory() -> 超图记忆检索
```

只要这两个函数的输入输出保持兼容，超图内部怎么实现都不会影响其他同学。

## 记忆记录基础字段

每条记忆至少保留：

```json
{
  "memory_id": "mem-0001",
  "source_agent": "summarizer",
  "created_at": "2026-05-25T...",
  "task_topic": "任务主题",
  "summary": "摘要描述",
  "tags": ["langgraph", "multi-agent"],
  "embedding": [0.1, -0.2],
  "embedding_provider": "local",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

不要删除这些字段。页面和 Agent 会依赖它们。

## 推荐增加的超图字段

可以在记忆记录里增加：

```json
{
  "entities": ["Alan Turing", "Bletchley Park"],
  "supporting_facts": ["..."],
  "evidence": ["..."],
  "state_refs": ["state-0001"],
  "answer": "United Kingdom",
  "reuse_count": 1,
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

这些字段都是可选扩展。已有 Agent 不应因为缺少这些字段而崩溃。

## 检索返回建议

`search_memory()` 返回 `list[dict]`。每条命中建议补充：

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

页面同学可以直接展示这些字段，实验报告也能解释“为什么命中这条记忆”。

## embedding 相似度融合检索

建议检索打分分三部分：

```text
final_score = 0.5 * hypergraph_score + 0.3 * embedding_score + 0.2 * tag_score
```

推荐流程：

1. `add_memory()` 写入记忆时保存 `embedding`。
2. `search_memory()` 收到 query 后调用 `embed_text(query)` 生成 query embedding。
3. 对 query embedding 和 memory embedding 计算 cosine similarity。
4. 结合超图路径命中、标签命中、关键词命中得到最终分数。
5. 返回 top-k 命中结果和解释路径。

权重可以调整，但要在实验报告中说明。

## 和 latent_states 的关系

`latent_states` 是当前任务内的非文本状态。超图记忆是跨任务长期记忆。

推荐关联方式：

```text
latent_states[state_id] -> 当前任务证据或 hidden state
memory[memory_id].state_refs -> 指向相关 state_id
hypergraph evidence/state_ref node -> 连接 memory 和 evidence
```

后续如果智能体侧新增 hidden latent，不需要改超图接口，只要把新的 `state_id` 放入 `state_refs` 即可。

## 验证标准

超图模块完成后，至少验证：

- 连续任务中 `memory_hits` 大于 0。
- `memory_hits` 包含 `retrieval_score` 或类似分数字段。
- `memory_hits` 包含 `matched_nodes`、`matched_hyperedges`、`reasoning_path`。
- 本地 embedding 模式下，记忆记录包含 `embedding_provider=local`。
- 页面可以直接读取 `memory_hits` 展示检索路径。

