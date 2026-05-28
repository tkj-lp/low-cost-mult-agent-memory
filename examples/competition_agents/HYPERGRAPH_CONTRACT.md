# 超图记忆对接文档

本文档是给超图记忆同学看的唯一最新版。旧的 `HYPERGRAPH_MEMORY_ALIGNMENT.md`、`HYPERGRAPH_MEMORY_UPDATE.md`、`HYPERGRAPH_TODO.md` 已废弃，避免多个版本互相冲突。

当前目标不是让超图模块绕过智能体主流程，而是让超图记忆给 `RetrieverAgent` 提供可解释、可裁剪、可复用的记忆命中结果。

## 1. 当前主链路

```text
PlannerAgent
-> RetrieverAgent
-> ReasonerAgent
-> SummarizerAgent
```

职责划分：

```text
PlannerAgent：生成计划，并写入 plan_ref。
RetrieverAgent：调用超图记忆检索，生成 evidence_ref，裁剪 memory_context。
ReasonerAgent：读取 plan、evidence、memory_context，生成 answer_ref。
SummarizerAgent：输出 final_answer，并通过 add_memory(..., trace=trace) 写入共享记忆。
```

## 2. 超图模块边界

超图同学主要维护：

```text
examples/competition_agents/hypergraph_memory_method/
```

多智能体主流程通过下面两个稳定入口接入超图记忆：

```python
add_memory(memory, *, source_agent, task_topic, summary, tags, trace=None) -> list[dict]
search_memory(memory, query, *, top_k) -> list[dict]
```

不要直接改这些 Agent 文件，除非和智能体同学确认：

```text
agents/planner/agent.py
agents/retriever/agent.py
agents/reasoner/agent.py
agents/summarizer/agent.py
competition_agents/graph.py
```

如果需要给 Agent 或页面增加信息，优先在 `search_memory()` 返回的 `memory_hits` 中追加兼容字段。

## 3. add_memory 写入内容

每轮任务结束时，`SummarizerAgent` 会调用：

```python
add_memory(
    memory,
    source_agent="summarizer",
    task_topic=state["task"],
    summary=summary,
    tags=[...],
    trace=trace,
)
```

`trace` 当前包含：

```text
task_meta
plan
messages
evidence
memory_hits
memory_context
latent_states
state_refs
execution_result
cognitive_gaps
agent_outputs
metrics
```

这意味着一条 memory 不只是答案文本，而是一条完整协作经验记录。超图可以从 trace 中建：

```text
task-agent-message-evidence-state_ref-memory_ref-outcome
```

## 4. memory record 基础字段

每条记忆至少保留：

```json
{
  "memory_id": "mem-0001",
  "source_agent": "summarizer",
  "created_at": "2026-05-28T...",
  "task_topic": "任务文本",
  "summary": "最终总结或可复用经验摘要",
  "tags": ["langgraph", "multi-agent", "protocol"],
  "embedding": [0.1, -0.2],
  "embedding_provider": "hash",
  "embedding_model": "hash-v1"
}
```

可选但推荐保留：

```json
{
  "entities": [],
  "supporting_facts": [],
  "evidence": [],
  "reasoning_path": [],
  "state_refs": ["state-0001", "state-0002"],
  "answer": "answer text",
  "reuse_count": 0,
  "trace": {},
  "hypergraph": {}
}
```

## 5. search_memory 返回内容

`RetrieverAgent` 调用：

```python
hits = search_memory(state["memory"], state["task"], top_k=2)
```

返回值保持 `list[dict]`。每个 hit 是完整 memory record 的兼容扩展，建议包含：

```json
{
  "memory_id": "mem-0001",
  "summary": "历史记忆摘要",
  "evidence": ["关键证据"],
  "state_refs": ["state-0002"],
  "retrieval_score": 75.5,
  "hypergraph_score": 65.75,
  "embedding_score": 0.59,
  "matched_nodes": ["artifact:evidence", "state:evidence"],
  "matched_hyperedges": ["hyperedge:mem-0001:evidence"],
  "reasoning_path": [
    {
      "query_node": "artifact:evidence",
      "hyperedge_id": "hyperedge:mem-0001:evidence",
      "memory_id": "mem-0001"
    }
  ]
}
```

字段作用：

```text
retrieval_score：最终排序分。
hypergraph_score：超图结构命中分。
embedding_score：query 与 memory embedding 的相似分。
matched_nodes：命中的结构节点，用于解释和页面高亮。
matched_hyperedges：命中的超边，用于解释和页面高亮。
reasoning_path：query 如何走到该 memory 的路径。
```

## 6. memory_hits 和 memory_context 的区别

现在在线链路同时保留两类数据：

```text
memory_hits：完整检索命中，用于 trace、离线分析、超图调试和完整展示。
memory_context：裁剪后的在线推理上下文，用于 RetrieverAgent 传给 ReasonerAgent。
```

`memory_hits` 可以较完整；`memory_context` 必须短、稳定、可解释。

当前链路：

```text
HyperGraphMemory.search()
-> RetrieverAgent 得到完整 memory_hits
-> RetrieverAgent 裁剪成 memory_context
-> RetrieverAgent 传 memory_refs + memory_context + evidence_ref
-> ReasonerAgent 使用 memory_context 推理
```

注意：`RetrieverAgent` 只把本轮 `evidence` 写入 latent state；`memory_context` 不再参与 evidence latent embedding。

## 7. memory_context 推荐 schema

`memory_context` 的目标不是把检索结果再压成向量，而是保留 Reasoner 正常推理需要的最小可解释信息：

```json
[
  {
    "memory_id": "mem-0001",
    "summary": "一句话说明这条记忆支持什么结论",
    "evidence": ["最多 1-2 条关键证据句"],
    "reasoning_path": [
      {
        "query_node": "artifact:evidence",
        "hyperedge_id": "hyperedge:mem-0001:evidence",
        "memory_id": "mem-0001"
      }
    ],
    "matched_nodes": ["artifact:evidence", "state:evidence", "tag:protocol"],
    "matched_hyperedges": ["hyperedge:mem-0001:evidence"],
    "retrieval_score": 75.5,
    "hypergraph_score": 65.75,
    "embedding_score": 0.59
  }
]
```

裁剪规则建议：

```text
top_k：2 或 3
summary：最多 200-300 字符
evidence：最多 2 条，每条最多 150-200 字符
reasoning_path：最多 3 条
matched_nodes：最多 8-12 个
matched_hyperedges：最多 6-8 个
score：全部保留
```

为什么保留这些字段：

```text
memory_id：后续 Agent 引用完整记忆，不直接传完整正文。
summary：告诉 Reasoner 这条记忆支持什么结论。
evidence：告诉 Reasoner 这条记忆凭什么可信。
reasoning_path：告诉 Reasoner 和页面为什么命中这条记忆。
matched_nodes / matched_hyperedges：用于超图局部视图高亮命中路径。
score：用于排序、置信度和收益解释。
```

不要把 `memory_context` 裁成纯向量：

```json
{"memory_id": "mem-0001", "vector": [0.12, 0.33]}
```

Reasoner 无法直接解释向量，还需要额外解码或解释层。

也不要把完整 record、完整 trace、完整 messages 放进 `memory_context`，否则会重新拉高通信开销。完整信息继续放在 `memory_hits` 和 trace 里。

## 8. 检索打分建议

当前检索应优先体现结构命中，再融合 embedding 分：

```text
retrieval_score =
  hypergraph_score
  + structured_bonus
  + embedding_score * weight
```

建议超图召回优先使用：

```text
task_meta
expected_reuse
supporting_facts
state_refs
memory_refs
evidence
strategy / experience
```

关键词和 embedding 可以作为补充，但不要让无关关键词或纯向量相似度覆盖结构信号。

## 9. 页面和实验展示字段

页面可以直接读取：

```text
memory_hits
memory_context
state_refs
latent_states
messages
metrics
```

其中：

```text
memory_hits：展示完整命中详情。
memory_context：展示在线推理实际使用的裁剪上下文。
matched_nodes / matched_hyperedges：画超图局部高亮。
reasoning_path：解释 query 到 memory 的命中路径。
```

## 10. 验证命令

Windows PowerShell：

```powershell
cd D:\qq\low-cost-memory-integrate
$env:PYTHONPATH="libs/langgraph;examples/competition_agents"
$env:LLM_PROVIDER="mock"
$env:LLM_PLANNER_PROVIDER="mock"
$env:LLM_REASONER_PROVIDER="mock"
$env:LLM_SUMMARIZER_PROVIDER="mock"
$env:EMBEDDING_PROVIDER="hash"

python examples\competition_agents\run_demo.py `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json `
  --limit 1 `
  --modes protocol `
  --show-memory `
  --show-latent `
  --show-state
```

重点看：

```text
turn-1: memory_hits=0
turn-2: memory_hits=1
turn-3: memory_hits=2
top_level_state_refs.evidence = state-xxxx
execution_result.memory_context 非空
retriever -> reasoner message 中有 memory_refs 和 memory_context
```

测试：

```powershell
$env:PYTHONPATH="libs/langgraph;examples/competition_agents"
python -m pytest -p no:cacheprovider `
  examples\competition_agents\tests\test_hypergraph_memory.py `
  examples\competition_agents\tests\test_retriever_memory_context.py
```

## 11. 当前结论

最新对接口径：

```text
超图记忆负责返回完整、可解释的 memory_hits。
RetrieverAgent 负责把 memory_hits 裁剪成 memory_context。
ReasonerAgent 使用 memory_context 推理。
完整 trace 继续写入 memory，供超图离线建图和页面展示。
```

这个设计兼顾三件事：

```text
1. Reasoner 有足够信息推理，不只看到 memory_id。
2. 在线消息不传完整 trace，不把检索结果重新 embedding。
3. 超图仍保留完整 memory_hits 和 trace，方便可视化和离线分析。
```
