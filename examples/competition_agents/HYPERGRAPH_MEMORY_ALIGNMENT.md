# 超图记忆与检�?Agent 对接说明

这份文档只说明一件事：超图记忆模块如何接入当前多智能体系统，重点是和 `RetrieverAgent` 对齐�?
## 完整目标流程

最终框架应是：

```text
PlannerAgent
  -> RetrieverAgent
  -> ReasonerAgent
  -> SummarizerAgent
```

各模块分工：

- `PlannerAgent`：拆解问题，生成推理计划�?- `RetrieverAgent`：调用超图记忆模块，检索相关记忆、证据、实体链�?supporting facts�?- `ReasonerAgent`：基�?`plan + evidence + memory_hits` 推理得到 `answer_candidate`�?- `SummarizerAgent`：输出最终答案，并把本轮结果写入共享记忆�?
超图记忆同学最重要的工作是�?*�?`RetrieverAgent` 能拿到真实可用的 `memory_hits` �?`evidence`�?*

## 你需要实现的核心接口

当前对接点在�?
```text
examples/competition_agents/competition_agents/memory.py
```

需要保持这两个函数签名�?
```python
def add_memory(
    memory: list[dict],
    *,
    source_agent: str,
    task_topic: str,
    summary: str,
    tags: list[str],
) -> list[dict]:
    ...


def search_memory(memory: list[dict], query: str, *, top_k: int) -> list[dict]:
    ...
```

你可以把内部实现替换成超图存储和超图检索，但返回格式要兼容现有 Agent�?
## add_memory 要写入什�?
`SummarizerAgent` 每轮结束后会调用 `add_memory()`�?
你需要把这次任务沉淀成一�?`MemoryUnit`，至少包含：

```json
{
  "memory_id": "mem-0001",
  "source_agent": "summarizer",
  "created_at": "2026-05-20T...",
  "task_topic": "original question or task",
  "summary": "final answer or reusable task summary",
  "tags": ["qa", "hotpot", "multi-hop"],
  "embedding": [0.1, -0.2]
}
```

建议额外支持这些字段�?
```json
{
  "entities": ["Alan Turing", "Bletchley Park", "United Kingdom"],
  "answer": "United Kingdom",
  "supporting_facts": [
    ["Alan Turing", "During World War II, Turing worked at Bletchley Park."],
    ["Bletchley Park", "Bletchley Park is located in the United Kingdom."]
  ],
  "state_refs": ["state-0001"],
  "hypergraph_node_id": "hg-node-001",
  "hyperedges": ["hg-edge-001"]
}
```

## search_memory 要返回什�?
`RetrieverAgent` 会调用：

```python
hits = search_memory(state["memory"], state["task"], top_k=2)
```

你返回的每个 hit 至少要有�?
```json
{
  "memory_id": "mem-0001",
  "task_topic": "previous related question",
  "summary": "reusable answer or reasoning summary",
  "tags": ["qa", "hotpot", "entity-bridge"],
  "embedding": [0.1, -0.2]
}
```

建议额外返回�?
```json
{
  "retrieval_score": 0.91,
  "entities": ["Alan Turing", "Bletchley Park"],
  "supporting_facts": [
    ["Alan Turing", "During World War II, Turing worked at Bletchley Park."]
  ],
  "evidence": [
    "During World War II, Turing worked at Bletchley Park.",
    "Bletchley Park is located in the United Kingdom."
  ],
  "reasoning_path": [
    "Alan Turing",
    "Bletchley Park",
    "United Kingdom"
  ],
  "hyperedges": ["hg-edge-001", "hg-edge-002"]
}
```

## RetrieverAgent 期望拿到的内�?
接入超图记忆后，`RetrieverAgent` 应该能从 `search_memory()` 结果里组装：

```python
evidence = [
    "During World War II, Turing worked at Bletchley Park.",
    "Bletchley Park is located in the United Kingdom.",
]

memory_hits = hits
memory_refs = [hit["memory_id"] for hit in hits]
```

然后传给 `ReasonerAgent`�?
```text
task + plan + evidence + memory_hits
```

也就是说，超图记忆模块不是只返回“相关记�?id”，还要尽量返回**可直接参与推理的证据�?*�?
## 推荐超图结构

HotpotQA / MuSiQue 这类多跳问答推荐建这些节点：

- `Question`
- `Answer`
- `Entity`
- `DocumentTitle`
- `SupportingFact`
- `MemoryUnit`
- `Agent`
- `Tag`

推荐超边�?
```text
(Question, Entity, Entity, Answer)
(Question, SupportingFact, Answer)
(MemoryUnit, Question, Answer)
(MemoryUnit, SupportingFact...)
(Entity, SupportingFact, DocumentTitle)
```

例子�?
```text
Question:
Alan Turing worked at an institution during World War II. Which country is that institution located in?

Reasoning path:
Alan Turing -> Bletchley Park -> United Kingdom

Supporting facts:
Alan Turing -> During World War II, Turing worked at Bletchley Park.
Bletchley Park -> Bletchley Park is located in the United Kingdom.
```

可以建成�?
```text
(question, Alan Turing, Bletchley Park, United Kingdom)
(Alan Turing, supporting_fact_1, Bletchley Park)
(Bletchley Park, supporting_fact_2, United Kingdom)
(memory_id, question, answer, supporting_fact_1, supporting_fact_2)
```

## 与协议字段的对齐

`RetrieverAgent` 发给 `ReasonerAgent` 的协议包里应包含�?
```json
{
  "action": "reason",
  "state_refs": ["state-0001"],
  "memory_refs": ["mem-0001"],
  "result": {
    "evidence_refs": ["state-0001"],
    "memory_hits": ["mem-0001"],
    "supporting_facts": [
      ["Alan Turing", "During World War II, Turing worked at Bletchley Park."]
    ]
  }
}
```

字段含义�?
- `memory_refs`：命中的历史记忆 id�?- `state_refs`：非文本状态引用，例如证据向量或压缩证据�?- `supporting_facts`：可解释推理证据�?- `evidence_refs`：证据状态引用�?
## 非文本状态如何接

如果你有超图节点向量或真�?embedding，可以替换当�?`state.py` 里的 hash 向量来源�?
当前非文本状态格式：

```json
{
  "state_id": "state-0001",
  "type": "latent_state_ref",
  "content_type": "latent.embedding",
  "producer": "retriever",
  "consumers": ["reasoner", "summarizer"],
  "topic": "current task",
  "summary": "compressed evidence representation",
  "embedding_provider": "hash-v1",
  "shape": [16],
  "vector": [0.1, -0.2],
  "checksum": "xxxxxxxxxxxxxxxx"
}
```

你可以把�?
```json
"embedding_provider": "hash-v1"
```

替换成：

```json
"embedding_provider": "hypergraph-node-embedding"
```

但请保留�?
- `state_id`
- `producer`
- `consumers`
- `summary`
- `vector`
- `checksum`

## 最小可交付版本

你最少需要交付：

1. `add_memory()` 能把任务结果写入超图�?2. `search_memory()` 能根据新问题返回相关 `MemoryUnit`�?3. 每个 hit 包含 `memory_id`、`summary`、`tags`�?4. 如果是问答任务，尽量返回 `supporting_facts` �?`evidence`�?5. 返回结果保持 list[dict]，不要改 Agent 调用方式�?
## 完整可交付版�?
更完整的版本应支持：

1. 实体抽取�?2. supporting facts 存储�?3. 多跳 reasoning path�?4. 超边检索�?5. embedding / 节点向量检索�?6. `memory_refs` �?`state_refs` 回传�?7. 检索分�?`retrieval_score`�?
## 验收标准

接入后我们应该能看到�?
- `RetrieverAgent` 回到主流程�?- `memory_hits` 不再总是 0�?- `ReasonerAgent` 输入中出现真�?`evidence`�?- 协议包中出现 `memory_refs`�?- 协议包中出现 `state_refs`�?- HotpotQA-style 样例仍能正确回答�?- 多轮任务中后面的任务能复用前面的记忆�?
