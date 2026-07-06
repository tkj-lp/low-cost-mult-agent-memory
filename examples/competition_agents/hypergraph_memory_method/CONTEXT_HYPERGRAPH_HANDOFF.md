# 上下文超图模块交接文档

## 1. 当前结论

上下文抽取并构成超图的功能已经可以交给智能体同学调用。

当前已完成：

- 从不同数据集字段中抽取上下文。
- 将上下文组织成轻量超图结构。
- 暴露统一检索接口 `search_context(...)`。
- 同时提供纯文本基线 `search_text_context(...)`，方便做消融对比。
- 提供 TopiOCQA 检索对比脚本，用于验证上下文超图和文本检索的差异。

当前不需要智能体同学理解超图内部结构，只需要调用接口并读取返回的 compact context hits。

## 2. 智能体同学需要调用的接口

推荐只调用这个接口：

```python
from hypergraph_memory_method import search_context

context_hits = search_context(
    task_meta,
    query=question,
    dataset_type="topiocqa",
    planner_focus=None,
    top_k=3,
)
```

参数说明：

```text
task_meta:
  当前任务元信息。可以直接传数据集里的单条样本，也可以传 run_demo 里整理后的 task_meta。

query:
  当前问题文本。建议传当前用户问题。

dataset_type:
  数据集类型。当前支持 topiocqa / hotpot / hotpot_continuous / codeact / generic。
  如果不传，模块会根据字段自动推断。

planner_focus:
  可选。Planner 如果已经识别出关键实体、问题意图、缺口，可以传字符串列表。

top_k:
  返回前 k 条上下文。
```

返回结果示例：

```python
[
    {
        "context_id": "chain-001:turn-2:supporting_passage:002:7c364253",
        "dataset_type": "topiocqa",
        "kind": "supporting_passage",
        "title": "Dunkirk (2017 film)",
        "section": "Introduction",
        "text": "Dunkirk is a 2017 war film ...",
        "context_score": 4.0,
        "matched_nodes": [
            "context:chain-001:turn-2:supporting-passage:002:7c364253",
            "keyword:film",
            "topic:dunkirk (2017 film)"
        ],
        "matched_hyperedges": [
            "context_hyperedge:chain-001:turn-2:supporting-passage:002:7c364253:task"
        ],
        "reasoning_path": [
            {
                "query_node": "keyword:film",
                "hyperedge_id": "context_hyperedge:chain-001:turn-2:supporting-passage:002:7c364253:task",
                "context_id": "chain-001:turn-2:supporting_passage:002:7c364253"
            }
        ]
    }
]
```

智能体侧最少只需要使用：

```text
hit["text"]
hit["title"]
hit["section"]
hit["context_score"]
```

如果前端或日志要展示“为什么命中”，再读取：

```text
matched_nodes
matched_hyperedges
reasoning_path
```

## 3. 支持的数据集字段

### TopiOCQA

当前支持：

```text
question
topic
topic_section
context
supporting_passage
chain_id
turn
answer
```

实际检索上下文主要来自：

```text
context -> dialogue_question / dialogue_answer
supporting_passage -> supporting_passage
topic -> topic node
topic_section -> section node
```

注意：`answer` 只用于测试评估，不作为在线检索上下文。

### Hotpot / continuous

当前支持：

```text
context[].title
context[].sentences[]
supporting_facts
question
task_id
group_id
turn_id
```

上下文会被拆成 sentence 级别候选。

### CodeAct

当前支持：

```text
task_type
inputs
allowed_imports
tags
```

会组织成一个 `tool_task_context` 候选上下文。

## 4. 超图内部组织方式

每个任务会临时构建一个上下文超图。

主要节点类型：

```text
task: 当前任务
dataset: 数据集类型
group: 连续任务链
turn: 当前轮次
question: 当前问题
context: 候选上下文片段
topic: 主题
section: 章节
entity: 实体
date: 日期
keyword: 关键词
```

主要超边类型：

```text
context_task_membership:
  把 dataset / task / group / turn / question 连起来。

task_context:
  把当前任务问题和候选上下文连起来。

context_topic:
  把上下文片段和 topic / section 连起来。

context_entities:
  把上下文片段和抽取到的实体、日期连起来。

context_keywords:
  把上下文片段和关键词连起来。

candidate_evidence:
  表示某个上下文片段是当前问题的候选证据。
```

智能体同学不需要直接操作这些节点和超边。

## 5. 纯文本基线接口

为了比较普通文本检索和超图检索，我们也暴露了纯文本上下文接口：

```python
from hypergraph_memory_method import search_text_context

text_hits = search_text_context(
    task_meta,
    query=question,
    dataset_type="topiocqa",
    top_k=3,
)
```

区别：

```text
search_text_context:
  只把上下文拍平成文本，根据词重叠排序。

search_context:
  会区分 task / topic / section / supporting_passage / dialogue history / entity / keyword。
```

当前 TopiOCQA 测试中，上下文超图比纯文本更容易命中真正的 `supporting_passage`。

## 6. 记忆检索相关接口

已有超图记忆接口：

```python
from hypergraph_memory_method import search_hypergraph_memory

memory_hits = search_hypergraph_memory(memory, query, top_k=3)
```

纯文本记忆基线接口：

```python
from hypergraph_memory_method import search_text_memory

text_memory_hits = search_text_memory(memory, query, top_k=3)
```

对比指标接口：

```python
from hypergraph_memory_method import compare_retrieval_sets

report = compare_retrieval_sets(
    text_hits=text_memory_hits,
    hypergraph_hits=memory_hits,
    expected_ids={"mem-0001"},
    top_k=3,
)
```

## 7. 推荐智能体接入方式

RetrieverAgent 可以按这个顺序接入：

```python
question = task_meta["question"]

context_hits = search_context(
    task_meta,
    query=question,
    dataset_type=task_meta.get("dataset_type"),
    top_k=3,
)

compact_context = [
    {
        "context_id": hit["context_id"],
        "title": hit["title"],
        "section": hit["section"],
        "text": hit["text"],
        "score": hit["context_score"],
    }
    for hit in context_hits
]
```

然后把 `compact_context` 放进 Retriever 给 Reasoner 的 evidence 或 memory_context 中。

如果要减少通信字符，可以只传：

```text
title
section
text
context_id
```

如果要前端解释命中路径，再额外传：

```text
matched_nodes
matched_hyperedges
reasoning_path
```

## 8. 测试脚本

TopiOCQA 正式模型测试脚本：

```powershell
python examples\competition_agents\hypergraph_memory_method\run_topiocqa_retrieval_compare.py `
  --memory-source agent `
  --experiment-mode protocol_api_embedding `
  --dataset-file examples\competition_agents\data\topiocqa_two_chains_10turns_refined.json `
  --limit 10 `
  --context-top-k 3 `
  --memory-top-k 3
```

不调用模型的离线检索链路测试：

```powershell
python examples\competition_agents\hypergraph_memory_method\run_topiocqa_retrieval_compare.py `
  --memory-source dataset `
  --experiment-mode protocol_api `
  --limit 2 `
  --context-top-k 2 `
  --memory-top-k 2
```

## 9. 已验证现象

在 TopiOCQA 10 轮测试中，当前结果显示：

```text
纯文本上下文 supporting passage 命中率：0.6
上下文超图 supporting passage 命中率：0.9
```

这说明上下文超图目前已经能体现优势。

但是记忆检索侧目前差异还不够明显：

```text
纯文本记忆命中率：0.9
超图记忆命中率：0.9
```

原因是当前 memory 主要还是自然语言 summary，结构化经验字段还不够强。

## 10. 当前未完成或需要后续优化

当前已经可用，但还有这些限制：

```text
1. 上下文超图是每次检索临时构建，还没有持久化缓存。

2. 记忆超图和纯文本记忆的差异还没有完全拉开。
   原因是 agent 写入 memory 时主要还是 summary 文本。

3. 如果要让记忆超图更强，需要 agent trace 里更明确写入：
   - 当前问题缺口
   - 指代解析结果
   - supporting_passage id
   - answer entity
   - reused memory id
   - 本轮复用收益

4. 当前检索速度指标只适合作为调试参考。
   真正比赛展示更应该看：
   - supporting_passage 命中率
   - exact_match
   - returned_chars
   - memory reuse precision
   - reasoning_path 可解释性
```

## 11. 给智能体同学的一句话说明

你只需要调用：

```python
from hypergraph_memory_method import search_context
```

传入当前 `task_meta` 和 `question`，拿返回的 `context_hits` 作为紧凑证据上下文即可。超图节点、超边、字段归一化都在方法层内部完成，不需要改智能体内部逻辑。

## 12. 记忆子系统 1.0 增量能力

本次增强没有修改智能体侧已经使用的 `add_memory(...)`、`search_memory(...)`、`refine_memory(...)` 和 `search_hypergraph_memory(...)` 接口。

新增能力如下：

```text
1. 统一记忆记录：补齐全局记忆编号、版本、细粒度记忆种类、来源、生命周期和检索文档。
2. 显式证据链：记录有序证据、推理结论、最终结论、依赖、支持、反驳和来源。
3. 证据链校验：检查空步骤、缺失来源、断裂依赖、循环依赖和无证据结论。
4. 紧凑复用投影：后续智能体可以只读取记忆编号、摘要和必要证据链，不需要传递完整轨迹。
5. 多路检索：关键词、标签、向量、超图和融合五种模式使用相同输入输出结构。
6. 超图检索：独立多路召回后执行跨记忆关系扩展、证据链质量约束、生命周期和复用反馈重排。
7. 持久化：提供带事务、增量索引、重启恢复、导入导出和显式复用反馈的本地数据库存储。
8. 公平评测：提供证据链感知检索指标和有记忆/无记忆配对因果净增益指标。
```

可选的新接口：

```python
from hypergraph_memory_method import (
    SQLiteHypergraphMemoryStore,
    evaluate_paired_memory_gain,
    evaluate_retrieval_modes,
    normalize_memory_record,
    search_memory_by_mode,
    validate_evidence_chain,
)
```

检索模式示例：

```python
hits = search_memory_by_mode(memory, query, top_k=3, mode="hypergraph")
```

当前两组十轮连续任务的离线公平检索结果中，超图模式均为最佳单路检索：第一组总体第一，第二组仅次于融合模式。正式报告仍应在目标操作系统和正式模型配置下重复运行并保留原始结果。

### 12.1 不改调用接口的可选持久化

原有写入和超图检索入口支持通过环境变量自动保存与加载：

```powershell
$env:HYPERGRAPH_MEMORY_DB='run_result/hypergraph_memory.sqlite3'
$env:HYPERGRAPH_MEMORY_AUTOPERSIST='1'
$env:HYPERGRAPH_MEMORY_AUTOLOAD='1'
```

配置后，`build_hypergraph_view(...)` 在原有记忆写入路径中自动写入数据库，`search_hypergraph_memory(...)` 会合并当前进程记忆和历史持久化记忆。未配置数据库路径时完全保持原来的纯内存行为。

注意：智能体侧 `search_memory(...)` 在传入空列表时会在调用超图模块前直接返回。受“不修改智能体侧文件”的边界约束，冷启动后的第一条空记忆查询需要由调用侧先执行 `SQLiteHypergraphMemoryStore.list()`；从产生第一条当前运行记忆开始，原检索链路会自动合并历史持久化记忆。
