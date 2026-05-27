# Hypergraph Memory Update

本文档说明本次超图记忆模块更新内容，以及智能体同学和前端界面同学现在可以获得的新能力。

## 1. 本次主要做了什么

本次更新把记忆模块从“保存答案并做关键词检索”推进到“保存多智能体协作经验，并基于结构化超图检索”。

公共接口保持不变：

```python
add_memory(memory, source_agent=..., task_topic=..., summary=..., tags=..., trace=...)
search_memory(memory, query, top_k=...)
```

也就是说，智能体侧仍然只需要调用原来的两个接口，不需要改接入方式。

本次新增和改造的核心文件：

```text
examples/competition_agents/hypergraph_memory_method/schema.py
examples/competition_agents/hypergraph_memory_method/query_profile.py
examples/competition_agents/hypergraph_memory_method/distill.py
examples/competition_agents/hypergraph_memory_method/visualization.py
examples/competition_agents/hypergraph_memory_method/visualize_run.py
examples/competition_agents/hypergraph_memory_method/evaluate_retrieval.py
examples/competition_agents/hypergraph_memory_method/builder.py
examples/competition_agents/hypergraph_memory_method/index.py
examples/competition_agents/hypergraph_memory_method/scoring.py
examples/competition_agents/hypergraph_memory_method/text.py
```

## 2. 记忆现在保存什么

每一轮任务结束后，`summarizer_agent` 会调用 `add_memory(..., trace=trace)`。本次更新会从真实 `trace` 中保存和建图：

```text
task_meta         任务编号、组号、轮次、期望复用、问题、支撑事实
messages          planner/retriever/reasoner/summarizer 之间的消息路径
state_refs        plan/evidence/answer 等状态引用
memory_hits       本轮命中过的历史记忆
cognitive_gaps    每个智能体当时的认知缺口
evidence          本轮证据
metrics           message_count、text_chars、latent_bytes 等通信指标
execution_result  reasoner 输出、retrieval_mode、answer_ref 等
```

因此，一条 memory 不再只是答案文本，而是一条“协作经验记录”。

## 3. 超图结构新增了什么

本次更新把一条 memory 转换成多类节点和超边。

主要节点类型：

```text
memory            记忆记录
task/group/turn   任务链、任务组、轮次
agent/action      智能体和动作
state/state_ref   计划、证据、答案等状态引用
memory_ref        历史记忆引用
gap               认知缺口
evidence          证据片段
supporting_fact   支撑事实
entity            证据链实体
strategy          协作策略
experience        可复用经验
reuse_benefit     复用收益
archive_scope     归档范围
lifecycle         生命周期阶段
```

主要超边类型：

```text
memory_event      主记忆事件，兼容旧索引
task_context      任务上下文
agent_message     智能体通信
state_flow        状态流
memory_reuse      记忆复用链路
cognitive_gap     认知缺口
evidence          证据
supporting_fact   支撑事实
strategy          协作策略
experience        可复用经验
reuse_benefit     复用收益
archive_scope     归档范围和生命周期
```

## 4. 检索现在怎么做

检索流程现在分三步：

```text
1. query_profile.py 先分析当前问题
2. index.py 用 query profile 生成的节点召回历史 memory
3. scoring.py 融合结构分、经验分、embedding 分进行排序
```

当前 query 会被分析成：

```text
intent            当前任务意图，例如 reuse / summarize / compare
artifacts         当前需要的产物，例如 evidence / answer / supporting-facts
gaps              当前可能存在的缺口，例如 missing-memory / needs-reasoning
strategies        可能需要的协作策略
experiences       可能需要复用的经验
keywords          低权重关键词
```

相比普通关键词或向量检索，现在检索目标不只是“文本相似”，而是：

```text
当前任务需要什么结构？
哪条历史记忆能覆盖这些结构需求？
```

## 5. 分数现在包含什么

检索结果中现在会返回这些解释分数：

```text
retrieval_score        最终排序分
hypergraph_score       超图结构命中分
embedding_score        query 与 memory embedding 的相似分
gap_coverage_score     认知缺口覆盖分
reuse_score            expected_reuse / memory_ref 命中分
evidence_score         evidence / supporting_fact / entity 命中分
state_ref_score        state / state_ref 命中分
strategy_score         strategy 命中分
experience_score       experience / reuse_benefit 命中分
```

`embedding_score` 已经融合进 `retrieval_score`，不是只做展示字段。

当前融合方式：

```text
retrieval_score =
  hypergraph_score
  + 结构化分数加成
  + embedding_score * 8.0
```

注意：embedding 只对已经被超图结构召回的候选 memory 做重排，不单独召回，避免纯向量噪声拉入无关记忆。

## 6. 智能体同学现在可以获得什么

智能体侧不需要新增接入接口。仍然调用：

```python
search_memory(state["memory"], state["task"], top_k=2)
```

但返回的 `memory_hits` 更丰富了。

每个 hit 现在包含：

```text
memory_id
summary
trace
hypergraph
retrieval_score
hypergraph_score
embedding_score
gap_coverage_score
reuse_score
evidence_score
state_ref_score
strategy_score
experience_score
matched_nodes
matched_hyperedges
reasoning_path
```

智能体同学可以直接使用：

```text
hit["summary"]             历史记忆摘要
hit["evidence"]            历史证据
hit["state_refs"]          历史状态引用
hit["reasoning_path"]      为什么命中这条记忆
hit["matched_nodes"]       命中了哪些结构节点
hit["retrieval_score"]     最终排序分
```

如果后续 reasoner 想更充分利用记忆，可以优先读取：

```text
trace.task_meta.supporting_facts
trace.evidence
trace.state_refs
hypergraph.distilled_experience
```

## 7. 智能体同学后续如果接 executor

现在已经预留 executor 字段。

如果未来 `trace["execution_result"]` 或 `trace["tool_trace"]` 中出现以下字段：

```text
sandbox_status
exit_code
stdout
stderr
tool_elapsed_ms
```

超图构建会自动生成：

```text
tool_execution
sandbox_outcome
```

也就是说，executor 接入时不需要改 `search_memory()` 接口，只要把执行结果放进 trace。

## 8. 前端同学现在可以获得什么

本次新增了真实运行可视化工具。

运行：

```powershell
python examples\competition_agents\hypergraph_memory_method\visualize_run.py `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json `
  --limit 1 `
  --modes protocol
```

会生成：

```text
examples/competition_agents/hypergraph_memory_method/artifacts/memory_graph.html
examples/competition_agents/hypergraph_memory_method/artifacts/memory_graph.json
```

其中：

```text
memory_graph.html   可直接打开查看超图构建和检索路径
memory_graph.json   可供前端二次渲染
```

可视化采用真实运行数据，不是手写样例。

前端可以从 JSON 中读取：

```text
memory_graph.nodes
memory_graph.links
memory_graph.hyperedges
retrieval_graphs[*].query
retrieval_graphs[*].query_nodes
retrieval_graphs[*].hits
retrieval_graphs[*].nodes
retrieval_graphs[*].links
```

节点中包含：

```text
id
kind
value
label
cn_label
description
color
highlight
```

超边中包含：

```text
id
type
description
memory_id
nodes
```

前端可以用这些字段画：

```text
整体记忆超图
单条 memory 的超边展开
某次 query 的检索路径
命中节点高亮
命中超边高亮
memory hit 详情面板
```

## 9. 评估脚本

本次新增真实运行检索评估：

```powershell
python examples\competition_agents\hypergraph_memory_method\evaluate_retrieval.py `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json `
  --limit 1 `
  --modes protocol `
  --top-k 2
```

会生成：

```text
examples/competition_agents/hypergraph_memory_method/artifacts/retrieval_eval.json
examples/competition_agents/hypergraph_memory_method/artifacts/retrieval_eval.md
```

当前统计指标包括：

```text
memory_hit_rate
expected_reuse_hit_rate
structured_hit_ratio
keyword_hit_ratio
keyword_noise_ratio
avg_retrieval_score
avg_hypergraph_score
avg_embedding_score
message_count
text_chars
latent_transfer_count
latent_bytes
```

这份评估用来回答：

```text
是否命中了应该复用的历史记忆？
命中是靠结构节点还是靠关键词？
是否还有数字/年份类噪声 keyword？
embedding 是否参与了最终排序？
通信成本是多少？
```

## 10. 当前真实运行结果

在 continuous 样例上，当前结果为：

```text
memory_hit_rate: 0.6667
expected_reuse_task_count: 2
hypergraph_expected_reuse_hit_rate: 1.0000
keyword_expected_reuse_hit_rate: 1.0000
embedding_expected_reuse_hit_rate: 1.0000
avg_structured_hit_ratio: 0.3233
avg_keyword_hit_ratio: 0.6363
avg_keyword_noise_ratio: 0.0000
avg_embedding_score: 0.5245
text_chars: 2640
latent_bytes: 576
```

需要诚实说明：

```text
当前 continuous 样例较简单，keyword 和 embedding baseline 也能命中 expected reuse。
所以目前不能仅凭这个样例证明最终准确率提升。
当前能证明的是：
1. 结构化记忆构建已经成立。
2. expected reuse 可以命中。
3. 数字/年份噪声 keyword 已经被压下去。
4. embedding_score 已经参与最终排序。
5. 检索路径可以被可视化解释。
```

后续如果要证明方法优势，需要构造更难的评估集，例如：

```text
关键词相似但证据链不同
问题表述不同但复用结构相同
多轮任务中需要复用 answer/state/evidence 的不同组合
```

## 12. 一句话总结

本次更新的核心是：

```text
把每轮多智能体协作从“答案文本”升级为“可解释的超图经验记录”，
并支持基于任务缺口、证据、状态引用、协作策略、经验和 embedding 的综合检索。
```
