# 多智能体低成本记忆系统统一评测指标

## 1. 评测目标

本项目不能只证明“流程能够运行”，而应同时回答以下问题：

1. 最终答案是否正确。
2. 多智能体通信是否真正降低了文本传输和运行成本。
3. 上下文检索是否找到了回答问题所需的背景证据。
4. 记忆检索是否找到了应该复用的历史经验，而不是仅仅返回相似文本。
5. 超图结构相比纯文本检索和纯向量检索是否带来稳定收益。
6. 检索结果是否短、可解释，并且确实被后续智能体使用。

因此，评测分为四层：

```text
任务效果层
通信与成本层
上下文/记忆检索层
超图结构专项层
```

技术报告中的核心结论应优先使用外部效果指标，例如准确率、正确复用率和通信成本。`retrieval_score`、`hypergraph_score` 等内部排序分数只能用于解释和消融，不能单独证明方法有效。

## 2. 对比方法

所有正式实验至少应包含以下三种记忆或检索方法：

| 方法 | 说明 | 作用 |
| --- | --- | --- |
| 纯文本基线 | 将上下文或记忆展平为文本，按词项重叠排序 | 代表常见轻量文本记忆 |
| 纯向量基线 | 使用 embedding 余弦相似度排序 | 验证纯语义相似检索能力 |
| 超图方法 | 使用结构节点、超边、经验复用信号和 embedding 融合排序 | 验证本项目方法 |

多智能体通信实验还应比较：

| 模式 | 说明 |
| --- | --- |
| `baseline_text` | 智能体之间主要传递文本 |
| `protocol_api` | 结构化协议通信，embedding 使用 hash 占位 |
| `protocol_api_embedding` | 结构化协议通信，并调用真实 embedding |
| `latent_hidden` | 传递 hidden-state 引用或紧凑状态引用 |

比较时必须保持数据集、题目顺序、模型、温度、`top_k` 和运行环境一致。

## 3. 任务效果层指标

### 3.1 Exact Match

最终答案与标准答案完全匹配的比例：

```text
Exact Match = 完全匹配的任务数 / 有标准答案的任务数
```

对应现有字段：

```text
runs[].exact_match
runs[].task_results[].exact_match
```

这是比赛展示的核心准确率指标。若数据集没有标准答案，必须显示 `N/A`，不能按错误处理。

### 3.2 宽松答案正确率

建议新增标准化后的答案正确率，例如忽略大小写、标点、冠词和首尾空格：

```text
Normalized Accuracy = 标准化后匹配的任务数 / 有标准答案的任务数
```

适用于回答表达略有差异、但语义正确的情况。当前项目主要输出 `exact_match`，该指标尚需补充实现。

### 3.3 记忆带来的答案收益

比较开启记忆和关闭记忆时的准确率：

```text
Memory Accuracy Gain = Accuracy_with_memory - Accuracy_without_memory
```

该指标用于证明“命中记忆”确实改善了答案，而不是仅增加了检索步骤。

建议进一步按是否命中正确记忆分组：

```text
Accuracy | correct memory hit
Accuracy | wrong memory hit
Accuracy | no memory hit
```

## 4. 通信与成本层指标

### 4.1 文本通信字符数

现有字段：

```text
metrics.text_chars
```

表示智能体消息中传递的文本字符总量。相对 `baseline_text` 的字符节省率为：

```text
字符节省率 =
  (baseline_text.text_chars - current.text_chars)
  / baseline_text.text_chars
```

字符节省率越高，说明结构化协议、状态引用和紧凑记忆上下文降低了通信负担。

### 4.2 消息数量

现有字段：

```text
metrics.message_count
```

用于确认字符降低是否只是通过减少协作轮次实现。报告中应同时展示消息数和字符数，避免把“少调用智能体”误认为“通信压缩更有效”。

### 4.3 Latent 传输量

现有字段：

```text
metrics.latent_transfer_count
metrics.latent_bytes
```

建议同时报告：

```text
平均每次 latent 传输字节数 =
  latent_bytes / latent_transfer_count

总通信近似量 =
  text_chars + latent_bytes
```

注意：字符数与字节数并不完全等价。正式报告中应注明该“总通信近似量”只是统一比较口径。

### 4.4 端到端耗时

现有字段：

```text
elapsed_ms
metrics.elapsed_ms
```

相对基线的耗时降低率：

```text
耗时降低率 =
  (baseline_text.elapsed_ms - current.elapsed_ms)
  / baseline_text.elapsed_ms
```

真实 API 受网络波动影响明显。正式实验应至少重复运行 3 次，报告平均值和标准差：

```text
mean elapsed_ms
std elapsed_ms
P50 latency
P95 latency
```

### 4.5 模型调用成本

建议新增：

```text
LLM 调用次数
embedding 调用次数
输入 token 数
输出 token 数
估算费用
```

该指标当前未完整接入，但对“低成本”比赛主题非常重要。仅展示 `text_chars` 不足以完整说明真实费用。

## 5. 上下文检索指标

上下文检索回答的是：“只给当前问题，系统能否从数据集背景中找到正确证据？”

### 5.1 Supporting Passage Hit Rate

正确支持段落是否出现在前 `k` 个结果中：

```text
Supporting Passage Hit Rate@k =
  前 k 个结果包含支持段落的任务数
  / 有支持段落标注的任务数
```

这是上下文超图最重要的指标。现有 TopiOCQA 调试结果曾观察到：

```text
纯文本上下文 supporting passage 命中率：0.6
上下文超图 supporting passage 命中率：0.9
```

该结果只能作为当前样例现象，正式报告需要扩大数据量并重复验证。

### 5.2 Precision@k

```text
Precision@k = 前 k 个结果中的相关上下文数 / k
```

用于衡量返回结果是否包含大量无关背景。

### 5.3 Recall@k

```text
Recall@k =
  前 k 个结果中的不同相关上下文数
  / 当前问题全部相关上下文数
```

多跳问答通常需要多个 supporting facts，因此 Recall@k 比单纯命中率更重要。

### 5.4 MRR

第一个正确上下文出现位置的倒数：

```text
MRR = mean(1 / 第一个相关结果的排名)
```

正确证据越靠前，MRR 越高。现有 `retrieval_metrics()` 已支持该指标。

## 6. 记忆检索与复用指标

记忆检索回答的是：“系统是否找到了当前任务真正应该复用的历史记忆？”

### 6.1 Memory Hit Rate

```text
Memory Hit Rate =
  返回至少一条记忆的任务数
  / 总任务数
```

对应现有字段：

```text
memory_hit_rate
metrics.memory_hits
```

该指标只能说明“有返回结果”，不能说明返回结果正确。命中率高但复用错误，反而可能损害最终答案。

### 6.2 Expected Reuse Hit Rate

对于带有 `expected_reuse` 标注的连续任务：

```text
Expected Reuse Hit Rate@k =
  前 k 个结果包含期望历史轮次的任务数
  / 带 expected_reuse 标注的任务数
```

现有 `evaluate_retrieval.py` 已比较：

```text
hypergraph_expected_reuse_hit_rate
keyword_expected_reuse_hit_rate
embedding_expected_reuse_hit_rate
```

这是证明超图记忆复用优于文本和纯向量检索的核心指标。

### 6.3 Memory Reuse Precision@k

```text
Memory Reuse Precision@k =
  前 k 个结果中真正应该复用的记忆数
  / 返回的记忆数
```

现有 `retrieval_metrics()` 已支持 `precision_at_k`，前提是数据集中存在可靠的 `expected_ids` 或 `expected_reuse` 标注。

### 6.4 Memory Reuse Recall@k

```text
Memory Reuse Recall@k =
  前 k 个结果中命中的不同正确记忆数
  / 当前任务全部应该复用的记忆数
```

现有 `retrieval_metrics()` 已支持 `recall_at_k`。

### 6.5 有效复用率

建议新增：

```text
有效复用率 =
  被 Reasoner 实际使用且最终答案正确的记忆命中数
  / 所有记忆命中数
```

该指标比普通命中率更严格，能够区分：

```text
检索到了但未使用
使用了但无收益
使用后改善了答案
```

当前系统已有 `memory_hits`、`memory_context` 和 `memory_refs`，但还缺少可靠的“本条记忆是否真正贡献答案”标注。

## 7. 返回成本与检索效率指标

### 7.1 Returned Chars

返回给后续智能体的上下文或记忆字符数：

```text
returned_chars
avg_returned_chars
```

该指标用于验证超图检索是否能返回更短、更聚焦的证据，而不是重新塞入完整历史上下文。

现有 `retrieval_metrics()` 已支持该指标。

### 7.2 检索时延

```text
retrieval_latency_ms
```

用于比较纯文本、纯向量和超图检索的查询速度。当前实现主要是内存结构和小规模数据，检索时延只适合作为调试参考，不能直接证明大规模性能。

正式报告建议在不同记忆规模下测试：

```text
100 条记忆
1,000 条记忆
10,000 条记忆
```

并报告：

```text
平均检索时延
P95 检索时延
索引构建耗时
峰值内存占用
```

### 7.3 准确率-成本联合指标

建议使用“每千字符正确回答数”描述低成本效果：

```text
Correct Answers per 1K Chars =
  正确回答数 / (text_chars / 1000)
```

也可以报告：

```text
Accuracy / elapsed_seconds
Accuracy / estimated_cost
Expected Reuse Hit Rate / returned_chars
```

联合指标只能作为辅助展示，主表仍应保留原始准确率和成本指标。

## 8. 超图结构专项指标

这些指标用于解释超图为什么命中，不直接代表最终任务效果。

### 8.1 Structured Hit Ratio

```text
Structured Hit Ratio =
  命中的结构化节点数
  / 全部命中节点数
```

结构化节点包括：

```text
expected_reuse
memory_ref
gap
evidence
supporting_fact
state_ref
strategy
experience
reuse_benefit
```

现有 `evaluate_retrieval.py` 已实现 `structured_hit_ratio`。

### 8.2 Keyword Hit Ratio

```text
Keyword Hit Ratio =
  keyword 节点命中数
  / 全部命中节点数
```

该值过高说明超图检索仍主要依赖关键词，结构创新尚未真正发挥作用。

### 8.3 Keyword Noise Ratio

```text
Keyword Noise Ratio =
  含数字等噪声的 keyword 命中数
  / 全部 keyword 命中数
```

现有 `evaluate_retrieval.py` 已实现该指标。该值越低越好。

### 8.4 解释路径完整率

建议新增：

```text
解释路径完整率 =
  同时包含 matched_nodes、matched_hyperedges 和 reasoning_path 的命中数
  / 全部命中数
```

现有代码已输出这些字段，但尚未统一计算完整率。

### 8.5 结构子分数

当前超图检索会输出：

```text
gap_coverage_score
reuse_score
evidence_score
state_ref_score
strategy_score
experience_score
embedding_score
hypergraph_score
retrieval_score
```

真实排序公式为：

```text
structural_score =
  gap_coverage_score
  + reuse_score
  + evidence_score
  + state_ref_score
  + strategy_score
  + experience_score

embedding_component = max(0, embedding_score) * 8

retrieval_score =
  hypergraph_score
  + min(structural_score, 20) * 0.25
  + embedding_component
```

这些分数用于解释排序和做消融实验。不同方法的内部排序分数尺度不同，不应直接横向比较平均 `retrieval_score`。

## 9. 推荐消融实验

为证明超图结构本身有效，至少进行以下消融：

| 实验 | 保留内容 | 要回答的问题 |
| --- | --- | --- |
| 文本基线 | 词项重叠 | 不使用图能达到什么效果 |
| 向量基线 | embedding 相似度 | 纯语义相似能达到什么效果 |
| 超图去 embedding | 仅结构匹配 | 收益是否来自结构 |
| 超图去经验节点 | 去掉 strategy/experience/reuse_benefit | 经验节点是否有效 |
| 超图去证据节点 | 去掉 evidence/supporting_fact/entity | 证据结构是否有效 |
| 完整超图 | 结构与 embedding 融合 | 最终方法效果 |

每个消融至少报告：

```text
Exact Match
Supporting Passage Hit Rate@k
Expected Reuse Hit Rate@k
Precision@k
Recall@k
MRR
returned_chars
retrieval_latency_ms
text_chars
elapsed_ms
```

## 10. 推荐技术报告主表

### 10.1 整体系统主表

| 模式 | Exact Match | Text Chars | 字符节省率 | Latent Bytes | Elapsed ms | Memory Hit Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_text` |  |  | 0 |  |  |  |
| `protocol_api` |  |  |  |  |  |  |
| `protocol_api_embedding` |  |  |  |  |  |  |
| `latent_hidden` |  |  |  |  |  |  |

### 10.2 检索质量主表

| 方法 | Supporting Hit@k | Expected Reuse Hit@k | Precision@k | Recall@k | MRR | Returned Chars | Retrieval ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 纯文本 |  |  |  |  |  |  |  |
| 纯向量 |  |  |  |  |  |  |  |
| 超图 |  |  |  |  |  |  |  |

### 10.3 超图解释性表

| 方法 | Structured Hit Ratio | Keyword Hit Ratio | Keyword Noise Ratio | 解释路径完整率 |
| --- | ---: | ---: | ---: | ---: |
| 超图 |  |  |  |  |

## 11. 当前实现状态

### 已实现并可直接统计

```text
exact_match
message_count
text_chars
latent_transfer_count
latent_bytes
elapsed_ms
memory_hit_rate
Supporting Passage Hit Rate
Expected Reuse Hit Rate
Precision@k
Recall@k
MRR
returned_chars
retrieval_latency_ms
structured_hit_ratio
keyword_hit_ratio
keyword_noise_ratio
各类超图结构子分数
```

主要实现位置：

```text
run_demo.py
RUNBOOK.md
hypergraph_memory_method/text_memory.py
hypergraph_memory_method/evaluate_retrieval.py
hypergraph_memory_method/run_topiocqa_retrieval_compare.py
hypergraph_memory_method/scoring.py
```

### 建议补充实现

```text
标准化答案正确率
LLM/embedding 调用次数与 token 数
真实费用估算
有效复用率
错误记忆导致的答案退化率
解释路径完整率
大规模检索速度和内存占用
多次运行的均值、标准差、P50、P95
```

## 12. 结论口径

只有同时满足以下条件，才能在报告中声称超图记忆优于普通记忆：

```text
1. Expected Reuse Hit Rate、Precision@k 或 Recall@k 明显提高；
2. Exact Match 不下降，最好能够提升；
3. returned_chars 或 text_chars 降低；
4. reasoning_path 能解释为什么命中；
5. 优势在多个数据集和多次运行中稳定存在。
```

如果超图只提高了内部 `retrieval_score`，但正确复用率、最终准确率和通信成本没有改善，则不能证明超图结构具有实际优势。
