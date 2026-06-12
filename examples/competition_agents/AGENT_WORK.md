# 智能体侧工作说明

本文档只记录当前智能体主流程已经实现、正在推进和下一阶段准备继续推进的内容。

如果和历史聊天记录不一致，以当前仓库代码、`RUNBOOK.md` 和本文档为准。

## 当前职责范围

智能体侧当前主要负责：

- 多 Agent 主流程组织与运行时接入
- `baseline_text`、`protocol_api`、`protocol_api_embedding`、`latent_hidden`、`codeact_tools` 等实验入口
- Agent 间结构化通信和非文本状态传递
- 共享记忆写入、检索接入与结果结构维护
- 前端结果 JSON 与评测指标输出

## 已经实现

### 1. 多种实验模式已接通

当前已支持：

- `baseline_text`
- `protocol_api`
- `protocol_api_embedding`
- `latent_hidden`
- `codeact_tools`

其中：
- `baseline_text` 作为纯文本协作基线
- `protocol_api` 作为结构化协议协作主链路
- `protocol_api_embedding` 作为结构化协议 + embedding 状态传递
- `latent_hidden` 作为本地模型 hidden state 传递路径
- `codeact_tools` 作为 CodeAct 执行路径

### 2. 共享记忆已经分成两层

当前共享记忆逻辑上已经分成：

- `evidence memory`
- `experience memory`

但物理上仍共用同一个 memory store 和同一套超图索引入口。

### 3. experience memory 已经接入主流程

当前主链路中，`SummarizerAgent` 会执行：

```text
先写 evidence memory
再做 refine_memory
再写 experience memory
```

这意味着经验记忆已经不是单独脚本实验，而是主流程里的正式步骤。

### 4. experience memory 已有基础字段

当前经验记忆已经稳定产出这些字段：

- `problem_pattern`
- `solution_pattern`
- `failure_pattern`
- `trigger_conditions`
- `experience_source`
- `abstraction_note`
- `experience_rule_card`
- `source_case`
- `reuse_count`
- `reuse_success_count`

### 5. 经验命中已经能进入 Retriever 和 Reasoner

当前 `RetrieverAgent` 已经分两路检索：

- `evidence_hits`
- `experience_hits`

并把经验规则整理成：

- `experience_rules`

再传给 `ReasonerAgent`。

### 6. Reasoner 已经显式消费经验规则

当前 `ReasonerAgent` 不只是“带着经验字段跑一遍”，而是已经把经验规则显式传入 reason prompt。

当前口径是：

- `solution_pattern` 用来指导答案抽取策略
- `failure_pattern` 用来规避历史错误
- `trigger_conditions` 用来判断规则是否匹配当前题
- 经验规则不能替代事实证据

### 7. experience 复用计数已经接上

当前已经区分两层复用信号：

- `reuse_count`
  - 被命中了多少次
- `reuse_success_count`
  - 命中之后成功帮助答对了多少次

这部分已经接入：

- memory record
- `increment_reuse_counts`
- continuous 汇总指标

### 8. experience 排序已开始使用复用效果

当前 `RetrieverAgent` 的经验重排已开始参考：

- family 匹配
- `trigger_conditions`
- `reuse_count`
- `reuse_success_count`

虽然这还不是最终版经验演化排序，但已经不是只看文本相似度。

### 9. memory refine case / debug 已拆层

当前结果输出已把下面几层拆开：

- `latest_memory_write`
- `memory_refine_case`
- `memory_refine_debug`
- `execution_result`

目的是避免再把“经验本体、案例输入、调试信息”混在一个字段里。

### 10. CodeAct 路径已接入主图

当前 CodeAct 已经不是独立实验，而是主图链路的一部分：

```text
planner -> retriever -> reasoner -> executor -> summarizer
```

并且：

- `executor` 只在真正有执行内容时展示
- `reasoner` 在 CodeAct 场景下主要负责 `execution_request`
- `summarizer` 以真实执行结果为准做最终呈现

## 当前还没完成

### 1. experience 内容质量还不够稳

虽然 experience memory 已经落地，但当前最大问题仍然是：

- 有时仍会退回到偏“题型标签”的模式
- 还没有完全稳定到 Evo-Memory 风格的“成功/失败后抽象策略”

当前已经做了第一轮约束和 fallback，但还需要继续打磨。

### 2. 还没有完整的经验演化生命周期

当前还没正式完成：

- `merge`
- `dedup`
- `prune`
- `decay`

这部分后续由超图侧主导实现。

### 3. 还没有完整的对照实验矩阵

当前还没正式整理好的对照包括：

- `NoMemory`
- `StaticMemory`
- `EvolvingMemory`
- `Hidden/EfficientMemory`

### 4. 前端“停止运行”还不是真停

当前前端停止按钮只会中断浏览器请求，不会真正终止后端实验进程。

这部分还需要补真正的后端 stop 机制。

## 下一阶段我这边继续做什么

### 1. 继续提升 experience distillation 质量

目标是让经验更像：

- 为什么成功
- 为什么失败
- 下次怎么办

而不是：

- 这是什么题

### 2. 强化 Reasoner 对经验规则的真实消费

下一步重点观察：

- 经验是否真的改变答案选择
- 没有经验时是否更容易犯错
- `experience_used_count` 和 `reuse_success_count` 是否开始增长

### 3. 持续收紧结果展示层

目标是让页面上更清楚地区分：

- 最终答案
- 经验卡片
- 检索命中
- 调试信息

### 4. 继续补 continuous 的记忆复用指标

后续重点保留并继续强化：

- `memory_write_count`
- `experience_write_count`
- `experience_hit_count`
- `experience_used_count`
- `reuse_success_count`
- `memory_growth`

## 超图同学下一阶段该做什么

超图侧建议优先推进：

1. 单独增强 `experience memory` 的排序
2. 排序中显式利用：
   - `problem_pattern`
   - `solution_pattern`
   - `failure_pattern`
   - `trigger_conditions`
   - `reuse_count`
   - `reuse_success_count`
3. 后续再做：
   - `merge`
   - `dedup`
   - `prune`
   - `decay`

## 当前协作边界

### 我这边负责

- 经验闭环接入
- experience 写入时机
- experience 检索时机
- Reasoner 消费经验
- 结果 JSON
- 指标与页面展示

### 超图同学负责

- 经验在超图中的建模
- 经验检索与排序
- 经验演化
- 经验去重与淘汰

## 一句话结论

当前智能体侧已经从“只有问答记忆”升级到“evidence + experience 双层记忆”，并且经验规则已经进入 Retriever 和 Reasoner 主链路。下一阶段不再是补字段，而是继续把 experience 做成真正能提升后续任务效果的可复用策略记忆。
