# 智能体侧工作说明

本文档用于记录当前智能体主流程相对赛题要求的完成情况、未完成项、协作边界与下一步工作重点。

如果与历史聊天记录、临时口头约定不一致，以当前仓库代码、`RUNBOOK.md`、`HYPERGRAPH_CONTRACT.md` 和本文档为准。

更新时间：2026-06-16

## 1. 当前职责范围

当前智能体侧主要负责：

- 多 Agent 主流程组织与运行时接入
- `baseline_text`、`protocol_api`、`protocol_api_embedding`、`latent_hidden`、`codeact_tools` 等实验入口
- Agent 间结构化通信与非文本状态传递主链路
- 共享记忆的写入时机、检索接入、Reasoner 消费方式
- 前端结果 JSON、评测指标、运行结果落盘格式
- continuous / CodeAct 数据集切分、复现实验与结果对齐

不直接负责的部分：

- 超图内部建模细节
- 经验记忆在超图内部的 `merge / dedup / prune / decay`
- 超图内部排序算法的后续独立优化

## 2. 对赛题要求的完成情况

下面按赛题原文逐条对齐。

### 2.1 不少于 3 个 Agent 协同运行

已完成。

当前已稳定运行的角色包括：

- `planner`
- `retriever`
- `reasoner`
- `summarizer`
- `executor`（仅在 CodeAct 或有真实执行内容时出现）

说明：

- 连续问答主链路至少使用 `planner + retriever + reasoner + summarizer`
- CodeAct 路径可进一步接入 `executor`

### 2.2 结构化通信机制替代自然语言透传

已完成基础版，仍可继续优化。

当前已实现：

- `baseline_text` 作为纯文本协作对照组
- `protocol_api` 作为结构化协议协作模式
- `protocol_api_embedding` 作为结构化协议 + embedding 传递模式
- 消息中已稳定包含：
  - `action`
  - `params / payload`
  - `result`
  - `capability / capability_required`
  - `state_refs`
  - `memory_refs`
  - `cognitive_gap`

当前不足：

- “握手 / 能力发现 / 协议映射”已有基础字段，但说明文档还需要写得更系统
- 结构化协议的“低开销优势”还需要统一实验表做更完整展示

### 2.3 同时支持纯文本协作模式和结构化协议协作模式

已完成。

当前模式：

- `baseline_text`
- `protocol_api`
- `protocol_api_embedding`

且已可在同一任务集上做对比实验。

### 2.4 实现非文本中间状态传递机制

已完成两条可运行路径。

当前已实现：

- `protocol_api_embedding`
  - 使用 embedding / latent state ref 作为非文本中间状态
- `latent_hidden`
  - 使用本地模型 hidden state / pooled hidden state 作为中间状态

已覆盖赛题要求中的四个问题：

- 生成方式：embedding 或本地 hidden state 抽取
- 传递方式：写入 `latent_states`，通过 `state_ref` 引用传递
- 接收方式：下游 Agent 根据 `state_ref` 恢复或消费对应状态
- 后续使用方式：Reasoner / Summarizer 通过 `state_ref` 和证据内容继续推理

当前不足：

- hidden 路径的统计口径还没有和协议路径完全统一
- hidden 路径的说明文档还可以进一步压缩成更适合答辩的话术

### 2.5 实现共享记忆模块

已完成基础版，且已经从“只有问答记忆”升级为“双层记忆”。

当前已实现：

- `evidence memory`
- `experience memory`

当前每条记忆至少包含：

- `memory_id`
- `memory_type`
- `source_agent`
- `created_at`
- `task_topic`
- `summary`

扩展字段包括：

- `problem_pattern`
- `solution_pattern`
- `failure_pattern`
- `trigger_conditions`
- `supporting_evidence`
- `experience_source`
- `abstraction_note`
- `reuse_count`
- `reuse_success_count`

### 2.6 支持按关键词、标签或语义相似度检索历史记忆

已完成基础版。

当前已实现：

- `search_memory(...)`
- evidence / experience 分路检索
- 关键词、标签、模式字段、触发条件、复用次数联合参与排序

当前已具备：

- `evidence_hits`
- `experience_hits`
- `experience_rules`

当前不足：

- 经验记忆的排序仍有继续提升空间
- 超图侧后续还应单独增强 `experience memory` 的演化检索能力

### 2.7 至少 2 组具有关联性的连续任务

已完成。

当前已接入并跑通：

- `TopiOCQA-1`
- `TopiOCQA-2`
- `CodeAct-1`
- `CodeAct-2`

说明：

- `TopiOCQA` 已拆成两个 10 轮连续任务分片
- `CodeAct` 已拆成两个 10 题分片
- `CodeAct` 不是简单按题号平分，而是按依赖关系重排，使每个分片都能独立运行

### 2.8 统计并展示通信开销、时延、记忆复用等指标

已完成基础版。

当前已经稳定输出的指标包括：

- `message_count`
- `text_chars`
- `latent_transfer_count`
- `latent_bytes`
- `hidden_state_refs`
- `hidden_bytes`
- `elapsed_ms`
- `memory_hits`
- `memory_hit_rate`
- `experience_hit_count`
- `experience_write_count`
- `reuse_success_count`

当前不足：

- hidden 路径和 protocol 路径的非文本状态统计字段仍有两套口径
- “整体性能提升情况”还需要整理成统一结果表和实验报告叙述

### 2.9 至少包含运行时、协议解析与调度、状态交换、共享记忆、评测模块

已完成。

当前架构已覆盖：

- 多 Agent 运行时
- 协议解析与调度
- 状态交换模块
- 共享记忆存储与检索模块
- 评测模块

且连续任务已稳定跑过不少于 10 轮。

### 2.10 支持 CodeAct 模式执行机制

已完成基础版。

当前已实现：

- 受限 Python CodeAct 执行
- `reasoner -> executor` 执行请求链路
- `executor` 结构化回传执行结果
- 前端仅在实际有执行内容时展示 `executor`

当前不足：

- 仍属于轻量原型，不是完整工业级沙箱
- 停止按钮虽然已经接了后端 stop 逻辑，但还需要继续观察极端情况下的中断稳定性

### 2.11 openEuler 24.03-LTS-SP3 可运行性

部分完成，尚未闭环验证。

当前情况：

- 代码整体按 Python / 前后端常规方式组织
- 目标环境要求已经明确
- 但尚未完成一次在 openEuler 目标环境上的完整复现实测

这是后续交付前必须补齐的一项。

## 3. 当前已经完成的关键里程碑

### 3.1 实验模式已接齐

当前已支持：

- `baseline_text`
- `protocol_api`
- `protocol_api_embedding`
- `latent_hidden`
- `codeact_tools`

### 3.2 双层记忆已经接入主流程

当前主流程已固定为：

```text
先写 evidence memory
再执行 refine_memory
再写 experience memory
```

这意味着 experience memory 已经不是独立脚本实验，而是主链路正式步骤。

### 3.3 experience memory 已经进入 Retriever 和 Reasoner

当前链路已经不是“只写经验不使用”，而是：

```text
Summarizer 写 experience
Retriever 检索 experience_hits
Reasoner 接收 experience_rules
Reasoner 按 solution_pattern / failure_pattern / trigger_conditions 消费经验
```

### 3.4 经验记忆已从“题型标签”升级为“可迁移策略”

已经完成的方向性改进：

- 不再只写“这是什么题”
- 已开始写“为什么成功 / 为什么失败 / 下次怎么避免”
- 已引入：
  - `experience_source`
  - `abstraction_note`
  - `source_case`
  - `reuse_count`
  - `reuse_success_count`

虽然质量还需要继续打磨，但方向已经对齐 Evo-Memory 风格的外部记忆学习。

### 3.5 hidden continuous 已对齐主链路

当前 `latent_hidden` 在 `TopiOCQA-1 / TopiOCQA-2` 上已经完成对齐：

- 统一使用 `refine_memory(...)`
- 统一 `trigger_conditions`
- 统一 `supporting_evidence` 清洗逻辑
- 统一 `latest_evidence_write / latest_experience_write / memory_refine_case / memory_refine_debug`
- 统一真实记忆命中口径

最新结果：

- `TopiOCQA-1`: `exact_match = 1.0`
- `TopiOCQA-2`: `exact_match = 1.0`

### 3.6 CodeAct 拆分数据集已重排完成

当前 `CodeAct-1 / CodeAct-2` 的切分已经不是简单按题号平均切分，而是已经按依赖重排：

- `codeact-019 -> codeact-003` 已放回同一分片
- `codeact-020 -> codeact-014` 已保留在同一分片

最新结果：

- `CodeAct-1`: `exact_match = 1.0`
- `CodeAct-2`: `exact_match = 1.0`

### 3.7 前端数据集入口已收口

前端页面当前只保留：

- `TopiOCQA-1`
- `TopiOCQA-2`
- `CodeAct-1`
- `CodeAct-2`

避免了无关数据集入口继续干扰展示与复现。

## 4. 当前还没完成的内容

### 4.1 experience 内容质量还不够稳定

虽然 experience memory 已经跑通，但当前仍存在：

- 有时仍会退回到偏“任务抽象 / 题型标签”的写法
- 部分经验仍然不够像“成功/失败后提炼出的可迁移策略”
- 相近经验之间仍有重复写入

这是当前最核心的“质量问题”。

### 4.2 经验演化生命周期未完成

当前还没有正式完成：

- `merge`
- `dedup`
- `prune`
- `decay`

这部分后续以超图侧为主导完成，但智能体侧要保证接口不再变动。

### 4.3 hidden 与 protocol 的统计口径尚未完全统一

当前 still open 的问题：

- protocol 路径更偏 `latent_transfer_count / latent_bytes`
- hidden 路径更偏 `hidden_state_refs / hidden_bytes`

后续要么统一口径，要么在报告中明确解释两者差异。

### 4.4 完整对照实验矩阵尚未整理完

虽然很多结果已经跑出来了，但还没有正式整理成统一实验矩阵：

- `baseline_text`
- `protocol_api`
- `protocol_api_embedding`
- `latent_hidden`

分别在：

- `TopiOCQA-1`
- `TopiOCQA-2`
- `CodeAct-1`
- `CodeAct-2`

上的完整对比表还需要正式汇总。

### 4.5 openEuler 目标环境实测尚未完成

这是交付前必须补的一项，目前不能算完成。

## 5. 当前结果结论

截至 2026-06-16，当前阶段可以明确说：

- 赛题要求中的三条主线已经都落地：
  - 低开销通信
  - 非文本状态传递
  - 共享记忆复用
- 连续任务和 CodeAct 两类核心实验都已跑通
- experience memory 已经进入真实的“写入 -> 检索 -> 消费 -> 统计”闭环
- hidden continuous 与 hidden codeact 的拆分结果都已可独立运行

当前还不能完全说已经“收工”的部分是：

- 经验内容质量仍需继续打磨
- 经验演化生命周期还没有做完
- 实验报告级别的统一对比表还没最终成文
- openEuler 目标环境验证还没收尾

## 6. 下一步优先级

建议按以下顺序推进。

### 第一优先级：整理实验矩阵和总表

把以下模式在 4 个拆分数据集上的结果统一收表：

- `baseline_text`
- `protocol_api`
- `protocol_api_embedding`
- `latent_hidden`

字段建议统一为：

- `method`
- `dataset`
- `task_count`
- `exact_match`
- `elapsed_ms`
- `message_count`
- `text_chars`
- `latent_transfer_count / latent_bytes`
- `memory_hits`
- `memory_hit_rate`
- `experience_hit_count`
- `experience_write_count`
- `reuse_success_count`

### 第二优先级：继续提升 experience distillation 质量

目标是让经验更像：

- 为什么成功
- 为什么失败
- 下次如何避免

而不是：

- 这是什么题

### 第三优先级：补文档与答辩口径

优先更新：

- `RUNBOOK.md`
- `HYPERGRAPH_CONTRACT.md`
- 前端 / 实验相关文档

要把“拆分数据集、双层记忆、统一接口、当前结果、未完成项”全部写清楚。

### 第四优先级：准备 openEuler 复现

包括：

- 环境依赖清单
- 启动命令
- 前后端运行步骤
- 结果文件位置
- 常见问题排查

## 7. 我和超图同学的协作边界

### 我这边负责

- experience 闭环接入
- experience 写入时机
- experience 检索时机
- Reasoner 消费经验
- 结果 JSON
- 指标与页面展示
- 实验入口、运行脚本、数据集切分

### 超图同学负责

- experience 在超图中的建模
- experience 检索与排序
- experience 演化
- experience 去重与淘汰

### 当前接口结论

当前与超图侧的共享接口已经冻结。

后续如果继续优化，默认原则是：

- 不改对外函数名
- 不改参数名
- 不改返回结构字段名
- 智能体侧后续优先做内部逻辑优化
- 超图侧后续优先做内部检索 / 排序 / 演化优化

## 8. 一句话结论

当前智能体侧已经从“只有问答记忆”升级到“evidence + experience 双层共享记忆”，并且 experience 已真实进入多 Agent 主链路；下一阶段的重点不再是补字段，而是把 experience 做得更稳、更能迁移，并把结果整理成能直接支撑答辩和实验报告的统一证据链。
