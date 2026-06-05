# 智能体部分后续工作文档

本文档用于记录智能体侧当前已经完成的内容、简单完成但还可补强的内容，以及后续仍需推进的工作。后续如与历史口头记录不一致，以本文档为准。

## 当前职责范围

智能体部分当前主要负责：

- 多 Agent 主流程组织
- `baseline_text` / `protocol_api` / `protocol_api_embedding` / `latent_hidden` 等协作模式
- LLM 与 embedding 调用链路
- 非文本中间状态传递
- 结果输出结构设计
- 与前端、记忆模块的字段对接稳定性

## 已完成

### 1. 纯文本协作模式

- 已支持 `baseline_text`
- 可作为结构化协议协作模式的对照组

### 2. 结构化协议协作模式

- 已支持 `protocol_api`
- Agent 间通信已经采用结构化字段，而不是纯自然语言长文本透传
- 当前稳定保留的核心字段包括：
  - `from`
  - `to`
  - `action`
  - `payload`
  - `state_refs`
  - `memory_refs`

### 3. 非文本中间状态传递机制

- 已支持 `protocol_api_embedding`
- 已支持 `latent_hidden`
- 已实现“中间状态先生成，再通过 `state_ref` 引用在 Agent 间传递”的机制
- `latent_hidden` 当前可表述为“结构化协议 + hidden state 引用传递”

### 4. 数据集与实验入口

- 已支持 `multihop`
- 已支持 `continuous`
- 已支持 `codeact`
- `run_demo.py` 已不再保留旧的内置十题 `builtin` 入口

### 5. 统一结果输出

- 已输出统一的 `run_result_json`
- 已稳定输出前端依赖的主要字段：
  - `agent_views`
  - `messages`
  - `retrieval_debug`
  - `execution_result`
  - `memory_hits`
  - `memory_context`
  - `latent_states`
  - `state_refs`

### 6. 前端对接说明

- 已补充 [FRONTEND_CONTRACT.md](D:/qq/low-cost-memory-integrate/examples/competition_agents/FRONTEND_CONTRACT.md)
- 已补充 [INTEGRATION.md](D:/qq/low-cost-memory-integrate/examples/competition_agents/INTEGRATION.md)
- 已明确前端按动态 Agent 渲染，不写死固定智能体名单

## 简单完成

这一部分表示：已经有实现或原型，可以运行，也可以用于比赛说明，但还没有完全打磨到最稳状态。

### 1. 协议中的能力描述

- 当前已经有：
  - `cap`
  - `capability_required`
  - `gap`
- 可以说明系统具备基础能力描述与能力需求表达
- 但“显式握手 / 能力发现 / 协议映射”的展示还不算特别完整

### 2. 纯文本模式与结构化协议模式的对比

- 现在两种模式都已经有
- 但正式的并排对比结果、统一口径总结还可以再收口

### 3. 非文本中间状态传递的正式说明

- 机制本身已经做了
- 但“生成方式、传递方式、接收方式、后续使用方式”的比赛化表述还可以再收硬一点

### 4. `latent_hidden` 的答辩表述

- 当前已经可以解释它如何用结构化消息传递 hidden state 引用
- 但还可以继续整理成更适合 README、答辩和汇报的标准表述

## 待完成

### 1. `protocol_api` 首轮答案选择稳定性

- 当前连续任务里最主要的问题仍然是首轮 canonical answer 选错
- 典型问题：
  - `Chief of Protocol`
  - 被选成 `United States ambassador to Ghana`

### 2. 错误记忆扩散控制

- 当前 `continuous` 链路中，首轮错答会被写入 memory 并持续复用
- 需要继续补：
  - memory write gate
  - supporting facts 与 final answer 分层写入
  - 低置信度结果不直接进入高优先级 memory

### 3. 正式对比结果收口

- 后续最好统一收成：
  - `baseline_text`
  - `protocol_api`
  - `protocol_api_embedding`
  - `latent_hidden`
- 并在相同数据集、相同 `limit` 下给出可复现实验对比

### 4. 新 Agent 扩展准备

- 后续可能新增：
  - `executor`
  - `experience_extractor`
  - `memory_graph`
- 需要在不破坏前端稳定字段的前提下继续扩展

## 当前建议优先级

1. 先修 `protocol_api` 首轮答案选择
2. 再控制错误答案写入 memory
3. 再补比赛用的正式说明和对比收口
4. 最后再继续扩展新 Agent 和记忆图能力

## 后续改代码时继续遵守的规则

### 1. 不改正式解析入口

保留：

```text
=== run_result_json ===
{...json...}
```

### 2. 不随意改前端已依赖的字段名

继续稳定保留：

- `agent_views`
- `messages`
- `retrieval_debug`
- `execution_result`
- `memory_hits`
- `memory_context`
- `latent_states`
- `state_refs`

### 3. `agent_views` 继续采用按需输出

- 哪个 Agent 本轮用了，就输出哪个 Agent
- 不强制所有 Agent 都带空字段
- 但一旦输出，尽量保持统一结构：

```json
{
  "display_text": "...",
  "structured": {},
  "state_ref": "..."
}
```

### 4. 新特殊能力优先走独立字段

例如未来新增：

- `experience_debug`
- `memory_graph_debug`
- `memory_refine_debug`

不要把所有新内容都硬塞进已有字段里。

## 一句话总结

当前阶段，智能体侧不是优先继续铺新功能，而是先把已经做出的结构化协议、非文本中间状态传递和连续任务链路收稳、讲清楚、对比完整。
