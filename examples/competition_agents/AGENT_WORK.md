# 智能体侧工作交接说明
9.赛题题目：一种面向多智能体协作的低开销通信、状态传递与共享记忆机制（社区赛题）
赛题说明：
随着大模型应用从单 Agent 问答逐步扩展到多 Agent 协同执行，智能系统正在从“单点生成”向“分工协作”演进。在检索增强生成、复杂任务规划、代码协作、办公自动化、知识分析等场景中，往往需要多个 Agent 分别承担规划、检索、执行、总结、生成等不同角色，并通过相互协作完成复杂任务。当前主流多 Agent 系统大多以自然语言或 JSON 作为通信媒介，即一个 Agent 将其中间结果组织成文本，再传递给其他 Agent 进行解析和继续处理。这种方式虽然通用性较好，但在多轮、多 Agent、复杂任务场景下存在明显不足：一是通信内容冗长、重复上下文多，token 消耗高；二是中间结果需要在“内部状态—文本—内部状态”之间反复转换，导致时延增加并可能带来语义损耗；三是任务执行过程中形成的中间知识和经验难以沉淀，系统在处理相似任务时往往仍需从头开始，缺乏持续积累和复用能力。
本赛题面向多智能体协作系统中的基础设施问题，要求选手围绕**低开销通信、非文本状态传递、共享记忆复用**三个方面，设计并实现一套可运行的原型系统。系统一方面需要通过结构化通信协议替代冗长自然语言交互，将 Agent 间传递的内容收敛为动作、参数、结果、能力等高密度语义单元，以降低通信成本和解析开销；另一方面需要探索 embedding、语义向量、隐藏状态特征或其他中间表示在 Agent 之间的直接传递机制，减少不必要的文本编解码过程，提高协作效率。在此基础上，还需将任务执行过程中形成的摘要、证据、策略、经验等内容沉淀为可标识、可检索、可复用的共享记忆单元，使系统具备跨任务的知识积累和协同增强能力。
本课题区别于一般的工作流编排类题目，重点不在于简单调用大模型接口和外部工具，而在于研究多智能体协作中的“系统层机制”：包括 Agent 间统一通信协议设计、中间状态表示与交换方式、共享记忆组织模型、跨任务复用机制以及整体运行效率验证。选手需面向开源操作系统或通用 Linux 环境完成原型实现，通过可复现实验验证该机制相较传统纯文本协作方式在通信开销、任务时延和记忆复用方面的改进效果。        
具体要求：
系统需支持不少于 3 个 Agent 协同运行，至少覆盖任务规划、信息检索、总结生成、工具执行等角色中的 3 类，并能够完成一个包含多步骤处理过程的复杂任务；
系统需设计并实现一套面向 Agent 间协作的结构化通信机制，通信内容至少包括动作类型、输入参数、返回结果和能力描述，并支持基本的握手、能力发现或协议映射机制，不得仅通过自然语言长文本直接透传全部协作信息；  
系统需同时支持“纯文本协作模式”和“结构化协议协作模式”，并在相同任务条件下完成可复现实验对比；  
系统需实现一种非文本中间状态传递机制，支持 embedding、语义向量、隐藏状态特征或其他中间表示在 Agent 间直接交换，并说明其生成方式、传递方式、接收方式及后续使用方式；
系统需实现共享记忆模块，能够将任务执行过程中的中间结果、摘要、经验片段、证据链、结论或策略保存为统一的记忆单元，并为每条记忆记录至少包含记忆 ID、来源 Agent、创建时间、任务主题和摘要描述等基本元数据；  
系统需支持按关键词、标签或语义相似度检索历史记忆，并允许不同 Agent 在后续任务中直接复用已有记忆；  
系统需至少设计 2 组具有关联性的连续任务，验证结构化通信、非文本状态传递和共享记忆复用在减少重复计算、降低协作开销和提升任务效率方面的实际效果；
系统需统计并展示 Agent 间消息次数、文本通信 token 或字符开销、非文本状态传递次数及数据规模、单任务总耗时、共享记忆命中率及整体性能提升情况； 
系统架构中至少应包含多 Agent 运行时、协议解析与调度模块、状态交换模块、共享记忆存储与检索模块和评测模块，并能够稳定执行不少于 10 轮连续任务；  
需提交完整源码、系统设计文档、部署文档、实验报告和演示视频，能够支持评审现，鼓励结合 IPC、共享内存、Socket、向量数据库、WASM/容器沙箱、eBPF 等系统技术提升实现质量。
鼓励系统能够支持基于 CodeAct 模式的 Agent 执行机制，允许 LLM 生成 Python 可执行代码，并在轻量沙箱中安全运行，实现低延迟、可隔离的代码执行与结果回传能力。
赛题要求：
系统支持不少于3个Agent协同运行，覆盖规划、检索、执行、总结等角色；
设计结构化通信协议替代自然语言交互；
实现非文本中间状态传递机制（embedding/语义向量/隐藏状态）；
实现共享记忆模块，支持记忆的存储、检索和复用；
至少设计2组关联性连续任务进行验证；
提供通信开销、任务时延、记忆复用等方面的性能对比数据。
评分细则（明确评审角度、标准和分值范围）：
通信效率（25分）：相比纯文本协作的token节省效果
状态传递创新（20分）：非文本状态传递机制的设计新颖性
记忆复用效果（20分）：跨任务记忆复用的准确性与效率
系统完整性（20分）：多Agent协作的稳定性与功能覆盖
实验验证（15分）：性能对比数据的说服力
交付要求：
最终交付的代码需在 openEuler 24.03-LTS-SP3 操作系统版本上能够正常编译、运行和测试。

更新时间：2026-06-24

本文档是新对话接手当前项目时的主入口。它记录当前已经完成的能力、尚未完成的事项、需要继续优化的点、团队分工、我负责的边界，以及后续应优先阅读的文档。

如果本文档与历史聊天记录、临时口头约定不一致，以当前仓库代码、`RUNBOOK.md`、`HYPERGRAPH_CONTRACT.md`、`competition_docs/` 和本文档为准。

## 1. 当前项目目标

赛题要求实现一种面向多智能体协作的低开销通信、非文本状态传递与共享记忆机制。当前项目围绕三条主线推进：

1. 低开销通信：用结构化协议减少纯文本长上下文传输。
2. 非文本状态传递：用 embedding / hidden state 引用传递中间状态。
3. 共享记忆复用：把任务过程中的事实、证据、策略和经验沉淀为可检索、可复用的记忆单元。

当前系统已经不是简单问答 demo，而是包含多 Agent 流程、协议通信、状态交换、共享记忆、经验提炼、前端展示、结果导出和评测指标的比赛原型。

## 2. 当前已完成内容

### 2.1 多 Agent 主流程

已经稳定接入以下角色：

- `planner`：任务规划、生成检索焦点。
- `retriever`：检索上下文、evidence memory 和 experience memory。
- `reasoner`：基于证据、经验规则和状态引用生成候选答案。
- `summarizer`：生成最终答案，写入 evidence memory 和 experience memory。
- `executor`：仅在 CodeAct 或真实工具执行场景出现。

连续问答主链路为：

```text
Planner -> Retriever -> Reasoner -> Summarizer -> Memory Write
```

CodeAct 链路为：

```text
Planner -> Retriever -> Reasoner -> Executor -> Summarizer -> Memory Write
```

前端时间线已经按实际使用的 Agent 展示，普通连续问答不应再显示空的 `executor`。

### 2.2 实验模式

当前已支持并跑通过：

- `baseline_text`
- `protocol_api`
- `protocol_api_embedding`
- `latent_hidden`
- `codeact_tools`

含义：

- `baseline_text`：纯文本协作基线。
- `protocol_api`：结构化协议协作。
- `protocol_api_embedding`：结构化协议 + embedding 状态传递。
- `latent_hidden`：本地模型生成/推理 + hidden state 传递。
- `codeact_tools`：CodeAct 工具执行链路。

### 2.3 数据集拆分

前端当前只保留比赛展示需要的数据集：

- `TopiOCQA-1`：10 轮连续问答。
- `TopiOCQA-2`：10 轮连续问答。
- `CodeAct-1`：10 题 CodeAct。
- `CodeAct-2`：10 题 CodeAct。

文件位置：

- `examples/competition_agents/data/topiocqa_1_10turns.json`
- `examples/competition_agents/data/topiocqa_2_10turns.json`
- `examples/competition_agents/data/codeact_1_10tasks.json`
- `examples/competition_agents/data/codeact_2_10tasks.json`

CodeAct 拆分不是简单按题号平分，而是按依赖关系重排，保证每个分片内部能独立展示连续复用。

### 2.4 双层共享记忆

当前记忆已经从“只存问答结果”升级为双层结构：

- `evidence memory`：保存事实、证据、答案、执行结果等可直接复用信息。
- `experience memory`：保存可迁移策略、避错规则和经验卡片。

主流程写回节奏为：

```text
先写 evidence memory
再执行 refine_memory
再写 experience memory
```

这意味着 experience memory 已经是主链路正式步骤，不是独立实验脚本。

### 2.5 experience memory 闭环

当前 experience memory 已经进入完整闭环：

```text
Summarizer 写 experience
Retriever 检索 experience_hits
Reasoner 接收 experience_rules
Reasoner 按 solution_pattern / failure_pattern / trigger_conditions 消费经验
前端与结果 JSON 展示命中、使用和写回情况
```

经验提炼已经从早期的“题型标签”改为更接近 Evo-Memory 的外部记忆学习：

- 基于本轮 `question / answer_candidate / gold / exact_match / evidence / prior hits` 提炼。
- 输出更偏“为什么成功、为什么失败、下次怎么避免”。
- 记录 `experience_source`、`source_case`、`abstraction_note`、`reuse_count`、`reuse_success_count`。

仍需继续优化的是经验质量、重复经验合并和经验生命周期管理。

### 2.6 hidden state 路径对齐

`latent_hidden` 路径已经对齐主链路：

- 使用本地模型生成规划/答案并提取 hidden state。
- 输出 `hidden_state_refs / hidden_bytes`。
- 接入 `refine_memory(...)`。
- 输出 `latest_evidence_write / latest_experience_write / memory_refine_case / memory_refine_debug`。
- 最新修复：hidden 路径导出给前端的 `memory_hits` / `memory_context` 不再清空，并保留评估需要的来源字段。

这次 hidden 修复的核心文件：

- `examples/competition_agents/run_hidden_latent_samples.py`

修复原因：

- 前端候选记忆能显示，但指标为 0，是因为 hidden 导出的记忆条目缺少 `source_case / trace / supporting_evidence` 等评估字段。
- 现在 `_hidden_memory_context(...)` 会保留轻量但足够的来源信息，让 `evaluate_run_reuse(...)` 能判断命中是否相关。

### 2.7 结果落盘

前端运行结果会写入 `run_result/`。

当前推荐结果命名风格：

- `frontend_baseline_text_topiocqa_1_10turns.json`
- `frontend_protocol_api_embedding_topiocqa_1_10turns.json`
- `frontend_latent_hidden_topiocqa_1_10turns.json`
- `frontend_latent_hidden_codeact_1.json`

说明：

- `run_result/` 是运行产物，不建议提交到 Git。
- 正式报告数据后续应在 openEuler 24.03-LTS-SP3 上重跑后填入。

## 3. 当前已知结果状态

截至当前阶段，本地 Windows 预实验中：

- `TopiOCQA-1` 在 `baseline_text` 和 `protocol_api_embedding` 上已经能达到较稳定结果。
- `TopiOCQA-2` 的答案后处理问题已经修过一轮，重点修了 Yes/No 误判、列表答案和序数/指代问题。
- `latent_hidden` 路径 TopiOCQA 能跑通，但前端记忆指标刚刚修复导出字段，需要重新跑 hidden 后确认指标是否恢复。
- `CodeAct-1 / CodeAct-2` 已经完成 10 题拆分与触发类型细化，主要用于展示工具执行与经验复用。

注意：

- 当前结果只能作为本地预实验结果。
- 最终比赛报告中的准确率、耗时、字符开销、hidden bytes、latent bytes 等必须以 openEuler 24.03-LTS-SP3 环境重跑为准。

## 4. 当前未完成内容

### 4.1 openEuler 正式复现

这是交付前必须完成的事项。

需要在 openEuler 24.03-LTS-SP3 上完成：

- 环境安装。
- 前端服务启动。
- 四个数据集的正式实验。
- 结果 JSON / TXT 留档。
- 截图、日志、命令记录。

### 4.2 完整实验矩阵汇总

需要整理成正式表格：

- 方法：`baseline_text`、`protocol_api`、`protocol_api_embedding`、`latent_hidden`。
- 数据集：`TopiOCQA-1`、`TopiOCQA-2`、`CodeAct-1`、`CodeAct-2`。
- 指标：准确率、文本字符数、非文本状态次数/规模、耗时、记忆命中率、经验命中数、复用成功数。

### 4.3 消融实验

建议补：

- `NoMemory`：不检索共享记忆。
- `NoExperience`：只用 evidence memory，不用 experience memory。
- `FullMemory`：evidence + experience 全开。

目的：

- 证明经验记忆不是“好看的日志”，而是真的影响 Reasoner 和结果。

### 4.4 experience 生命周期

当前仍未完整实现：

- `merge`
- `dedup`
- `prune`
- `decay`

这部分以后以超图同学为主导。我方需要保证接口稳定，不再随意改字段名和函数签名。

### 4.5 文档和报告定稿

当前 `competition_docs/` 是草稿级文档，方法部分和设计部分已经能读，但实验数据还不能定稿。

需要后续补：

- openEuler 正式结果表。
- 消融实验表。
- 系统架构图。
- 共享记忆结构图。
- 前端展示截图。
- 演示视频脚本。

## 5. 当前需要继续优化的点

### 5.1 hidden 前端指标复核

最新代码已经修复 hidden 导出字段，但还需要重新跑一遍 hidden：

```powershell
python examples\competition_agents\run_hidden_latent_samples.py `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\topiocqa_1_10turns.json `
  --context-source hypergraph `
  --answer-mode generate
```

或者通过前端选择 `Hidden state 引用传递` + `TopiOCQA 第 1 组 / 第 2 组` 运行。

确认点：

- 记忆检索模块候选记忆不为空。
- 超图记忆指标不再全部为 0。
- `memory_evaluation.summary.relevant_memory_hit_count` 能正常统计。

### 5.2 experience 内容质量

当前经验质量已经比早期明显好，但仍要防止退回到“题型标签”。

理想经验应该像：

```text
当问题包含多个候选日期时，必须同时匹配介质、地区和时间约束，避免只取最显眼的日期。
```

不理想经验是：

```text
这是一个日期查询题。
```

后续优化应优先调整 distillation prompt 和 trigger 归一化，不要轻易改共享接口。

### 5.3 trigger 条件收口

当前已经尽量去掉 `general`，但仍要检查是否出现：

- `none`
- `memory_grounded_qa`
- 太泛的 `question_answering`

如果出现，应在我方内部做映射兜底，不要求超图同学改接口。

### 5.4 答案后处理

已经修过：

- `can you mention...` 不应被误判成 Yes/No。
- `can you tell me...` 不应被误判成 Yes/No。
- 列表题不应被强制缩成 Yes/No。

后续如果 TopiOCQA-2 再掉点，优先看：

- 原始 `answer_candidate`
- 后处理后的 `prediction`
- `final_answer`
- `exact_match`
- `memory_refine_case.exact_match`

确保判分口径一致。

## 6. 团队分工

### 6.1 我方负责

我方负责 Agent 主流程和实验闭环：

- Planner / Retriever / Reasoner / Summarizer / Executor 的流程组织。
- 什么时候写入 evidence memory。
- 什么时候 refine 并写入 experience memory。
- Retriever 如何检索 evidence / experience。
- Reasoner 如何消费 experience_rules。
- hidden state / embedding 状态传递主链路。
- 结果 JSON 字段。
- 前端需要的运行结果。
- 指标统计口径。
- 数据集拆分。
- RUNBOOK、比赛文档、实验报告框架。
- openEuler 最终复现实验组织。

### 6.2 超图同学负责

超图同学负责共享记忆子系统内部能力：

- memory record 的超图建模。
- evidence / experience 的检索排序。
- pattern、trigger、source_case 等字段在超图里的组织。
- 记忆解释路径。
- merge / dedup / prune / decay。
- 提升记忆检索质量和复用质量。

### 6.3 前端同学负责

前端同学负责页面展示和交互：

- 运行模式和数据集选择。
- 时间线和 Agent 输出展示。
- 记忆检索模块展示。
- 超图记忆指标卡片。
- 停止按钮调用后端 `/api/stop`。
- 页面布局和视觉交互。

注意：

- 前端不应自己推断记忆指标，优先使用后端返回的 `memory_evaluation`。
- 如果指标不对，先检查后端导出的 `memory_hits / memory_context / expected_reuse / source_case / trace`。

## 7. 与超图同学的冻结接口

共享接口当前视为冻结，后续不要随便改函数名、参数名和核心字段名。

冻结函数：

- `add_memory(...)`
- `search_memory(...)`
- `refine_memory(...)`
- `increment_reuse_counts(...)`

冻结字段：

- `memory_id`
- `memory_type`
- `source_agent`
- `created_at`
- `task_topic`
- `summary`
- `problem_pattern`
- `solution_pattern`
- `failure_pattern`
- `trigger_conditions`
- `supporting_evidence`
- `confidence`
- `reuse_count`
- `reuse_success_count`
- `source_case`
- `experience_rule_card`
- `trace`

运行结果字段：

- `memory_hits`
- `memory_context`
- `experience_hits`
- `experience_rules`
- `latest_evidence_write`
- `latest_experience_write`
- `memory_refine_case`
- `memory_refine_debug`
- `memory_evaluation`

后续我方可以内部优化：

- prompt
- trigger 清洗
- Reasoner 使用规则
- answer postprocess
- 前端展示字段
- 指标汇总

但不要改冻结接口。

## 8. 新对话接手时优先阅读哪些文档

建议新对话按这个顺序读：

1. `examples/competition_agents/AGENT_WORK.md`

当前总进度、分工、已完成/未完成/优化项，都在这里。

2. `examples/competition_agents/HYPERGRAPH_CONTRACT.md`

这是我方和超图同学之间的接口契约，后续不要轻易改。

3. `examples/competition_agents/competition_docs/07_当前实现与接口说明.md`

给比赛文档用的当前实现摘要，适合快速理解系统已经做了什么。

4. `examples/competition_agents/competition_docs/01_系统设计文档.md`

了解系统架构、Agent 分工、通信、状态传递和共享记忆设计。

5. `examples/competition_agents/competition_docs/05_指标定义.md`

了解每个指标怎么解释，尤其是 `text_chars`、`latent_bytes`、`hidden_bytes`、`memory_evaluation`。

6. `examples/competition_agents/competition_docs/06_运行命令汇总.md`

快速找到各模式的运行命令。

7. `examples/competition_agents/RUNBOOK.md`

正式运行和排查时以 RUNBOOK 为准。

8. `examples/competition_agents/LATENT_HIDDEN_FRAMEWORK.md`

如果要继续改 hidden state 路径，再读这个。

## 9. 新对话建议从哪里继续

最推荐的新对话起点：

1. 重新跑 hidden state 的 TopiOCQA-1 / TopiOCQA-2，确认前端超图记忆指标恢复。
2. 整理本地预实验矩阵，把当前四个数据集和各方法结果汇总成表。
3. 设计并运行 `NoExperience` 消融实验。
4. 准备 openEuler 24.03-LTS-SP3 复现环境。
5. 把 openEuler 正式结果填进 `competition_docs/03_实验报告框架.md`。

当前最不建议优先做的事：

- 继续大改超图接口。
- 继续频繁改 experience 字段名。
- 在没有 openEuler 正式结果前，把实验报告结论写死。

## 10. 一句话状态

当前项目已经完成比赛原型的主干：多 Agent、结构化通信、embedding/hidden 非文本状态传递、evidence + experience 双层共享记忆、前端展示、结果导出和基础评测都已跑通。下一阶段重点不是继续加字段，而是确认 hidden 指标、补消融实验、整理正式实验矩阵，并在 openEuler 目标环境完成最终复现。
