# 多智能体低开销通信实验系统

本仓库是社区赛题“一种面向多智能体协作的低开销通信、状态传递与共享记忆机制”的参赛作品。系统围绕多 Agent 协作中的通信开销、非文本状态传递和共享记忆复用三个问题，实现了可运行的实验控制台、后端执行器、共享记忆模块、CodeAct 工具执行链路、对比方法适配和多模型实验结果展示。

本项目主体位于 `examples/competition_agents`，原仓库中的 LangGraph 基础代码仅作为工程底座保留，最终评审和复现实验请以本 README、根目录 `RUNBOOK.md` 和 `examples/competition_agents/competition_docs` 为准。

## 核心能力

- 多 Agent 协作链路：`Planner -> Retriever -> Reasoner -> Summarizer -> MemoryManager`。
- CodeAct 工具任务链路：由 `Reasoner` 触发 `Executor`，执行受限 Python 代码并返回结构化结果。
- 结构化协议通信：使用 `action / params / result / state_refs / memory_refs / meta` 表示 Agent 间消息。
- 非文本状态传递：支持 embedding 状态、hidden state、`state_ref` 和非文本状态规模统计。
- 共享记忆复用：支持 evidence memory、experience memory、历史摘要、任务结论和超图关联关系。
- 对比方法适配：支持纯文本、LLMLingua-2、AutoForm、结构化协议、协议+向量、隐状态传递等实验模式。
- 前端展示：支持实验运行、协作链路查看、记忆复用查看、指标评估、结果文件导入和多模型结果对比。

## 目录结构

```text
.
├── examples/competition_agents/       参赛系统主体代码
│   ├── agents/                        Planner、Retriever、Reasoner、Executor、Summarizer 等 Agent
│   ├── baselines/                     LLMLingua-2、AutoForm 对比方法适配
│   ├── competition_agents/            协议、记忆、状态、评测、本地模型模块
│   ├── competition_docs/              设计文档、实验报告、架构图和部署说明
│   ├── data/                          TopiOCQA、CodeAct 实验数据
│   ├── frontend/                      Vue + Vite 实验控制台
│   ├── runner_server.py               前端运行后端
│   ├── run_demo.py                    主实验入口
│   └── start_local_runner.ps1         Windows 本地启动脚本
├── run_result/                        已保存的实验结果，按模型分目录
├── RUNBOOK.md                         运行手册
└── README.md                          项目总说明
```

## 实验模式

| 页面选项 | 后端模式 | 说明 |
| --- | --- | --- |
| 纯文本 | `baseline_text` | 自然语言长文本通信基线 |
| AutoForm | `autoform_text` | 基于 AutoForm 思想的紧凑通信格式生成 |
| LLMLingua-2 | `llmlingua_text` | 生成后压缩 Agent 间文本消息 |
| 结构化协议 | `protocol_api` | 使用结构化协议包传递 Agent 中间结果 |
| 协议 + 向量 | `protocol_api_embedding` | 结构化协议结合 embedding 状态引用 |
| 隐状态传递 | `hidden_api` | 基于本地模型 hidden state 的状态引用传递 |

## 数据集

| 数据集 | 任务类型 | 规模 |
| --- | --- | --- |
| TopiOCQA 第 1 组 | 连续问答 | 10 轮 |
| TopiOCQA 第 2 组 | 连续问答 | 10 轮 |
| CodeAct 第 1 组 | 工具执行任务 | 10 题 |
| CodeAct 第 2 组 | 工具执行任务 | 10 题 |

## 本地模型

系统通过 HuggingFace `AutoTokenizer` 和 `AutoModelForCausalLM` 加载本地模型。页面可选择的模型包括：

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- `microsoft/Phi-3.5-mini-instruct`
- `deepseek-ai/deepseek-coder-1.3b-instruct`

正式报告中应保证同一组对比实验使用相同模型、相同数据集、相同任务数量和相同检索配置。

## 快速启动

启动后端：

```powershell
cd D:\qq\low-cost-memory-integrate-codex
D:\qq\low-cost-memory-integrate\.venv\Scripts\Activate.ps1
powershell -ExecutionPolicy Bypass -File examples\competition_agents\start_local_runner.ps1 -UseOfficialLLMLingua2
```

启动前端：

```powershell
cd D:\qq\low-cost-memory-integrate-codex\examples\competition_agents\frontend
npm.cmd install
npm.cmd run dev
```

打开页面：

```text
http://localhost:5173/
```

完整运行说明见 [RUNBOOK.md](./RUNBOOK.md)。

## 交付文档

正式交付材料集中在 `examples/competition_agents/competition_docs`：

| 文件 | 内容 |
| --- | --- |
| `项目设计文档.docx` | 正式项目设计文档，来源为 `D:\qq\项目设计文档 (2).docx` |
| `01_系统设计文档.md` | 系统目标、总体架构、Agent 分工、结构化通信、状态传递、共享记忆和评测设计 |
| `02_部署文档草稿.md` | 环境准备、依赖安装、前后端启动、openEuler 复现步骤 |
| `03_实验报告框架.md` | 实验设置、对比方法、实验结果、结果分析和报告结构 |
| `04_方法部分.md` | 结构化协议、非文本状态传递、超图记忆、经验提炼和对比方法实现说明 |
| `05_指标定义.md` | 通信、状态、记忆、准确率等指标定义 |
| `06_运行命令汇总.md` | 常用运行命令 |
| `optimized_multi_agent_architecture.drawio` | 当前正式系统架构图，可编辑 |
| `optimized_multi_agent_architecture_white.png` | 当前正式系统架构图，白底图片 |

## 技术报告说明

比赛要求的技术报告应覆盖作品链接、设计方案、实现方案、运行效果/测试结果、演示视频和特色创新。`项目设计文档.docx` 可以作为设计方案和实现方案的正式基础，但完整技术报告还需要合并实验结果、运行截图、作品链接、演示视频链接和创新点说明。

