> 当前比赛入口已于 2026-09-27 固定为三条主要方案及两种保留对照。以 [当前方案](examples/competition_agents/README.md) 和 [当前运行手册](examples/competition_agents/RUNBOOK.md) 为准。下文为历史版本说明，其中旧入口及协议向量实验已经退役。

# 多智能体低开销通信实验系统

本仓库是社区赛题“一种面向多智能体协作的低开销通信、状态传递与共享记忆机制”的参赛作品。系统围绕多 Agent 协作中的通信开销、非文本状态传递和共享记忆复用三个问题，实现了可运行的实验控制台、后端执行器、共享记忆模块、CodeAct 工具执行链路、对比方法适配和多模型实验结果展示。

本项目主体位于 `examples/competition_agents`。最终评审和复现实验请以本 README、根目录 `RUNBOOK.md`、根目录 `部署文档.md` 和 `examples/competition_agents/competition_docs` 中的正式材料为准。

## 核心能力

- 多 Agent 协作链路：`Planner -> Retriever -> Reasoner -> Summarizer -> MemoryManager`。
- CodeAct 工具任务链路：由 `Reasoner` 触发 `Executor`，执行受限 Python 代码并返回结构化结果。
- 结构化协议通信：使用 `action / params / result / state_refs / memory_refs / meta` 表示 Agent 间消息。
- 非文本状态传递：支持 embedding 状态、hidden state、`state_ref` 和非文本状态规模统计。
- 共享记忆复用：支持 evidence memory、experience memory、历史摘要、任务结论和超图关联关系。
- 对比方法适配：支持纯文本、LLMLingua-2、AutoForm、结构化协议、协议 + 向量、隐状态传递等实验模式。
- 前端展示：支持实验运行、协作链路查看、记忆复用查看、指标评估、结果文件导入和多模型结果对比。

非文本模式采用控制面与数据面分离：Agent 消息仅携带 `state_ref`，计划、检索结果和答案载荷写入进程内状态池；下游 Agent 通过受控解析接口读取载荷，hidden 向量用于知识候选重排和计划—证据一致性约束。当前统计的是逻辑协议与引用包规模，不代表 Socket/IPC 物理流量。

## 目录结构

```text
.
├── examples/competition_agents/       参赛系统主体代码
│   ├── agents/                        Planner、Retriever、Reasoner、Executor、Summarizer 等 Agent
│   ├── baselines/                     LLMLingua-2、AutoForm 对比方法适配
│   ├── competition_agents/            协议、记忆、状态、评测、本地模型模块
│   ├── competition_docs/              设计文档、实验报告、架构图等材料
│   ├── data/                          TopiOCQA、CodeAct 实验数据
│   ├── frontend/                      Vue + Vite 实验控制台
│   ├── runner_server.py               前端运行后端
│   ├── run_demo.py                    主实验入口
│   ├── setup_openeuler_env.sh         openEuler/Linux 环境初始化脚本
│   └── start_local_runner.ps1         Windows 本地启动脚本
├── run_result/                        已保存的实验结果，按模型分目录
├── RUNBOOK.md                         运行手册
├── 部署文档.md                        部署文档
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
| 隐状态传递 | `hidden_api` | 本地模型 hidden state 写入状态池，下游按引用读取并用于证据重排与推理约束 |

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

## openEuler 快速部署

赛题交付环境要求为 `openEuler 24.03-LTS-SP3`。在 openEuler 或 Linux 环境中，推荐从仓库根目录执行：

```bash
bash examples/competition_agents/setup_openeuler_env.sh
```

脚本会安装或复用 Conda 环境，安装 Python 与前端依赖，并下载默认模型。若只安装依赖、不下载模型：

```bash
DOWNLOAD_MODELS=0 bash examples/competition_agents/setup_openeuler_env.sh
```

启动后端：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate competition-agents
python examples/competition_agents/runner_server.py
```

启动前端：

```bash
cd examples/competition_agents/frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

浏览器访问：

```text
http://<服务器 IP>:5173/
```

完整部署流程见 [部署文档.md](./部署文档.md)，完整运行说明见 [RUNBOOK.md](./RUNBOOK.md)。

## Windows 本地调试

Windows 本地调试可以从仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r examples\competition_agents\requirements.txt
```

启动后端：

```powershell
powershell -ExecutionPolicy Bypass -File examples\competition_agents\start_local_runner.ps1 -UseOfficialLLMLingua2
```

启动前端：

```powershell
cd examples\competition_agents\frontend
npm.cmd install
npm.cmd run dev
```

浏览器访问：

```text
http://localhost:5173/
```

## 交付文档

正式交付材料集中在仓库根目录和 `examples/competition_agents/competition_docs`：

| 文件 | 内容 |
| --- | --- |
| `部署文档.md` | 正式部署文档，覆盖 openEuler/Linux 和 Windows 调试 |
| `RUNBOOK.md` | 运行手册，覆盖模型准备、启动、运行、导入和排错 |
| `examples/competition_agents/competition_docs/项目设计文档.docx` | 正式项目设计文档 |
| `examples/competition_agents/competition_docs/optimized_multi_agent_architecture.drawio` | 当前正式系统架构图，可编辑 |
| `examples/competition_agents/competition_docs/optimized_multi_agent_architecture_white.png` | 当前正式系统架构图，白底图片 |

## 技术报告说明

比赛要求的技术报告应覆盖作品链接、设计方案、实现方案、运行效果/测试结果、演示视频和特色创新。`项目设计文档.docx` 可以作为设计方案和实现方案的正式基础，但完整技术报告还需要合并实验结果、运行截图、作品链接、演示视频链接和创新点说明。
