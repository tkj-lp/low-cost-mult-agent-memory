# 多智能体低开销通信实验系统运行手册

本文档是项目最终运行手册，覆盖环境准备、模型下载、前后端启动、实验运行、结果保存、结果导入和常见问题排查。

## 1. 环境要求

### 1.1 Python 环境

Windows 本地推荐使用已有虚拟环境：

```powershell
D:\qq\low-cost-memory-integrate\.venv\Scripts\python.exe
```

检查 Python、PyTorch、Transformers 和 CUDA：

```powershell
D:\qq\low-cost-memory-integrate\.venv\Scripts\python.exe -c "import sys, torch, transformers; print(sys.executable); print(torch.__version__); print(transformers.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

重新安装依赖：

```powershell
cd D:\qq\low-cost-memory-integrate-codex
D:\qq\low-cost-memory-integrate\.venv\Scripts\python.exe -m pip install -r examples\competition_agents\requirements.txt
```

### 1.2 前端环境

```powershell
cd D:\qq\low-cost-memory-integrate-codex\examples\competition_agents\frontend
npm.cmd install
```

## 2. 模型准备

本项目通过 HuggingFace 缓存加载本地模型。建议先下载模型，再离线运行。

设置国内镜像：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

下载任务模型示例：

```powershell
D:\qq\low-cost-memory-integrate\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-0.5B-Instruct')"
```

可选模型：

```text
Qwen/Qwen2.5-0.5B-Instruct
Qwen/Qwen2.5-1.5B-Instruct
TinyLlama/TinyLlama-1.1B-Chat-v1.0
HuggingFaceTB/SmolLM2-1.7B-Instruct
microsoft/Phi-3.5-mini-instruct
deepseek-ai/deepseek-coder-1.3b-instruct
```

下载 LLMLingua-2 官方压缩模型：

```powershell
D:\qq\low-cost-memory-integrate\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank')"
```

## 3. 启动系统

### 3.1 清理旧后端

如果 8765 端口已有旧进程，先停止：

```powershell
Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

### 3.2 启动后端

推荐使用项目脚本启动。脚本会统一设置本地模型、离线加载、embedding、LLMLingua-2 等运行环境。

```powershell
cd D:\qq\low-cost-memory-integrate-codex
D:\qq\low-cost-memory-integrate\.venv\Scripts\Activate.ps1
powershell -ExecutionPolicy Bypass -File examples\competition_agents\start_local_runner.ps1 -UseOfficialLLMLingua2
```

如果不运行 LLMLingua-2 对比方法，可以去掉 `-UseOfficialLLMLingua2`：

```powershell
powershell -ExecutionPolicy Bypass -File examples\competition_agents\start_local_runner.ps1
```

启动成功后应看到：

```text
runner server: http://127.0.0.1:8765
```

确认后端使用的是虚拟环境：

```powershell
Get-NetTCPConnection -LocalPort 8765 -State Listen |
  ForEach-Object {
    Get-CimInstance Win32_Process -Filter "ProcessId=$($_.OwningProcess)" |
      Select-Object ProcessId,ExecutablePath,CommandLine
  }
```

`ExecutablePath` 应指向：

```text
D:\qq\low-cost-memory-integrate\.venv\Scripts\python.exe
```

### 3.3 启动前端

```powershell
cd D:\qq\low-cost-memory-integrate-codex\examples\competition_agents\frontend
npm.cmd run dev
```

浏览器打开：

```text
http://localhost:5173/
```

## 4. 页面实验流程

在“实验控制台”依次选择：

```text
数据集
任务数量
协作方式
记忆组织方式
检索方式
模型后端
本地模型
```

点击“启动实验”后，前端调用 `runner_server.py`，后端启动 `run_demo.py` 并把运行事件写入：

```text
examples/competition_agents/frontend/public/runtime/run_events.jsonl
examples/competition_agents/frontend/public/runtime/latest_run_stdout.log
```

这两个文件是临时运行日志，不作为正式结果提交。

## 5. 实验模式

| 页面选项 | 后端模式 | 说明 |
| --- | --- | --- |
| 纯文本 | `baseline_text` | 自然语言通信基线 |
| AutoForm | `autoform_text` | 模型直接生成更紧凑的通信表示 |
| LLMLingua-2 | `llmlingua_text` | 对发送消息进行生成后压缩 |
| 结构化协议 | `protocol_api` | 结构化字段通信 |
| 协议 + 向量 | `protocol_api_embedding` | 结构化协议 + embedding 状态引用 |
| 隐状态传递 | `hidden_api` | hidden state 状态引用传递 |

## 6. 数据集

| 页面名称 | 后端数据文件 |
| --- | --- |
| TopiOCQA 第 1 组 | `examples/competition_agents/data/topiocqa_1_10turns.json` |
| TopiOCQA 第 2 组 | `examples/competition_agents/data/topiocqa_2_10turns.json` |
| CodeAct 第 1 组 | `examples/competition_agents/data/codeact_1_10tasks.json` |
| CodeAct 第 2 组 | `examples/competition_agents/data/codeact_2_10tasks.json` |

## 7. 结果保存和导入

实验完成后，结果保存到：

```text
run_result/<模型目录>/
```

示例：

```text
run_result/Qwen2.5-0.5B-Instruct/
run_result/SmolLM2-1.7B-Instruct/
run_result/Phi-3.5-mini-instruct/
```

结果文件名包含实验模式、数据集和检索方式，例如：

```text
frontend_baseline_text_topiocqa-1_fusion.json
frontend_protocol_api_topiocqa-1_fusion.json
frontend_hidden_api_codeact-2_fusion.json
frontend_llmlingua_text_0.75_topiocqa-1_fusion.json
```

前端“实验记录 / 导入结果”支持按模型文件夹选择结果文件。导入后，指标评估页面会按当前选择的数据集展示对应结果。

## 8. 指标口径

| 指标 | 含义 |
| --- | --- |
| 文本字符数 | Agent 间显式文本通信字符数 |
| 通信载荷 | 文本通信、协议消息和状态引用的综合开销 |
| 非文本状态次数 | embedding、hidden state 等非文本状态传递次数 |
| 非文本状态规模 | 非文本状态传递或存储规模，单位通常为 KB |
| 单任务耗时 | 每个任务从开始到完成的端到端耗时 |
| 准确率 | 与数据集标准答案的 exact match 结果 |
| 记忆复用率 | 有记忆命中的任务比例 |
| 记忆召回率 | 期望复用记忆被召回的比例 |
| 记忆精度 | 命中记忆中有效记忆的比例 |

当前系统主要统计字符和载荷规模，不把字符数直接等同于 token 数。报告中如写 token，需要额外说明 tokenizer 和统计方法。

## 9. 命令行运行

不经过前端，也可以直接运行主实验入口：

```powershell
cd D:\qq\low-cost-memory-integrate-codex

D:\qq\low-cost-memory-integrate\.venv\Scripts\python.exe examples\competition_agents\run_demo.py `
  --experiment-mode baseline_text `
  --llm-backend local `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\topiocqa_1_10turns.json `
  --limit 1 `
  --context-source hypergraph `
  --memory-retrieval-mode fusion `
  --show-memory `
  --show-latent `
  --show-state
```

CodeAct 示例：

```powershell
D:\qq\low-cost-memory-integrate\.venv\Scripts\python.exe examples\competition_agents\run_demo.py `
  --experiment-mode protocol_api `
  --llm-backend local `
  --dataset-type codeact `
  --dataset-file examples\competition_agents\data\codeact_2_10tasks.json `
  --limit 1 `
  --context-source hypergraph `
  --memory-retrieval-mode fusion `
  --show-memory `
  --show-latent `
  --show-state
```

## 10. openEuler 运行

openEuler 环境可参考：

```text
examples/competition_agents/setup_openeuler_env.sh
examples/competition_agents/competition_docs/02_部署文档.md
```

基本流程：

```bash
cd ~/code/low-cost-mult-agent-memory
python3 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r examples/competition_agents/requirements.txt
```

前端需要安装 Node.js 和 npm 后运行：

```bash
cd examples/competition_agents/frontend
npm install
npm run dev -- --host 0.0.0.0
```

## 11. 常见问题

### 11.1 页面选择本地模型但仍请求网络

检查后端启动环境是否正确：

```powershell
Get-NetTCPConnection -LocalPort 8765 -State Listen |
  ForEach-Object {
    Get-CimInstance Win32_Process -Filter "ProcessId=$($_.OwningProcess)" |
      Select-Object ProcessId,ExecutablePath,CommandLine
  }
```

如果显示 `D:\python3.11\python.exe`，说明旧后端或错误环境仍在运行。停止旧进程后重新用根目录脚本启动。

### 11.2 GPU 没有占用

先确认环境支持 CUDA：

```powershell
D:\qq\low-cost-memory-integrate\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

运行过程中再观察：

```powershell
nvidia-smi -l 1
```

### 11.3 LLMLingua-2 启动慢

第一次使用官方 LLMLingua-2 压缩模型时会加载 `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`。如果模型未提前下载，会触发网络请求。正式实验建议提前下载并用脚本离线启动。

### 11.4 导入结果没有显示

确认导入的是 `.json` 结果文件，并且页面当前选择的数据集与结果文件的数据集一致。例如 `frontend_baseline_text_codeact-2_fusion.json` 只会在 CodeAct 第 2 组下展示。
