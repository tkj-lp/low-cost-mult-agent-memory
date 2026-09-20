# 多智能体低开销通信实验系统运行手册

本文档是项目最终运行手册，覆盖环境准备、模型下载、前后端启动、实验运行、结果保存、结果导入和常见问题排查。所有命令默认从仓库根目录执行，不依赖本机绝对路径。

## 1. 环境要求

### 1.1 openEuler/Linux 环境

赛题指定交付环境为：

```text
openEuler 24.03-LTS-SP3
```

推荐依赖：

- Python 3.10 或 3.11。
- Node.js 20 及以上。
- npm。
- Git。
- 可选 NVIDIA GPU 与可用 CUDA 驱动。

推荐使用项目脚本完成环境初始化：

```bash
bash examples/competition_agents/setup_openeuler_env.sh
```

脚本默认创建 Conda 环境 `competition-agents`。初始化完成后激活环境：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate competition-agents
```

检查 Python、PyTorch、Transformers 和 CUDA：

```bash
python -c "import sys, torch, transformers; print(sys.executable); print(torch.__version__); print(transformers.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

### 1.2 Windows 本地调试环境

Windows 环境用于本地调试和演示录制。推荐在仓库根目录创建虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r examples\competition_agents\requirements.txt
```

检查环境：

```powershell
python -c "import sys, torch, transformers; print(sys.executable); print(torch.__version__); print(transformers.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

### 1.3 前端依赖

```bash
cd examples/competition_agents/frontend
npm install
cd ../../..
```

Windows 下也可使用：

```powershell
cd examples\competition_agents\frontend
npm.cmd install
cd ..\..\..
```

## 2. 模型准备

本项目通过 HuggingFace 缓存加载本地模型。建议先下载模型，再离线运行，避免正式实验中触发网络请求。

### 2.1 设置镜像

Linux/openEuler：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Windows PowerShell：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

### 2.2 下载任务模型

以 Qwen2.5 0.5B 为例：

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-0.5B-Instruct')"
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

下载其他模型时替换 `repo_id`。

### 2.3 下载 LLMLingua-2 压缩模型

运行 LLMLingua-2 对比方法时需要官方压缩模型：

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank')"
```

说明：

- 页面选择的本地模型仍是任务 LLM。
- LLMLingua-2 模型只负责压缩 Agent 间传输文本。
- 对比实验中应保证同一数据集、同一任务模型和同一检索配置。

### 2.4 离线运行

模型下载完成后，正式实验推荐离线运行。

Linux/openEuler：

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export LOCAL_HIDDEN_LOCAL_FILES_ONLY=1
```

Windows PowerShell：

```powershell
$env:TRANSFORMERS_OFFLINE="1"
$env:HF_HUB_OFFLINE="1"
$env:LOCAL_HIDDEN_LOCAL_FILES_ONLY="1"
```

## 3. 启动系统

### 3.1 openEuler/Linux 启动

启动后端：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate competition-agents
python examples/competition_agents/runner_server.py
```

启动成功后应看到：

```text
runner server: http://127.0.0.1:8765
```

另开终端启动前端：

```bash
cd examples/competition_agents/frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

浏览器访问：

```text
http://<服务器 IP>:5173/
```

### 3.2 Windows 本地启动

清理旧后端：

```powershell
Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

启动后端：

```powershell
.\.venv\Scripts\Activate.ps1
powershell -ExecutionPolicy Bypass -File examples\competition_agents\start_local_runner.ps1 -UseOfficialLLMLingua2
```

如果不运行 LLMLingua-2 对比方法，可以去掉 `-UseOfficialLLMLingua2`：

```powershell
powershell -ExecutionPolicy Bypass -File examples\competition_agents\start_local_runner.ps1
```

启动前端：

```powershell
cd examples\competition_agents\frontend
npm.cmd run dev
```

浏览器访问：

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
| 完整协议包 | `logical_protocol_bytes`，逻辑协议对象的 UTF-8 序列化长度 |
| 引用通信 | `reference_wire_bytes`，最小状态引用控制包的估算长度，不等同于物理网络流量 |
| 非文本状态次数 | embedding、hidden state 等非文本状态传递次数 |
| 非文本状态规模 | `latent_payload_bytes`，向量或 hidden payload 的逻辑数据规模 |
| 状态池规模 | `state_store_estimated_bytes`，进程内状态池载荷的估算规模 |
| 单任务耗时 | 每个任务从开始到完成的端到端耗时 |
| 准确率 | 与数据集标准答案的 exact match 结果 |
| 记忆命中任务占比 | 旧口径 `memory_hit_rate`：有任意记忆命中的任务比例，不判断召回是否相关 |
| QA 记忆召回率 | 有 `expected_reuse` 标注时，应找回的历史 QA 中实际找回的比例 |
| QA 记忆精度 | 有 `expected_reuse` 标注时，找回的 QA 记忆中真正相关的比例 |
| 有效 QA 复用率 | 相关 QA 记忆被显式采用且最终答案 EM 正确的任务比例 |

当前系统主要统计字符和逻辑载荷规模，不把字符数直接等同于 token 数。系统尚未接入跨进程 Socket/IPC 测量，因此不得把引用通信字节数描述为物理网络传输量。`elapsed_ms` 是正式端到端耗时；`elapsed_excluding_state_generation_ms` 只用于状态生成开销消融，不作为端到端主指标。报告中如写 token，需要额外说明 tokenizer 和统计方法。

对已有实验结果，可用离线脚本补算 QA 复用指标，不覆盖原始 JSON：

```bash
python examples/competition_agents/reassess_reuse.py --annotate-datasets --results run_result
```

该命令会：

1. 把 `data/qa_reuse_annotations.json` 写回 factoid 数据集的 `expected_reuse`
2. 为匹配的 `run_result/**/frontend_*.json` 生成同名 `.reuse-evaluation` 侧车
3. 前端/导入结果时读取侧车，补显示 QA 召回率与精度

## 9. 命令行运行

不经过前端，也可以直接运行主实验入口。

TopiOCQA 示例：

```bash
python examples/competition_agents/run_demo.py \
  --experiment-mode baseline_text \
  --llm-backend local \
  --dataset-type continuous \
  --dataset-file examples/competition_agents/data/topiocqa_1_10turns.json \
  --limit 10 \
  --context-source hypergraph \
  --memory-retrieval-mode fusion \
  --show-memory \
  --show-latent \
  --show-state
```

CodeAct 示例：

```bash
python examples/competition_agents/run_demo.py \
  --experiment-mode protocol_api \
  --llm-backend local \
  --dataset-type codeact \
  --dataset-file examples/competition_agents/data/codeact_2_10tasks.json \
  --limit 10 \
  --context-source hypergraph \
  --memory-retrieval-mode fusion \
  --show-memory \
  --show-latent \
  --show-state
```

Windows PowerShell 下把反斜杠续行改成反引号即可：

```powershell
python examples\competition_agents\run_demo.py `
  --experiment-mode baseline_text `
  --llm-backend local `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\topiocqa_1_10turns.json `
  --limit 10 `
  --context-source hypergraph `
  --memory-retrieval-mode fusion `
  --show-memory `
  --show-latent `
  --show-state
```

## 10. 运行验证

### 10.1 后端进程验证

Linux/openEuler：

```bash
ss -ltnp | grep 8765
```

Windows：

```powershell
Get-NetTCPConnection -LocalPort 8765 -State Listen |
  ForEach-Object {
    Get-CimInstance Win32_Process -Filter "ProcessId=$($_.OwningProcess)" |
      Select-Object ProcessId,ExecutablePath,CommandLine
  }
```

### 10.2 GPU 验证

```bash
nvidia-smi -l 1
```

如果运行本地模型时 GPU 显存上升，说明模型已加载到 GPU。若显存无变化，需要检查后端环境、模型加载配置和旧进程占用情况。

### 10.3 结果验证

完成一轮 10 任务实验后，应检查：

1. 页面任务进度达到 `10 / 10`。
2. `run_result/<模型名称>/` 下生成对应 JSON 文件。
3. 指标评估页可以显示文本字符数、单任务耗时、通信开销、记忆复用率、准确率等指标。
4. 导入结果后，切换数据集时只显示对应数据集的结果。

## 11. 常见问题

### 11.1 页面选择本地模型但仍请求网络

原因通常是模型或 LLMLingua-2 压缩模型没有提前下载，或后端没有以离线模式启动。处理方式是先下载模型，再设置离线变量并重启后端。

### 11.2 端口 8765 被旧后端占用

Linux/openEuler：

```bash
pkill -f "examples/competition_agents/runner_server.py" || true
```

Windows：

```powershell
Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

### 11.3 LLMLingua-2 启动慢

第一次使用官方 LLMLingua-2 压缩模型时会加载 `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`。如果模型未提前下载，会触发网络请求。正式实验建议提前下载并离线启动。

### 11.4 导入结果没有显示

确认导入的是 `.json` 结果文件，并且页面当前选择的数据集与结果文件的数据集一致。例如 `frontend_baseline_text_codeact-2_fusion.json` 只会在 CodeAct 第 2 组下展示。
