# 运行手册

这份文档只回答一个问题：不同实验应该怎么跑。页面同学可以把实验配置理解成两个选择框：

```text
运行方式：quick / api / api_embedding / latent_hidden
数据集：competition_10 / hotpot_multihop / hotpot_continuous / hidden_hotpot_5 / hidden_continuous
```

当前推荐的统一入口是 `run_demo.py`。它可以跑三类数据：内置 10 轮比赛主流程、多跳问答样例、连续关联任务。`latent_hidden` 使用单独脚本，因为它验证的是本地 LLM 最后一层 hidden state 的引用传递机制。

## 1. 四种运行方式

| 运行方式 | 含义 | 是否调用 API | 是否生成 embedding | 是否读取 LLM hidden state |
| --- | --- | --- | --- | --- |
| `quick` | 全部 Agent 使用 mock，最快最稳定，适合先验证流程和页面 | 否 | 默认 hash 或按环境变量决定 | 否 |
| `api` | Planner / Reasoner / Summarizer 使用真实 API 生成文字 | 是 | 否 | 否 |
| `api_embedding` | API 负责生成文字，本地 embedding 把中间文本压成向量并通过 `state_ref` 传递 | 是 | 是 | 否 |
| `latent_hidden` | 本地 Qwen 前向推理后直接读取 `outputs.hidden_states[-1]`，消息里只传 `hidden_state_ref` | 否 | 否 | 是 |

注意：`api_embedding` 不是直接传 LLM hidden state，它是“文本 -> embedding 向量 -> state_ref”。`latent_hidden` 才是读取本地 LLM 最后一层 hidden state。

## 2. 统一数据集入口

`run_demo.py` 现在支持命令行选择数据：

```powershell
python examples\competition_agents\run_demo.py --dataset-type builtin
python examples\competition_agents\run_demo.py --dataset-type multihop --dataset-file examples\competition_agents\data\hotpot_multihop_sample.json
python examples\competition_agents\run_demo.py --dataset-type continuous --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json
```

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--dataset-type builtin` | 跑内置 10 轮比赛主流程，默认值 |
| `--dataset-type multihop` | 跑 HotpotQA 抽样多跳问答 |
| `--dataset-type continuous` | 跑从 HotpotQA 构造的连续关联任务，用于验证记忆复用 |
| `--dataset-file` | 指定 JSON 数据文件，`multihop` 和 `continuous` 必填 |
| `--limit` | 限制样例数；连续任务中表示限制 group 数 |
| `--modes` | 指定 `text`、`protocol` 或 `text,protocol` |

## 3. quick：快速测试

用途：不调用 API，不下载大模型，先确认多 Agent 流程、结构化协议和指标输出可以跑通。

```powershell
cd D:\qq\low-cost-memory-integrate
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH="libs/langgraph;examples/competition_agents"

$env:LLM_PROVIDER="mock"
$env:LLM_PLANNER_PROVIDER="mock"
$env:LLM_REASONER_PROVIDER="mock"
$env:LLM_SUMMARIZER_PROVIDER="mock"
$env:EMBEDDING_PROVIDER="hash"
```

跑 10 轮比赛主流程：

```powershell
python examples\competition_agents\run_demo.py --dataset-type builtin
```

跑 1 组连续关联任务，适合快速看记忆命中：

```powershell
python examples\competition_agents\run_demo.py `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json `
  --limit 1 `
  --modes protocol
```

## 4. api：真实 API 生成文字

用途：让 Planner / Reasoner / Summarizer 使用真实大模型 API，验证输出质量。API 返回的是文字，不是 hidden state。

```powershell
cd D:\qq\low-cost-memory-integrate
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH="libs/langgraph;examples/competition_agents"

$env:LLM_PROVIDER="mock"
$env:LLM_PLANNER_PROVIDER="openai_compatible"
$env:LLM_REASONER_PROVIDER="openai_compatible"
$env:LLM_SUMMARIZER_PROVIDER="openai_compatible"

$env:LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:LLM_MODEL="qwen-plus-2025-07-28"
$env:LLM_API_KEY="your_api_key"
$env:EMBEDDING_PROVIDER="hash"
```

跑多跳问答，并输出 `gold` 和 `exact_match`：

```powershell
python examples\competition_agents\run_demo.py `
  --dataset-type multihop `
  --dataset-file examples\competition_agents\data\hotpot_multihop_sample.json `
  --limit 5
```

## 5. api_embedding：真实 API + 本地 embedding

用途：API 负责生成计划、推理和总结文字；本地 embedding 模型把中间文本压成 384 维向量，存入 latent state，Agent 消息中传 `state_ref`。

第一次运行前安装：

```powershell
pip install sentence-transformers
```

配置：

```powershell
$env:LLM_PROVIDER="mock"
$env:LLM_PLANNER_PROVIDER="openai_compatible"
$env:LLM_REASONER_PROVIDER="openai_compatible"
$env:LLM_SUMMARIZER_PROVIDER="openai_compatible"

$env:LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:LLM_MODEL="qwen-plus-2025-07-28"
$env:LLM_API_KEY="your_api_key"

$env:EMBEDDING_PROVIDER="local"
$env:LOCAL_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

跑 10 轮比赛主流程：

```powershell
python examples\competition_agents\run_demo.py --dataset-type builtin
```

跑连续任务：

```powershell
python examples\competition_agents\run_demo.py `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json `
  --limit 5
```

## 6. latent_hidden：本地 LLM hidden state 引用传递

用途：验证“非文本状态传递”中更接近模型内部状态的一种实现。它会读取本地 Qwen 的最后一层 hidden state，把完整 tensor 放进进程内共享 latent memory，Agent 消息中只传 `hidden-0001` 这样的引用。

第一次运行前安装：

```powershell
pip install torch transformers
```

配置：

```powershell
$env:LOCAL_HIDDEN_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
```

单条机制验证：

```powershell
python examples\competition_agents\run_latent_hidden_demo.py
```

5 条 HotpotQA 样例的 hidden-state reference transfer 实验：

```powershell
python examples\competition_agents\run_hidden_latent_samples.py
```

连续关联任务的 hidden-state reference transfer + 记忆复用实验：

```powershell
python examples\competition_agents\run_hidden_latent_samples.py `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json `
  --limit 1
```

重要说明：当前 hidden state 只作为“非文本消息传递对象”和“状态引用机制”使用，不声明 Reasoner 已经完全依赖 hidden state 直接推理。答案质量仍由 API/mock/evidence 文本链路验证。

## 7. 页面同学推荐接入组合

| 页面选择：运行方式 | 页面选择：数据集 | 推荐命令 |
| --- | --- | --- |
| `quick` | `competition_10` | `python examples\competition_agents\run_demo.py --dataset-type builtin` |
| `quick` | `hotpot_continuous` | `python examples\competition_agents\run_demo.py --dataset-type continuous --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json --limit 1 --modes protocol` |
| `api` | `hotpot_multihop` | `python examples\competition_agents\run_demo.py --dataset-type multihop --dataset-file examples\competition_agents\data\hotpot_multihop_sample.json --limit 5` |
| `api_embedding` | `competition_10` | `python examples\competition_agents\run_demo.py --dataset-type builtin` |
| `api_embedding` | `hotpot_continuous` | `python examples\competition_agents\run_demo.py --dataset-type continuous --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json --limit 5` |
| `latent_hidden` | `hidden_hotpot_5` | `python examples\competition_agents\run_hidden_latent_samples.py` |
| `latent_hidden` | `hidden_continuous` | `python examples\competition_agents\run_hidden_latent_samples.py --dataset-type continuous --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json --limit 1` |

## 8. 输出字段

页面和实验报告优先读取这些字段：

```text
messages[].from
messages[].to
messages[].action
messages[].char_cost
metrics.message_count
metrics.text_chars
metrics.latent_transfer_count
metrics.latent_bytes
memory_hits
memory_hit_rate
latent_states
elapsed_ms
exact_match
```

`payload` 内部结构后续可能继续压缩优化，页面不要强依赖 payload 的具体子字段。稳定字段是 `from`、`to`、`action`、`char_cost`、`metrics`、`memory_hits`、`latent_states`。
