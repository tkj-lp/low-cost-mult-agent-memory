# 运行手册

本文档回答两个问题：

1. 不同实验模式应该怎么跑。
2. 页面上的收益统计应该怎么和基线结果对齐。

本文档同时是当前唯一正式运行手册。凡是运行命令、结果文件命名、`run_result_json` 输出口径、前后端稳定字段约定，统一以本文档为准；页面字段细节再结合 `FRONTEND_CONTRACT.md` 与 `INTEGRATION.md` 一起看。

当前正式实验入口采用两层选择：

```text
experiment_mode：baseline_text / protocol_api / protocol_api_embedding / latent_hidden / codeact_tools
dataset: hotpot_multihop / hotpot_continuous / topiocqa_continuous / hidden_hotpot_5 / hidden_continuous / codeact_20
```

`mock` 只用于终端自测流程是否跑通，不作为页面正式运行方式。

## 0. 当前状态总览

截至当前版本，下面这些正式方法都已经可以在至少一个正式数据集上跑通，并输出前端可直接解析的 `run_result_json`：

| experiment_mode | 当前已跑通的数据集口径 | 备注 |
| --- | --- | --- |
| `baseline_text` | `hotpot_multihop` / `hotpot_continuous` / `topiocqa_continuous` | 作为收益统计基线。 |
| `protocol_api` | `hotpot_multihop` / `hotpot_continuous` / `topiocqa_continuous` | 结构化协议主链。 |
| `protocol_api_embedding` | `hotpot_multihop` / `hotpot_continuous` / `topiocqa_continuous` | 已支持 `context_source=hypergraph`。 |
| `latent_hidden` | `hidden_hotpot_5` / `hidden_continuous` / `topiocqa_continuous` | 已支持 `context_source=hypergraph`。 |
| `codeact_tools` | `codeact_20` | 独立执行器与主图闭环都可跑。 |

补充说明：

```text
1. “跑通”指命令可正常完成，并在输出末尾带有 `=== run_result_json ===` 机器可读区块。
2. “跑通”不等于“当前效果最好”。例如 TopiOCQA 上 `protocol_api_embedding` 和 `latent_hidden` 都已跑通，但准确率仍会随检索和生成策略变化。
3. 前端正式解析入口始终只有 `run_result_json`，不要依赖普通日志文本。
```

## 1. 正式实验模式

| experiment_mode | 含义 | 入口 |
| --- | --- | --- |
| `baseline_text` | 纯文本协作基线。真实 API 生成文字，Agent 间用自然语言长文本通信。 | `run_demo.py` |
| `protocol_api` | 结构化协议协作。真实 API 生成文字，Agent 间用结构化协议包通信，embedding 使用 hash 占位。 | `run_demo.py` |
| `protocol_api_embedding` | 协议 + API embedding。真实 API 生成文字，并用 OpenAI-compatible embedding 生成 `state_ref`。 | `run_demo.py` |
| `latent_hidden` | 本地 LLM hidden state 引用传递。消息里只传 `hidden_state_ref`。 | `run_hidden_latent_samples.py` |
| `codeact_tools` | 受限 Python CodeAct 执行。使用 `reference_code` 或后续 LLM 生成代码，经 AST 安全检查和临时目录子进程执行后返回结构化结果。 | `run_codeact_samples.py` |

`run_demo.py` 在设置 `--experiment-mode` 时会自动读取：

```text
examples/competition_agents/.env.local
```

并把 Planner / Reasoner / Summarizer 设置为真实 API：

```text
LLM_PROVIDER=openai_compatible
LLM_PLANNER_PROVIDER=openai_compatible
LLM_REASONER_PROVIDER=openai_compatible
LLM_SUMMARIZER_PROVIDER=openai_compatible
LLM_REQUIRE_REAL=1
```

因此正式实验前需要保证 `.env.local` 至少包含：

```text
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=你的模型名
LLM_API_KEY=你的 key
```

如果运行 `protocol_api_embedding`，还会使用：

```text
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_MODEL=text-embedding-v4
```

如果没有单独设置 `EMBEDDING_API_KEY`，embedding 会复用 `LLM_API_KEY`。

## 2. 数据集

| dataset | run_demo.py 参数 | 含义 |
| --- | --- | --- |
| `hotpot_multihop` | `--dataset-type multihop --dataset-file examples\competition_agents\data\hotpot_multihop_sample.json` | HotpotQA 抽样多跳问答。 |
| `hotpot_continuous` | `--dataset-type continuous --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json` | 从 `hotpot_dev_distractor_v1.json` 固定抽样构造的连续关联任务，当前为 2 个 chain，每个 chain 10 turn，共 20 turn。 |
| `topiocqa_continuous` | `--dataset-type continuous --dataset-file examples\competition_agents\data\topiocqa_two_chains_10turns_refined.json` | TopiOCQA 风格的连续追问任务，当前为 2 个 chain，每个 chain 10 turn，共 20 turn。 |
| `hidden_hotpot_5` | `run_hidden_latent_samples.py` 默认 | 5 条 hidden-state reference transfer 样例。 |
| `hidden_continuous` | `run_hidden_latent_samples.py --dataset-type continuous --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json` | 同一连续任务数据集上的 hidden-state reference transfer + memory reuse。 |
| `codeact_20` | `run_codeact_samples.py --dataset-file examples\competition_agents\data\codeact_tasks.json` | 自制 20 条安全工具执行任务，覆盖统计、表格、字符串、日期、证据链处理和记忆复用。 |

`hotpot_continuous_tasks.json` 由脚本从 HotpotQA dev distractor 原始数据集中固定抽样生成，不手写事实：

```powershell
python examples\competition_agents\build_hotpot_continuous_tasks.py `
  --source examples\competition_agents\data\hotpot_dev_distractor_v1.json `
  --output examples\competition_agents\data\hotpot_continuous_tasks.json
```

当前固定样本：

```text
chain-001: 5a8c7595554299585d9e36b6, Kiss and Tell (1945 film) -> Shirley Temple, 10 turn
chain-002: 5a85ea095542994775f606a8, The Hork-Bajir Chronicles -> Animorphs, 10 turn
```

运行时 `--limit 1` 表示只跑 `chain-001` 的 10 turn；不加 `--limit` 会跑 2 个 chain，共 20 turn。

`topiocqa_two_chains_10turns_refined.json` 当前也使用相同的连续任务分组格式：

```text
chain-001: Dunkirk (2017 film) -> Urk -> North Sea, 10 turn
chain-002: Metabolism -> Catabolism -> Hormone -> Cortisol / Insulin / Melatonin, 10 turn
```

运行时 `--limit 1` 同样表示只跑 `chain-001` 的 10 turn；不加 `--limit` 会跑 2 个 chain，共 20 turn。

## 3. 可用组合

| experiment_mode | 可用 dataset |
| --- | --- |
| `baseline_text` | `hotpot_multihop`, `hotpot_continuous`, `topiocqa_continuous` |
| `protocol_api` | `hotpot_multihop`, `hotpot_continuous`, `topiocqa_continuous` |
| `protocol_api_embedding` | `hotpot_multihop`, `hotpot_continuous`, `topiocqa_continuous` |
| `latent_hidden` | `hidden_hotpot_5`, `hidden_continuous`, `topiocqa_continuous` |
| `codeact_tools` | `codeact_20` |

不建议组合：

```text
latent_hidden + hotpot_multihop
baseline_text + hidden_*
protocol_api + hidden_*
protocol_api_embedding + hidden_*
codeact_tools + non-codeact datasets
```

原因：`latent_hidden` 当前使用独立脚本，验证的是本地 LLM hidden state 引用传递；它已可运行连续任务正式数据集，但入口仍独立于 `run_demo.py`。

## 4. baseline_text

多跳问答：

```powershell
python examples\competition_agents\run_demo.py `
  --experiment-mode baseline_text `
  --dataset-type multihop `
  --dataset-file examples\competition_agents\data\hotpot_multihop_sample.json `
  --limit 5
```

连续任务：

```powershell
python examples\competition_agents\run_demo.py `
  --experiment-mode baseline_text `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json `
  --limit 1
```

TopiOCQA 风格连续任务：

```powershell
python examples\competition_agents\run_demo.py `
  --experiment-mode baseline_text `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\topiocqa_two_chains_10turns_refined.json `
  --limit 1
```
 2>&1 | Tee-Object run_result\baseline_topiocqa_run.txt

这里 `--limit 1` 表示运行 1 个连续任务组；当前示例数据文件中的该任务组包含 10 个 turn。

`baseline_text` 是页面收益统计的统一参照项。前端需要保存同一 dataset 下的 baseline 结果，用于和其他实验模式计算收益。

## 5. protocol_api

多跳问答：

```powershell
python examples\competition_agents\run_demo.py `
  --experiment-mode protocol_api `
  --dataset-type multihop `
  --dataset-file examples\competition_agents\data\hotpot_multihop_sample.json `
  --limit 5
```

连续任务，带超图/latent 展示：

```powershell
python examples\competition_agents\run_demo.py `
  --experiment-mode protocol_api `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json `
  --limit 1 `
  --show-memory `
  --show-latent `
  --show-state
```

TopiOCQA 风格连续任务，带超图/latent 展示：

```powershell
python examples\competition_agents\run_demo.py `
  --experiment-mode protocol_api `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\topiocqa_two_chains_10turns_refined.json `
  --limit 1 `
  --show-memory `
  --show-latent `
  --show-state
```
2>&1 | Tee-Object run_result\api_topiocqa_run.txt

这里 `--limit 1` 表示运行 1 个连续任务组；当前示例数据文件中的该任务组包含 10 个 turn。

## 6. protocol_api_embedding

多跳问答：

```powershell
python examples\competition_agents\run_demo.py `
  --experiment-mode protocol_api_embedding `
  --dataset-type multihop `
  --dataset-file examples\competition_agents\data\hotpot_multihop_sample.json `
  --limit 5 
```
 2>&1 | Tee-Object run_result\protocol_api_embedding_multihop_run.txt

连续任务：

```powershell
python examples\competition_agents\run_demo.py `
  --experiment-mode protocol_api_embedding `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json `
  --limit 1 `
  --show-memory `
  --show-latent `
  --show-state 
```
2>&1 | Tee-Object run_result\protocol_api_embedding_continuous_run.txt

TopiOCQA 风格连续任务：

```powershell
python examples\competition_agents\run_demo.py `
  --experiment-mode protocol_api_embedding `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\topiocqa_two_chains_10turns_refined.json `
  --limit 1 `
  --context-source hypergraph `
  --show-memory `
  --show-latent `
  --show-state
```
 2>&1 | Tee-Object run_result\api_embedding_TopiOCQA_run.txt

这里 `--limit 1` 表示运行 1 个连续任务组；当前示例数据文件中的该任务组包含 10 个 turn。

当前推荐口径：

```text
TopiOCQA 连续任务请显式使用 `--context-source hypergraph`。
这表示顶层任务文本只保留 Question，
Retriever 再从超图里取回 supporting passage / context hits。
```

该模式重点展示：

```text
latent_transfer_count
latent_bytes
latent_states.*.embedding_provider
latent_states.*.embedding_model
state_refs
```

## 7. latent_hidden

这一部分当前有两个入口：

1. `run_hidden_generation_demo.py`
   只验证“同一个本地模型生成文本，并在同一条生成流程里直接拿最后一层 hidden state”。
2. `run_hidden_latent_samples.py`
   用连续任务数据集验证 hidden-state 引用传递、共享记忆复用和存储开销。

### 7.1 最小 demo

```powershell
$env:PYTHONPATH="libs/langgraph;examples/competition_agents"
$env:LOCAL_HIDDEN_MODEL="Qwen/Qwen2.5-0.5B-Instruct"

python examples\competition_agents\run_hidden_generation_demo.py
```

这个 demo 不跑数据集，只验证：

```text
同一个本地模型生成答案
同一条生成流程里直接拿最后一层 hidden state
```

### 7.2 数据集运行

如果第一次运行缺少依赖，需要先安装：

```powershell
pip install torch transformers
```

轻量模式配置：

```powershell
$env:PYTHONPATH="libs/langgraph;examples/competition_agents"
$env:LOCAL_HIDDEN_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
$env:LOCAL_HIDDEN_STORAGE_MODE="pooled_only"
```

说明：

```text
默认推荐模式。
只保留 pooled hidden、pooled_shape、checksum、shape 等元信息，
不在进程内保留完整 [seq_len, hidden_dim] tensor。
更适合前端展示和连续任务演示。
```

Tensor 配置：

```powershell
$env:PYTHONPATH="libs/langgraph;examples/competition_agents"
$env:LOCAL_HIDDEN_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
$env:LOCAL_HIDDEN_STORAGE_MODE="full"
```

说明：

```text
调试模式。
会在进程内保留完整 hidden tensor，并把 hidden_bytes 按完整存储规模统计。
更慢、更占内存，不推荐作为默认演示模式。
```


TopiOCQA 连续任务，推荐跑法：

```powershell
python examples\competition_agents\run_hidden_latent_samples.py `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\topiocqa_two_chains_10turns_refined.json `
  --limit 1 `
  --context-source hypergraph `
  --answer-mode generate
```

`pooled_only` 推荐结果文件：

```powershell
$env:LOCAL_HIDDEN_STORAGE_MODE="pooled_only"

python examples\competition_agents\run_hidden_latent_samples.py `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\topiocqa_two_chains_10turns_refined.json `
  --limit 1 `
  --context-source hypergraph `
  --answer-mode generate `
```
  2>&1 | Tee-Object run_result\hidden_pooled_only_TopiOCQA_run.txt


`full` 调试结果文件：

```powershell
$env:LOCAL_HIDDEN_STORAGE_MODE="full"

python examples\competition_agents\run_hidden_latent_samples.py `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\topiocqa_two_chains_10turns_refined.json `
  --limit 1 `
  --context-source hypergraph `
  --answer-mode generate `
```
  2>&1 | Tee-Object run_result\hidden_full_TopiOCQA_run.txt


这里：

```text
`--limit 1` 表示只跑 chain-001 的 10 个 turn。
`--answer-mode generate` 表示由同一个本地模型生成答案，并直接提取这次生成流程的 hidden state。
`--context-source hypergraph` 表示 hidden 路径也采用“Question only + 从超图取证据”的正式口径。
```

Hotpot 连续任务，推荐跑法：

```powershell
python examples\competition_agents\run_hidden_latent_samples.py `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json `
  --limit 1 `
  --answer-mode generate
```


### 7.5 说明口径

```text
`run_hidden_latent_samples.py --answer-mode generate` 现在已经不是旧的关键词抽取主路径，
而是“同一个本地模型生成答案 + 直接提取 hidden state”的数据集运行方式。

如果显式改成 `--answer-mode extract`，才会退回旧的确定性 evidence extractor 模式。
```

## 8. codeact_tools

CodeAct 工具执行建议按 4 种场景理解，避免把“是否走主图”“是否使用真实模型”“是否真实执行代码”混在一起。数据集位置：

```text
examples/competition_agents/data/codeact_tasks.json
```

### 8.1 独立执行器验证

用途：

```text
只验证 restricted Python CodeAct 执行器本身是否可用。
不强调多智能体主图，不依赖 LLM 规划或总结质量。
真实执行代码；默认执行数据集中的 reference_code。
```

运行 20 条任务：

```powershell
python examples\competition_agents\run_codeact_samples.py `
  --dataset-file examples\competition_agents\data\codeact_tasks.json
```

快速运行前 5 条：

```powershell
python examples\competition_agents\run_codeact_samples.py `
  --dataset-file examples\competition_agents\data\codeact_tasks.json `
  --limit 5
```

### 8.2 主图闭环

用途：

```text
验证多智能体主图中的完整闭环：
Planner -> Retriever -> Reasoner -> Executor -> Summarizer
其中 Executor 会真实执行 restricted_python_codeact。
```

基础命令：

```powershell
python examples\competition_agents\run_demo.py `
  --dataset-type codeact `
  --dataset-file examples\competition_agents\data\codeact_tasks.json `
  --limit 20 `
  --modes protocol
```

通过主图快速运行前 5 条：

```powershell
python examples\competition_agents\run_demo.py `
  --dataset-type codeact `
  --dataset-file examples\competition_agents\data\codeact_tasks.json `
  --limit 5 `
  --modes protocol
```

主图闭环分成两种口径：

`mock` 口径：

```text
验证“主图流程是否跑通”，不验证真实模型效果。
Planner / Reasoner / Summarizer 使用 mock；
Executor 仍会真实执行 restricted_python_codeact。
适合本地开发、自测和排查主图链路。
```

运行示例：

```powershell
$env:LLM_PROVIDER="mock"
$env:LLM_PLANNER_PROVIDER="mock"
$env:LLM_REASONER_PROVIDER="mock"
$env:LLM_SUMMARIZER_PROVIDER="mock"
$env:EMBEDDING_PROVIDER="hash"

python examples\competition_agents\run_demo.py `
  --dataset-type codeact `
  --dataset-file examples\competition_agents\data\codeact_tasks.json `
  --limit 20 `
  --modes protocol
```

`真实 API` 口径：

```text
验证“真实模型 + 多智能体主图 + 真实 CodeAct 执行器”整条链路。
这是最接近比赛展示口径的 CodeAct 主图运行方式。
如果模型接口请求失败，流程会在 Planner/Reasoner 阶段提前报错，
此时还没有进入 Executor 执行代码。
```

运行示例：

```powershell
$env:LLM_PROVIDER="openai_compatible"
$env:LLM_PLANNER_PROVIDER="openai_compatible"
$env:LLM_REASONER_PROVIDER="openai_compatible"
$env:LLM_SUMMARIZER_PROVIDER="openai_compatible"
$env:LLM_API_KEY="你的 key"
$env:LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:LLM_MODEL="你的模型名"
$env:EMBEDDING_PROVIDER="hash"

python examples\competition_agents\run_demo.py `
  --dataset-type codeact `
  --dataset-file examples\competition_agents\data\codeact_tasks.json `
  --limit 20 `
  --modes protocol
```

补充说明：

```text
`--modes protocol` 只决定走协议主图，不负责自动设置模型。
如果没有预先设置 `LLM_PROVIDER` 等环境变量，`run_demo.py` 默认会回落到 mock。
`--experiment-mode codeact_tools` 当前会强制设置为 mock，适合可复现实验，不适合声明为真实 API 主图运行。
```

任务覆盖：

```text
数学计算：平均数、方差、比例、增长率
表格处理：筛选、排序、聚合
字符串处理：词频、关键词、数字提取
日期计算：间隔天数、星期判断
证据链处理：从 evidence list 找最终答案
记忆复用：复用上一轮工具执行结果
```

输出会包含：

```text
execution_result.execution_mode = restricted_python_codeact
execution_result.sandbox_status = ok / error
execution_result.allowed_imports
execution_result.code_source
execution_result.code
execution_result.exit_code
execution_result.elapsed_ms
execution_result.result
execution_result.safety_checks
```

说明口径：

```text
当前实现是 restricted Python CodeAct，不是裸 exec LLM 代码。
代码必须定义 result，不能自行 print；Executor 使用 AST 检查、危险 import/call 黑名单、while 禁止、临时目录子进程、timeout 和输出长度限制执行，并统一输出 JSON。
为了可复现实验，当前 deterministic 模式默认使用 reference_code；后续真实 API 模式可由 Reasoner 生成 execution_request.code。
`run_demo.py --dataset-type codeact --modes protocol` 已接入主图，用于演示四角色闭环。
`run_demo.py --dataset-type codeact` 的 `run_result_json.runs[].task_results[]` 会包含每题的 `execution_result`、`trace`、`memory_hits`、`memory_context` 和 `metrics`。
看到运行输出中的
LLM providers: default=..., planner=..., reasoner=..., summarizer=...
即可判断当前到底是 mock 还是真实 API。
```

## 9. 终端自测模式

mock 只用于确认流程能跑通，不进入页面正式选项。

```powershell
$env:PYTHONPATH="libs/langgraph;examples/competition_agents"
$env:LLM_PROVIDER="mock"
$env:LLM_PLANNER_PROVIDER="mock"
$env:LLM_REASONER_PROVIDER="mock"
$env:LLM_SUMMARIZER_PROVIDER="mock"
$env:EMBEDDING_PROVIDER="hash"

python examples\competition_agents\run_demo.py `
  --dataset-type multihop `
  --dataset-file examples\competition_agents\data\hotpot_multihop_sample.json `
  --modes text,protocol
```

快速看连续任务记忆命中：

```powershell
python examples\competition_agents\run_demo.py `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json `
  --limit 1 `
  --modes protocol `
  --show-memory `
  --show-latent `
  --show-state
```

## 10. 页面收益统计怎么显示

右下角“收益统计”建议永远显示 **当前实验模式相对同数据集 `baseline_text` 的收益**。

也就是说，前端需要先拿到：

```text
baseline = experiment_mode=baseline_text + 当前 dataset 的运行结果
current = 当前 experiment_mode + 当前 dataset 的运行结果
```

然后按下面公式计算。

### 10.1 字符节省

```text
字符节省率 = (baseline.metrics.text_chars - current.metrics.text_chars)
           / baseline.metrics.text_chars
```

展示建议：

```text
字符节省
42%
较 baseline_text
```

如果结果为负数，显示为：

```text
字符增加
12%
较 baseline_text
```

### 10.2 耗时降低

```text
耗时降低率 = (baseline.elapsed_ms - current.elapsed_ms)
           / baseline.elapsed_ms
```

展示建议：

```text
耗时降低
31%
较 baseline_text
```

如果结果为负数，说明当前方式更慢，显示为：

```text
耗时增加
8%
较 baseline_text
```

注意：真实 API 耗时受网络波动影响很大，报告里建议重复跑多次取平均值；页面实时展示可以直接显示本次结果。

### 10.3 memory hit rate

这里建议展示当前实验的绝对命中率，并在小字里显示相对 baseline 的变化。

```text
current_memory_hit_rate = current.memory_hit_rate
delta = current.memory_hit_rate - baseline.memory_hit_rate
```

展示建议：

```text
memory hit rate
0.83
较 baseline_text +0.12
```

如果没有 baseline，就只显示：

```text
memory hit rate
0.83
```

### 10.4 exact match

这里建议展示当前实验的准确率，并在小字里显示相对 baseline 的变化。

```text
current_exact_match = current.exact_match
delta = current.exact_match - baseline.exact_match
```

展示建议：

```text
exact match
0.67
较 baseline_text -0.03
```

如果任务没有 gold，`exact_match` 为 `None`，页面显示：

```text
exact match
N/A
无 gold
```

### 10.5 latent_hidden 的收益统计

`latent_hidden` 不完全等价于 `run_demo.py` 的 API 问答流程，因此右下角建议这样显示：

| 卡片 | 显示内容 |
| --- | --- |
| 字符节省 | 如果存在同 dataset 的 `baseline_text`，按 `text_chars` 公式计算；否则显示 N/A。 |
| 耗时降低 | 如果存在同 dataset 的 `baseline_text`，按 `elapsed_ms` 公式计算；否则显示 N/A。 |
| memory hit rate | 显示 `latent_hidden` 当前命中率。 |
| exact match | 显示 `latent_hidden` 当前 exact_match，并注明 deterministic extractor。 |

同时在 hidden 专属区域额外展示：

```text
hidden_state_refs
hidden_bytes
pooled_similarity
```

不要把 `hidden_bytes` 当作通信文本开销，它表示共享 latent memory 中保存的 hidden tensor 规模。

## 11. 输出字段

本项目当前正式输出契约必须同时满足：

```text
examples/competition_agents/FRONTEND_CONTRACT.md
examples/competition_agents/INTEGRATION.md
```

也就是说，无论是：

```text
run_demo.py
run_hidden_latent_samples.py
run_codeact_samples.py
```

都必须保证输出末尾存在：

```text
=== run_result_json ===
{...json...}
```

并且前端只从这一段解析，不从普通日志文本反向抠字段。

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
state_refs
elapsed_ms
exact_match
```

`payload` 内部结构后续可能继续压缩优化，页面不要强依赖 payload 的具体子字段。

补充稳定字段要求：

```text
1. 顶层必须稳定保留：schema_version / experiment_mode / dataset_type / dataset_file / limit / task_count / runs
2. runs[0] 必须稳定保留：mode / experiment_mode / metrics / task_results
3. task_results[] 必须稳定保留：task_id / question / final_answer / gold / exact_match / messages / metrics / agent_views / retrieval_debug
4. 如果当前链路支持上下文来源切换，应保留：context_source
5. 如果当前链路支持检索调试，应在 retrieval_debug 中保留：retrieval_query / planner_focus / context_hit_ids / context_hits
6. hidden 路径与 protocol 路径都应继续保留：latent_states / state_refs
```

当前已确认符合文档口径的输出脚本：

| 脚本 | 已确认口径 |
| --- | --- |
| `run_demo.py` | 已输出统一 `run_result_json`，并保留 `agent_views` / `retrieval_debug` / `memory_hits` / `memory_context` / `latent_states` / `state_refs`。 |
| `run_hidden_latent_samples.py` | 已输出统一 `run_result_json`，并与正式连续任务口径对齐 `context_source`、`agent_views`、`retrieval_debug`、`latent_states`、`state_refs`。 |
| `run_codeact_samples.py` | 已输出统一 `run_result_json`，并保留 `execution_result` 等 CodeAct 专属字段。 |

## 12. 超图记忆裁剪口径

超图记忆检索可以保留两类输出：

```text
memory_hits：完整检索命中，用于 trace、离线分析、超图调试和完整展示。
memory_context：裁剪后的在线推理上下文，用于 RetrieverAgent 传给 ReasonerAgent。
```

`memory_context` 的目标不是把检索结果再压成向量，而是保留 Reasoner 能正常推理所需的最小可解释信息。当前推荐 schema：

```json
[
  {
    "memory_id": "mem-0001",
    "summary": "一句话说明这条记忆支持什么结论",
    "evidence": ["最多 1-2 条关键证据句"],
    "reasoning_path": [
      {
        "query_node": "artifact:evidence",
        "hyperedge_id": "hyperedge:mem-0001:evidence",
        "memory_id": "mem-0001"
      }
    ],
    "matched_nodes": ["artifact:evidence", "state:evidence", "tag:protocol"],
    "matched_hyperedges": ["hyperedge:mem-0001:evidence"],
    "retrieval_score": 75.5,
    "hypergraph_score": 65.75,
    "embedding_score": 0.59
  }
]
```

裁剪规则建议：

```text
top_k：2 或 3
summary：最多 200-300 字符
evidence：最多 2 条，每条最多 150-200 字符
reasoning_path：最多 3 条
matched_nodes：最多 8-12 个
matched_hyperedges：最多 6-8 个
score：全部保留
```

字段作用：

```text
memory_id：后续 Agent 引用完整记忆，不直接传完整正文。
summary：告诉 Reasoner 这条记忆支持什么结论。
evidence：告诉 Reasoner 这条记忆凭什么可信。
reasoning_path：告诉 Reasoner 和页面为什么命中这条记忆。
matched_nodes / matched_hyperedges：用于超图局部视图高亮命中路径。
retrieval_score / hypergraph_score / embedding_score：用于排序、置信度和收益解释。
```

不要把 `memory_context` 裁成纯向量：

```json
{"memory_id": "mem-0001", "vector": [0.12, 0.33]}
```

Reasoner 无法直接解释向量，还需要额外解码或解释层。

也不要把完整 record、完整 trace、完整 messages 直接塞进 `memory_context`，否则会重新拉高通信开销。完整信息继续放在 `memory_hits` 和 trace 里即可。

当前在线链路是：

```text
HyperGraphMemory.search()
-> RetrieverAgent 得到完整 memory_hits
-> RetrieverAgent 裁剪成 memory_context
-> RetrieverAgent 传 memory_refs + memory_context + evidence_ref
-> ReasonerAgent 使用 memory_context 推理
```

注意：RetrieverAgent 只把本轮 `evidence` 写入 latent state；`memory_context` 不再参与 evidence latent embedding。

## 13. 常用验证命令

Python 语法检查：

```powershell
python -m py_compile examples\competition_agents\run_demo.py
```

超图记忆测试：

```powershell
python -m pytest -p no:cacheprovider examples\competition_agents\tests\test_hypergraph_memory.py
```

前端构建：

```powershell
cd examples\competition_agents\frontend
npm.cmd install
npm.cmd run build
```
