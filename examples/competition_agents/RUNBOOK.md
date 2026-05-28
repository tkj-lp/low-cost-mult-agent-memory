# 运行手册

本文档回答两个问题：

1. 不同实验模式应该怎么跑。
2. 页面上的收益统计应该怎么和基线结果对齐。

当前正式实验入口采用两层选择：

```text
experiment_mode：baseline_text / protocol_api / protocol_api_embedding / latent_hidden
dataset：competition_10 / hotpot_multihop / hotpot_continuous / hidden_hotpot_5 / hidden_continuous
```

`mock` 只用于终端自测流程是否跑通，不作为页面正式运行方式。

## 1. 正式实验模式

| experiment_mode | 含义 | 入口 |
| --- | --- | --- |
| `baseline_text` | 纯文本协作基线。真实 API 生成文字，Agent 间用自然语言长文本通信。 | `run_demo.py` |
| `protocol_api` | 结构化协议协作。真实 API 生成文字，Agent 间用结构化协议包通信，embedding 使用 hash 占位。 | `run_demo.py` |
| `protocol_api_embedding` | 协议 + API embedding。真实 API 生成文字，并用 OpenAI-compatible embedding 生成 `state_ref`。 | `run_demo.py` |
| `latent_hidden` | 本地 LLM hidden state 引用传递。消息里只传 `hidden_state_ref`。 | `run_hidden_latent_samples.py` |

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
| `competition_10` | `--dataset-type builtin` | 内置 10 轮比赛主流程。 |
| `hotpot_multihop` | `--dataset-type multihop --dataset-file examples\competition_agents\data\hotpot_multihop_sample.json` | HotpotQA 抽样多跳问答。 |
| `hotpot_continuous` | `--dataset-type continuous --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json` | 连续关联任务，用于验证跨轮记忆复用。 |
| `hidden_hotpot_5` | `run_hidden_latent_samples.py` 默认 | 5 条 hidden-state reference transfer 样例。 |
| `hidden_continuous` | `run_hidden_latent_samples.py --dataset-type continuous --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json` | 连续任务上的 hidden-state reference transfer + memory reuse。 |

## 3. 可用组合

| experiment_mode | 可用 dataset |
| --- | --- |
| `baseline_text` | `competition_10`, `hotpot_multihop`, `hotpot_continuous` |
| `protocol_api` | `competition_10`, `hotpot_multihop`, `hotpot_continuous` |
| `protocol_api_embedding` | `competition_10`, `hotpot_continuous` |
| `latent_hidden` | `hidden_hotpot_5`, `hidden_continuous` |

不建议组合：

```text
latent_hidden + competition_10
latent_hidden + hotpot_multihop
baseline_text + hidden_*
protocol_api + hidden_*
protocol_api_embedding + hidden_*
```

原因：`latent_hidden` 当前使用独立脚本，验证的是本地 LLM hidden state 引用传递，不接入 `run_demo.py` 的 10 轮主流程。

## 4. baseline_text

10 轮主流程：

```powershell
python examples\competition_agents\run_demo.py `
  --experiment-mode baseline_text `
  --dataset-type builtin
```

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

`baseline_text` 是页面收益统计的统一参照项。前端需要保存同一 dataset 下的 baseline 结果，用于和其他实验模式计算收益。

## 5. protocol_api

10 轮主流程：

```powershell
python examples\competition_agents\run_demo.py `
  --experiment-mode protocol_api `
  --dataset-type builtin
```

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

## 6. protocol_api_embedding

10 轮主流程：

```powershell
python examples\competition_agents\run_demo.py `
  --experiment-mode protocol_api_embedding `
  --dataset-type builtin
```

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

该模式重点展示：

```text
latent_transfer_count
latent_bytes
latent_states.*.embedding_provider
latent_states.*.embedding_model
state_refs
```

## 7. latent_hidden

单条机制验证：

```powershell
python examples\competition_agents\run_latent_hidden_demo.py
```

5 条 hidden Hotpot 样例：

```powershell
python examples\competition_agents\run_hidden_latent_samples.py
```

连续任务：

```powershell
python examples\competition_agents\run_hidden_latent_samples.py `
  --dataset-type continuous `
  --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json `
  --limit 1
```

`latent_hidden` 输出中的 `exact_match` 使用确定性 evidence extractor，用于隔离 hidden-state 传递机制的开销；完整 LLM 问答质量由 `baseline_text` / `protocol_api` / `protocol_api_embedding` 路径验证。

## 8. 终端自测模式

mock 只用于确认流程能跑通，不进入页面正式选项。

```powershell
$env:PYTHONPATH="libs/langgraph;examples/competition_agents"
$env:LLM_PROVIDER="mock"
$env:LLM_PLANNER_PROVIDER="mock"
$env:LLM_REASONER_PROVIDER="mock"
$env:LLM_SUMMARIZER_PROVIDER="mock"
$env:EMBEDDING_PROVIDER="hash"

python examples\competition_agents\run_demo.py `
  --dataset-type builtin `
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

## 9. 页面收益统计怎么显示

右下角“收益统计”建议永远显示 **当前实验模式相对同数据集 `baseline_text` 的收益**。

也就是说，前端需要先拿到：

```text
baseline = experiment_mode=baseline_text + 当前 dataset 的运行结果
current = 当前 experiment_mode + 当前 dataset 的运行结果
```

然后按下面公式计算。

### 9.1 字符节省

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

### 9.2 耗时降低

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

### 9.3 memory hit rate

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

### 9.4 exact match

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

### 9.5 latent_hidden 的收益统计

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

## 10. 输出字段

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

## 11. 常用验证命令

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
