# 运行手册

本文档回答两个问题：

1. 不同实验模式应该怎么跑。
2. 页面上的收益统计应该怎么和基线结果对齐。

当前正式实验入口采用两层选择：

```text
experiment_mode：baseline_text / protocol_api / protocol_api_embedding / latent_hidden / codeact_tools
dataset: hotpot_multihop / hotpot_continuous / hidden_hotpot_5 / hidden_continuous / codeact_20
```

`mock` 只用于终端自测流程是否跑通，不作为页面正式运行方式。

## 1. 正式实验模式

| experiment_mode          | 含义                                                                                                                       | 入口                           |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| `baseline_text`          | 纯文本协作基线。真实 API 生成文字，Agent 间用自然语言长文本通信。                                                          | `run_demo.py`                  |
| `protocol_api`           | 结构化协议协作。真实 API 生成文字，Agent 间用结构化协议包通信，embedding 使用 hash 占位。                                  | `run_demo.py`                  |
| `protocol_api_embedding` | 协议 + API embedding。真实 API 生成文字，并用 OpenAI-compatible embedding 生成 `state_ref`。                               | `run_demo.py`                  |
| `latent_hidden`          | 本地 LLM hidden state 引用传递。消息里只传 `hidden_state_ref`。                                                            | `run_hidden_latent_samples.py` |
| `codeact_tools`          | 受限 Python CodeAct 执行。使用 `reference_code` 或后续 LLM 生成代码，经 AST 安全检查和临时目录子进程执行后返回结构化结果。 | `run_codeact_samples.py`       |

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

`.env.local` 是本机私有配置文件，推荐放真实 key 并保持不提交；
`.env.example` 是示例配置文件。前端 HTTP 服务默认优先读取 `.env.local`，
如果该文件不存在，会回退读取 `.env.example`，方便本地联调。

如果运行 `protocol_api_embedding`，还会使用：

```text
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_MODEL=text-embedding-v4
```

如果没有单独设置 `EMBEDDING_API_KEY`，embedding 会复用 `LLM_API_KEY`。

## 2. 数据集

| dataset             | run_demo.py 参数                                                                                                                      | 含义                                                                                                               |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `hotpot_multihop`   | `--dataset-type multihop --dataset-file examples\competition_agents\data\hotpot_multihop_sample.json`                                 | HotpotQA 抽样多跳问答。                                                                                            |
| `hotpot_continuous` | `--dataset-type continuous --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json`                              | 从 `hotpot_dev_distractor_v1.json` 固定抽样构造的连续关联任务，当前为 2 个 chain，每个 chain 10 turn，共 20 turn。 |
| `hidden_hotpot_5`   | `run_hidden_latent_samples.py` 默认                                                                                                   | 5 条 hidden-state reference transfer 样例。                                                                        |
| `hidden_continuous` | `run_hidden_latent_samples.py --dataset-type continuous --dataset-file examples\competition_agents\data\hotpot_continuous_tasks.json` | 同一连续任务数据集上的 hidden-state reference transfer + memory reuse。                                            |
| `codeact_20`        | `run_codeact_samples.py --dataset-file examples\competition_agents\data\codeact_tasks.json`                                           | 自制 20 条安全工具执行任务，覆盖统计、表格、字符串、日期、证据链处理和记忆复用。                                   |

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

## 3. 可用组合

| experiment_mode          | 可用 dataset                           |
| ------------------------ | -------------------------------------- |
| `baseline_text`          | `hotpot_multihop`, `hotpot_continuous` |
| `protocol_api`           | `hotpot_multihop`, `hotpot_continuous` |
| `protocol_api_embedding` | `hotpot_multihop`, `hotpot_continuous` |
| `latent_hidden`          | `hidden_hotpot_5`, `hidden_continuous` |
| `codeact_tools`          | `codeact_20`                           |

不建议组合：

```text
latent_hidden + hotpot_multihop
baseline_text + hidden_*
protocol_api + hidden_*
protocol_api_embedding + hidden_*
codeact_tools + non-codeact datasets
```

原因：`latent_hidden` 当前使用独立脚本，验证的是本地 LLM hidden state 引用传递，不接入 `run_demo.py` 的正式 HotpotQA 主流程。

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

这里 `--limit 1` 表示运行 1 个连续任务组；当前示例数据文件中的该任务组包含 10 个 turn。

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

这里 `--limit 1` 会在 hidden-state reference transfer 实验中运行 `chain-001` 的 10 个 turn。

`latent_hidden` 输出中的 `exact_match` 使用确定性 evidence extractor，用于隔离 hidden-state 传递机制的开销；完整 LLM 问答质量由 `baseline_text` / `protocol_api` / `protocol_api_embedding` 路径验证。

## 8. codeact_tools

CodeAct 工具执行当前有两层验证：独立入口 `run_codeact_samples.py` 用于 20 题可复现实验；主图中的 `ExecutorAgent` 也已复用同一套受限 Python CodeAct 沙箱，用于验证 `Planner -> Retriever -> Reasoner -> Executor -> Summarizer` 四角色闭环。数据集位置：

```text
examples/competition_agents/data/codeact_tasks.json
```

运行 20 条任务：

```powershell
python examples\competition_agents\run_codeact_samples.py `
  --dataset-file examples\competition_agents\data\codeact_tasks.json
```

通过多智能体主图运行 20 条任务：

```powershell
python examples\competition_agents\run_demo.py `
  --dataset-type codeact `
  --dataset-file examples\competition_agents\data\codeact_tasks.json `
  --limit 20 `
  --modes protocol
```

快速运行前 5 条：

```powershell
python examples\competition_agents\run_codeact_samples.py `
  --dataset-file examples\competition_agents\data\codeact_tasks.json `
  --limit 5
```

通过主图快速运行前 5 条：

```powershell
python examples\competition_agents\run_demo.py `
  --dataset-type codeact `
  --dataset-file examples\competition_agents\data\codeact_tasks.json `
  --limit 5 `
  --modes protocol
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
为了可复现实验，当前 deterministic 模式使用 reference_code；后续真实 API 模式可由 Reasoner 生成 execution_request.code。
`run_demo.py --dataset-type codeact` 已接入主图，用于演示四角色闭环。
`run_demo.py --dataset-type codeact` 的 `run_result_json.runs[].task_results[]` 会包含每题的 `execution_result`、`trace`、`memory_hits`、`memory_context` 和 `metrics`。
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

| 卡片            | 显示内容                                                                        |
| --------------- | ------------------------------------------------------------------------------- |
| 字符节省        | 如果存在同 dataset 的 `baseline_text`，按 `text_chars` 公式计算；否则显示 N/A。 |
| 耗时降低        | 如果存在同 dataset 的 `baseline_text`，按 `elapsed_ms` 公式计算；否则显示 N/A。 |
| memory hit rate | 显示 `latent_hidden` 当前命中率。                                               |
| exact match     | 显示 `latent_hidden` 当前 exact_match，并注明 deterministic extractor。         |

同时在 hidden 专属区域额外展示：

```text
hidden_state_refs
hidden_bytes
pooled_similarity
```

不要把 `hidden_bytes` 当作通信文本开销，它表示共享 latent memory 中保存的 hidden tensor 规模。

## 11. 输出字段

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
{ "memory_id": "mem-0001", "vector": [0.12, 0.33] }
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

## 14. 前端运行 HTTP API

页面不能直接执行 Python 脚本，因此本目录提供了一个零第三方依赖的本地
HTTP 服务，用于把页面请求转换成正式实验命令，并返回 `run_result_json`。

页面点击“运行”后的完整链路如下：

```text
前端按钮
-> POST http://127.0.0.1:8765/api/run
-> frontend_server.py 校验 experiment_mode / dataset / limit
-> frontend_server.py 读取 .env.local；如果不存在则回退 .env.example
-> frontend_server.py 组装 PYTHONPATH、LLM_*、EMBEDDING_* 等环境变量
-> frontend_server.py 根据实验模式生成 run_demo.py / run_hidden_latent_samples.py / run_codeact_samples.py 命令
-> subprocess.run(...) 启动实验脚本并捕获 stdout
-> 实验脚本运行 PlannerAgent / RetrieverAgent / ReasonerAgent / SummarizerAgent
-> 实验脚本打印 === run_result_json === 和后一行 JSON
-> frontend_server.py 从 stdout 中解析 run_result_json
-> 前端读取 result.runs[0]，更新时间线、图表、收益统计和任务明细
```

关键入口：

- 前端点击处理：`frontend/src/App.vue` 中的 `toggleRunning()`
- 后端请求处理：`frontend_server.py` 中的 `do_POST()` 和 `run_experiment()`
- 命令生成：`frontend_server.py` 中的 `_build_command()`
- 环境变量生成：`frontend_server.py` 中的 `_build_env()`
- 主实验脚本：`run_demo.py` 中的 `main()` 和 `run_mode()`
- JSON 输出：`run_demo.py` 中的 `print_run_result_json()`

启动服务：

```powershell
python examples\competition_agents\frontend_server.py --host 127.0.0.1 --port 8765
```

如果希望前端点击“运行”后，也在启动服务的终端看到完整实验输出和
`run_result_json`，启动时加上 `--echo-run-output`：

```powershell
python examples\competition_agents\frontend_server.py --host 127.0.0.1 --port 8765 --echo-run-output
```

终端中会出现：

```text
=== run_result_json ===
{...}
```

也可以不经过前端，直接运行实验脚本查看 `run_result_json`：

```powershell
python examples\competition_agents\run_demo.py `
  --experiment-mode baseline_text `
  --env-file examples\competition_agents\.env.example `
  --dataset-type multihop `
  --dataset-file examples\competition_agents\data\hotpot_multihop_sample.json `
  --limit 1
```

健康检查：

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/health" -Method Get
```

查询页面可用选项：

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/options" -Method Get
```

正式运行请求：

```powershell
$body = @{
  experiment_mode = "protocol_api_embedding"
  dataset = "hotpot_continuous"
  limit = 1
  timeout_seconds = 600
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8765/api/run" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

返回格式：

```json
{
  "ok": true,
  "command": ["python", "examples/competition_agents/run_demo.py", "..."],
  "result": {
    "schema_version": 1,
    "experiment_mode": "protocol_api_embedding",
    "dataset_type": "continuous",
    "runs": [
      {
        "metrics": {},
        "task_results": []
      }
    ]
  },
  "stdout_tail": "..."
}
```

如果 `.env.local` 缺少 `LLM_API_KEY` 或 API 调用失败，接口会返回：

```json
{
  "ok": false,
  "error": "Experiment process exited with code 1.",
  "stdout_tail": "..."
}
```

仅用于本地联调的 mock 成功路径：

```powershell
$body = @{
  experiment_mode = "protocol_api"
  dataset = "competition_10"
  limit = 1
  developer_mode = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8765/api/run" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

`developer_mode=true` 只用于确认 HTTP 链路和页面解析能跑通，会强制使用 mock
LLM 和 hash embedding；正式页面默认不要传这个字段。
