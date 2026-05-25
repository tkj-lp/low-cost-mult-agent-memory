# 运行方式说明

这份文档只说明怎么跑。页面同学可以把运行入口理解成两个选择框：

```text
运行方式：快速测试 / API / API+embedding / latent
数据集：10 轮比赛主流程 / 5 个多跳问答样例
```

其中 `latent` 指的是直接读取本地 LLM 最后一层 hidden state，目前是独立实验，不和两个数据集完全合并。

## 一、页面选项设计

### 选择框 1：运行方式

| 运行方式 | 含义 | 是否调用 API | 是否使用本地 embedding | 是否直接读取 LLM 最后一层 hidden state |
| --- | --- | --- | --- | --- |
| `quick` 快速测试 | 全部 Agent 用 mock，最快最稳定 | 否 | 默认不强制 | 否 |
| `api` 真实 API | Planner / Reasoner / Summarizer 用 API 生成文字 | 是 | 否 | 否 |
| `api_embedding` API+embedding | API 生成文字，本地 embedding 把中间文本压成向量 | 是 | 是 | 否 |
| `latent_hidden` hidden latent | 本地 Qwen 直接提取 `outputs.hidden_states[-1]` | 否 | 否 | 是 |

### 选择框 2：数据集 / 实验任务

| 数据集选项 | 实际脚本 | 内容 | 主要输出 |
| --- | --- | --- | --- |
| `competition_10` | `run_demo.py` | 10 轮比赛主流程 | messages、text_chars、latent_transfers、latent_bytes、memory_hits、elapsed_ms |
| `hotpot_5` | `run_hotpot_samples.py` | 5 个多跳问答样例 | final_answer、gold、exact_match、messages、text_chars、latent_transfers、memory_hits |
| `hidden_demo` | `run_latent_hidden_demo.py` | 本地 LLM hidden state 机制验证 | hidden_state_refs、hidden shape、hidden bytes、pooled_similarity |

### 推荐组合

| 运行方式 | 数据集 | 是否推荐 | 说明 |
| --- | --- | --- | --- |
| `quick` | `competition_10` | 推荐 | 最稳定，适合页面调试和主流程指标展示 |
| `quick` | `hotpot_5` | 推荐 | 不花 API，检查多跳问答链路 |
| `api` | `competition_10` | 可用 | 真实模型跑 10 轮，可能较慢或 API 超时 |
| `api` | `hotpot_5` | 推荐 | 真实模型问答效果展示 |
| `api_embedding` | `competition_10` | 推荐 | API 生成文字 + 本地 embedding 状态传递 + 主流程指标 |
| `api_embedding` | `hotpot_5` | 推荐 | API 生成文字 + 本地 embedding + exact_match |
| `latent_hidden` | `hidden_demo` | 推荐 | 证明直接读取本地 LLM 最后一层 hidden state |
| `latent_hidden` | `competition_10` | 暂不支持 | hidden state 还没有接进 10 轮主流程 |
| `latent_hidden` | `hotpot_5` | 暂不支持 | hidden state 还没有接进多跳问答主流程 |

## 二、运行方式对应的前置配置

### 方式 1：`quick` 快速测试

用途：不调用 API，不下载大模型，先验证多 Agent 流程和指标统计。

```powershell
cd D:\qq\low-cost-memory-integrate
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH="libs/langgraph;examples/competition_agents"

$env:LLM_PROVIDER="mock"
$env:LLM_PLANNER_PROVIDER="mock"
$env:LLM_REASONER_PROVIDER="mock"
$env:LLM_SUMMARIZER_PROVIDER="mock"
```

这种配置下：

```text
Planner / Reasoner / Summarizer：mock 生成文字
Retriever：本地检索
Agent 通信：由运行脚本里的 text/protocol 模式决定
```

### 方式 2：`api` 真实 API 生成文字

用途：让 Planner / Reasoner / Summarizer 使用真实大模型 API 生成文字。

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
```

这种配置下：

```text
Planner / Reasoner / Summarizer：API 生成文字
Retriever：本地检索
API 输出：是文字，不是 hidden state
Agent 通信：由运行脚本里的 text/protocol 模式决定
```

注意：API 一般不会返回模型内部最后一层 hidden state。

### 方式 3：`api_embedding` 真实 API + 本地 embedding 向量

用途：API 负责生成文字，本地 embedding 负责把中间文本状态压成 384 维向量。

先执行 `api` 的配置，再额外执行：

```powershell
pip install sentence-transformers
$env:EMBEDDING_PROVIDER="local"
$env:LOCAL_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

这种配置下：

```text
API：生成计划、推理、总结文字
本地 embedding：文本内容 -> 384 维 embedding 向量 -> state_ref
```

它不是 LLM 最后一层 hidden state。

### 方式 4：`local_embedding` 仅本地 embedding 向量

用途：把文本压成 384 维语义向量，再通过 `state_ref` 传递。

第一次运行前安装：

```powershell
pip install sentence-transformers
```

配置：

```powershell
$env:EMBEDDING_PROVIDER="local"
$env:LOCAL_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

这种配置下：

```text
文本内容 -> sentence-transformers/all-MiniLM-L6-v2 -> 384 维 embedding 向量 -> state_ref
```

它不是 LLM 最后一层 hidden state。

这个方式可以和 `quick` 搭配，也可以和 `api` 搭配。页面如果只保留一个简化选项，建议使用上面的 `api_embedding`。

### 方式 5：`latent_hidden` 本地 LLM 最后一层 hidden state

用途：直接读取本地 Qwen 前向推理产生的最后一层 hidden state，再通过 `state_ref` 传递。

第一次运行前安装：

```powershell
pip install torch transformers
```

配置：

```powershell
$env:LOCAL_HIDDEN_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
```

这种配置下：

```text
文本输入 -> 本地 Qwen 前向推理 -> outputs.hidden_states[-1] -> state_ref
```

这是“直接传模型内部最后一层向量引用”的实验。

## 三、数据集对应的运行脚本

### 数据集 1：`competition_10`

实际脚本：`run_demo.py`

运行命令：

```powershell
python examples\competition_agents\run_demo.py
```

它跑什么：

```text
10 个连续相关任务
text 模式跑一遍
protocol 模式跑一遍
最后输出两种模式的通信开销、latent 传递次数、记忆命中率、耗时对比
```

适合做什么：

```text
比赛主 demo
实验报告里的通信效率和记忆复用指标
网页同学读取 messages / metrics 做可视化
```

它默认不跑 hidden-state demo。它主要使用 embedding-based latent state 或 hash latent state。

### 数据集 2：`hotpot_5`

实际脚本：`run_hotpot_samples.py`

运行命令：

```powershell
python examples\competition_agents\run_hotpot_samples.py
```

它跑什么：

```text
多跳问答样例
text 模式跑一遍
protocol 模式跑一遍
输出 final_answer、gold、exact_match
```

适合做什么：

```text
证明系统不仅能跑指标，还能完成多跳推理任务
检查 Planner / Retriever / Reasoner / Summarizer 输出是否合理
检查本地 embedding latent state 是否生成
```

如果同时使用配置 B 和配置 C：

```text
API 负责生成文字答案
本地 embedding 负责把中间文本压成向量
Agent 消息通过 text 或 protocol 模式传递
```

### 数据集 3：`hidden_demo`

实际脚本：`run_latent_hidden_demo.py`

运行命令：

```powershell
python examples\competition_agents\run_latent_hidden_demo.py
```

它跑什么：

```text
只跑一个 hidden-state 传递小实验
Planner 输入文本到本地 Qwen
提取 outputs.hidden_states[-1]
把完整 hidden tensor 放进进程内共享 latent memory
消息里只传 hidden-0001 这样的 state_ref
Reasoner 再生成自己的 hidden state
输出 hidden state 的 shape、checksum、字节规模和相似度
```

适合做什么：

```text
证明“不是文本压成 embedding”，而是直接读取本地 LLM 最后一层 hidden state
作为论文方向的机制验证
```

它目前不跑 10 轮任务，也不跑超图记忆检索。

## 四、可直接复制的常用组合

### 组合 1：`quick + competition_10`

先执行 `quick` 配置，然后运行：

```powershell
python examples\competition_agents\run_demo.py
```

含义：

```text
mock 生成文字
跑 10 轮任务
比较 text 和 protocol
```

### 组合 2：`api + competition_10`

先执行 `api` 配置，然后运行：

```powershell
python examples\competition_agents\run_demo.py
```

含义：

```text
API 生成文字
跑 10 轮任务
比较 text 和 protocol
不代表传递 API 的 hidden state
```

### 组合 3：`api_embedding + competition_10`

先执行 `api_embedding` 配置，然后运行：

```powershell
python examples\competition_agents\run_demo.py
```

含义：

```text
API 生成文字
sentence-transformers 把中间文本压成 384 维向量
跑 10 轮比赛主流程
统计通信开销、latent 传递、记忆命中和耗时
```

### 组合 4：`api_embedding + hotpot_5`

先执行 `api_embedding` 配置，然后运行：

```powershell
python examples\competition_agents\run_hotpot_samples.py
```

含义：

```text
API 生成文字
sentence-transformers 把中间文本压成 384 维向量
通过 state_ref 传递 embedding 向量引用
检查 exact_match
```

### 组合 5：`latent_hidden + hidden_demo`

先执行 `latent_hidden` 配置，然后运行：

```powershell
python examples\competition_agents\run_latent_hidden_demo.py
```

含义：

```text
本地 Qwen 前向推理
直接读取 outputs.hidden_states[-1]
通过 state_ref 传递 hidden state 引用
不调用 API
不等同于 sentence-transformers embedding
```

## 五、三种传输到底传的是什么

| 名称 | 产生方式 | 传输内容 | 是否是 LLM 最后一层 hidden state |
| --- | --- | --- | --- |
| text 模式 | Agent 生成自然语言 | 长文本 | 否 |
| protocol 模式 | Agent 生成结构化字段 | JSON-like 协议、`state_ref`、`memory_ref` | 不一定 |
| 本地 embedding | 文本进入 sentence-transformers | 384 维 embedding 的 `state_ref` | 否 |
| 本地 hidden state | 文本进入本地 Qwen，读取 `outputs.hidden_states[-1]` | hidden tensor 的 `state_ref` | 是 |

一句话总结：

```text
API 负责生成文字；embedding 负责把文本压成向量；hidden-state demo 负责直接读取本地 LLM 内部最后一层向量。
```
