# 多智能体比赛原型

这个目录演示如何把 4 个比赛 Agent 接入 LangGraph 风格的多智能体运行时：

- `PlannerAgent`：拆解任务，并选择下一步 Agent。
- `RetrieverAgent`：检索资料和可复用共享记忆。
- `ExecutorAgent`：执行轻量 CodeAct 风格工具步骤。
- `SummarizerAgent`：生成最终总结，并把结果写入共享记忆。

本原型代码全部放在 `examples/competition_agents` 下，不修改 LangGraph 原本核心代码。LangGraph 用作有状态多 Agent 运行时，结构化协议包、状态引用、记忆记录和评测指标都放在本目录中。

给队友看的接口说明见 `INTEGRATION.md`。文件树说明见 `FILE_TREE.md`。

## 运行方式

在仓库根目录运行：

```bash
PYTHONPATH=libs/langgraph:examples/competition_agents python examples/competition_agents/run_demo.py
```

Windows PowerShell：

```powershell
$env:PYTHONPATH="libs/langgraph;examples/competition_agents"
python examples/competition_agents/run_demo.py
```

默认使用 `LLM_PROVIDER=mock`，不需要 API key，所有队友都能直接运行。

## 接真实模型 API

建议先只让 `SummarizerAgent` 调 API，`PlannerAgent` 继续用 mock。这样验证更快，也能减少外部 API 延迟对实验指标的影响。

Linux/macOS：

```bash
export LLM_PROVIDER=mock
export LLM_PLANNER_PROVIDER=mock
export LLM_SUMMARIZER_PROVIDER=openai_compatible
export LLM_API_KEY=your_api_key
export LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export LLM_MODEL=qwen-plus-2025-07-28
PYTHONPATH=libs/langgraph:examples/competition_agents python examples/competition_agents/run_demo.py
```

Windows PowerShell：

```powershell
$env:LLM_PROVIDER="mock"
$env:LLM_PLANNER_PROVIDER="mock"
$env:LLM_SUMMARIZER_PROVIDER="openai_compatible"
$env:LLM_API_KEY="your_api_key"
$env:LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:LLM_MODEL="qwen-plus-2025-07-28"
$env:PYTHONPATH="libs/langgraph;examples/competition_agents"
python examples/competition_agents/run_demo.py
```

注意：不要把 API key 写进代码、文档或 GitHub。只通过环境变量配置。

## 系统流程

当前图流程如下：

```text
START
  -> planner
  -> retriever
  -> executor
  -> summarizer
  -> END
```

原型支持两种模式：

- `text`：纯文本协作模式。Agent 之间传较长自然语言消息，用作 baseline。
- `protocol`：结构化协议协作模式。Agent 之间传结构化协议包、状态引用和记忆引用，用来降低通信开销。

`run_demo.py` 会连续运行 10 个相关任务，并输出：

- Agent 间消息次数
- 文本通信字符开销
- 非文本状态传递次数
- 非文本状态数据规模
- 单任务耗时
- 共享记忆命中情况
- `text` 模式和 `protocol` 模式的总对比

## 当前原型范围

默认模式故意不依赖外部 LLM API，便于在干净的 Linux 或 Windows 环境中运行。`LLM_PROVIDER=mock` 时行为是确定性的：

- 规划使用 `MockLLMClient`。
- 检索使用本地共享记忆关键词匹配。
- CodeAct 执行由 Python 统计步骤模拟。
- 总结使用 `MockLLMClient`。
- 非文本状态传递使用 `simple_embedding`，这是一个确定性的 16 维 hash 向量。

`simple_embedding` 是真实 embedding 的轻量占位实现。它用于先演示“非文本状态引用传递”的机制，后续可以替换成真实 embedding API、本地 embedding 模型或超图记忆模块提供的向量表示。

## 后续接入步骤

建议顺序：

1. 保持当前 mock 模式，作为可复现实验 baseline。
2. 使用 `LLM_SUMMARIZER_PROVIDER=openai_compatible` 验证真实模型 API。
3. 由超图记忆同学替换 `memory.py` 中的 `add_memory()` 和 `search_memory()`。
4. 将 `simple_embedding` 替换为真实 embedding API 或本地 embedding 模型。
5. 修复完整 LangGraph 依赖环境，去掉 fallback runner。
6. 在 openEuler 24.03-LTS-SP3 上验证同样命令可以运行。
## Embedding 与 latent collaboration

当前示例支持三种 embedding provider：

```env
EMBEDDING_PROVIDER=hash
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_PROVIDER=local
```

第一版推荐使用 Qwen / OpenAI-compatible embedding API：

```env
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_API_KEY=replace_with_your_api_key
```

如果没有单独设置 `EMBEDDING_API_KEY`，代码会复用 `LLM_API_KEY`。`protocol` 模式下，`PlannerAgent` 会把 plan 写入 `state["latent_states"]` 并向 `ReasonerAgent` 传递 `plan_ref`；`ReasonerAgent` 会把 answer / reasoning 写入 `state["latent_states"]` 并向 `SummarizerAgent` 传递 `answer_ref`。这就是当前的 embedding-based latent collaboration。

运行 HotpotQA-style 对比实验：

```powershell
.\run_hotpot.ps1
```

输出会展示：

```text
latent states:
  state-0001: producer=planner, provider=openai_compatible, model=text-embedding-v4, shape=[1024], checksum=...
  state-0002: producer=reasoner, provider=openai_compatible, model=text-embedding-v4, shape=[1024], checksum=...

metrics: messages=2, text_chars=..., latent_transfers=2, latent_bytes=8192, memory_hits=0, elapsed_ms=...
```

默认会同时跑 `text` 和 `protocol` 两组，并输出：

```text
=== protocol vs text ===
text_chars_delta=...
latent_transfer_delta=...
latent_bytes_delta=...
elapsed_ms_delta=...
```

第二版可以切换到本地轻量英文 embedding：

```powershell
pip install sentence-transformers
$env:EMBEDDING_PROVIDER="local"
$env:LOCAL_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
.\run_hotpot.ps1
```

`sentence-transformers/all-MiniLM-L6-v2` 会输出 384 维向量，适合作为本地离线 embedding 对照。第一次运行会下载模型，后续会复用本地缓存。也可以设置 `LOCAL_EMBEDDING_CACHE_DIR` 指定模型缓存目录。

当前实现没有修改 LangGraph 框架内核，而是在 LangGraph-style `StateGraph` 的共享状态里扩展了 `latent_states`、`state_refs` 和 `metrics`。因此它属于应用层的非文本协作传输：Agent 节点通过结构化协议传 `state_ref`，再从共享 state 中读取 embedding latent state。
# 当前实现状态

以本节为准。当前 HotpotQA-style 主实验流程是：

```text
PlannerAgent -> ReasonerAgent -> SummarizerAgent
```

`RetrieverAgent` 和 `ExecutorAgent` 仍保留在代码与能力注册中，但当前 HotpotQA-style 主流程暂时没有接回它们。后续接入超图记忆后，目标流程会扩展为：

```text
PlannerAgent -> RetrieverAgent -> ReasonerAgent -> SummarizerAgent
```

当前已经实现 Qwen / OpenAI-compatible LLM 调用、OpenAI-compatible embedding API、本地 `sentence-transformers/all-MiniLM-L6-v2` embedding provider，以及 text/protocol 两种通信模式对比。`protocol` 模式会通过 `state_ref` 传递 `plan_ref` 和 `answer_ref`，并在 `state["latent_states"]` 中保存 embedding latent state。

OSC / CKM 当前仍是轻量 `cognitive_gaps` trace，用于记录 `known`、`unknown`、`needed`、`gap_type` 和 `confidence`。它还没有成为自动路由或自动决策模块。
