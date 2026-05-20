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
