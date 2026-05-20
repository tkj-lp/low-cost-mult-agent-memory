# 多智能体原型接口说明

这份文档说明当前多智能体原型的代码结构，以及网页可视化同学、超图记忆同学应该如何和智能体部分对接。

## 当前范围

代码位于 `examples/competition_agents`，没有修改 LangGraph 原本核心代码。

当前包含 4 个 Agent：

- `PlannerAgent`：拆解任务，并向检索 Agent 发起请求。
- `RetrieverAgent`：检索共享记忆，生成证据，并写入一个非文本状态。
- `ExecutorAgent`：模拟 CodeAct 或工具执行步骤。
- `SummarizerAgent`：生成最终总结，并把结果写入共享记忆。

当前支持两种运行模式：

- `text`：纯文本协作模式，用较长自然语言消息作为 baseline。
- `protocol`：结构化协议模式，用结构化字段和状态引用减少通信开销。

## 运行入口

在仓库根目录运行：

```bash
PYTHONPATH=libs/langgraph:examples/competition_agents python examples/competition_agents/run_demo.py
```

Windows PowerShell 运行：

```powershell
$env:PYTHONPATH="libs/langgraph;examples/competition_agents"
python examples/competition_agents/run_demo.py
```

默认使用 `LLM_PROVIDER=mock`，不需要 API key，所有队友都可以直接运行。

后续如果要接 OpenAI-compatible API，建议先只让 `SummarizerAgent` 调 API，`PlannerAgent` 继续用 mock，这样验证更快：

```bash
export LLM_PROVIDER=mock
export LLM_PLANNER_PROVIDER=mock
export LLM_SUMMARIZER_PROVIDER=openai_compatible
export LLM_API_KEY=your_api_key
export LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export LLM_MODEL=qwen-plus-2025-07-28
```

Windows PowerShell：

```powershell
$env:LLM_PROVIDER="mock"
$env:LLM_PLANNER_PROVIDER="mock"
$env:LLM_SUMMARIZER_PROVIDER="openai_compatible"
$env:LLM_API_KEY="your_api_key"
$env:LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:LLM_MODEL="qwen-plus-2025-07-28"
```

`run_demo.py` 会连续执行 10 个相关任务，分别跑 `text` 模式和 `protocol` 模式，并输出对比指标。

## 模型接口

模型适配层位于 `examples/competition_agents/competition_agents/llm`。

当前包含：

- `MockLLMClient`：默认使用，不依赖网络和 API key。
- `OpenAICompatibleClient`：后续可接 OpenAI、通义千问兼容接口、DeepSeek 兼容接口等。

Agent 不直接绑定某个模型，而是通过 `get_llm_client(role)` 获取统一接口。可以用 `LLM_PLANNER_PROVIDER` 和 `LLM_SUMMARIZER_PROVIDER` 分别控制不同 Agent 的模型来源。当前已经接入模型接口的 Agent：

- `PlannerAgent`：通过 `client.plan(task)` 生成任务步骤。
- `SummarizerAgent`：通过 `client.summarize(...)` 生成最终总结。

暂时保持本地逻辑的 Agent：

- `RetrieverAgent`：后续优先和超图记忆、真实 embedding 对接。
- `ExecutorAgent`：后续再接 CodeAct 沙箱或真实工具执行。

## 共享状态接口

共享状态定义在 `examples/competition_agents/competition_agents/state.py`。

重要字段如下：

- `task`：当前任务文本。
- `mode`：运行模式，取值为 `text` 或 `protocol`。
- `plan`：规划 Agent 输出的任务步骤。
- `evidence`：检索 Agent 找到的证据。
- `execution_result`：工具执行或 CodeAct 执行结果。
- `final_answer`：总结 Agent 输出的最终答案。
- `messages`：当前任务中的 Agent 间通信消息。
- `latent_states`：非文本中间状态，目前是确定性的 embedding 向量。
- `memory`：跨任务累积的共享记忆。
- `memory_hits`：当前任务命中的历史记忆。
- `metrics`：通信开销和状态传递统计指标。

## 给网页同学的消息接口

每条消息保存在 `state["messages"]` 中，格式如下：

```json
{
  "from": "planner",
  "to": "retriever",
  "action": "retrieve",
  "payload": "... 或结构化协议包 ...",
  "char_cost": 305
}
```

建议网页可视化这样使用：

- 节点来源：`from`
- 节点目标：`to`
- 边标签：`action`
- 边权重或角标：`char_cost`
- 详情面板：`payload`

当前通信路径是：

```text
planner -> retriever -> executor -> summarizer
```

## 结构化协议包接口

结构化协议包由 `examples/competition_agents/competition_agents/protocol.py` 创建。

协议包格式如下：

```json
{
  "msg_id": "msg-xxxxxxxx",
  "from": "retriever",
  "to": "executor",
  "action": "execute",
  "params": {
    "tool": "codeact_stats",
    "evidence_count": 2
  },
  "result": {
    "evidence_refs": ["state-0001"],
    "memory_hits": ["mem-0001"]
  },
  "state_refs": ["state-0001"],
  "capability_required": ["tool_execution"],
  "created_at": "2026-05-19T..."
}
```

网页端可以把 `params`、`result`、`state_refs`、`capability_required` 展示成结构化字段，而不是当成长文本显示。

## 指标接口

每个任务返回的 `state["metrics"]` 格式如下：

```json
{
  "message_count": 3,
  "text_chars": 934,
  "latent_transfer_count": 1,
  "latent_bytes": 64
}
```

其他可展示指标：

- 当前任务记忆命中数：`len(state["memory_hits"])`
- 当前任务耗时：在 `run_demo.py` 中统计为 `elapsed_ms`

最终对比会输出：

- `text` 模式总字符数
- `protocol` 模式总字符数
- 平均字符节省率
- 总耗时提升
- 记忆命中率

## 给超图记忆同学的记忆接口

记忆记录由 `examples/competition_agents/competition_agents/memory.py` 创建。

当前记忆格式如下：

```json
{
  "memory_id": "mem-0001",
  "source_agent": "summarizer",
  "created_at": "2026-05-19T...",
  "task_topic": "Design a LangGraph multi-agent protocol with shared memory",
  "summary": "Task ... was completed ...",
  "tags": ["langgraph", "multi-agent", "protocol", "protocol"],
  "embedding": [0.1, -0.2, "..."]
}
```

建议超图映射方式：

- 任务节点：`task_topic`
- Agent 节点：`source_agent`
- 记忆节点：`memory_id`
- 标签节点：`tags` 中的每个值
- 证据或状态节点：来自 `latent_states`
- 超边：`(task, source_agent, memory, tags, evidence_refs)`

推荐替换点：

- 替换 `add_memory()`：把记忆写入超图记忆存储。
- 替换 `search_memory()`：按关键词、标签、语义相似度或超图扩展进行检索。
- 尽量保持当前记忆字段兼容，这样 Agent 和网页代码不用大改。

## 非文本状态接口

非文本状态由 `state.py` 中的 `put_latent_state()` 创建。

当前格式如下：

```json
{
  "state_id": "state-0001",
  "type": "embedding",
  "producer": "retriever",
  "topic": "current task",
  "shape": [16],
  "vector": [0.1, -0.2, "..."]
}
```

当前 `simple_embedding()` 是一个确定性的轻量占位实现，不是真实大模型 embedding。后续可以替换为真实 embedding API 或本地 embedding 模型。

建议保持以下字段稳定：

- `state_id`
- `type`
- `producer`
- `topic`
- `shape`
- `vector`

这样网页模块和超图记忆模块可以继续使用同一套接口。

## 分工建议

智能体同学负责：

- 维护 Agent 流程和结构化协议字段。
- 保持 `messages`、`metrics`、`latent_states`、`memory` 这些字段稳定。
- 提供 10 轮连续任务的实验输出。

超图记忆同学负责：

- 在 `add_memory()` 和 `search_memory()` 后面接入超图存储与检索。
- 保持 Agent 当前使用的记忆字段不变。
- 需要新增超图专用字段时，建议作为可选字段添加。

网页可视化同学负责：

- 读取 `messages` 展示 Agent 通信链路。
- 读取 `metrics` 展示通信开销和状态传递指标。
- 读取 `memory_hits` 和 `memory` 展示共享记忆复用。
- 读取 `latent_states` 展示非文本状态传递。

## 后续环境工作

openEuler 和完整 LangGraph 依赖环境可以在接口稳定后再整理。

当前原型包含 fallback runner，所以即使本机 LangGraph 依赖暂时不完整，队友也可以先基于这些接口开发超图记忆和网页可视化模块。
