# 文件树说明

本文档说明 `examples/competition_agents` 目录下各文件的作用。

```text
examples/competition_agents
├── README.md
├── INTEGRATION.md
├── FILE_TREE.md
├── run_demo.py
├── agents
│   ├── __init__.py
│   ├── planner
│   │   ├── __init__.py
│   │   └── agent.py
│   ├── retriever
│   │   ├── __init__.py
│   │   └── agent.py
│   ├── executor
│   │   ├── __init__.py
│   │   └── agent.py
│   └── summarizer
│       ├── __init__.py
│       └── agent.py
└── competition_agents
    ├── __init__.py
    ├── graph.py
    ├── memory.py
    ├── protocol.py
    ├── state.py
    └── llm
        ├── __init__.py
        ├── base.py
        ├── client.py
        ├── mock.py
        └── openai_compatible.py
```

## 顶层文件

- `README.md`：项目原型的基本说明，包括运行命令、两种模式、当前原型范围和后续接入步骤。
- `INTEGRATION.md`：给网页同学和超图记忆同学看的接口文档，说明 `messages`、`metrics`、`memory`、`latent_states` 等字段怎么对接。
- `FILE_TREE.md`：当前文件树说明文档。
- `run_demo.py`：实验入口脚本，连续运行 10 个相关任务，分别测试 `text` 模式和 `protocol` 模式，并输出通信开销、记忆命中、耗时等指标。

## agents 目录

- `agents/__init__.py`：Agent 包入口说明。

### planner

- `agents/planner/__init__.py`：导出 `planner_agent`，供 `graph.py` 注册节点。
- `agents/planner/agent.py`：实现 `PlannerAgent`。负责拆解任务、生成计划，并把检索请求发送给 `RetrieverAgent`。

### retriever

- `agents/retriever/__init__.py`：导出 `retriever_agent`，供 `graph.py` 注册节点。
- `agents/retriever/agent.py`：实现 `RetrieverAgent`。负责检索共享记忆、构造证据、生成非文本 `latent state`，并把结果发送给 `ExecutorAgent`。

### executor

- `agents/executor/__init__.py`：导出 `executor_agent`，供 `graph.py` 注册节点。
- `agents/executor/agent.py`：实现 `ExecutorAgent`。当前模拟 CodeAct 或工具执行，统计消息数、记忆命中数和非文本状态数量，后续可替换成真实沙箱执行。

### summarizer

- `agents/summarizer/__init__.py`：导出 `summarizer_agent`，供 `graph.py` 注册节点。
- `agents/summarizer/agent.py`：实现 `SummarizerAgent`。负责生成最终答案，并把任务结果写入共享记忆。

## competition_agents 目录

- `competition_agents/__init__.py`：比赛原型核心包入口说明。
- `competition_agents/graph.py`：构建多 Agent 工作流。优先使用 LangGraph `StateGraph`，如果本地依赖不完整，则使用顺序 fallback runner。
- `competition_agents/memory.py`：共享记忆模块。当前用 Python list 保存记忆，并用简单关键词和标签匹配检索；后续超图记忆同学主要替换这里的 `add_memory()` 和 `search_memory()`。
- `competition_agents/protocol.py`：结构化通信协议模块。负责生成包含 `from`、`to`、`action`、`params`、`result`、`state_refs`、`capability_required` 的协议包。
- `competition_agents/state.py`：共享状态和指标定义。包含 `CompetitionState`、`Metrics`、初始状态创建、轻量 embedding、latent state 创建和消息统计逻辑。

## competition_agents/llm 目录

- `competition_agents/llm/__init__.py`：导出统一模型选择入口 `get_llm_client`。
- `competition_agents/llm/base.py`：定义最小模型接口 `LLMClient`，包括 `generate()`、`plan()`、`summarize()`。
- `competition_agents/llm/client.py`：根据环境变量选择模型客户端。支持全局 `LLM_PROVIDER`，也支持 `LLM_PLANNER_PROVIDER`、`LLM_SUMMARIZER_PROVIDER` 按 Agent 单独配置。
- `competition_agents/llm/mock.py`：本地 mock 模型，不需要 API key，用于可复现实验和队友开发。
- `competition_agents/llm/openai_compatible.py`：OpenAI-compatible API 客户端，可用于阿里云 DashScope 兼容模式、OpenAI 或其他兼容接口。

## 不需要提交的运行产物

运行 Python 后可能出现 `__pycache__` 目录和 `.pyc` 文件，这些是解释器缓存，不是源码文件，不需要写进文档，也不建议提交到 GitHub。
