# 多智能体比赛原型

这个目录是比赛用的多智能体协作原型，当前重点验证三件事：

- 用结构化协议减少 Agent 之间的长文本通信。
- 用 `state_ref` / `hidden_state_ref` 传递非文本中间状态。
- 用共享记忆和超图检索支持跨任务复用。

当前代码都放在 `examples/competition_agents` 下，不修改 LangGraph 核心代码。

## 文档导航

这套文档现在按“入口 / 运行 / 前端 / 协议”分工：

| 文档 | 作用 | 
| --- | --- | 
| `README.md` | 项目简介和文档导航 | 
| `RUNBOOK.md` | 唯一运行手册：怎么配、怎么跑、怎么看结果 | 
| `FRONTEND_CONTRACT.md` | 前端字段契约：读哪些字段、怎么解析 `run_result_json`、页面怎么展示 | 
| `PROTOCOL_V0_2.md` | 协议结构、`state_ref`、`memory_ref`、`cognitive_gap` 定义 | 
| `HYPERGRAPH_CONTRACT.md` | 超图记忆模块接口和对接口径 | 

如果只想快速上手：

1. 先看 `RUNBOOK.md`
2. 需要接前端时再看 `FRONTEND_CONTRACT.md`
3. 需要看协议字段来源时再看 `PROTOCOL_V0_2.md`

## Agent 分工

- `PlannerAgent`：拆解任务，生成执行计划。
- `RetrieverAgent`：检索共享记忆和证据，并写入 latent state。当前检索采用“超图结构匹配 + embedding 相似度融合排序”。
- `ReasonerAgent`：基于计划、证据和记忆命中结果生成候选答案。
- `SummarizerAgent`：生成最终答案，并把结果沉淀为共享记忆。

主流程是：

```text
PlannerAgent -> RetrieverAgent -> ReasonerAgent -> SummarizerAgent
```

如果本地 LangGraph 依赖不可用，代码会退回顺序运行器，保证 demo 可以先跑通。

## 当前结论

当前原型已经完成：

- 多 Agent 原型流程
- `baseline_text` / `protocol_api` / `protocol_api_embedding` / `latent_hidden` / `codeact_tools` 几类实验入口
- 文本开销、latent 传递、记忆命中、耗时、`exact_match` 等指标统计
- 连续任务上的共享记忆复用实验
- Hidden latent 引用传递实验
- 受限 Python CodeAct 执行实验

具体命令、数据集组合、收益统计和输出字段，不再放在这里重复描述，统一看 `RUNBOOK.md` 和 `FRONTEND_CONTRACT.md`。
