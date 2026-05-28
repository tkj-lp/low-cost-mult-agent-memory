# 多智能体比赛原型

这个目录是比赛用的多智能体协作原型，目标是验证三件事：

- 用结构化协议减少 Agent 之间的长文本通信。
- 用本地或 API embedding 生成非文本 latent state，并通过 `state_ref` 在 Agent 间传递。
- 用共享记忆和超图记忆检索支持跨任务复用。

当前代码都放在 `examples/competition_agents` 下，不修改 LangGraph 原本核心代码。

如果只是想快速知道每种方式怎么配置、每个脚本具体跑什么，请先看 `RUNBOOK.md`。

## 团队文档入口

不同同学优先阅读不同文档：

| 角色 | 需要阅读 |
| --- | --- |
| 智能体同学 | `RUNBOOK.md`、`AGENT_WORK.md` |
| 超图记忆同学 | `HYPERGRAPH_CONTRACT.md` |
| 页面可视化同学 | `FRONTEND_CONTRACT.md` |
| 所有人 | `README.md`、`RUNBOOK.md` |

## Agent 分工

- `PlannerAgent`：拆解任务，生成执行计划。
- `RetrieverAgent`：检索共享记忆和证据，并写入 latent embedding state。
- `ReasonerAgent`：基于计划、证据、记忆命中结果生成候选答案。
- `SummarizerAgent`：生成最终答案，并把结果沉淀为共享记忆。

主流程目标是：

```text
PlannerAgent -> RetrieverAgent -> ReasonerAgent -> SummarizerAgent
```

如果本地 LangGraph 依赖不可用，代码会退回到顺序运行器，保证比赛 demo 可以先跑通。

## 快速运行

Windows PowerShell：

```powershell
cd D:\qq\low-cost-memory-integrate
$env:PYTHONPATH="libs/langgraph;examples/competition_agents"
python examples\competition_agents\run_demo.py
```

Linux / openEuler：

```bash
cd /path/to/low-cost-memory-integrate
export PYTHONPATH=libs/langgraph:examples/competition_agents
python examples/competition_agents/run_demo.py
```

`run_demo.py` 会连续运行 10 个相关任务，并输出：

- Agent 消息路径。
- 文本通信字符开销。
- latent state 传递次数和字节规模。
- 共享记忆命中次数。
- text 模式和 protocol 模式对比结果。

## 运行方式
看RUNBOOK.md

## 给超图记忆同学

看 `examples\competition_agents\HYPERGRAPH_CONTRACT.md`。这是给超图记忆同学的唯一最新版文档，旧的 alignment/update/todo 文档已经删除，避免多个版本互相冲突。

超图模块应尽量只对外暴露两个稳定入口：

```python
build_hypergraph_view(record: dict) -> dict
search_hypergraph_memory(memory: list[dict], query: str, top_k: int) -> list[dict]
```

## 当前完成情况

已经完成：

- 多 Agent 原型流程。
- text / protocol 两种通信模式。
- 消息次数、字符开销、latent 传递次数、latent 字节数、记忆命中等指标统计。
- OpenAI-compatible LLM API 适配。
- 本地 embedding provider。
- `state_ref` 形式的非文本 latent state 传递。

还需要继续完善：

- 超图记忆和 embedding 相似度融合检索。
- 更清晰的 protocol 模式压缩策略，提升字符节省率。
- 更真实的 CodeAct 沙箱执行。
- openEuler 24.03-LTS-SP3 上的部署验证。
- 网页控制台读取 `messages`、`metrics`、`memory_hits`、`latent_states` 并可视化。

## 给前端同学

看examples\competition_agents\FRONTEND_CONTRACT.md和examples\competition_agents\RUNBOOK.md
