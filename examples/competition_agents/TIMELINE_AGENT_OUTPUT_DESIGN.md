# 运行进度时间线 Agent 输出展示方案

本文档整理“点击运行进度时间线中每一个圆圈，都能展示出这个对话的智能体输出”的前端实现方案。

字段契约以 `FRONTEND_CONTRACT.md` 为准，本文只描述页面布局、交互、数据归一化和实现注意事项。

## 1. 功能目标

运行进度时间线本质是一个二维矩阵：

```text
task_result[i] x agentKey
```

每个圆圈代表某一轮任务中某个 Agent 的一次输出入口。用户点击圆圈后，页面应展示：

- 当前轮问题
- 当前 Agent 的主要输出
- 当前 Agent 的结构化输出
- 当前 Agent 使用或产生的 `state_ref`
- 当前轮检索信息与记忆上下文
- 当前轮与该 Agent 相关的通信消息
- 当前轮执行结果、最终答案、gold 和 exact_match

页面必须只读取 `run_result_json` 中的稳定字段，不从终端自然语言日志里切字符串。

## 2. 页面布局

控制台采用三行布局。

第一行：

```text
运行进度时间线面板
├── 左侧：task x agent 圆圈矩阵
└── 右侧：当前选中圆圈的 Agent 输出详情
```

第二行：

```text
记忆检索解释 | 超图局部视图
```

第三行：

```text
耗时折线图 | 字符数折线图 | 收益统计
```

时间线详情不使用抽屉或弹窗。右侧详情固定在时间线面板内部，内容区域内部滚动，避免长文本撑开整个控制台。

窄屏时，时间线面板内部从左右分栏切换为上下结构。

## 3. 时间线圆圈语义

每行对应一轮任务：

```text
runs[0].task_results[i]
```

每列对应一个 Agent：

```text
planner
retriever
reasoner
executor
summarizer
```

Agent 列优先从当前运行结果的 `messages` 中动态提取。如果没有结果，则使用当前运行方式的默认 Agent 顺序。

圆圈状态：

| 状态 | 含义 |
| --- | --- |
| `waiting` | 该轮尚未运行，不可点击。 |
| `running` | 当前正在运行，可展示运行中占位。 |
| `done` | 该轮已有结果，可点击。 |
| `selected` | 当前右侧详情正在展示的圆圈。 |
| `missing` | 该轮已完成，但该 Agent 没有可展示的 `agent_views` 输出。 |

## 4. 选中状态

前端维护一个选中单元格：

```ts
type TimelineCell = {
  taskIndex: number;
  agentKey: string;
};
```

点击圆圈时写入：

```ts
selectedTimelineCell = { taskIndex, agentKey };
```

当切换数据集、运行方式、运行结果或 Agent 列变化时，需要保证选中状态仍然有效：

- 如果当前选中任务还存在，并且 Agent 列还存在，保持当前选中。
- 如果当前选中失效，默认选中第一轮第一个 Agent。
- 如果当前没有运行结果且未运行，清空选中状态。

## 5. 右侧详情字段来源

右侧详情的基础对象：

```ts
const task = activeRun.task_results[taskIndex];
const agentView = task.agent_views?.[agentKey] ?? fallbackFromAgentOutputs(task, agentKey);
```

### 5.1 顶部标题

展示：

```text
task N / agentKey
状态：已完成 / 运行中 / 待运行
```

来源：

```text
selectedTimelineCell.taskIndex
selectedTimelineCell.agentKey
runs[0].task_results[i]
```

### 5.2 问题

展示当前轮问题。

来源：

```text
runs[0].task_results[i].question
```

如果为空，显示“当前轮次暂无问题文本”。

### 5.3 Agent 输出

优先来源：

```text
runs[0].task_results[i].agent_views[agentKey].display_text
```

常见路径：

```text
agent_views.planner.display_text
agent_views.retriever.display_text
agent_views.reasoner.display_text
agent_views.executor.display_text
agent_views.summarizer.display_text
```

如果没有 `agent_views`，回退到 `agent_outputs`。

兼容两种旧形态：

```ts
task.agent_outputs?.[agentKey]
```

以及：

```ts
[
  { agent: "PlannerAgent", raw_output: "..." },
  { agent: "RetrieverAgent", retrieval_query: "..." }
]
```

旧式列表需要将 `PlannerAgent` 归一化为 `planner`，并把 `raw_output`、`answer_candidate`、`reasoning_summary`、`retrieval_query` 等字段转为 `display_text`。

### 5.4 结构化内容

来源：

```text
runs[0].task_results[i].agent_views[agentKey].structured
```

展示前格式化为 JSON：

```ts
JSON.stringify(value, null, 2)
```

常见内容：

- planner: plan、raw_output
- retriever: retrieval_query、planner_focus、memory_hit_ids、memory_context
- reasoner: answer_candidate、reasoning_summary、execution_request
- executor: execution_result
- summarizer: final_answer、latest_memory_summary

### 5.5 state_ref

优先来源：

```text
runs[0].task_results[i].agent_views[agentKey].state_ref
```

回退来源：

```text
runs[0].task_results[i].state_refs[agentKey]
```

如果存在，显示在详情顶部元信息区域。

### 5.6 检索与记忆

检索词来源：

```text
runs[0].task_results[i].retrieval_debug.retrieval_query
```

辅助字段：

```text
runs[0].task_results[i].retrieval_debug.planner_focus
runs[0].task_results[i].retrieval_debug.memory_hit_ids
runs[0].task_results[i].retrieval_debug.context_hit_ids
```

真正传给 Reasoner 的轻量上下文来源：

```text
runs[0].task_results[i].memory_context
```

完整调试命中来源：

```text
runs[0].task_results[i].memory_hits
```

右侧详情只展示紧凑内容。完整展示可联动下方“记忆检索解释”面板。

### 5.7 通信消息

来源：

```text
runs[0].task_results[i].messages
```

稳定字段：

```text
messages[].from
messages[].to
messages[].action
messages[].char_cost
```

详情区只展示与当前 Agent 相关的消息：

```ts
message.from === agentKey || message.to === agentKey
```

注意：`messages[].payload` 不作为前端稳定展示依赖。

### 5.8 执行结果

来源：

```text
runs[0].task_results[i].execution_result
```

主要用于 `codeact_tools` 或存在 Executor 的运行方式。

常见字段：

```text
execution_mode
sandbox_status
allowed_imports
code_source
code
exit_code
elapsed_ms
result
safety_checks
```

### 5.9 答案与评测

来源：

```text
runs[0].task_results[i].final_answer
runs[0].task_results[i].gold
runs[0].task_results[i].exact_match
```

展示：

```text
final: ...
gold: ...
exact_match: true / false / N/A
```

## 6. 下方面板联动

点击圆圈后，不只更新右侧详情，也更新第二排面板。

### 6.1 记忆检索解释

读取当前选中任务：

```ts
const task = activeRun.task_results[selectedTimelineCell.taskIndex];
```

展示：

```text
retrieval_debug.retrieval_query
retrieval_debug.memory_hit_ids
memory_hits.length
memory_context.length
retrieval_debug.planner_focus
```

如果没有选中任务，展示引导文案。

### 6.2 超图局部视图

超图当前可以先使用静态 SVG 骨架，但节点标签应尽量从当前任务派生：

```text
task label: task N
memory nodes: retrieval_debug.memory_hit_ids
state nodes: task_result.state_refs
outcome node: task_result.final_answer
```

后续如果需要更真实的超图高亮，可继续读取：

```text
memory_hits[].matched_nodes
memory_hits[].matched_hyperedges
memory_hits[].reasoning_path
```

## 7. 后端摘要要求

`frontend_server.py` 不能只返回指标摘要，否则前端无法展示圆圈详情。

每个 `task_result` 至少需要透传：

```text
task_id
group_id
turn_id
dataset_type
question
final_answer
gold
exact_match
messages
metrics
agent_views
agent_outputs
retrieval_debug
memory_hits
memory_context
latent_states
state_refs
execution_result
```

`messages` 摘要中必须保留：

```text
from
to
action
char_cost
```

如果后端仍返回旧结构，例如只包含：

```text
messages
metrics
exact_match
execution_result
```

那么前端只能显示“暂无问题文本”和“暂无 agent_views 输出”。这说明正在运行的 API 服务没有加载新的字段透传代码，需要重启服务。

## 8. 运行服务重启注意事项

前端页面连接的是：

```text
http://127.0.0.1:8765/api/run
```

如果修改了 `frontend_server.py`，必须重启占用 8765 端口的 Python 服务。

验证接口是否返回完整字段：

```text
task_results[0].question 应存在
task_results[0].agent_views 应存在
task_results[0].agent_views.planner.display_text 应存在
```

如果这些字段不存在，优先检查是否还是旧服务进程。

## 9. 前端归一化建议

推荐抽象出这些函数：

```ts
getTaskResult(taskIndex)
getAgentView(task, agentKey)
hasAgentOutput(taskIndex, agentKey)
normalizeMessagesForAgent(task, agentKey)
safeStringify(value)
```

其中 `getAgentView` 应遵循：

```text
1. task.agent_views[agentKey]
2. task.agent_outputs[agentKey]
3. task.agent_outputs 列表中 agent 名称归一化匹配
```

Agent 名称归一化规则：

```text
PlannerAgent -> planner
RetrieverAgent -> retriever
ReasonerAgent -> reasoner
ExecutorAgent -> executor
SummarizerAgent -> summarizer
```

## 10. 验证清单

功能完成后至少验证：

1. 运行 `baseline_text + hotpot_continuous` 后，点击 planner 圆圈能看到问题和 planner 输出。
2. 点击 retriever 圆圈能看到检索词、planner_focus 或 memory_context。
3. 点击 reasoner 圆圈能看到推理输出或结构化内容。
4. 点击 summarizer 圆圈能看到 final_answer。
5. `codeact_tools` 中点击 executor 圆圈能看到 `execution_result`。
6. 第二排记忆检索解释随选中任务更新。
7. 缺少 `agent_views` 但存在旧式 `agent_outputs` 时，前端仍能展示兜底输出。
8. 后端 `/api/run` 返回的 `task_results[0]` 包含 `question` 和 `agent_views`。

构建检查：

```powershell
cd examples\competition_agents\frontend
npm run build
```

后端语法检查：

```powershell
python -m py_compile examples\competition_agents\frontend_server.py
```
