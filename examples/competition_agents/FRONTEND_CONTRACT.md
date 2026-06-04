# 前端字段契约

本文档只回答前端真正需要的三件事：

1. 从结果文件里解析哪一段内容
2. 每个页面模块应该读取哪些稳定字段
3. 如何把字段归一化成可以直接渲染的前端数据

运行命令、实验边界、数据集说明统一看 `RUNBOOK.md`。本文档不重复维护运行手册。

## 1. 唯一解析入口

前端不要从终端日志文本中反向提取字段，不要解析：

- `summary: ...`
- `metrics: ...`
- `Answer: ...`
- 普通调试日志

前端只解析每个输出文件最后的机器可读区块：

```text
=== run_result_json ===
{...json...}
```

适用脚本：

- `run_demo.py`
- `run_hidden_latent_samples.py`
- `run_codeact_samples.py`

推荐解析方式：

```ts
export function parseRunResultText(raw: string) {
  const marker = "=== run_result_json ===";
  const idx = raw.lastIndexOf(marker);
  if (idx < 0) {
    throw new Error("未找到 run_result_json 区块");
  }

  const jsonText = raw.slice(idx + marker.length).trim();
  return JSON.parse(jsonText);
}
```

## 2. 顶层稳定结构

前端优先依赖这些稳定字段：

```text
schema_version
experiment_mode
dataset_type
dataset_file
limit
task_count
runs[]
runs[].metrics
runs[].task_results[]
```

单次运行最常用的读取入口通常是：

```ts
const data = parseRunResultText(fileText);
const run = data.runs[0];
const turns = run.task_results;
```

## 3. 每一轮稳定字段

每个 `task_result` / 每一轮前端优先依赖：

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
memory_hits
memory_context
latent_states
state_refs
execution_result
agent_views
retrieval_debug
```

最常用基础字段：

```ts
type TaskResult = {
  task_id: string;
  group_id?: string | null;
  turn_id?: string | null;
  question?: string;
  final_answer?: string;
  gold?: string;
  exact_match?: boolean | null;
  messages?: unknown[];
  metrics?: Record<string, unknown>;
  memory_hits?: unknown[];
  memory_context?: unknown[];
  latent_states?: Record<string, unknown>;
  state_refs?: Record<string, string>;
  execution_result?: Record<string, unknown>;
  agent_views?: Record<string, unknown>;
  retrieval_debug?: Record<string, unknown>;
};
```

## 4. 页面模块与字段路径

### 4.1 问题与答案模块

读取：

```text
runs[0].task_results[i].task_id
runs[0].task_results[i].group_id
runs[0].task_results[i].turn_id
runs[0].task_results[i].question
runs[0].task_results[i].final_answer
runs[0].task_results[i].gold
runs[0].task_results[i].exact_match
runs[0].task_results[i].metrics
```

### 4.2 Agent 输出模块

前端优先读取：

```text
runs[0].task_results[i].agent_views
```

常见结构：

```text
agent_views.planner.display_text
agent_views.planner.structured
agent_views.planner.state_ref

agent_views.retriever.display_text
agent_views.retriever.structured
agent_views.retriever.state_ref

agent_views.reasoner.display_text
agent_views.reasoner.structured
agent_views.reasoner.state_ref

agent_views.summarizer.display_text
agent_views.summarizer.structured
agent_views.summarizer.state_ref
```

如果 `agent_views` 不存在，再回退到：

```text
runs[0].task_results[i].agent_outputs
```

## 5. 你当前最关心的 3 类展示字段

这一节是给前端直接写抽取代码用的。

### 5.1 每一轮各个智能体的输出内容

优先路径：

```text
runs[0].task_results[i].agent_views
```

前端应展示：

- `planner.display_text`
- `retriever.display_text`
- `reasoner.display_text`
- `summarizer.display_text`

如果要展开结构化内容，再显示：

- `planner.structured`
- `retriever.structured`
- `reasoner.structured`
- `summarizer.structured`

推荐归一化函数：

```ts
export function normalizeAgentPanels(taskResult: any) {
  const views = taskResult?.agent_views;
  if (!views || typeof views !== "object") return [];

  const order = ["planner", "retriever", "reasoner", "summarizer", "executor"];

  return order
    .filter((key) => views[key])
    .map((key) => ({
      key,
      title: key,
      text: views[key]?.display_text ?? "",
      structured: views[key]?.structured ?? null,
      stateRef: views[key]?.state_ref ?? null,
    }));
}
```

### 5.2 检索智能体给超图记忆的检索词

直接读取：

```text
runs[0].task_results[i].retrieval_debug.retrieval_query
```

推荐同时展示：

```text
runs[0].task_results[i].retrieval_debug.planner_focus
runs[0].task_results[i].retrieval_debug.memory_hit_ids
```

推荐归一化函数：

```ts
export function normalizeRetrievalDebug(taskResult: any) {
  const debug = taskResult?.retrieval_debug ?? {};
  return {
    retrievalQuery: debug.retrieval_query ?? "",
    plannerFocus: Array.isArray(debug.planner_focus) ? debug.planner_focus : [],
    memoryHitIds: Array.isArray(debug.memory_hit_ids) ? debug.memory_hit_ids : [],
    memoryHits: Array.isArray(debug.memory_hits) ? debug.memory_hits : [],
  };
}
```

### 5.3 超图记忆的返回内容

分两层展示：

#### A. 完整命中结果

读取：

```text
runs[0].task_results[i].memory_hits
```

常见子字段：

```text
memory_id
summary
evidence
reasoning_path
matched_nodes
matched_hyperedges
retrieval_score
hypergraph_score
embedding_score
```

这部分适合“调试视角”页面。

#### B. 真正传给 Reasoner 的轻量上下文

读取：

```text
runs[0].task_results[i].memory_context
```

这部分适合“在线推理视角”页面。

推荐归一化函数：

```ts
export function normalizeMemoryPanel(taskResult: any) {
  return {
    memoryHits: Array.isArray(taskResult?.memory_hits) ? taskResult.memory_hits : [],
    memoryContext: Array.isArray(taskResult?.memory_context) ? taskResult.memory_context : [],
  };
}
```

## 6. `protocol_api*` 与 `hidden_*` 的差异

### `protocol_api` / `protocol_api_embedding`

这两类结果里，前端重点读取：

```text
agent_views
retrieval_debug
memory_hits
memory_context
messages
execution_result
latent_states
state_refs
```

### `hidden_multihop` / `hidden_continuous`

这两类结果也有 `agent_views` 和 `retrieval_debug`，但它们更偏 hidden-state 机制展示。

前端重点读取：

```text
agent_views
retrieval_debug
latent_states
state_refs
memory_hits
memory_context
```

`hidden_*` 里同样可以直接显示：

- `planner.display_text`
- `retriever.display_text`
- `reasoner.display_text`
- `summarizer.display_text`

## 7. 通信路径模块

如果页面需要画 agent 通信图，读取：

```text
runs[0].task_results[i].messages
```

前端只稳定依赖：

```text
messages[].from
messages[].to
messages[].action
messages[].char_cost
```

不要强依赖 `messages[].payload` 内部结构，因为它后续可能继续压缩。

推荐归一化函数：

```ts
export function normalizeMessages(taskResult: any) {
  const messages = Array.isArray(taskResult?.messages) ? taskResult.messages : [];
  return messages.map((msg: any, index: number) => ({
    id: `${index}-${msg.from}-${msg.to}-${msg.action}`,
    from: msg.from ?? "",
    to: msg.to ?? "",
    action: msg.action ?? "",
    charCost: msg.char_cost ?? null,
  }));
}
```

## 8. 指标模块

运行级指标：

```text
runs[0].metrics
```

单轮指标：

```text
runs[0].task_results[i].metrics
```

常用字段：

```text
message_count
text_chars
latent_transfer_count
latent_bytes
hidden_state_refs
hidden_bytes
memory_hits
memory_hit_rate
elapsed_ms
exact_match
```

## 9. 推荐给前端 AI 的直接实现骨架

下面这段可以直接作为前端 AI 生成代码的输入约束：

```ts
export function buildTurnViewModel(taskResult: any) {
  return {
    id: taskResult.task_id,
    groupId: taskResult.group_id ?? null,
    turnId: taskResult.turn_id ?? null,
    question: taskResult.question ?? "",
    finalAnswer: taskResult.final_answer ?? "",
    gold: taskResult.gold ?? "",
    exactMatch: taskResult.exact_match ?? null,
    metrics: taskResult.metrics ?? {},
    agentPanels: normalizeAgentPanels(taskResult),
    retrieval: normalizeRetrievalDebug(taskResult),
    memory: normalizeMemoryPanel(taskResult),
    messages: normalizeMessages(taskResult),
    latentStates: taskResult.latent_states ?? {},
    stateRefs: taskResult.state_refs ?? {},
    executionResult: taskResult.execution_result ?? null,
  };
}

export function buildRunViewModel(parsed: any) {
  const run = parsed?.runs?.[0] ?? {};
  const turns = Array.isArray(run.task_results) ? run.task_results : [];

  return {
    schemaVersion: parsed?.schema_version ?? 1,
    experimentMode: parsed?.experiment_mode ?? "",
    datasetType: parsed?.dataset_type ?? "",
    datasetFile: parsed?.dataset_file ?? "",
    limit: parsed?.limit ?? null,
    taskCount: parsed?.task_count ?? turns.length,
    metrics: run?.metrics ?? {},
    turns: turns.map(buildTurnViewModel),
  };
}
```

## 10. 前端实现原则

1. 只从 `run_result_json` 解析
2. 只依赖文档中列出的稳定字段
3. `agent_views` 优先于 `agent_outputs`
4. `memory_context` 是在线推理视角，`memory_hits` 是完整调试视角
5. `retrieval_debug.retrieval_query` 是检索词的唯一正式入口
6. 不要从自然语言日志块里做字符串切片解析

## 11. 你当前这批结果里，前端最该显示什么

针对你当前已经跑出的结果文件，页面至少要能稳定显示：

- 每轮 `question / final_answer / gold / exact_match`
- 每轮四个 Agent 的 `display_text`
- `retrieval_debug.retrieval_query`
- `memory_hits`
- `memory_context`
- `messages`
- `metrics.elapsed_ms / text_chars / memory_hits / latent_bytes / hidden_bytes`

这样前端就已经能完整呈现：

- 智能体逐轮输出
- 检索词
- 超图记忆返回
- 实际送入推理的记忆
- 通信路径
- 每轮指标
