# 集成约束说明

本文档写给三类协作者：

- 前端同学
- 智能体链路同学
- 记忆/超图同学

目标只有两个：

1. 说明当前给前端的结果输出，是否已经足够做出目标页面
2. 规定后续后端改动必须遵守哪些稳定字段和规则，避免把前端解析打崩

## 1. 当前结论

当前前端同学如果已经拿到：

- `FRONTEND_CONTRACT.md`
- 真实输出结果文件
- “智能体队列做动态渲染”的说明

那么**已经足够实现你现在想要的页面效果**，包括：

- 每一轮的问题、答案、gold、exact_match
- 每一轮各个智能体的输出内容
- 检索智能体给记忆系统的检索词
- 超图/记忆系统的返回内容
- 通信链路
- latent / hidden state 元数据
- CodeAct 执行结果

前提是前端遵守两条实现原则：

1. 只解析 `=== run_result_json ===` 后面的 JSON
2. 按动态 agent 渲染，不要写死固定智能体名单

## 2. 前端动态渲染原则

前端不要写死：

```ts
showPlanner()
showRetriever()
showReasoner()
showSummarizer()
```

而要改成：

- 从结果里动态发现有哪些 agent
- 有谁就渲染谁
- 没有谁就不显示谁

优先读取入口：

1. `task_result.agent_views`
2. `task_result.messages[].from / to`
3. `task_result.agent_outputs`

### 推荐顺序

前端可以维护一个可扩展顺序表：

```ts
const AGENT_ORDER = [
  "planner",
  "retriever",
  "reasoner",
  "executor",
  "experience_extractor",
  "memory_graph",
  "summarizer",
];
```

规则：

- 已知 agent 按这个顺序排
- 未知 agent 放到最后
- 不要求后端必须返回所有 agent 的空字段

这意味着当前最稳的前后端约定是：

> `agent_views` 采用按需输出；前端根据实际存在的 key 动态渲染。

## 3. 后续后端改动必须遵守的稳定字段

后续你怎么改 agent 数量、执行链路、记忆系统实现都可以，但下面这些字段**必须尽量保持稳定**。

### 3.1 顶层入口

结果文件最后必须继续保留：

```text
=== run_result_json ===
{...json...}
```

前端只依赖这一段 JSON，不依赖普通日志文本。

### 3.2 顶层字段

以下字段必须继续保留：

```text
schema_version
experiment_mode
dataset_type
dataset_file
limit
task_count
runs[]
```

### 3.3 每轮任务字段

以下字段必须继续保留：

```text
task_id
group_id
turn_id
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

可以新增字段，但尽量不要删除这些字段。

## 4. `agent_views` 的约束

后续新增 agent 时，前端最依赖的是 `agent_views`。

### 规则 1：按需输出

当前约定不是“所有 agent 都输出空字段”，而是：

- 本轮用到哪个 agent，就输出哪个 agent
- 没用到就不输出

也就是：

```json
{
  "planner": {...},
  "retriever": {...},
  "reasoner": {...},
  "summarizer": {...}
}
```

如果本轮没有 `executor`，那就可以没有 `executor`。

### 规则 2：一旦输出，结构尽量统一

每个出现的 agent，都尽量保持下面这个结构：

```json
{
  "display_text": "...",
  "structured": {...},
  "state_ref": "..."
}
```

说明：

- `display_text`：给前端直接展示的主要文本
- `structured`：给前端展开调试内容的结构化对象
- `state_ref`：如果该 agent 产出了状态引用，就填；没有可以省略或为 `null`

### 规则 3：不要把页面展示依赖建立在 `agent_outputs` 上

`agent_outputs` 可以继续保留作兼容或调试用途，但前端主入口应始终是：

```text
task_result.agent_views
```

## 5. `messages` 的约束

前端通信图和消息链路依赖 `messages`。

后续必须继续保留：

```json
{
  "from": "...",
  "to": "...",
  "action": "...",
  "char_cost": 123
}
```

说明：

- `payload` 可以继续变化、压缩、精简
- 但 `from / to / action / char_cost` 尽量稳定

前端只按实际消息边画图，不假设固定链路一定是：

```text
planner -> retriever -> reasoner -> summarizer
```

因为后续可能变成：

```text
planner -> retriever -> reasoner -> executor -> summarizer
planner -> retriever -> experience_extractor -> reasoner -> executor -> summarizer
```

## 6. 特殊能力字段的约束

前端除了动态 agent 区，还需要几个独立能力面板。

因此后端后续应继续保持“特殊能力走独立字段”的规则。

### 检索

继续保留：

```text
retrieval_debug
```

前端当前主要依赖：

```text
retrieval_debug.retrieval_query
retrieval_debug.planner_focus
retrieval_debug.memory_hit_ids
```

### 记忆

继续保留：

```text
memory_hits
memory_context
```

含义约定：

- `memory_hits`：完整命中的记忆，适合调试视角
- `memory_context`：真正传给推理器的轻量记忆，适合在线推理视角

### 执行

继续保留：

```text
execution_result
```

即使某轮没有执行，也不要把这个字段彻底改成别的名字。

### latent / hidden state

继续保留：

```text
latent_states
state_refs
```

### 未来新增

如果后续加入新的专项模块，推荐新开独立字段，不要强塞进已有字段。

例如：

- 经验抽取：`experience_debug`
- 记忆图：`memory_graph_debug`
- 记忆精炼：`memory_refine_debug`

## 7. 你后续改代码时要遵守的规则

这部分是最关键的“以后别把前端搞崩”的约束。

### 可以改的

你可以改：

- agent 数量
- agent 顺序
- 是否加入 `executor`
- 是否加入 `experience_extractor`
- 是否加入 `memory_graph`
- `payload` 内部结构
- 内部推理逻辑
- memory 检索实现
- embedding / hidden state 实现

### 尽量不要改的

你尽量不要改：

- `run_result_json` 作为唯一正式解析入口
- `task_results[]` 这一层结构
- `agent_views` 这个主展示入口名称
- `messages` 这个通信入口名称
- `retrieval_debug` 这个检索入口名称
- `execution_result` 这个执行入口名称
- `memory_hits` / `memory_context` 这两个记忆入口名称
- `latent_states` / `state_refs` 这两个状态入口名称

### 如果必须改字段名

如果以后确实必须改字段名，建议遵守：

1. 先保留旧字段一个兼容期
2. 新旧字段同时输出
3. 在 `FRONTEND_CONTRACT.md` 明确标记弃用时间

不要今天直接把前端依赖字段删掉。

## 8. 给前端同学的最终承诺

如果后端遵守本文档，前端可以长期基于下面 4 类结构稳定开发：

```text
messages
agent_views
retrieval_debug
execution_result
```

以及这 4 组辅助结构：

```text
memory_hits
memory_context
latent_states
state_refs
```

前端不需要假设：

- 一定有 `reasoner`
- 一定只有 4 个 agent
- 一定有固定链路
- 一定每轮都有 `executor`

只需要：

- 动态发现 agent
- 动态渲染 agent
- 依赖稳定字段名

## 9. 当前最推荐的协作方式

### 你给前端的要求

1. 只解析 `run_result_json`
2. 动态渲染 `agent_views`
3. 通信图只读 `messages`
4. 检索面板读 `retrieval_debug`
5. 记忆面板读 `memory_hits` 和 `memory_context`
6. 执行面板读 `execution_result`

### 你给自己后续改动的要求

1. 新 agent 可以加，但尽量继续写入 `agent_views`
2. 新链路可以变，但继续保留 `messages`
3. 新专项能力尽量走独立字段
4. 不要让前端重新去抠自然语言日志

## 10. 结论

当前你给前端同学发的说明，加上 `FRONTEND_CONTRACT.md`，**已经足够她做出你想要的效果**。

后续为了让前端一直省心，你最需要遵守的不是“不要再加 agent”，而是：

> 可以持续演化内部实现，但要保持结果输出入口稳定、主字段稳定、按需动态输出 agent、专项能力走独立字段。
