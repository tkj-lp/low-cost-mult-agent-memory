# 智能体部分工作文档

本文档是智能体同学自己的后续工作清单，主要用于记录当前边界、后续要做什么，以及哪些接口不能随便改。

## 当前职责

智能体部分负责：

- 多 Agent 主流程。
- text / protocol 两种通信模式。
- LLM API 和 mock provider 选择。
- embedding-based latent state 写入和传递。
- hidden-state latent 实验。
- 指标统计。
- 和超图、页面同学保持接口稳定。

## 当前主流程

目标流程：

```text
PlannerAgent -> RetrieverAgent -> ReasonerAgent -> SummarizerAgent
```

各 Agent 职责：

- `PlannerAgent`：生成任务计划。
- `RetrieverAgent`：检索证据和历史记忆，写入 latent state。
- `ReasonerAgent`：基于计划、证据、记忆命中结果生成候选答案。
- `SummarizerAgent`：生成最终答案，并写入共享记忆。

## 不要随便改的接口

为了不影响超图和页面同学，不要破坏这些结构：

```text
messages: list[dict]
metrics: dict
memory: list[dict]
memory_hits: list[dict]
latent_states: dict[state_id, metadata]
```

`messages` 至少保留：

```json
{
  "from": "planner",
  "to": "retriever",
  "action": "retrieve",
  "payload": "...",
  "char_cost": 305
}
```

`metrics` 至少保留：

```json
{
  "message_count": 3,
  "text_chars": 3283,
  "latent_transfer_count": 3,
  "latent_bytes": 4608
}
```

新增字段可以，但不要删除旧字段。

## 后续任务

### 当前方向收束

hidden state 后续只作为“非文本消息传递对象”和“状态引用机制”使用，不再尝试在本阶段实现“Reasoner 完全依赖 hidden state 直接推理”。

也就是说，当前比赛原型采用三层分工：

```text
推理质量：由 mock / API / evidence 文本链路验证
非文本状态传递：由 embedding state 和 hidden state_ref 验证
共享记忆复用：由 memory / hypergraph memory 验证
```

这样做的原因：

- `outputs.hidden_states[-1]` 可以被提取和传递，但不能直接作为 `generate()` 的输入继续推理。
- 真正让下游模型消费上游 latent 需要 KV cache、prefix adapter 或 projector，超出当前原型优先级。
- 比赛要求是实现非文本中间状态传递机制，不要求完整复现 latent-only reasoning。

因此后续 hidden state 相关工作只做：

- 提取本地 LLM 最后一层 hidden state。
- 存入共享 latent memory。
- Agent 消息只传 `state_ref`。
- 统计 hidden refs、hidden bytes、message chars、elapsed time。
- 页面展示 hidden state 元数据。

不做：

- 不宣称 hidden state 已经完全替代文本推理。
- 不把 hidden tensor 直接塞进 `ReasonerAgent` 当作唯一输入。
- 不在当前阶段实现 KV cache 续写。

### 1. 稳定接口冻结

目标：保证超图和页面同学可以并行推进，不被智能体侧频繁改动影响。

冻结字段：

```text
messages[].from
messages[].to
messages[].action
messages[].payload
messages[].char_cost

metrics.message_count
metrics.text_chars
metrics.latent_transfer_count
metrics.latent_bytes

memory[].memory_id
memory[].source_agent
memory[].created_at
memory[].task_topic
memory[].summary
memory[].tags

memory_hits: list[dict]
latent_states: dict[state_id, metadata]
```

后续只允许：

- 新增字段。
- 新增 `content_type`。
- 新增脚本。
- 新增文档。

不允许：

- 删除核心字段。
- 重命名核心字段。
- 改变 `memory_hits` 和 `latent_states` 的基本类型。

### 2. protocol 模式压缩

目标：提升 text/protocol 字符节省率。

当前已完成一版压缩：

```text
新增 make_compact_packet()
protocol 模式只传短字段、state_ref、memory_ref 和压缩 cognitive_gap
修正 message path 为 planner -> retriever -> reasoner -> summarizer
```

一次 mock/hash `run_demo.py` 验证结果：

```text
text chars: 30019
protocol chars: 8816
average character saving: 70.63%
```

可做方向：

- protocol payload 中少放自然语言解释。
- 长文本改为 `state_ref`、`memory_ref`。
- `cognitive_gap` 只传必要字段。
- 大段证据只存 latent state，消息中只传 evidence_ref。

指标：

```text
average character saving
text_chars_delta
protocol_chars
```

### 3. hidden state 消息传递实验

当前状态：

```text
run_latent_hidden_demo.py 已能提取本地 Qwen 的 outputs.hidden_states[-1]
run_hidden_latent_samples.py 用 5 个多跳样例验证 hidden-state reference transfer
完整 hidden tensor 存在进程内共享 latent memory
消息中只传 hidden state_ref
```

定位：

```text
hidden state 只作为非文本消息传递和状态引用实验。
答案推理仍由可控 evidence 文本或 API/mock 链路完成。
```

后续方向：

- 统计 hidden_state_refs、hidden_bytes、elapsed_ms。
- 和 embedding 传递做对比：embedding 是 384 维句向量，hidden 是本地 LLM 最后一层 `[1, seq_len, hidden_size]`。
- 把 `run_hidden_latent_samples.py` 的输出格式稳定下来，方便页面展示。
- 文档里统一表述为 `hidden-state reference transfer`。

不要让 hidden latent 改坏 `latent_states` 对外结构。只通过新增 `content_type` 扩展：

```json
{
  "content_type": "latent.llm_hidden_state"
}
```

暂不做：

- 暂不做 KV cache / `past_key_values` 续写。
- 暂不训练 adapter/projector。
- 暂不声称 hidden state 完全替代文字推理。

### 4. 超图记忆对接支持

智能体侧需要配合超图同学，但不直接实现全部超图逻辑。

要保持：

```python
add_memory(...)
search_memory(...)
```

后续智能体侧只需要读取超图同学返回的：

```text
retrieval_score
hypergraph_score
embedding_score
matched_nodes
matched_hyperedges
reasoning_path
```

并把这些字段保留在 `memory_hits` 中，供页面展示。

### 5. CodeAct 沙箱

目标：从模拟执行升级为更真实的轻量沙箱执行。

建议返回字段：

```json
{
  "stdout": "...",
  "stderr": "...",
  "exit_code": 0,
  "elapsed_ms": 12.3,
  "safety_status": "ok"
}
```

第一版可以用：

- Python 子进程。
- 超时限制。
- 禁止危险模块。
- 限制文件写入目录。

### 6. openEuler 验证

目标：保证在 openEuler 24.03-LTS-SP3 可运行。

需要整理：

- Python 版本。
- 依赖安装命令。
- 环境变量配置。
- `run_demo.py` 运行截图或日志。
- `run_hotpot_samples.py` 运行截图或日志。

### 7. 实验报告指标

需要保留这些输出：

- `message_count`
- `text_chars`
- `latent_transfer_count`
- `latent_bytes`
- `memory_hits`
- `memory_hit_rate`
- `elapsed_ms`
- `exact_match`
- `average character saving`

### 推荐推进顺序

当前阶段建议按这个顺序做：

1. 固定接口文档，通知超图和页面同学开始并行。
2. 优化 protocol payload，提升字符节省率。
3. 协助超图同学把 `search_memory()` 接入超图 + embedding 相似度融合。
4. 让页面同学先接 `messages`、`metrics`、`memory_hits`、`latent_states`。
5. 补 CodeAct 沙箱最小实现。
6. 整理 openEuler 部署文档。
7. 最后再考虑 hidden state 的 KV cache 增强，不作为当前主线。

## 和超图同学的关系

超图同学只改 `memory.py` 后面的实现，不直接改 Agent。

智能体侧只调用：

```python
add_memory(...)
search_memory(...)
```

如果需要更多检索解释字段，由超图同学添加到 `memory_hits` 的 dict 中。

## 和页面同学的关系

页面同学只读取运行结果，不参与 Agent 内部逻辑。

页面稳定读取：

- `messages`
- `metrics`
- `memory_hits`
- `latent_states`
- `final_answer`

后续新增 hidden state、KV cache、CodeAct 结果，都尽量作为新增字段，而不是改旧字段。


## run_demo.py 统一入口

当前建议把 `run_demo.py` 作为主实验入口，避免页面同学需要记多个主流程脚本。

支持的数据类型：

```text
--dataset-type builtin      内置 10 轮比赛主流程
--dataset-type multihop     HotpotQA 抽样多跳问答
--dataset-type continuous   HotpotQA 构造的连续关联任务
```

支持的稳定参数：

```text
--dataset-file  指定 JSON 数据文件
--limit         限制样例数量；continuous 模式下限制 group 数
--modes         text、protocol 或 text,protocol
```

这个改动不改变对外状态接口，页面和超图同学仍然读取：

```text
messages
metrics
memory
memory_hits
latent_states
final_answer
```

hidden state 仍然保留为独立实验入口：`run_latent_hidden_demo.py` 和 `run_hidden_latent_samples.py`。当前阶段不把 hidden state 强行并入 10 轮主流程，避免影响页面和超图同学已经对接的接口。

`run_hidden_latent_samples.py` 现在支持两类 hidden latent 数据：

```text
默认模式：hidden_hotpot_5，多跳问答样例的 hidden-state reference transfer
--dataset-type continuous：hidden_continuous，连续关联任务的 hidden-state reference transfer + memory_hits
```

## 共享记忆生命周期设计

后续共享记忆不要拆成“连续任务记忆一套、经验记忆一套”两个系统。推荐统一使用 `MemoryUnit`，再用字段区分生命周期：

```json
{
  "memory_id": "mem-0001",
  "memory_type": "episodic",
  "scope": "chain-001",
  "status": "active",
  "ttl": "session",
  "task_topic": "Kiss and Tell -> Shirley Temple",
  "summary": "本轮任务得到答案 Chief of Protocol。",
  "evidence": ["..."],
  "strategy": "先定位作品中的人物，再检索人物履历属性。",
  "experience": "作品 -> 人物 -> 职位类问题可复用两跳检索策略。",
  "source_agent": "summarizer",
  "created_at": "...",
  "tags": ["hotpot", "multi-hop"],
  "state_refs": ["state-0001", "hidden-0002"]
}
```

字段含义：

- `memory_type=episodic`：短期连续任务记忆，由每一轮任务生成，优先在当前 `scope` 内复用。
- `memory_type=experience`：长期经验记忆，由一组连续任务结束后蒸馏得到，用于跨任务复用。
- `scope`：记忆作用域，连续任务使用 `chain-001` 这类 group id，长期经验使用 `global`。
- `status`：`active` 表示可检索，`archived` 表示已归档但仍可用于复盘。
- `ttl`：`session` 表示短期，`persistent` 表示长期。

智能体侧后续建议增加一个轻量生命周期流程：

```text
1. 每轮任务结束：SummarizerAgent 写入 memory_type=episodic、scope=当前 group_id 的记忆。
2. 同一 group 后续任务：RetrieverAgent 优先检索 scope=当前 group_id、status=active 的 episodic memory。
3. 当前连续任务结束：把该 scope 下的 episodic memory 标记为 archived。
4. 任务结束后蒸馏：从 archived episodic memory 中提炼 memory_type=experience、scope=global 的长期经验。
5. 新任务开始：RetrieverAgent 可以检索 scope=global 的 experience memory。
```

智能体侧要保证：

- 不删除原有 `memory_id/source_agent/created_at/task_topic/summary/tags` 字段。
- 只新增生命周期字段，不破坏 `memory_hits` 和 `latent_states` 的类型。
- `memory_hits` 可以同时返回 episodic 和 experience，但需要保留 `memory_type`，方便页面和报告区分。

推荐新增但不强制立刻实现的函数：

```python
archive_scope(memory, scope) -> list[dict]
distill_experience(memory, scope) -> list[dict]
```

第一阶段可以先在实验报告里说明该生命周期设计；第二阶段再把 `archive_scope()` 和 `distill_experience()` 接入连续任务结束逻辑。
