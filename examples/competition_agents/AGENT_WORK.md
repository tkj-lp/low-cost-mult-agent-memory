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

### 1. protocol 模式压缩

目标：提升 text/protocol 字符节省率。

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

### 2. hidden latent 增强

当前状态：

```text
run_latent_hidden_demo.py 已能提取本地 Qwen 的 outputs.hidden_states[-1]
完整 hidden tensor 存在进程内共享 latent memory
消息中只传 hidden state_ref
```

后续方向：

- 新增正式多样例 hidden latent 实验脚本。
- 统计 hidden_state_refs、hidden_bytes、elapsed_ms。
- 明确说明当前 hidden state 是机制验证，不等同于完整 KV-cache 继续推理。
- 如果时间允许，再尝试 `past_key_values` / KV cache 传递实验。

不要让 hidden latent 改坏 `latent_states` 对外结构。只通过新增 `content_type` 扩展：

```json
{
  "content_type": "latent.llm_hidden_state"
}
```

### 3. CodeAct 沙箱

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

### 4. openEuler 验证

目标：保证在 openEuler 24.03-LTS-SP3 可运行。

需要整理：

- Python 版本。
- 依赖安装命令。
- 环境变量配置。
- `run_demo.py` 运行截图或日志。
- `run_hotpot_samples.py` 运行截图或日志。

### 5. 实验报告指标

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

