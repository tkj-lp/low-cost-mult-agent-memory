# 超图记忆对接契约

本文档是智能体主流程与超图记忆模块之间当前唯一有效的对接说明。

目标只有两个：
- 明确现在已经落地且必须保持稳定的接口
- 明确后续优化时你这边和超图同学各自负责什么

如果后续口头沟通与代码不一致，以当前仓库代码和本文档为准。

## 1. 职责边界

### 智能体主流程侧负责

- `PlannerAgent -> RetrieverAgent -> ReasonerAgent -> SummarizerAgent` 主链路
- 什么时候写入共享记忆
- 什么时候检索 `evidence memory` 和 `experience memory`
- Reasoner 如何消费经验规则
- 结果 JSON、前端展示、评测指标

### 超图记忆侧负责

- memory record 的超图建模
- 超图检索与排序
- `distill_experience()` 等经验蒸馏辅助逻辑
- 经验记忆的后续 `merge / dedup / prune / decay`
- 检索解释字段的构造

### 当前分工结论

- 我这边负责“经验闭环接入和验证”
- 超图同学负责“经验在超图里的组织、检索、排序和演化”

## 2. 当前共享入口

当前主流程通过下面这个文件调用超图能力：

- [memory.py](/d:/qq/low-cost-memory-integrate/examples/competition_agents/competition_agents/memory.py)

超图同学自己的主目录是：

- [hypergraph_memory_method](/d:/qq/low-cost-memory-integrate/examples/competition_agents/hypergraph_memory_method)

后续超图内部实现可以继续优化，但下面这些共享入口的函数名、参数名、返回结构不要再改。

## 3. 必须稳定的共享函数

### `add_memory`

```python
add_memory(
    memory: list[dict],
    *,
    source_agent: str,
    task_topic: str,
    summary: str,
    tags: list[str],
    trace: dict[str, Any] | None = None,
    memory_type: str = "evidence",
    problem_pattern: str = "",
    solution_pattern: str = "",
    failure_pattern: str = "",
    trigger_conditions: list[str] | None = None,
    supporting_evidence: list[str] | None = None,
    confidence: float = 0.0,
    archive_scope: str = "task",
    lifecycle_stage: str = "episodic",
    strategies: list[str] | None = None,
    experiences: list[str] | None = None,
    reuse_benefits: list[str] | None = None,
    experience_rule_card: dict[str, Any] | None = None,
    experience_source: str = "",
    distillation_case: dict[str, Any] | None = None,
    abstraction_note: str = "",
    distillation_mode: str = "",
    distillation_raw_output: str = "",
) -> list[dict]
```

### `refine_memory`

```python
refine_memory(
    *,
    task_topic: str,
    summary: str,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]
```

### `search_memory`

```python
search_memory(
    memory: list[dict],
    query: str,
    *,
    top_k: int,
    memory_types: Iterable[str] | None = None,
) -> list[dict]
```

### `increment_reuse_counts`

```python
increment_reuse_counts(
    memory: list[dict],
    memory_ids: Iterable[str],
    *,
    successful_memory_ids: Iterable[str] | None = None,
) -> list[dict]
```

## 4. 当前记忆分层

当前不是两套物理存储，而是同一个共享记忆池中的两类逻辑记忆：

### `evidence memory`

作用：
- 存放摘要、证据、问答结果、执行结果等可直接复用事实
- 给 Reasoner 提供事实答案支撑

### `experience memory`

作用：
- 存放可迁移策略和避错经验
- 给 Reasoner 提供“怎么选答案”的规则

关键字段：
- `problem_pattern`
- `solution_pattern`
- `failure_pattern`
- `trigger_conditions`
- `reuse_count`
- `reuse_success_count`

## 5. 当前 memory record 最小契约

### 基础字段

```json
{
  "memory_id": "mem-0001",
  "memory_type": "evidence",
  "source_agent": "summarizer",
  "created_at": "2026-06-11T00:00:00+00:00",
  "task_topic": "Question: ...",
  "summary": "可复用摘要",
  "tags": ["langgraph", "multi-agent"],
  "embedding": [],
  "embedding_provider": "hash",
  "embedding_model": "hash-v1"
}
```

### 兼容字段

```json
{
  "entities": [],
  "supporting_facts": [],
  "supporting_evidence": [],
  "evidence": [],
  "reasoning_path": [],
  "state_refs": [],
  "answer": "answer text",
  "reuse_count": 0,
  "reuse_success_count": 0,
  "trace": {},
  "hypergraph": {}
}
```

### experience memory 额外字段

```json
{
  "memory_type": "experience",
  "problem_pattern": "multiple_temporal_candidates_with_constraints",
  "solution_pattern": "choose_date_matching_all_constraints",
  "failure_pattern": "confuse_digital_release_with_disc_release",
  "trigger_conditions": ["question_when", "release_date", "medium_constraint"],
  "confidence": 0.8,
  "experience_source": "success-derived",
  "abstraction_note": "When multiple release dates exist, follow the medium and region constraint before selecting a date.",
  "source_case": {
    "question": "when was ...",
    "group_id": "chain-001",
    "dataset_type": "continuous",
    "case_outcome": "success",
    "exact_match": true
  },
  "experience_rule_card": {
    "problem_pattern": "multiple_temporal_candidates_with_constraints",
    "solution_pattern": "choose_date_matching_all_constraints",
    "failure_pattern": "confuse_digital_release_with_disc_release",
    "trigger_conditions": ["question_when", "release_date", "medium_constraint"],
    "supporting_evidence": [],
    "confidence": 0.8,
    "abstraction_note": "..."
  }
}
```

## 6. 当前主流程怎样消费超图返回

### Retriever

当前固定分两路查：

```python
evidence_hits = search_memory(..., memory_types=["evidence"])
experience_hits = search_memory(..., memory_types=["experience"])
```

要求：
- 这两个入口必须持续可用
- `experience_hits` 返回的每条记录必须保留完整经验字段

### Reasoner

Reasoner 当前已经显式接收：

- `evidence`
- `memory_hit_details`
- `experience_rules`

其中 `experience_rules` 的用途不是补充事实，而是：
- 用 `solution_pattern` 指导答案抽取
- 用 `failure_pattern` 规避历史错误
- 用 `trigger_conditions` 判断规则是否匹配当前题

### Summarizer

当前固定节奏是：

```text
先写 evidence memory
再做 refine_memory
再写 experience memory
```

超图侧后续即使做 `merge / dedup / prune / decay`，也必须兼容这个双写节奏。

## 7. 当前必须稳定的结果字段

### task 结果层

- `memory_hits`
- `memory_hit_count`
- `experience_hits`
- `experience_hit_count`
- `memory_refine_case`
- `memory_refine_debug`
- `latest_evidence_write`
- `latest_experience_write`
- `latest_memory_write`

说明：
- `latest_evidence_write` 和 `latest_experience_write` 是当前前后端展示应优先使用的稳定字段
- `latest_memory_write` 仅作为兼容旧前端和旧结果文件的保留字段，后续不再作为主展示出口

### metrics 层

- `memory_write_count`
- `experience_write_count`
- `experience_hit_count`
- `experience_used_count`
- `reuse_success_count`
- `memory_hit_rate`
- `memory_growth`

### retrieval_debug 层

- `memory_hit_ids`
- `evidence_hit_ids`
- `experience_hit_ids`
- `experience_hits`
- `experience_rules`
- `memory_context`
- `matched_nodes`
- `matched_hyperedges`
- `reasoning_path`

## 8. 超图同学后续优先做什么

按当前代码状态，超图侧下一阶段最值得做的是：

1. 单独加强 `experience memory` 的排序
2. 排序时使用：
   - `problem_pattern`
   - `solution_pattern`
   - `failure_pattern`
   - `trigger_conditions`
   - `reuse_count`
   - `reuse_success_count`
3. 后续再做：
   - `merge`
   - `dedup`
   - `prune`
   - `decay`

## 9. 一句话结论

当前已经稳定对齐的口径是：

`evidence memory` 和 `experience memory` 现在逻辑上已经分层，但物理上仍共用同一个 memory store 与 hypergraph pipeline。后续超图内部实现可以继续增强，但共享函数签名、关键字段名和结果 JSON 结构不再变动。

## 10. 冻结接口与内部可调边界

这一节用于明确：哪些内容从现在开始视为“冻结接口”，哪些属于我方内部实现，可继续调整但不影响超图侧对接。

### 10.1 冻结接口

以下内容从当前版本开始视为稳定契约，后续优化不应再改字段名、主语义或函数签名：

#### 共享函数签名

- `add_memory(...)`
- `search_memory(...)`
- `refine_memory(...)`

#### memory record 核心字段

- `memory_id`
- `memory_type`
- `source_agent`
- `created_at`
- `task_topic`
- `summary`
- `problem_pattern`
- `solution_pattern`
- `failure_pattern`
- `trigger_conditions`
- `supporting_evidence`
- `confidence`
- `reuse_count`
- `reuse_success_count`
- `source_case`
- `experience_rule_card`

#### 运行态结果字段

- `memory_hits`
- `experience_hits`
- `experience_rules`
- `memory_context`
- `latest_evidence_write`
- `latest_experience_write`
- `memory_refine_case`
- `memory_refine_debug`

#### 字段语义也冻结

- `memory_type="evidence"` 表示事实/证据型记忆
- `memory_type="experience"` 表示经验/策略型记忆
- `trigger_conditions` 表示经验触发条件，不得改成别的用途
- `experience_rule_card` 表示供 Reasoner 消费的规则卡片，不得退化成普通日志容器

### 10.2 我方内部可调实现

以下内容后续可以继续优化，但默认只在我方链路内部调整，不要求超图侧同步改接口：

- experience distillation prompt
- `problem_pattern / solution_pattern / failure_pattern` 的生成策略
- `trigger_conditions` 的补全、清洗与泛化规则
- experience rerank / soft match / 打分权重
- Reasoner 如何消费 `experience_rules`
- evidence / experience 的写回节奏
- 前端展示结构与文案
- `merge / dedup / prune / decay` 的内部策略

原则：
- 可以新增向后兼容字段
- 不可以删除冻结字段
- 不可以改变冻结字段的主语义
- 不可以修改共享函数名和入参名

### 10.3 协作边界

从现在起建议按下面的边界协作：

- 超图同学负责：
  - `add_memory / search_memory / refine_memory` 这类共享记忆子系统内部能力增强
  - experience/evidence 的排序、合并、去重、淘汰、衰减
  - 超图检索命中质量优化

- 我方负责：
  - 多 Agent 主流程接入
  - Planner / Retriever / Reasoner / Summarizer 如何使用这些接口
  - 经验提炼策略
  - 实验入口、结果结构、前端展示、评测口径

### 10.4 后续默认协作规则

若后续仅涉及以下内容：

- prompt 调整
- trigger 生成逻辑调整
- 经验规则卡片提炼优化
- Reasoner 消费策略优化
- 前端字段展示优化

则默认视为我方内部调整，不要求超图侧改接口。

若后续需要变更以下内容：

- 共享函数签名
- 冻结字段名
- 冻结字段语义
- `memory_type` 取值含义

则必须先同步并更新本合同文档，再做代码变更。
