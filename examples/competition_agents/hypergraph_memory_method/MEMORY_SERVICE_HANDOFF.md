# 分层记忆服务方案与接口交接

本次实现集中在 `hypergraph_memory_method`。统一入口是 `MemoryService`，接口类型在 `contracts.py`。智能体主流程、角色提示词、协议消息和状态池均由调用方负责。本目录既不调度角色，也不决定下一次检索是否执行。

## 一、已落实的方案

同一个持久化库保存三层记忆，每条记录都有独立的全局标识、来源、任务、批次、任务类型、数据集、层次和审核状态。数据库只是存储介质，检索范围由作用域约束，不会因为写入同一个文件就让所有数据集互相复用。

服务构造时必须显式给出数据库文件路径，例如 `memory_store/shared_memory.sqlite3`；不再默认使用进程结束即消失的内存库。正式运行可以按项目或实验系列使用一个数据库文件，无须为每个数据集复制一套库。检索前先依据命名空间和当前批次确定任务类型，再由数据库完成作用域过滤，因此不同任务类型即使位于同一个物理文件中也不会互相召回。只有测试代码才应显式使用 `:memory:`。

| 层次 | 写入内容 | 默认检索范围 | 批次结束后的处理 |
| --- | --- | --- | --- |
| 知识层 | 当前批次提供的事实、文档片段、工具资料 | 当前批次 | 从活动库及索引删除 |
| 问答层 | 任务结果、中间产物、答案和证据链 | 当前批次 | 默认删除，可配置保留归档 |
| 经验层 | 问题模式、处理策略、适用条件、失败限制及来源记忆 | 已核验长期经验；候选需显式开启 | 核验、去重、隔离问题经验、控制容量并持久化 |

经验默认按 `task_family` 跨批次复用。同一类型的两个数据集可以共享经验，例如 `topiocqa-1` 和 `topiocqa-2` 都声明为 `continuous_qa`；`codeact-1` 声明为 `code_execution`，不会检索到连续问答经验。`dataset_id` 保留具体来源。若某种经验只对一个数据集成立，可以设置 `BatchPolicy(experience_scope="dataset")`，将它限制在相同 `dataset_id` 内。换言之，默认复用关系是“同一任务类型、不同运行批次”，而不是“数据库中任意数据集之间共享”。

`namespace` 隔离不同项目、实验系列或用户，`task_family` 隔离任务能力类型，`dataset_id` 标记具体数据集，`batch_id` 隔离一次运行。每次独立运行使用新的 `batch_id`。同一任务内的局部记忆编号可以重复出现在其他任务中；接口返回的 `memory_id` 是全局标识，调用方应使用该值建立引用，不要依靠局部编号查找“最新一条”。

**建图。** 保留调用方提供的实体及别名，统一大小写、空白和全角形式，显式别名归到同一节点。没有显式实体时使用轻量规则补充，记录来源是规则推断，不能据此宣称实体抽取完全正确。经验来源和知识引用必须指向可访问的已有记忆。普通来源标签、角色名、最终答案等公共节点不参与关系扩展。

**检索。** 写入时建立全文倒排与节点关联索引；查询先召回少量种子，再沿实体关联、知识引用和经验来源构建局部子图，用带查询重启分布的加权随机游走排序。融合模式增加真实向量召回，使用名次融合，并给予已核验反馈不超过约 5% 的乘法调整。候选数量、扩展层数、高度数节点及返回字节数均有上限。只有被选入候选的正文会被读取。

默认向量后端是缓存的精确矩阵计算，适合当前小规模记忆。它仍是向量计算上的线性扫描，不等于近似索引；正文不会被逐条重新解析或编码。可显式选择 `vector_backend="hnsw"`，依赖缺失时直接报错。近似索引按可访问的活动快照构建，数据发生变化时失效；高频写入、大规模数据下的重建成本需要另外实测。

**结果利用。** 检索返回紧凑投影，包含结论、支持步骤、来源、层次、排序通道及实际关系路径，不返回完整轨迹和向量副本。证据过多时保留结论并明确列出省略的依赖，不能把截断片段标成完整证据链。调用方可根据全局记忆标识获取完整链，或把这份紧凑结果写入其状态池再传状态引用。

**维护。** 每轮通过反馈接口登记采用与核验。批次结束时由调用方执行一次 `finalize_batch`，记忆模块在事务内完成清理、审核、合并、归档和索引更新。失败时整批回滚，重试返回已保存的结果。没有计时器、固定十轮判断或后台自动调度。

## 二、主流程调用顺序

1. 启动运行：调用 `begin_batch`，明确传入批次、任务类型、具体数据集和策略；按需调用 `write_knowledge`。
2. 规划或检索阶段：调用 `retrieve`，指定检索层并保存返回的 `retrieval_id`。是否继续检索下一层，由主流程决定。
3. 任务产生结果：可以分别调用 `write_episode`、`propose_experience` 和 `record_feedback`；正式接入优先使用 `commit_task_memory`，把同一任务的上述内容放在一个可重试事务中。
4. 任务反馈可用：调用 `record_feedback` 并提交 `retrieval_ids` 和实际采用编号。接口从持久化检索回执恢复真实候选，拒绝伪造或不一致的检索集合。即使不知道答案正确性，也能登记实际采用，但正确性字段应为 `None`。
5. 批次结束：可以先调用 `preview_maintenance` 查看处理清单，再调用 `finalize_batch`。取消批次调用 `abort_batch`。
6. 下一批运行：创建新批次，已核验经验可以被检索；上一批临时知识和问答不再进入默认检索。

`write_episode` 和 `propose_experience` 应先于针对这些记录的核验反馈。一个任务的反馈是不可变事件；同一批次、同一任务重复提交相同反馈不会重复计数，提交不同内容会报错。独立复核使用新的任务或复核编号，并注明核验来源。

## 三、公共接口

| 接口 | 调用方输入 | 返回与约束 |
| --- | --- | --- |
| `begin_batch` | 批次编号、任务类型、数据集编号、任务组、`BatchPolicy` | 批次状态；同编号配置不一致时报错 |
| `write_knowledge` | 知识单元列表、操作编号 | 每条知识的全局标识 |
| `write_episode` | 任务编号、记忆单元、操作编号 | 问答或中间结果的全局标识 |
| `propose_experience` | 任务编号、经验、来源记忆、操作编号 | 候选经验标识，不直接确认为正确经验 |
| `retrieve` | `RetrievalRequest` | 紧凑候选、来源关系、排序说明和检索统计 |
| `record_feedback` | `TaskFeedback` | 实际检索数、采用数；持久化更新反馈 |
| `commit_task_memory` | `TaskMemoryCommit` | 原子写入本轮问答、候选经验和反馈；任一步失败时整体回滚 |
| `review_experience` | `ExperienceReview` | 独立登记人工、工具或离线评测对候选经验的核验结果 |
| `preview_maintenance` | 批次编号 | 清理、提升、待核验、拒绝、合并和容量处理清单，不改数据库 |
| `finalize_batch` | 批次编号 | 原子提交维护结果，重复调用返回相同报告 |
| `abort_batch` | 批次编号、取消原因 | 删除该批次临时记录；已提交的逐任务反馈仍留作审计 |
| `get_memory` | 批次编号、全局记忆标识 | 完整记录；可读取待核验经验供外部复核 |
| `get_evidence_path` | 批次编号、记忆标识 | 原始证据链及跨记忆来源，列出未解析引用 |
| `supersede_memory` | 旧记忆、替代内容、操作编号 | 创建新版本并停止旧版本参与检索，保留双向审计关系 |
| `update_memory_relations` | `MemoryRelationUpdate` | 版本化增加或移除知识引用、经验来源关系 |
| `get_batch_status` | 批次编号 | 批次状态、操作数、检索数、反馈数及维护报告 |
| `get_batch_memory_stats` | 批次编号 | 记忆检索、采用、正确采用、返回字节和检索耗时统计 |
| `list_active_experiences` | 批次编号、数量上限 | 当前批次可访问的已核验长期经验 |
| `list_pending_reviews` | 批次编号、数量上限 | 当前批次可访问的候选或待核验记忆 |
| `reindex_embeddings` | 批次编号 | 用当前编码器重建可访问记忆的向量，供更换模型或补建索引时调用 |
| `export_snapshot` | 无 | 当前命名空间的记录、批次、反馈、核验档案及向量 |
| `restore_snapshot` | 快照字典 | 仅允许恢复到空命名空间，重建派生索引，不覆盖现有实验 |

写入单元至少包含 `source_agent`、`task_topic` 和 `summary`。经验还需要 `problem_pattern`、`solution_pattern`、`source_memory_ids`，建议提供 `trigger_conditions` 和 `failure_pattern`。知识引用使用 `knowledge_refs`。接口不会读取状态池；`state_refs` 只作为调用方提供的来源引用保留。

`RetrievalRequest` 的关键字段：

- `layers`：最终允许返回的记忆层。
- `bridge_layers`：允许用于关系连接、但不返回正文的中间层，默认没有。
- `seed_memory_ids`：已知的有效来源或前序记忆，不是评测标准答案；可用于连续指代的关系检索。
- `mode`：关键词、标签、语义、局部超图或融合，对应 `keyword`、`tag`、`semantic`、`hypergraph`、`fusion`。
- `layer_limits`：各返回层的数量上限，不是保底数量。
- `candidate_limit`、`graph_hops`、`max_node_degree`：限定子图大小。
- `max_evidence_steps`、`max_payload_bytes`：限定紧凑结果规模。装不下的单元整条跳过并报告原因，不截断事实文本。
- `include_candidates`：显式允许读取待核验候选；默认关闭，不能把这类候选当作已核验经验。

这些上限约束计算量与输出量，不承诺任意硬件下的固定毫秒延迟。`payload_bytes` 是返回候选列表的 UTF-8 JSON 字节数，不是智能体间网络通信量，也不包含诊断字段。

## 四、核验与经验维护口径

`retrieve` 会持久化本次请求和候选，并返回唯一的 `retrieval_id`。`record_feedback` 优先根据 `retrieval_ids` 恢复真实候选；兼容旧调用时仍可直接传 `retrieved_ids`。若两者同时提供，内容必须完全一致。`adopted_ids` 表示下游实际使用的记忆，必须是真实候选的子集。检索返回本身不会自动增加采用次数，反馈接口按任务去重计数。

`answer_correct` 只表示本任务答案或执行结果的核验结果。`memory_verdicts` 按记忆标识给出对该条记忆的单独核验：`True` 为确认适用或有支撑，`False` 为确认错误，`None` 为未知。`validation_source` 记录人工复核、工具断言、证据校验器或离线评分器的名称。只填来源名称并不能证明真实性，调用方必须实际执行所声明的核验。

经验获得一次成功核验，需要本任务结果正确，并且该经验的单独核验为真。只有答案正确时，不会自动把所有候选或已采用经验都标成正确。采用后答错、但无法归因到某条记忆时，经验转为待核验，不直接判定为错误。被检索但未采用的记忆不受质量惩罚。

默认一次明确成功核验允许经验进入活动长期库，可用 `min_verified_successes` 提高门槛；明确失败两次时标记为拒绝，拒绝记录不再参与检索，但保留审计信息。没有任何核验的候选可以持久保存用于以后复核，默认不进入下一批正常检索。

主流程明确报告任务答错时，该任务刚写入的问答结果会进入待核验状态，避免继续作为可信答案复用。该处理不等于把其中每一项事实都判成错误，也不会处罚未被采用的其他候选。

自动合并要求问题模式、适用条件、失败限制和任务类型一致，且处理策略规范化后等价。语义相似仅生成合并建议；提供 `merge_verifier` 后，该回调明确确认等价的相似策略才会合并。该接口只在批次维护阶段调用，不修改角色提示词。合并保留原策略、全部来源、旧编号映射和去重后的核验事件。

批次清理前，长期经验引用的来源及其依赖会保存到不可检索的溯源档案。因此清除临时知识并不等于抹掉长期经验所需的来源证明。这些来源不会作为普通知识候选参与后续检索。`get_evidence_path` 检查的是来源链是否能解析、依赖结构是否完整，事实是否正确仍由外部核验负责。

## 五、最小接入示例

```python
from hypergraph_memory_method import MemoryService, RetrievalRequest, TaskFeedback

with MemoryService("memory.sqlite3", namespace="experiment-series-1") as memory:
    memory.begin_batch(
        "run-001",
        task_family="continuous_qa",
        dataset_id="topiocqa-1",
        group_id="group-1",
    )
    receipt = memory.write_episode(
        "run-001", "task-1",
        {
            "source_agent": "summarizer",
            "task_topic": "Resolve project location",
            "summary": "Project Orion is based in Paris.",
            "evidence": ["Project registry lists Paris as the location."],
            "entities": ["Project Orion", "Paris"],
        },
        operation_id="task-1:episode",
    )
    source_id = receipt["records"][0]["memory_id"]
    result = memory.retrieve(RetrievalRequest(
        batch_id="run-001", query="Project Orion location", layers=("episodic",),
    ))
    memory.record_feedback(TaskFeedback(
        batch_id="run-001", task_id="task-2",
        retrieved_ids=tuple(hit["memory_id"] for hit in result["hits"]),
        adopted_ids=(source_id,),
        answer_correct=None,
    ))
    report = memory.finalize_batch("run-001")
```

主流程需要语义或融合检索时，可以注入自己管理的编码器，也可以使用本目录的 `SentenceTransformerEncoder`。编码器提供 `model_id` 与批量 `encode(texts)`。模型标识必须包括模型版本和池化方式；同维度但不同模型的向量也不能混用。

```python
from hypergraph_memory_method import MemoryService, SentenceTransformerEncoder

encoder = SentenceTransformerEncoder(
    "/models/local-embedding-model",
    model_id="chosen-model:revision:mean-pooling:normalized",
    device="cpu",
)
memory = MemoryService("memory.sqlite3", encoder=encoder)
```

编码器只加载已有本地模型；模型目录或依赖缺失时直接报错，不自动联网下载，也不使用哈希向量冒充语义向量。向量在写入时生成并持久化，查询向量有有界缓存。更换编码器后需要显式重建向量，不能在查询过程中偷偷遍历正文重新编码。

## 六、验证与接入边界

目录内测试覆盖三层隔离、十轮批次维护、核验归因、幂等反馈、故障回滚、重启和快照恢复、跨记忆关系召回、公共节点抑制、中英文及数字检索、向量缓存和空间校验、证据链截断、合并来源保留。测试中的编码器和核验器是可控替身，用来检查接口契约，不代表真实模型的语义质量或答案准确率。

在仓库根目录设置 `PYTHONPATH=examples/competition_agents:examples/competition_agents/src` 后运行：

```bash
python -m pytest examples/competition_agents/hypergraph_memory_method/tests -q
```

独立检索评测入口为 `evaluate_service.py`。输入活动批次快照和独立标注的查询列表，标注与检索请求分开。输出前列相关率、召回率、首个相关结果名次、检索延迟、返回字节数和读取记录数。相关标注不会拼入查询；无记忆的冷启动任务也不会访问未来记录。

```json
[
  {
    "case_id": "case-1",
    "request": {
      "batch_id": "run-001",
      "query": "What is the location of Project Orion?",
      "layers": ["episodic"],
      "top_k": 3
    },
    "relevant_memory_ids": ["在快照中实际存在的全局记忆标识"]
  }
]
```

```bash
python -m hypergraph_memory_method.evaluate_service snapshot.json cases.json \
  --modes keyword hypergraph --repeats 3 --output retrieval_report.json
```

旧检索函数与 SQLite 存储接口保留，已有主流程不会自动切换到新服务。智能体同学需要接入上面的调用点，才能在完整任务运行中使用批次清理、持久化子图检索及明确的采用反馈。更快的检索不等于整题必然更快；端到端耗时和答案准确率须在接入后，以相同模型、提示和任务顺序复测。

关键词和局部超图检索不依赖额外模型；语义和融合模式要求 NumPy 与真实编码器。可选近似索引、真实向量模型的质量与硬件延迟需要在目标环境另测。本次没有改动或验证智能体之间的实际状态交换行为。

### 本次验证记录

2026 年 9 月 8 日，在现有 `langgraph` 环境执行本目录测试，54 项通过，包含本地 `all-MiniLM-L6-v2` 模型的真实融合检索与查询缓存测试。该真实模型测试使用三条人工构造材料验证语义召回接入，不代表正式数据集准确率。近似向量索引依赖当前未安装，其分支尚未完成实机验证。

另外，用 1000 条合成记忆进行了 20 次精确查询。旧关键词接口和新索引接口都找对 20 次；平均查询耗时分别为 136.552 毫秒和 0.674 毫秒，新接口每次最多读取一条候选正文。该检查只说明本例中避免全量正文准备的收益，不代表语义排序质量、整个超图检索的加速比例或智能体端到端收益。完整数值保存在 [索引检查记录](tests/fixtures/service_smoke_20260908.json)。

2026 年 9 月 14 日补齐分层输入输出、检索回执、任务原子提交、独立经验复核、版本化纠错、关系更新和批次统计接口后，本目录测试结果为 62 项通过、1 项因未配置真实本地向量模型而跳过。再次使用 1000 条合成记忆执行 20 次精确查询，旧接口与索引接口均正确 20 次，平均查询耗时分别为 140.055 毫秒和 0.773 毫秒，索引接口每次最多读取一条候选正文。该结果仍只用于验证索引和接口边界。

复测索引规模：

```bash
python -m hypergraph_memory_method.run_service_smoke --records 1000 --queries 20
```

显式开启本地模型接入测试：

```bash
MEMORY_TEST_MODEL_PATH=/models/all-MiniLM-L6-v2 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m pytest examples/competition_agents/hypergraph_memory_method/tests -q
```

## 七、方法来源与实现范围

- [SQLite FTS5](https://www.sqlite.org/fts5.html)：使用现成全文倒排及 BM25 排名，不自行模拟全文检索引擎。
- [MemORAI](https://aclanthology.org/2026.findings-acl.1408/)：借鉴查询相关稀疏子图和关系传播的思路；本实现使用现有超图的引用、实体关联，不宣称复现其整套算法或达到论文结果。
- [句向量模型接口文档](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)：本地轻量编码器适配依据。

子图传播采用记忆与关系关联的有界投影，关系权重和反馈幅度是可复核的工程选择，尚未在独立大规模检索集上完成调优。后续测试重点应是同一批记忆中正确结果的前列排序、真实查询延迟和采用后的答案变化，而不是只提高返回数量。
