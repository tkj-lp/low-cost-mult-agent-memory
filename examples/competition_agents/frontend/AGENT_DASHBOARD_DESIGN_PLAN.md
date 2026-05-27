# 智能体可视化控制台（Dashboard）页面设计方案

本方案基于 `FRONTEND_CONTRACT.md` 的稳定字段与展示建议进行设计，旨在将 Agent 内部复杂的隐状态（Latent/Hidden State）和记忆复用机制通过直观的图表呈现给用户，屏蔽底层复杂实现。

---

## 🗺️ 页面布局总览 (Page Layout)

页面整体建议采用 **「左侧/顶部实验配置栏 + 主体三段式」** 结构：
1. **控制与总览区（Top/Config Bar & Metric Cards）**：选择实验并查看大盘核心指标。
2. **核心流向与分析区（Middle-Left: Topology / Middle-Right: Chart Panels）**：展示 Agent 通信拓扑图和趋势图表。
3. **数据明细面板（Bottom: Multi-Tabs / Tables）**：展示具体任务组、消息、记忆和隐藏状态的元数据表格。

---

## 🏗️ 详细模块设计

### 1. 实验选择与参数控制区 (Experiment Selector)
位于页面最上方或左侧边栏，提供两个核心级联/独立选择框，并配有对应的**运行命令提示**（方便前端调试或后端异步触发）。

* **选择框 A：运行方式 (Execution Mode)**
  * `quick` (跑通流程/调试) | `api` (真实 API 生成) | `api_embedding` (向量状态传递) | `latent_hidden` (共享隐状态传递)
* **选择框 B：数据集 (Dataset)**
  * `competition_10` | `hotpot_multihop` | `hotpot_continuous` | `hidden_hotpot_5` | `hidden_continuous`

> ⚠️ **前端交互拦截**：当运行方式选择 `latent_hidden` 时，数据集下拉框应高亮推荐 `hidden_hotpot_5` 和 `hidden_continuous`。若用户强行组合 `competition_10`，页面应弹出 Toast 提示：*“当前第一版不建议组合该配置，Hidden State 目前为独立实验入口。”*

---

### 2. 总览指标卡片区 (Global Metrics Dashboard)
以 **「指标卡片 (KPI Cards)」** 形式平铺在页面顶部，突出非文本传输（低开销）与高记忆命中的技术优势。以 `hidden_continuous` 为例：

| 指标名称 | 绑定字段 / 计算逻辑 | 样式推荐 | 业务解释口径 (Tooltip) |
| :--- | :--- | :--- | :--- |
| **测试样例数** | `samples` (如: 15) | 基础数字 | 本次实验运行的总 Turn/样例数 |
| **任务准确率** | `exact_match` (如: 100%) | 绿色高亮数字/进度条 | 由确定性 evidence extractor 验证的准确率 |
| **总消息数** | `metrics.message_count` (如: 45) | 基础数字 | Agent 之间发生的通信总次数 |
| **文本通信开销** | `metrics.text_chars` (如: 8,490) | 带有 `chars` 单位 | 值越小，说明 Agent 间文本通信越低开销 |
| **Hidden 引用数** | `hidden_state_refs` (如: 45) | 蓝色数字 | 消息中传递的状态引用（`state_ref`）次数 |
| **Hidden 状态规模** | `metrics.latent_bytes` (如: 39.70 MB) | 带有 `MB` 单位 | 共享 Latent Memory 中保存的完整 hidden tensor 规模 |
| **记忆命中数** | `memory_hits` 总数 (如: 15) | 火焰或命中图标 | 连续任务后续轮次复用前序记忆的次数 |
| **总耗时** | `elapsed_ms` (如: 152.11 s) | 带有 `s` 单位 | 整个实验流程的耗时 |

> 💡 **页面醒目位置补充「方案文案说明（Caveat）」**：
> *“本实验在 5 组连续任务、15 个 turn 中完成 hidden-state reference transfer... 当前 exact_match 由确定性 evidence extractor 验证，用于隔离传递机制开销；完整问答质量请参考 API 主流程。”*

---

### 3. 核心可视化图表区 (Charts & Topology)
此区域为页面的核心视觉痛点，建议分为左右两栏。

#### 左栏：Agent 串行通信链路图 (Topology Graph)
点击下方明细表格中的某一个具体 Case 时，在此处动态渲染该 Turn 的拓扑关系。
* **节点 (Nodes)**：固定横向排列：`Planner` ➔ `Retriever` ➔ `Reasoner` ➔ `Summarizer`
* **连线 (Edges)**：
  * 线上标签 (Label) 渲染 `action`（如 `action=retrieve`）。
  * 线上 Hover Tooltip 渲染 `refs` 列表（如 `refs=hidden-0001, hidden-0002`）以及文本开销 `char_cost`。
* **节点下方挂载**：该 Agent 产生的 `state_id` 标签（点击可联动跳转到下方元数据表）。

#### 右栏：对比与趋势图表组 (Analysis Charts)
按文档 9.6 要求，并排或切换展示以下三张图表：
1. **Memory Hits 折线图 (核心亮点)**
   * **X 轴**：`turn-1` / `turn-2` / `turn-3`
   * **Y 轴**：`memory_hits` 数量 (0 -> 1 -> 2)
   * **表现形式**：多条折线（每个连续任务组 `group` 一条线），清晰展现出随轮次推进，记忆命中率稳步上升的趋势。
2. **Text Chars 文本开销柱状图**
   * 展示各个 Turn 的文本通信量，直观体现出因为使用了 `state_ref`，文本传输一直保持在极低的水平。
3. **Hidden Bytes 隐状态规模柱状图**
   * 展示每个 Turn 写入共享 Latent Memory 的 Tensor 规模，与文本开销图形成对比。

---

### 4. 数据明细与元数据面板 (Data & Metadata Tabs)
位于页面底部，采用 **Tab 标签页** 切换不同的数据维度。

#### Tab 1：连续任务组视图（Group & Turn Hierarchical Table）
采用 **树形折叠表格（Tree Table）**：
* **第一层（父节点）**：连续任务组卡片（如 `chain-001 Kiss and Tell -> Shirley Temple`），右侧简要展示该组的记忆命中趋势（0➔1➔2）。
* **第二层（展开子节点）**：单轮任务结果表。
  * **列定义**：`Turn` \| `Question` \| `Final Answer` \| `Gold` \| `Exact Match (True/False)` \| `Memory Hits` \| `Text Chars` \| `Hidden Refs` \| `Hidden Bytes` \| `Elapsed(ms)`

#### Tab 2：Hidden State 元数据表（Latent States Metadata）
展示当前 Case 产生的隐状态列表，**绝对不展示完整向量/Tensor 数组**，只展示结构化属性：
* **列定义**：`state_id` \| `producer (生成 Agent)` \| `content_type` \| `model` \| `layer` \| `shape` \| `pooled_shape` \| `checksum`

#### Tab 3：记忆命中面板（Memory Hits Panel）
当用户点击某轮任务中的 `Memory Hits > 0` 时，可在此查看抓取到的历史记忆详情卡片。
* **基础信息**：记忆ID（`memory_id`）、来源（`source_agent`）、时间戳、任务主题、摘要（`summary`）、标签（`tags`）。
* **高级超图解释（按需渲染）**：检索得分（`retrieval_score`）、超图得分、命中节点/超边（`matched_nodes/hyperedges`）以及推理路径（`reasoning_path`）。

---

## 🛠️ 前端实现技术栈推荐

1. **大盘与表格**：使用 `Ant Design` 或 `Element Plus`。其 `Table` 组件原生支持树形数据展示（非常适合呈现 Group -> Turn 的层开销次）。
2. **拓扑链路图**：使用 `AntV G6`、`React Flow` 或 `Apache ECharts (Graph树图)`。由于本项目的 Agent 通信链路固定为单向串行（`Planner -> Retriever -> Reasoner -> Summarizer`），甚至可以用标准的步骤条（`Steps`）或简单的 CSS Flexbox 连线来实现，轻量且不易出错。
3. **趋势图表**：使用 `ECharts` 或 `ApexCharts`，折线图开启多系列（Multi-series）以支持多条 Group 曲线并行对比。
