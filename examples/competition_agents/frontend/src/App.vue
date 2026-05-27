<script setup>
import { computed, ref, watch } from "vue";
import { ElMessage } from "element-plus";
import {
  Check,
  Connection,
  Cpu,
  DataLine,
  Document,
  Operation,
  Search,
  Timer,
  Upload,
} from "@element-plus/icons-vue";

const executionModes = [
  { label: "quick", value: "quick" },
  { label: "api", value: "api" },
  { label: "api_embedding", value: "api_embedding" },
  { label: "latent_hidden", value: "latent_hidden" },
];

const datasets = [
  { label: "competition_10", value: "competition_10", hidden: false },
  { label: "hotpot_multihop", value: "hotpot_multihop", hidden: false },
  { label: "hotpot_continuous", value: "hotpot_continuous", hidden: false },
  { label: "hidden_hotpot_5", value: "hidden_hotpot_5", hidden: true },
  { label: "hidden_continuous", value: "hidden_continuous", hidden: true },
];

const fileInput = ref(null);
const selectedMode = ref("latent_hidden");
const selectedDataset = ref("hidden_continuous");
const activeTab = ref("groups");
const selectedCaseId = ref("chain-001-turn-3");

const cases = ref([
  {
    id: "chain-001-turn-1",
    groupId: "chain-001",
    groupLabel: "Kiss and Tell -> Shirley Temple",
    turn: "turn-1",
    question: "Who wrote Kiss and Tell?",
    finalAnswer: "F. Hugh Herbert",
    gold: "F. Hugh Herbert",
    exactMatch: true,
    memoryHits: 0,
    textChars: 1450,
    hiddenRefs: 3,
    hiddenBytes: 8.1,
    elapsedMs: 8920,
  },
  {
    id: "chain-001-turn-2",
    groupId: "chain-001",
    groupLabel: "Kiss and Tell -> Shirley Temple",
    turn: "turn-2",
    question: "Which film connects that writer to Shirley Temple?",
    finalAnswer: "Rebecca of Sunnybrook Farm",
    gold: "Rebecca of Sunnybrook Farm",
    exactMatch: true,
    memoryHits: 1,
    textChars: 1610,
    hiddenRefs: 3,
    hiddenBytes: 8.2,
    elapsedMs: 9340,
  },
  {
    id: "chain-001-turn-3",
    groupId: "chain-001",
    groupLabel: "Kiss and Tell -> Shirley Temple",
    turn: "turn-3",
    question: "What shared memory helped answer the final relation?",
    finalAnswer: "The reused writer and film relation.",
    gold: "The reused writer and film relation.",
    exactMatch: true,
    memoryHits: 2,
    textChars: 1588,
    hiddenRefs: 3,
    hiddenBytes: 8.3,
    elapsedMs: 10180,
  },
  {
    id: "chain-002-turn-1",
    groupId: "chain-002",
    groupLabel: "Olympic host -> architect trail",
    turn: "turn-1",
    question: "Which city hosted the referenced event?",
    finalAnswer: "Barcelona",
    gold: "Barcelona",
    exactMatch: true,
    memoryHits: 0,
    textChars: 1354,
    hiddenRefs: 3,
    hiddenBytes: 7.8,
    elapsedMs: 7840,
  },
  {
    id: "chain-002-turn-2",
    groupId: "chain-002",
    groupLabel: "Olympic host -> architect trail",
    turn: "turn-2",
    question: "Which architect is associated with that city?",
    finalAnswer: "Antoni Gaudi",
    gold: "Antoni Gaudi",
    exactMatch: true,
    memoryHits: 1,
    textChars: 1496,
    hiddenRefs: 3,
    hiddenBytes: 8.0,
    elapsedMs: 8260,
  },
  {
    id: "chain-002-turn-3",
    groupId: "chain-002",
    groupLabel: "Olympic host -> architect trail",
    turn: "turn-3",
    question: "Which prior facts were reused?",
    finalAnswer: "The city and architect pair.",
    gold: "The city and architect pair.",
    exactMatch: true,
    memoryHits: 2,
    textChars: 1512,
    hiddenRefs: 3,
    hiddenBytes: 8.1,
    elapsedMs: 8610,
  },
]);

const topologyEdges = [
  {
    from: "Planner",
    to: "Retriever",
    action: "plan",
    refs: ["hidden-0001"],
    charCost: 312,
  },
  {
    from: "Retriever",
    to: "Reasoner",
    action: "retrieve",
    refs: ["hidden-0002", "memory:mem-0003"],
    charCost: 486,
  },
  {
    from: "Reasoner",
    to: "Summarizer",
    action: "reason",
    refs: ["hidden-0003"],
    charCost: 790,
  },
];

const agents = [
  { name: "Planner", stateId: "hidden-0001", tone: "blue" },
  { name: "Retriever", stateId: "hidden-0002", tone: "purple" },
  { name: "Reasoner", stateId: "hidden-0003", tone: "purple" },
  { name: "Summarizer", stateId: "memory:mem-0003", tone: "green" },
];

const latentStates = [
  {
    state_id: "hidden-0001",
    producer: "Planner",
    content_type: "hidden_state",
    model: "Qwen2.5-0.5B-Instruct",
    layer: "last",
    shape: "[1, 128, 896]",
    pooled_shape: "[896]",
    checksum: "8f31c9a2",
  },
  {
    state_id: "hidden-0002",
    producer: "Retriever",
    content_type: "embedding",
    model: "all-MiniLM-L6-v2",
    layer: "pooled",
    shape: "[384]",
    pooled_shape: "[384]",
    checksum: "c2a773e4",
  },
  {
    state_id: "hidden-0003",
    producer: "Reasoner",
    content_type: "hidden_state",
    model: "Qwen2.5-0.5B-Instruct",
    layer: "last",
    shape: "[1, 160, 896]",
    pooled_shape: "[896]",
    checksum: "41dd09f8",
  },
];

const memoryHitCards = [
  {
    memory_id: "mem-0001",
    source_agent: "summarizer",
    task_topic: "Kiss and Tell relation",
    summary: "F. Hugh Herbert wrote Kiss and Tell and can anchor later relation questions.",
    tags: ["writer", "film", "reuse"],
    retrieval_score: 0.92,
    hypergraph_score: 0.86,
    matched_nodes: ["tag:film", "keyword:writer", "memory:mem-0001"],
  },
  {
    memory_id: "mem-0003",
    source_agent: "summarizer",
    task_topic: "Barcelona architecture relation",
    summary: "Barcelona related turns reused the city and architect pair across follow-up tasks.",
    tags: ["city", "architecture", "openeuler-demo"],
    retrieval_score: 0.88,
    hypergraph_score: 0.81,
    matched_nodes: ["tag:city", "keyword:gaudi", "hyperedge:mem-0003"],
  },
];

const commandMap = {
  quick:
    "python examples\\competition_agents\\run_demo.py --dataset-type builtin",
  api:
    "python examples\\competition_agents\\run_demo.py --dataset-type multihop --dataset-file examples\\competition_agents\\data\\hotpot_multihop_sample.json --limit 5",
  api_embedding:
    "python examples\\competition_agents\\run_demo.py --dataset-type continuous --dataset-file examples\\competition_agents\\data\\hotpot_continuous_tasks.json --limit 5",
  latent_hidden:
    "python examples\\competition_agents\\run_hidden_latent_samples.py --dataset-type continuous --dataset-file examples\\competition_agents\\data\\hotpot_continuous_tasks.json --limit 1",
};

const selectedCase = computed(
  () => cases.value.find((item) => item.id === selectedCaseId.value) ?? cases.value[0],
);

const totals = computed(() => {
  const totalCases = cases.value.length;
  const exact = cases.value.filter((item) => item.exactMatch).length;
  const messageCount = totalCases * 3;
  const textChars = cases.value.reduce((sum, item) => sum + item.textChars, 0);
  const hiddenRefs = cases.value.reduce((sum, item) => sum + item.hiddenRefs, 0);
  const latentBytes = cases.value.reduce((sum, item) => sum + item.hiddenBytes, 0);
  const memoryHits = cases.value.reduce((sum, item) => sum + item.memoryHits, 0);
  const elapsedMs = cases.value.reduce((sum, item) => sum + item.elapsedMs, 0);

  return {
    totalCases,
    exactRate: Math.round((exact / totalCases) * 100),
    messageCount,
    textChars,
    hiddenRefs,
    latentBytes,
    memoryHits,
    elapsedSeconds: elapsedMs / 1000,
  };
});

const metrics = computed(() => [
  {
    label: "测试样例数",
    value: totals.value.totalCases,
    unit: "cases",
    icon: Document,
    tone: "blue",
    tip: "本次实验运行的样例或 turn 数。",
  },
  {
    label: "任务准确率",
    value: `${totals.value.exactRate}%`,
    unit: "exact match",
    icon: Check,
    tone: "green",
    tip: "由确定性 evidence extractor 验证的准确率。",
  },
  {
    label: "总消息数",
    value: totals.value.messageCount,
    unit: "messages",
    icon: Connection,
    tone: "blue",
    tip: "Agent 之间发生的通信总次数。",
  },
  {
    label: "文本通信开销",
    value: totals.value.textChars.toLocaleString(),
    unit: "chars",
    icon: DataLine,
    tone: "blue",
    tip: "Agent 间传递的文本字符总量。",
  },
  {
    label: "Hidden 引用数",
    value: totals.value.hiddenRefs,
    unit: "state refs",
    icon: Cpu,
    tone: "purple",
    tip: "消息中传递的 hidden state 或 embedding 引用次数。",
  },
  {
    label: "Hidden 状态规模",
    value: totals.value.latentBytes.toFixed(1),
    unit: "MB",
    icon: Operation,
    tone: "purple",
    tip: "共享 Latent Memory 中保存的 hidden tensor 规模。",
  },
  {
    label: "记忆命中数",
    value: totals.value.memoryHits,
    unit: "hits",
    icon: Search,
    tone: "green",
    tip: "连续任务中复用历史记忆的次数。",
  },
  {
    label: "总耗时",
    value: totals.value.elapsedSeconds.toFixed(1),
    unit: "s",
    icon: Timer,
    tone: "blue",
    tip: "整个实验流程的累计耗时。",
  },
]);

const groupedRows = computed(() => {
  const groups = new Map();
  for (const item of cases.value) {
    if (!groups.has(item.groupId)) {
      groups.set(item.groupId, {
        id: item.groupId,
        groupLabel: item.groupLabel,
        turn: item.groupId,
        question: "连续任务组",
        finalAnswer: "Memory hits: 0 -> 1 -> 2",
        gold: "稳定复用",
        exactMatch: true,
        memoryHits: 0,
        textChars: 0,
        hiddenRefs: 0,
        hiddenBytes: 0,
        elapsedMs: 0,
        children: [],
      });
    }

    const group = groups.get(item.groupId);
    group.memoryHits += item.memoryHits;
    group.textChars += item.textChars;
    group.hiddenRefs += item.hiddenRefs;
    group.hiddenBytes += item.hiddenBytes;
    group.elapsedMs += item.elapsedMs;
    group.children.push(item);
  }
  return Array.from(groups.values());
});

const memorySeries = computed(() => {
  const groups = new Map();
  for (const item of cases.value) {
    if (!groups.has(item.groupId)) groups.set(item.groupId, []);
    groups.get(item.groupId).push(item.memoryHits);
  }
  return Array.from(groups, ([name, values], index) => ({ name, values, index }));
});

const maxTextChars = computed(() =>
  Math.max(...cases.value.map((item) => item.textChars)),
);

const maxHiddenBytes = computed(() =>
  Math.max(...cases.value.map((item) => item.hiddenBytes)),
);

const currentCommand = computed(() => commandMap[selectedMode.value]);

watch(selectedMode, (mode) => {
  if (mode === "latent_hidden" && !selectedDataset.value.startsWith("hidden_")) {
    ElMessage.warning("当前版本建议 latent_hidden 搭配 hidden_hotpot_5 或 hidden_continuous。");
  }
});

watch(selectedDataset, (dataset) => {
  if (selectedMode.value === "latent_hidden" && !dataset.startsWith("hidden_")) {
    ElMessage.warning("Hidden State 当前为独立实验入口，建议切换到 hidden 数据集。");
  }
});

function selectCase(row) {
  if (!row.children) {
    selectedCaseId.value = row.id;
  }
}

function pointFor(value, index, total) {
  const x = 28 + index * (244 / Math.max(total - 1, 1));
  const y = 116 - value * 42;
  return `${x},${y}`;
}

function linePoints(values) {
  return values.map((value, index) => pointFor(value, index, values.length)).join(" ");
}

function formatMs(ms) {
  return `${(ms / 1000).toFixed(2)}s`;
}

function triggerImport() {
  fileInput.value?.click();
}

function normalizeImportedCase(item, index) {
  const metrics = item.metrics ?? {};
  const latentStatesValue = item.latent_states ?? [];
  const latentCount = Array.isArray(latentStatesValue)
    ? latentStatesValue.length
    : Object.keys(latentStatesValue).length;

  return {
    id: item.id ?? item.case_id ?? `${item.group_id ?? "imported"}-${item.turn_id ?? index + 1}`,
    groupId: item.group_id ?? item.groupId ?? "imported-chain",
    groupLabel: item.group_label ?? item.groupLabel ?? item.group_id ?? "Imported chain",
    turn: item.turn_id ?? item.turn ?? `turn-${index + 1}`,
    question: item.question ?? item.input ?? "Imported question",
    finalAnswer: item.final_answer ?? item.finalAnswer ?? item.answer ?? "",
    gold: item.gold ?? "",
    exactMatch: Boolean(item.exact_match ?? item.exactMatch),
    memoryHits: Array.isArray(item.memory_hits)
      ? item.memory_hits.length
      : Number(item.memory_hits ?? item.memoryHits ?? 0),
    textChars: Number(metrics.text_chars ?? item.text_chars ?? item.textChars ?? 0),
    hiddenRefs: Number(
      item.hidden_state_refs ??
        metrics.latent_transfer_count ??
        item.hiddenRefs ??
        latentCount,
    ),
    hiddenBytes: Number(
      metrics.latent_bytes_mb ??
        item.hidden_bytes_mb ??
        item.hiddenBytes ??
        (Number(metrics.latent_bytes ?? 0) / 1024 / 1024),
    ),
    elapsedMs: Number(item.elapsed_ms ?? item.elapsedMs ?? 0),
  };
}

async function importJson(event) {
  const file = event.target.files?.[0];
  if (!file) return;

  try {
    const payload = JSON.parse(await file.text());
    if (payload.mode) selectedMode.value = payload.mode;
    if (payload.dataset) selectedDataset.value = payload.dataset;

    const importedCases = Array.isArray(payload.cases)
      ? payload.cases.map(normalizeImportedCase)
      : [];
    if (importedCases.length > 0) {
      cases.value = importedCases;
      selectedCaseId.value = importedCases[0].id;
      activeTab.value = "groups";
      ElMessage.success(`已导入 ${importedCases.length} 条实验结果`);
    } else {
      ElMessage.warning("JSON 中没有找到 cases 数组");
    }
  } catch (error) {
    ElMessage.error(`导入失败：${error.message}`);
  } finally {
    event.target.value = "";
  }
}
</script>

<template>
  <main class="dashboard-shell">
    <section class="topbar surface">
      <div class="identity">
        <span class="system-mark">LG</span>
        <div>
          <p class="eyebrow">Agent Experiment Console</p>
          <h1>多智能体可视化控制台</h1>
        </div>
      </div>

      <div class="control-grid">
        <label class="control-field">
          <span>运行方式</span>
          <el-select v-model="selectedMode" size="large">
            <el-option
              v-for="item in executionModes"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </label>

        <label class="control-field">
          <span>数据集</span>
          <el-select v-model="selectedDataset" size="large">
            <el-option
              v-for="item in datasets"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            >
              <div class="dataset-option">
                <span>{{ item.label }}</span>
                <el-tag v-if="item.hidden" size="small" effect="plain" type="info">
                  hidden
                </el-tag>
              </div>
            </el-option>
          </el-select>
        </label>

        <div class="command-card">
          <span>推荐命令</span>
          <code>{{ currentCommand }}</code>
        </div>

        <el-button
          class="import-button"
          :icon="Upload"
          size="large"
          @click="triggerImport"
        >
          导入 JSON
        </el-button>
        <input
          ref="fileInput"
          class="file-input"
          type="file"
          accept="application/json,.json"
          @change="importJson"
        />
      </div>
    </section>

    <section class="metrics-grid">
      <el-tooltip
        v-for="metric in metrics"
        :key="metric.label"
        :content="metric.tip"
        placement="top"
      >
        <article class="metric-card surface" :class="`tone-${metric.tone}`">
          <div class="metric-icon">
            <component :is="metric.icon" />
          </div>
          <div>
            <span>{{ metric.label }}</span>
            <strong>{{ metric.value }}</strong>
            <small>{{ metric.unit }}</small>
          </div>
        </article>
      </el-tooltip>
    </section>

    <section class="analysis-grid">
      <article class="surface topology-panel">
        <div class="panel-heading">
          <div>
            <p class="eyebrow">Selected Case</p>
            <h2>{{ selectedCase.groupId }} / {{ selectedCase.turn }}</h2>
          </div>
          <el-tag type="success" effect="plain">Exact Match</el-tag>
        </div>

        <div class="agent-flow">
          <template v-for="(agent, index) in agents" :key="agent.name">
            <div class="agent-node" :class="`node-${agent.tone}`">
              <span>{{ agent.name }}</span>
              <button type="button" @click="activeTab = 'latent'">
                {{ agent.stateId }}
              </button>
            </div>
            <el-tooltip
              v-if="topologyEdges[index]"
              :key="topologyEdges[index].action"
              placement="top"
            >
              <template #content>
                refs={{ topologyEdges[index].refs.join(", ") }}<br />
                char_cost={{ topologyEdges[index].charCost }}
              </template>
              <div class="edge-line">
                <span>{{ topologyEdges[index].action }}</span>
              </div>
            </el-tooltip>
          </template>
        </div>

        <div class="case-summary">
          <div>
            <span>Question</span>
            <p>{{ selectedCase.question }}</p>
          </div>
          <div>
            <span>Final Answer</span>
            <p>{{ selectedCase.finalAnswer }}</p>
          </div>
        </div>
      </article>

      <article class="surface chart-panel">
        <div class="panel-heading">
          <div>
            <p class="eyebrow">Analysis</p>
            <h2>记忆命中与状态传输趋势</h2>
          </div>
        </div>

        <div class="chart-stack">
          <div class="line-chart">
            <div class="chart-title">Memory Hits</div>
            <svg viewBox="0 0 300 140" role="img" aria-label="Memory hits line chart">
              <line x1="28" y1="116" x2="272" y2="116" class="axis" />
              <line x1="28" y1="20" x2="28" y2="116" class="axis" />
              <polyline
                v-for="series in memorySeries"
                :key="series.name"
                :points="linePoints(series.values)"
                :class="series.index === 0 ? 'series-primary' : 'series-secondary'"
              />
              <circle
                v-for="item in cases"
                :key="`${item.id}-point`"
                :cx="28 + (Number(item.turn.split('-')[1]) - 1) * 122"
                :cy="116 - item.memoryHits * 42"
                r="4"
                class="chart-dot"
              />
            </svg>
          </div>

          <div class="bar-grid">
            <div class="mini-chart">
              <div class="chart-title">Text Chars</div>
              <div class="bars">
                <span
                  v-for="item in cases"
                  :key="`${item.id}-chars`"
                  :style="{ height: `${(item.textChars / maxTextChars) * 100}%` }"
                  :title="`${item.turn}: ${item.textChars} chars`"
                />
              </div>
            </div>

            <div class="mini-chart">
              <div class="chart-title">Hidden Bytes</div>
              <div class="bars purple-bars">
                <span
                  v-for="item in cases"
                  :key="`${item.id}-bytes`"
                  :style="{ height: `${(item.hiddenBytes / maxHiddenBytes) * 100}%` }"
                  :title="`${item.turn}: ${item.hiddenBytes} MB`"
                />
              </div>
            </div>
          </div>
        </div>
      </article>
    </section>

    <section class="surface detail-panel">
      <el-tabs v-model="activeTab">
        <el-tab-pane label="连续任务组" name="groups">
          <el-table
            :data="groupedRows"
            row-key="id"
            default-expand-all
            class="dashboard-table"
            @row-click="selectCase"
          >
            <el-table-column prop="turn" label="Turn" min-width="130" />
            <el-table-column prop="question" label="Question" min-width="260" show-overflow-tooltip />
            <el-table-column prop="finalAnswer" label="Final Answer" min-width="220" show-overflow-tooltip />
            <el-table-column prop="gold" label="Gold" min-width="160" show-overflow-tooltip />
            <el-table-column label="Exact" width="92">
              <template #default="{ row }">
                <el-tag :type="row.exactMatch ? 'success' : 'danger'" effect="plain">
                  {{ row.exactMatch ? "True" : "False" }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="memoryHits" label="Memory Hits" width="118" />
            <el-table-column prop="textChars" label="Text Chars" width="112" />
            <el-table-column prop="hiddenRefs" label="Hidden Refs" width="112" />
            <el-table-column label="Hidden Bytes" width="122">
              <template #default="{ row }">{{ row.hiddenBytes.toFixed(1) }} MB</template>
            </el-table-column>
            <el-table-column label="Elapsed" width="104">
              <template #default="{ row }">{{ formatMs(row.elapsedMs) }}</template>
            </el-table-column>
          </el-table>
        </el-tab-pane>

        <el-tab-pane label="Hidden State 元数据" name="latent">
          <el-table :data="latentStates" class="dashboard-table">
            <el-table-column prop="state_id" label="state_id" min-width="140" />
            <el-table-column prop="producer" label="producer" min-width="120" />
            <el-table-column prop="content_type" label="content_type" min-width="130" />
            <el-table-column prop="model" label="model" min-width="220" show-overflow-tooltip />
            <el-table-column prop="layer" label="layer" width="90" />
            <el-table-column prop="shape" label="shape" min-width="140" />
            <el-table-column prop="pooled_shape" label="pooled_shape" min-width="130" />
            <el-table-column prop="checksum" label="checksum" min-width="120" />
          </el-table>
        </el-tab-pane>

        <el-tab-pane label="记忆命中" name="memory">
          <div class="memory-grid">
            <article
              v-for="memory in memoryHitCards"
              :key="memory.memory_id"
              class="memory-card"
            >
              <div class="memory-card-header">
                <strong>{{ memory.memory_id }}</strong>
                <span>{{ memory.source_agent }}</span>
              </div>
              <h3>{{ memory.task_topic }}</h3>
              <p>{{ memory.summary }}</p>
              <div class="tag-row">
                <el-tag v-for="tag in memory.tags" :key="tag" effect="plain">
                  {{ tag }}
                </el-tag>
              </div>
              <div class="score-row">
                <span>retrieval {{ memory.retrieval_score }}</span>
                <span>hypergraph {{ memory.hypergraph_score }}</span>
              </div>
              <code>{{ memory.matched_nodes.join(" / ") }}</code>
            </article>
          </div>
        </el-tab-pane>
      </el-tabs>
    </section>
  </main>
</template>
