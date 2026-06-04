<template>
  <el-config-provider namespace="ep">
    <main class="lab-console">
      <header class="topbar">
        <div class="brand">
          <span class="brand-mark">⌘</span>
          <h1>多智能体低成本记忆实验控制台</h1>
        </div>
        <div class="top-actions">
          <button class="ghost-button">▦ 实验记录</button>
          <button class="user-button">♟ 研究员⌄</button>
        </div>
      </header>

      <section class="panel config-panel">
        <div class="section-title">配置运行区</div>
        <div class="config-grid">
          <label class="field">
            <span>运行方式 <b>ⓘ</b></span>
            <select v-model="selectedMode">
              <option
                v-for="mode in experimentModes"
                :key="mode.value"
                :value="mode.value"
              >
                {{ mode.label }}
              </option>
            </select>
          </label>

          <label class="field">
            <span>数据集 <b>ⓘ</b></span>
            <select v-model="selectedDataset">
              <option
                v-for="dataset in availableDatasets"
                :key="dataset.value"
                :value="dataset.value"
              >
                {{ dataset.label }}
              </option>
            </select>
          </label>

          <label class="field api-field api-url">
            <span><em>API 配置</em> base_url <b>ⓘ</b></span>
            <input v-model="apiBaseUrl" aria-label="base_url" />
          </label>

          <label class="field api-field">
            <span>model</span>
            <input v-model="modelName" aria-label="model" />
          </label>

          <label class="field api-field">
            <span>api_key</span>
            <div class="password-field">
              <input :type="showApiKey ? 'text' : 'password'" v-model="apiKey" />
              <button type="button" @click="showApiKey = !showApiKey">◎</button>
            </div>
          </label>

          <div class="run-actions">
            <button class="run-button" :disabled="isRunning" @click="toggleRunning">
              ▶ {{ isRunning ? "运行中" : "开始运行" }}
            </button>
            <button class="stop-button" :class="{ enabled: isRunning }" @click="stopRun">
              ■ 停止
            </button>
          </div>
        </div>
      </section>

      <section class="dashboard-grid">
        <article class="panel timeline-panel">
          <div class="section-title">运行进度时间线</div>
          <div class="agent-header" :style="timelineGridStyle">
            <span class="clock-dot">◷</span>
            <span
              v-for="agent in activeAgents"
              :key="agent.key"
              class="agent-name"
              :style="{ color: agent.color }"
            >
              {{ agent.icon }} {{ agent.label }}
            </span>
          </div>
          <div class="timeline">
            <div
              v-for="task in timelineTasks"
              :key="task.id"
              class="task-row"
              :class="{ active: task.status === 'running' }"
              :style="timelineGridStyle"
            >
              <span class="rail-dot" :class="task.status"></span>
              <span class="task-label">{{ task.label }}</span>
              <div
                v-for="agent in activeAgents"
                :key="`${task.id}-${agent.key}`"
                class="agent-step"
                :class="task.status"
              >
                <span v-if="task.status === 'done'" class="done-dot">✓</span>
                <span v-else-if="task.status === 'running'" class="spinner"></span>
                <span v-else class="empty-dot"></span>
              </div>
              <span class="task-state">{{ task.state }}</span>
            </div>
          </div>
        </article>

        <article class="panel chart-panel">
          <div class="section-title">耗时对比（每轮） <b>ⓘ</b></div>
          <div ref="timeChartRef" class="chart"></div>
        </article>

        <article class="panel chart-panel">
          <div class="section-title">字符数对比（每轮） <b>ⓘ</b></div>
          <div ref="charChartRef" class="chart"></div>
        </article>

        <article class="panel memory-panel">
          <div class="section-title">记忆检索解释</div>
          <div class="query-card">
            <strong>当前 query</strong>
            <p>蒙娜丽莎的创作时间、作者以及现藏地点分别是什么？</p>
          </div>
          <div class="memory-hit">
            <strong>命中记忆</strong>
            <div>
              <span class="pill blue">memory-0003</span>
              <span class="pill green">相关度 0.92</span>
            </div>
          </div>
          <div class="reason-table">
            <div
              v-for="item in memoryReasons"
              :key="item.tag"
              class="reason-row"
            >
              <span class="pill" :class="item.className">{{ item.tag }}</span>
              <span>命中</span>
              <strong>{{ item.score }}</strong>
              <span>⌄</span>
            </div>
          </div>
        </article>

        <article class="panel graph-panel">
          <div class="section-title">超图局部视图</div>
          <div class="legend">
            <span v-for="item in graphLegend" :key="item.name">
              <i :style="{ borderColor: item.color }"></i>{{ item.name }}
            </span>
          </div>
          <div class="graph-canvas" aria-label="hypergraph local view">
            <svg viewBox="0 0 760 290" role="img">
              <g class="grid-lines">
                <line v-for="x in graphColumns" :key="x" :x1="x" y1="24" :x2="x" y2="264" />
              </g>
              <g class="edges">
                <path d="M58 153 L158 76 L260 92 L360 50" />
                <path d="M58 153 L158 126 L260 145 L360 110" />
                <path d="M58 153 L158 181 L260 188 L360 178" />
                <path d="M58 153 L158 232 L260 218 L360 235" />
                <path d="M360 50 L476 92 L596 92 L696 153" />
                <path d="M360 178 L476 218 L596 218 L696 153" />
                <path d="M596 92 L596 218" />
              </g>
              <circle cx="58" cy="153" r="24" class="node task" />
              <text x="58" y="158" text-anchor="middle">task-3</text>
              <GraphNode
                v-for="node in graphNodes"
                :key="node.label"
                v-bind="node"
              />
              <GraphDiamond :x="318" :y="96" label="e1" />
              <GraphDiamond :x="318" :y="184" label="e2" />
              <GraphDiamond :x="472" :y="92" label="e4" />
              <GraphDiamond :x="472" :y="218" label="e3" />
              <GraphDiamond :x="596" :y="153" label="e6" />
            </svg>
          </div>
        </article>

        <article class="panel gains-panel">
          <div class="section-title">收益统计</div>
          <div class="gain-grid">
            <div
              v-for="metric in gainMetrics"
              :key="metric.label"
              class="gain-card"
            >
              <span>{{ metric.label }}</span>
              <strong :style="{ color: metric.color }">{{ metric.value }}</strong>
              <small>{{ metric.caption }}</small>
            </div>
          </div>
          <p v-if="gainNotice" class="gain-notice">{{ gainNotice }}</p>
          <div class="updated-at">更新时间： {{ lastUpdatedAt }}  ↻</div>
        </article>
      </section>
    </main>
  </el-config-provider>
</template>

<script setup lang="ts">
import { computed, defineComponent, h, nextTick, onMounted, onUnmounted, ref, watch } from "vue";
import type { PropType } from "vue";
import * as echarts from "echarts";
import { ElMessage } from "element-plus";

type ExperimentMode =
  | "baseline_text"
  | "protocol_api"
  | "protocol_api_embedding"
  | "latent_hidden"
  | "codeact_tools";

type Dataset =
  | "hotpot_multihop"
  | "hotpot_continuous"
  | "hotpot_continuous_3turn"
  | "topiocqa_continuous"
  | "hidden_hotpot_5"
  | "hidden_continuous"
  | "codeact_20";

interface GraphNodeSpec {
  x: number;
  y: number;
  label: string;
  kind: "agent" | "evidence" | "state" | "memory" | "outcome";
}

interface RunComparison {
  mode: ExperimentMode;
  label: string;
  color: string;
  timeMs: number[];
  textChars: number[];
}

interface BackendMetricBag {
  elapsed_ms?: number | string;
  text_chars?: number | string;
  memory_hit_rate?: number | string;
  exact_match?: number | string | null;
  [key: string]: unknown;
}

interface BackendTaskResult {
  messages?: Array<{
    from?: string;
    to?: string;
    action?: string;
  }>;
  metrics?: BackendMetricBag;
  execution_result?: unknown;
  exact_match?: boolean | number | null;
}

interface BackendRunSummary {
  mode?: string;
  experiment_mode?: string;
  metrics?: BackendMetricBag;
  task_results?: BackendTaskResult[];
  memory_hit_rate?: number | string;
  exact_match?: number | string | null;
  elapsed_ms?: number | string;
}

interface AgentSpec {
  key: string;
  label: string;
  icon: string;
  color: string;
}

interface TimelineTask {
  id: number;
  label: string;
  status: "done" | "running" | "waiting";
  state: string;
}

interface GainMetric {
  label: string;
  value: string;
  caption: string;
  color: string;
}

interface BackendRunPayload {
  ok?: boolean;
  error?: string;
  result?: {
    runs?: BackendRunSummary[];
  };
  stdout_tail?: string;
}

interface BackendExperimentModeOption {
  value: ExperimentMode;
  label: string;
  script?: string;
  datasets: Dataset[];
}

interface BackendDatasetOption {
  value: Dataset;
  label: string;
  default_limit: number;
  preview_task_count: number;
  script_dataset_type?: string;
  dataset_file?: string;
}

interface BackendOptionsPayload {
  ok?: boolean;
  experiment_modes?: BackendExperimentModeOption[];
  datasets?: BackendDatasetOption[];
}

const GraphNode = defineComponent({
  name: "GraphNode",
  props: {
    x: { type: Number, required: true },
    y: { type: Number, required: true },
    label: { type: String, required: true },
    kind: {
      type: String as PropType<GraphNodeSpec["kind"]>,
      required: true,
    },
  },
  setup(props) {
    const palette = {
      agent: ["#e8f7eb", "#21a854"],
      evidence: ["#fff2d8", "#f5a623"],
      state: ["#f1e7ff", "#8e5ad7"],
      memory: ["#dff7f7", "#35b6b6"],
      outcome: ["#ffe5ef", "#ef5a8a"],
    } as const;

    return () => {
      const [fill, stroke] = palette[props.kind];
      return h("g", [
        h("rect", {
          x: props.x - 35,
          y: props.y - 17,
          width: 70,
          height: 34,
          rx: 6,
          fill,
          stroke,
          "stroke-width": 1.8,
        }),
        h(
          "text",
          {
            x: props.x,
            y: props.y + 5,
            "text-anchor": "middle",
            class: "graph-label",
          },
          props.label,
        ),
      ]);
    };
  },
});

const GraphDiamond = defineComponent({
  name: "GraphDiamond",
  props: {
    x: { type: Number, required: true },
    y: { type: Number, required: true },
    label: { type: String, required: true },
  },
  setup(props) {
    return () =>
      h("g", [
        h("polygon", {
          points: `${props.x},${props.y - 18} ${props.x + 18},${props.y} ${props.x},${props.y + 18} ${props.x - 18},${props.y}`,
          fill: "#ffffff",
          stroke: "#7e8996",
          "stroke-width": 1.8,
        }),
        h(
          "text",
          {
            x: props.x,
            y: props.y + 5,
            "text-anchor": "middle",
            class: "graph-label",
          },
          props.label,
        ),
      ]);
  },
});

const fallbackExperimentModes = [
  { value: "baseline_text", label: "baseline_text" },
  { value: "protocol_api", label: "protocol_api" },
  { value: "protocol_api_embedding", label: "protocol_api_embedding" },
  { value: "latent_hidden", label: "latent_hidden" },
  { value: "codeact_tools", label: "codeact_tools" },
] satisfies Array<{ value: ExperimentMode; label: string }>;

const fallbackDatasets = [
  {
    value: "hotpot_multihop",
    label: "HotpotQA 多跳问答",
    default_limit: 5,
    preview_task_count: 5,
    script_dataset_type: "multihop",
    dataset_file: "examples/competition_agents/data/hotpot_multihop_sample.json",
  },
  {
    value: "hotpot_continuous",
    label: "continuous HotpotQA",
    default_limit: 1,
    preview_task_count: 10,
    script_dataset_type: "continuous",
    dataset_file: "examples/competition_agents/data/hotpot_continuous_tasks.json",
  },
  {
    value: "hotpot_continuous_3turn",
    label: "continuous HotpotQA 3-turn",
    default_limit: 1,
    preview_task_count: 3,
    script_dataset_type: "continuous",
    dataset_file: "examples/competition_agents/data/hotpot_continuous_tasks_3turn.json",
  },
  {
    value: "topiocqa_continuous",
    label: "TopiOCQA 连续追问",
    default_limit: 1,
    preview_task_count: 10,
    script_dataset_type: "continuous",
    dataset_file: "examples/competition_agents/data/topiocqa_two_chains_10turns_refined.json",
  },
  {
    value: "hidden_hotpot_5",
    label: "Hidden Hotpot 样例",
    default_limit: 0,
    preview_task_count: 5,
    script_dataset_type: "samples",
    dataset_file: "",
  },
  {
    value: "hidden_continuous",
    label: "Hidden 连续任务",
    default_limit: 1,
    preview_task_count: 10,
    script_dataset_type: "continuous",
    dataset_file: "examples/competition_agents/data/hotpot_continuous_tasks.json",
  },
  {
    value: "codeact_20",
    label: "CodeAct 20 题",
    default_limit: 0,
    preview_task_count: 20,
    script_dataset_type: "codeact",
    dataset_file: "examples/competition_agents/data/codeact_tasks.json",
  },
] satisfies BackendDatasetOption[];

const fallbackDatasetMap: Record<ExperimentMode, Dataset[]> = {
  baseline_text: ["hotpot_multihop", "hotpot_continuous", "hotpot_continuous_3turn", "topiocqa_continuous"],
  protocol_api: ["hotpot_multihop", "hotpot_continuous", "hotpot_continuous_3turn", "topiocqa_continuous"],
  protocol_api_embedding: ["hotpot_multihop", "hotpot_continuous", "hotpot_continuous_3turn", "topiocqa_continuous"],
  latent_hidden: ["hidden_hotpot_5", "hidden_continuous", "topiocqa_continuous"],
  codeact_tools: ["codeact_20"],
};

const modeMeta: Record<ExperimentMode, { label: string; color: string }> = {
  baseline_text: { label: "baseline_text", color: "#2576f3" },
  protocol_api: { label: "protocol_api", color: "#ff9418" },
  protocol_api_embedding: { label: "protocol_api_embedding", color: "#8e5ad7" },
  latent_hidden: { label: "latent_hidden", color: "#18a957" },
  codeact_tools: { label: "codeact_tools", color: "#ef5a8a" },
};

const selectedMode = ref<ExperimentMode>("baseline_text");
const selectedDataset = ref<Dataset>("hotpot_continuous");
const apiBaseUrl = ref("");
const modelName = ref("");
const apiKey = ref("");
const showApiKey = ref(false);
const isRunning = ref(false);
const lastUpdatedAt = ref("尚未运行");
const lastRunResult = ref<BackendRunSummary | null>(null);
const backendOptions = ref<BackendOptionsPayload | null>(null);
const activeRunController = ref<AbortController | null>(null);
let runMessage: { close: () => void } | null = null;
let stopMessageShown = false;
const runResults = ref<Record<Dataset, Partial<Record<ExperimentMode, BackendRunSummary>>>>({
  hotpot_multihop: {},
  hotpot_continuous: {},
  hotpot_continuous_3turn: {},
  topiocqa_continuous: {},
  hidden_hotpot_5: {},
  hidden_continuous: {},
  codeact_20: {},
});
const comparisonRuns = ref<Record<Dataset, RunComparison[]>>({
  hotpot_multihop: [],
  hotpot_continuous: [],
  hotpot_continuous_3turn: [],
  topiocqa_continuous: [],
  hidden_hotpot_5: [],
  hidden_continuous: [],
  codeact_20: [],
});

const experimentModes = computed(() =>
  backendOptions.value?.experiment_modes?.map((mode) => ({
    value: mode.value,
    label: mode.label,
  })) ?? fallbackExperimentModes,
);

const datasetOptions = computed(() => backendOptions.value?.datasets ?? fallbackDatasets);

const datasetMap = computed<Record<ExperimentMode, Dataset[]>>(() => {
  const options = backendOptions.value?.experiment_modes;
  if (!options || options.length === 0) {
    return fallbackDatasetMap;
  }
  const nextMap = { ...fallbackDatasetMap };
  options.forEach((mode) => {
    nextMap[mode.value] = mode.datasets;
  });
  return nextMap;
});

const datasetInfoMap = computed<Record<Dataset, BackendDatasetOption>>(() => {
  return datasetOptions.value.reduce(
    (acc, dataset) => {
      acc[dataset.value] = dataset;
      return acc;
    },
    {} as Record<Dataset, BackendDatasetOption>,
  );
});

const availableDatasets = computed(() =>
  datasetOptions.value.filter((dataset) => datasetMap.value[selectedMode.value].includes(dataset.value)),
);

const activeComparisonRuns = computed(() => comparisonRuns.value[selectedDataset.value]);

watch(selectedMode, () => {
  if (!datasetMap.value[selectedMode.value].includes(selectedDataset.value)) {
    selectedDataset.value = datasetMap.value[selectedMode.value][0];
  }
});

const agentPalette: Record<string, Omit<AgentSpec, "key" | "label">> = {
  planner: { icon: "♙", color: "#2d7cff" },
  retriever: { icon: "⌘", color: "#1baa55" },
  reasoner: { icon: "✣", color: "#8e5ad7" },
  executor: { icon: "⚙", color: "#ef5a8a" },
  summarizer: { icon: "♘", color: "#ff9418" },
  user: { icon: "◇", color: "#64748b" },
  system: { icon: "◆", color: "#475467" },
};

const fallbackAgentPalette = ["#2d7cff", "#1baa55", "#8e5ad7", "#ef5a8a", "#ff9418", "#35b6b6"];

const defaultAgentKeysByMode: Record<ExperimentMode, string[]> = {
  baseline_text: ["planner", "retriever", "reasoner", "summarizer"],
  protocol_api: ["planner", "retriever", "reasoner", "summarizer"],
  protocol_api_embedding: ["planner", "retriever", "reasoner", "summarizer"],
  latent_hidden: ["planner", "retriever", "reasoner", "summarizer"],
  codeact_tools: ["planner", "retriever", "reasoner", "executor", "summarizer"],
};

const memoryReasons = [
  { tag: "expected_reuse", score: "0.93", className: "blue" },
  { tag: "supporting_fact", score: "0.91", className: "green" },
  { tag: "state_ref", score: "0.89", className: "purple" },
  { tag: "strategy", score: "0.86", className: "orange" },
];

const graphLegend = [
  { name: "task", color: "#2d7cff" },
  { name: "agent", color: "#21a854" },
  { name: "evidence", color: "#f5a623" },
  { name: "state_ref", color: "#8e5ad7" },
  { name: "memory", color: "#35b6b6" },
  { name: "outcome", color: "#ef5a8a" },
  { name: "超边", color: "#7e8996" },
];

const graphColumns = [115, 245, 365, 495, 625];

const graphNodes: GraphNodeSpec[] = [
  { x: 158, y: 76, label: "planner", kind: "agent" },
  { x: 158, y: 126, label: "retriever", kind: "agent" },
  { x: 158, y: 181, label: "reasoner", kind: "agent" },
  { x: 158, y: 232, label: "summarizer", kind: "agent" },
  { x: 260, y: 92, label: "ev-0012", kind: "evidence" },
  { x: 260, y: 145, label: "ev-0045", kind: "evidence" },
  { x: 260, y: 188, label: "ev-0078", kind: "evidence" },
  { x: 260, y: 218, label: "ev-0090", kind: "evidence" },
  { x: 548, y: 92, label: "memory-0003", kind: "memory" },
  { x: 548, y: 218, label: "memory-0011", kind: "memory" },
  { x: 548, y: 153, label: "", kind: "memory" },
  { x: 410, y: 92, label: "state-003", kind: "state" },
  { x: 410, y: 218, label: "state-007", kind: "state" },
  { x: 696, y: 153, label: "outcome-3", kind: "outcome" },
];

const timeChartRef = ref<HTMLDivElement | null>(null);
const charChartRef = ref<HTMLDivElement | null>(null);
let timeChart: echarts.ECharts | null = null;
let charChart: echarts.ECharts | null = null;

const FRONTEND_SERVER_URL = "http://127.0.0.1:8765";
const USE_DEVELOPER_MODE = false;
const toMetricNumber = (value: unknown) => {
  const parsed = Number(value ?? 0);
  return Number.isFinite(parsed) ? parsed : 0;
};

const makeRounds = (runs: RunComparison[]) => {
  if (runs.length === 0) {
    return [];
  }
  return Array.from(
    { length: Math.max(...runs.flatMap((run) => [run.timeMs.length, run.textChars.length])) },
    (_, index) => index + 1,
  );
};

const makeLineSeries = (runs: RunComparison[], metric: "timeMs" | "textChars") =>
  runs.map((run) => ({
    name: run.label,
    type: "line",
    smooth: true,
    symbolSize: 6,
    data: run[metric],
  }));

const activeRunForSelection = computed(
  () => runResults.value[selectedDataset.value]?.[selectedMode.value] ?? null,
);

const toAgentSpec = (key: string, index: number): AgentSpec => {
  const normalizedKey = key.trim().toLowerCase();
  const palette = agentPalette[normalizedKey];
  return {
    key: normalizedKey,
    label: normalizedKey,
    icon: palette?.icon ?? "●",
    color: palette?.color ?? fallbackAgentPalette[index % fallbackAgentPalette.length],
  };
};

const extractAgentKeys = (run: BackendRunSummary | null) => {
  const keys: string[] = [];
  const ignored = new Set(["", "user", "system", "task", "memory"]);

  run?.task_results?.forEach((task) => {
    task.messages?.forEach((message) => {
      [message.from, message.to].forEach((value) => {
        const key = String(value ?? "").trim().toLowerCase();
        if (!ignored.has(key) && !keys.includes(key)) {
          keys.push(key);
        }
      });
    });
  });

  if (run?.task_results?.some((task) => task.metrics?.tool_execution_count || "execution_result" in task)) {
    keys.push("executor");
  }

  return keys;
};

const activeAgents = computed(() => {
  const extractedKeys = extractAgentKeys(activeRunForSelection.value);
  const keys = extractedKeys.length > 0 ? extractedKeys : defaultAgentKeysByMode[selectedMode.value];
  return keys.map(toAgentSpec);
});

const timelineGridStyle = computed(() => ({
  gridTemplateColumns: `112px repeat(${activeAgents.value.length}, minmax(68px, 1fr)) 104px`,
}));

const formatElapsed = (value: unknown) => {
  const elapsed = toMetricNumber(value);
  if (!elapsed) {
    return "已完成";
  }
  return elapsed >= 1000 ? `已完成 ${(elapsed / 1000).toFixed(1)}s` : `已完成 ${Math.round(elapsed)}ms`;
};

const timelineTasks = computed<TimelineTask[]>(() => {
  const run = activeRunForSelection.value;
  const taskResults = run?.task_results ?? [];
  const previewTaskCount =
    Number(datasetInfoMap.value[selectedDataset.value]?.preview_task_count) ||
    (selectedDataset.value === "codeact_20" ? 20 : 10);

  if (taskResults.length > 0) {
    return taskResults.map((task, index) => ({
      id: index + 1,
      label: `task ${index + 1}`,
      status: "done",
      state: formatElapsed(task.metrics?.elapsed_ms),
    }));
  }

  if (isRunning.value) {
    return [
      { id: 1, label: "task 1", status: "running", state: "运行中" },
      ...Array.from({ length: Math.max(previewTaskCount - 1, 0) }, (_, index) => ({
        id: index + 2,
        label: `task ${index + 2}`,
        status: "waiting" as const,
        state: "等待中",
      })),
    ];
  }

  return Array.from({ length: previewTaskCount }, (_, index) => ({
    id: index + 1,
    label: `task ${index + 1}`,
    status: "waiting",
    state: "待运行",
  }));
});

const getRunMetricTotal = (
  run: BackendRunSummary | null | undefined,
  metric: keyof Pick<BackendMetricBag, "elapsed_ms" | "text_chars">,
) => {
  const taskValues =
    run?.task_results
      ?.map((task) => toMetricNumber(task.metrics?.[metric]))
      .filter((value) => value > 0) ?? [];

  if (taskValues.length > 0) {
    return taskValues.reduce((sum, value) => sum + value, 0);
  }

  return toMetricNumber(run?.metrics?.[metric] ?? (metric === "elapsed_ms" ? run?.elapsed_ms : 0));
};

const getRunRatioMetric = (
  run: BackendRunSummary | null | undefined,
  metric: keyof Pick<BackendMetricBag, "memory_hit_rate" | "exact_match">,
) => {
  const rawValue = run?.metrics?.[metric] ?? run?.[metric];
  if (rawValue !== undefined && rawValue !== null && rawValue !== "") {
    return toMetricNumber(rawValue);
  }

  if (metric === "exact_match") {
    const values =
      run?.task_results
        ?.map((task) => task.exact_match)
        .filter((value): value is boolean | number => value !== null && value !== undefined) ?? [];
    if (values.length > 0) {
      return values.filter(Boolean).length / values.length;
    }
  }

  return null;
};

const formatPercent = (value: number) => `${Math.abs(value * 100).toFixed(0)}%`;
const formatRatio = (value: number | null) => (value === null ? "N/A" : value.toFixed(2));

const baselineRunForSelection = computed(
  () => runResults.value[selectedDataset.value]?.baseline_text ?? null,
);

const hasBaselineComparableDataset = computed(() =>
  datasetMap.value.baseline_text.includes(selectedDataset.value),
);

const gainNotice = computed(() => {
  if (!hasBaselineComparableDataset.value) {
    return "当前数据集不属于 baseline_text 可运行范围，收益统计仅展示当前实验指标。";
  }
  if (!baselineRunForSelection.value) {
    return "请先在当前数据集运行 baseline_text，再查看相对收益。";
  }
  if (!activeRunForSelection.value) {
    return "请运行当前实验模式后查看收益统计。";
  }
  return "";
});

const gainMetrics = computed<GainMetric[]>(() => {
  const current = activeRunForSelection.value;
  const baseline = baselineRunForSelection.value;
  const currentText = getRunMetricTotal(current, "text_chars");
  const currentElapsed = getRunMetricTotal(current, "elapsed_ms");
  const baselineText = getRunMetricTotal(baseline, "text_chars");
  const baselineElapsed = getRunMetricTotal(baseline, "elapsed_ms");
  const memoryHitRate = getRunRatioMetric(current, "memory_hit_rate");
  const exactMatch = getRunRatioMetric(current, "exact_match");

  const canCompare = Boolean(current && baseline && baselineText > 0 && baselineElapsed > 0);
  const textDelta = canCompare ? (baselineText - currentText) / baselineText : null;
  const elapsedDelta = canCompare ? (baselineElapsed - currentElapsed) / baselineElapsed : null;

  return [
    {
      label: textDelta === null || textDelta >= 0 ? "字符节省" : "字符增加",
      value: textDelta === null ? "N/A" : formatPercent(textDelta),
      caption: baseline ? "较 baseline_text" : "等待 baseline_text",
      color: textDelta !== null && textDelta < 0 ? "#e15959" : "#19b45b",
    },
    {
      label: elapsedDelta === null || elapsedDelta >= 0 ? "耗时降低" : "耗时增加",
      value: elapsedDelta === null ? "N/A" : formatPercent(elapsedDelta),
      caption: baseline ? "较 baseline_text" : "等待 baseline_text",
      color: elapsedDelta !== null && elapsedDelta < 0 ? "#e15959" : "#247cff",
    },
    {
      label: "memory hit rate",
      value: formatRatio(memoryHitRate),
      caption: memoryHitRate === null ? "等待运行结果" : "当前实验",
      color: "#8a61df",
    },
    {
      label: "exact match",
      value: formatRatio(exactMatch),
      caption: exactMatch === null ? "无 gold 或未返回" : "当前实验",
      color: "#ff9418",
    },
  ];
});

const getVisibleRuns = () => {
  return activeComparisonRuns.value;
};

const getAxisMax = (runs: RunComparison[], metric: "timeMs" | "textChars") => {
  if (runs.length === 0) {
    return metric === "textChars" ? 40000 : 1000;
  }
  const maxValue = Math.max(...runs.flatMap((run) => run[metric]), 0);
  if (metric === "textChars") {
    return Math.max(40000, Math.ceil(maxValue / 5000) * 5000);
  }
  return Math.max(1000, Math.ceil(maxValue / 1000) * 1000);
};

const baseChartOption = {
  tooltip: { trigger: "axis" },
  legend: {
    top: 2,
    right: 16,
    itemWidth: 12,
    itemHeight: 8,
    type: "scroll",
    textStyle: { color: "#344054", fontSize: 11 },
  },
  grid: { left: 48, right: 24, top: 54, bottom: 38 },
  xAxis: {
    type: "category",
    name: "任务轮次",
    nameLocation: "middle",
    nameGap: 28,
    data: [],
    axisLine: { lineStyle: { color: "#d8e1ec" } },
    axisTick: { show: false },
    axisLabel: { color: "#536174" },
  },
  yAxis: {
    type: "value",
    axisLine: { show: true, lineStyle: { color: "#d8e1ec" } },
    splitLine: { lineStyle: { color: "#e8edf3" } },
    axisLabel: { color: "#536174" },
  },
};

const updateCharts = () => {
  if (!timeChartRef.value || !charChartRef.value) {
    return;
  }

  if (!timeChart) {
    timeChart = echarts.init(timeChartRef.value);
  }

  if (!charChart) {
    charChart = echarts.init(charChartRef.value);
  }

  const runs = getVisibleRuns();
  const colors = runs.map((run) => run.color);
  const rounds = makeRounds(runs);

  timeChart.setOption({
    ...baseChartOption,
    color: colors,
    xAxis: {
      ...baseChartOption.xAxis,
      data: rounds,
    },
    yAxis: {
      ...baseChartOption.yAxis,
      name: "耗时 (ms)",
      min: 0,
      max: getAxisMax(runs, "timeMs"),
    },
    series: makeLineSeries(runs, "timeMs"),
  }, { notMerge: true });

  charChart.setOption({
    ...baseChartOption,
    color: colors,
    xAxis: {
      ...baseChartOption.xAxis,
      data: rounds,
    },
    yAxis: {
      ...baseChartOption.yAxis,
      name: "字符数",
      axisLabel: {
        color: "#536174",
        formatter: (value: number) => `${Math.round(value / 1000)}K`,
      },
      min: 0,
      max: getAxisMax(runs, "textChars"),
    },
    series: makeLineSeries(runs, "textChars"),
  }, { notMerge: true });
};

const resizeCharts = () => {
  timeChart?.resize();
  charChart?.resize();
};

const closeRunMessage = () => {
  runMessage?.close();
  runMessage = null;
};

const showRunLoadingMessage = () => {
  closeRunMessage();
  runMessage = ElMessage({
    message: "正在请求实验运行...",
    type: "info",
    duration: 0,
    showClose: true,
    customClass: "run-request-message",
  });
};

const summarizeStdoutTail = (stdoutTail?: string) =>
  stdoutTail
    ?.split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(-4)
    .join(" / ");

const pickMetricSeries = (
  run: BackendRunSummary,
  metric: keyof Pick<BackendMetricBag, "elapsed_ms" | "text_chars">,
) => {
  const taskValues =
    run.task_results
      ?.map((task) => toMetricNumber(task.metrics?.[metric]))
      .filter((value) => value > 0) ?? [];

  if (taskValues.length > 0) {
    return taskValues;
  }

  const runValue = toMetricNumber(run.metrics?.[metric]);
  return runValue > 0 ? [runValue] : [0];
};

const addRunResultToComparison = (
  run: BackendRunSummary,
  mode: ExperimentMode,
  dataset: Dataset,
) => {
  const meta = modeMeta[mode];
  const datasetRuns = comparisonRuns.value[dataset];
  const existingIndex = datasetRuns.findIndex((item) => item.mode === mode);
  runResults.value[dataset][mode] = run;
  const nextRun: RunComparison = {
    mode,
    label: meta.label,
    color: meta.color,
    timeMs: pickMetricSeries(run, "elapsed_ms"),
    textChars: pickMetricSeries(run, "text_chars"),
  };

  if (existingIndex >= 0) {
    datasetRuns.splice(existingIndex, 1, nextRun);
  } else {
    datasetRuns.push(nextRun);
  }

  updateCharts();
};

const loadBackendOptions = async () => {
  const response = await fetch(`${FRONTEND_SERVER_URL}/api/options`);
  const payload = (await response.json()) as BackendOptionsPayload;
  if (!response.ok || !payload.ok) {
    throw new Error("无法读取前端运行配置。");
  }
  backendOptions.value = payload;

  const supportedModes = payload.experiment_modes?.map((mode) => mode.value) ?? [];
  if (supportedModes.length > 0 && !supportedModes.includes(selectedMode.value)) {
    selectedMode.value = supportedModes[0];
  }

  const allowedDatasets = datasetMap.value[selectedMode.value] ?? [];
  if (allowedDatasets.length > 0 && !allowedDatasets.includes(selectedDataset.value)) {
    selectedDataset.value = allowedDatasets[0];
  }
};

const toggleRunning = async () => {
  if (isRunning.value) {
    return;
  }

  const mode = selectedMode.value;
  const dataset = selectedDataset.value;
  const controller = new AbortController();

  isRunning.value = true;
  activeRunController.value = controller;
  stopMessageShown = false;
  showRunLoadingMessage();

  try {
    const response = await fetch(`${FRONTEND_SERVER_URL}/api/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        experiment_mode: mode,
        dataset,
        limit: 1,
        timeout_seconds: 600,
        ...(USE_DEVELOPER_MODE ? { developer_mode: true } : {}),
        base_url: apiBaseUrl.value,
        model: modelName.value,
        api_key: apiKey.value,
      }),
    });
    const payload = (await response.json()) as BackendRunPayload;

    if (!response.ok || !payload.ok) {
      const stdoutSummary = summarizeStdoutTail(payload.stdout_tail);
      throw new Error(
        [payload.error || `实验运行失败，HTTP ${response.status}`, stdoutSummary]
          .filter(Boolean)
          .join("："),
      );
    }

    const run = payload.result?.runs?.[0];
    if (!run) {
      throw new Error("后端返回结果缺少 result.runs[0]");
    }

    lastRunResult.value = run;
    addRunResultToComparison(run, mode, dataset);
    lastUpdatedAt.value = new Date().toLocaleString();
    closeRunMessage();
    ElMessage.success(
      `${modeMeta[mode].label} / ${datasetInfoMap.value[dataset]?.label ?? dataset} 已完成`,
    );
  } catch (error) {
    closeRunMessage();
    if (error instanceof DOMException && error.name === "AbortError") {
      if (!stopMessageShown) {
        ElMessage.warning("已停止本次前端请求");
      }
      return;
    }
    ElMessage.error(error instanceof Error ? error.message : String(error));
  } finally {
    if (activeRunController.value === controller) {
      activeRunController.value = null;
    }
    isRunning.value = false;
  }
};

const stopRun = () => {
  activeRunController.value?.abort();
  isRunning.value = false;
  closeRunMessage();
  if (activeRunController.value && !stopMessageShown) {
    stopMessageShown = true;
    ElMessage.warning("已停止本次前端请求");
  }
};

onMounted(async () => {
  try {
    await loadBackendOptions();
  } catch (error) {
    ElMessage.warning(error instanceof Error ? error.message : "读取运行配置失败，已回退到本地默认选项。");
  }
  await nextTick();
  updateCharts();
  window.addEventListener("resize", resizeCharts);
});

watch(selectedDataset, () => {
  nextTick(() => updateCharts());
});

onUnmounted(() => {
  window.removeEventListener("resize", resizeCharts);
  timeChart?.dispose();
  charChart?.dispose();
});
</script>
