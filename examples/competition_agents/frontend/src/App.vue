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

          <label class="field api-url">
            <span>API 配置 <b>ⓘ</b></span>
            <input v-model="apiBaseUrl" aria-label="base_url" />
            <small>base_url</small>
          </label>

          <label class="field">
            <span>&nbsp;</span>
            <input v-model="modelName" aria-label="model" />
            <small>model</small>
          </label>

          <label class="field">
            <span>&nbsp;</span>
            <div class="password-field">
              <input :type="showApiKey ? 'text' : 'password'" v-model="apiKey" />
              <button type="button" @click="showApiKey = !showApiKey">◎</button>
            </div>
            <small>api_key</small>
          </label>

          <div class="run-actions">
            <button class="run-button" @click="toggleRunning">
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
          <div class="agent-header">
            <span class="clock-dot">◷</span>
            <span
              v-for="agent in agents"
              :key="agent.key"
              class="agent-name"
              :style="{ color: agent.color }"
            >
              {{ agent.icon }} {{ agent.label }}
            </span>
          </div>
          <div class="timeline">
            <div
              v-for="task in tasks"
              :key="task.id"
              class="task-row"
              :class="{ active: task.status === 'running' }"
            >
              <span class="rail-dot" :class="task.status"></span>
              <span class="task-label">{{ task.label }}</span>
              <div
                v-for="agent in agents"
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
          <div class="updated-at">更新时间： 2025-05-30 11:24:35  ↻</div>
        </article>
      </section>
    </main>
  </el-config-provider>
</template>

<script setup lang="ts">
import { computed, defineComponent, h, nextTick, onMounted, onUnmounted, ref, watch } from "vue";
import type { PropType } from "vue";
import * as echarts from "echarts";

type ExperimentMode =
  | "baseline_text"
  | "protocol_api"
  | "protocol_api_embedding"
  | "latent_hidden";

type Dataset =
  | "competition_10"
  | "hotpot_multihop"
  | "hotpot_continuous"
  | "hidden_hotpot_5"
  | "hidden_continuous";

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

const experimentModes = [
  { value: "baseline_text", label: "text_full_context" },
  { value: "protocol_api", label: "protocol_api" },
  { value: "protocol_api_embedding", label: "api_embedding" },
  { value: "latent_hidden", label: "latent_hidden" },
] satisfies Array<{ value: ExperimentMode; label: string }>;

const datasets = [
  { value: "competition_10", label: "10 轮主流程" },
  { value: "hotpot_multihop", label: "HotpotQA 多跳问答" },
  { value: "hotpot_continuous", label: "continuous HotpotQA" },
  { value: "hidden_hotpot_5", label: "Hidden Hotpot 样例" },
  { value: "hidden_continuous", label: "Hidden 连续任务" },
] satisfies Array<{ value: Dataset; label: string }>;

const datasetMap: Record<ExperimentMode, Dataset[]> = {
  baseline_text: ["competition_10", "hotpot_multihop", "hotpot_continuous"],
  protocol_api: ["competition_10", "hotpot_multihop", "hotpot_continuous"],
  protocol_api_embedding: ["competition_10", "hotpot_continuous"],
  latent_hidden: ["hidden_hotpot_5", "hidden_continuous"],
};

const modeMeta: Record<ExperimentMode, { label: string; color: string }> = {
  baseline_text: { label: "text_full_context", color: "#2576f3" },
  protocol_api: { label: "protocol_api", color: "#ff9418" },
  protocol_api_embedding: { label: "api_embedding", color: "#8e5ad7" },
  latent_hidden: { label: "latent_hidden", color: "#18a957" },
};

const runProfiles: Record<ExperimentMode, Omit<RunComparison, "mode" | "label" | "color">> = {
  baseline_text: {
    timeMs: [820, 1350, 1960, 2700, 3480, 4260, 5080, 5920, 6680, 7350],
    textChars: [10200, 18400, 26900, 35500, 46600, 57500, 67400, 77200, 86200, 95300],
  },
  protocol_api: {
    timeMs: [520, 890, 1320, 1830, 2380, 2960, 3540, 4170, 4810, 5480],
    textChars: [6200, 10600, 15800, 21400, 27600, 33800, 40200, 46800, 53600, 60600],
  },
  protocol_api_embedding: {
    timeMs: [440, 760, 1110, 1510, 1980, 2440, 2920, 3460, 4010, 4580],
    textChars: [5200, 8900, 13200, 17800, 22800, 28100, 33600, 39300, 45200, 51400],
  },
  latent_hidden: {
    timeMs: [240, 420, 580, 790, 990, 1150, 1330, 1540, 1740, 1930],
    textChars: [1050, 1900, 2900, 4100, 5400, 6700, 7900, 9100, 10500, 11900],
  },
};

const selectedMode = ref<ExperimentMode>("baseline_text");
const selectedDataset = ref<Dataset>("hotpot_continuous");
const apiBaseUrl = ref("https://api.example.com/v1");
const modelName = ref("gpt-4o-mini");
const apiKey = ref("sk-competition-console");
const showApiKey = ref(false);
const isRunning = ref(false);
const comparisonRuns = ref<Record<Dataset, RunComparison[]>>({
  competition_10: [],
  hotpot_multihop: [],
  hotpot_continuous: [
    {
      mode: "baseline_text",
      label: modeMeta.baseline_text.label,
      color: modeMeta.baseline_text.color,
      ...runProfiles.baseline_text,
    },
    {
      mode: "latent_hidden",
      label: modeMeta.latent_hidden.label,
      color: modeMeta.latent_hidden.color,
      ...runProfiles.latent_hidden,
    },
  ],
  hidden_hotpot_5: [],
  hidden_continuous: [],
});

const availableDatasets = computed(() =>
  datasets.filter((dataset) => datasetMap[selectedMode.value].includes(dataset.value)),
);

const activeComparisonRuns = computed(() => comparisonRuns.value[selectedDataset.value]);

watch(selectedMode, () => {
  if (!datasetMap[selectedMode.value].includes(selectedDataset.value)) {
    selectedDataset.value = datasetMap[selectedMode.value][0];
  }
});

const agents = [
  { key: "planner", label: "planner", icon: "♙", color: "#2d7cff" },
  { key: "retriever", label: "retriever", icon: "⌘", color: "#1baa55" },
  { key: "reasoner", label: "reasoner", icon: "✣", color: "#8e5ad7" },
  { key: "summarizer", label: "summarizer", icon: "♘", color: "#ff9418" },
];

const tasks = [
  { id: 1, label: "task 1", status: "done", state: "已完成 12.4s" },
  { id: 2, label: "task 2", status: "done", state: "已完成 18.7s" },
  { id: 3, label: "task 3", status: "running", state: "运行中 9.3s" },
  ...Array.from({ length: 7 }, (_, index) => ({
    id: index + 4,
    label: `task ${index + 4}`,
    status: "waiting",
    state: "等待中",
  })),
];

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

const gainMetrics = [
  { label: "字符节省", value: "42%", caption: "较 text_full_context", color: "#19b45b" },
  { label: "耗时降低", value: "31%", caption: "较 text_full_context", color: "#247cff" },
  { label: "memory hit rate", value: "0.83", caption: "命中率", color: "#8a61df" },
  { label: "exact match", value: "0.67", caption: "匹配准确率", color: "#ff9418" },
];

const timeChartRef = ref<HTMLDivElement | null>(null);
const charChartRef = ref<HTMLDivElement | null>(null);
let timeChart: echarts.ECharts | null = null;
let charChart: echarts.ECharts | null = null;

const rounds = Array.from({ length: 10 }, (_, index) => index + 1);

const makeLineSeries = (runs: RunComparison[], metric: "timeMs" | "textChars") =>
  runs.map((run) => ({
    name: run.label,
    type: "line",
    smooth: true,
    symbolSize: 6,
    data: run[metric],
  }));

const getVisibleRuns = () => {
  if (activeComparisonRuns.value.length > 0) {
    return activeComparisonRuns.value;
  }

  const meta = modeMeta[selectedMode.value];
  const profile = runProfiles[selectedMode.value];
  return [
    {
      mode: selectedMode.value,
      label: `${meta.label}（待运行）`,
      color: meta.color,
      timeMs: profile.timeMs,
      textChars: profile.textChars,
    },
  ];
};

const getAxisMax = (runs: RunComparison[], metric: "timeMs" | "textChars") => {
  const maxValue = Math.max(...runs.flatMap((run) => run[metric]), 0);
  if (metric === "textChars") {
    return Math.max(100000, Math.ceil(maxValue / 10000) * 10000);
  }
  return Math.max(8000, Math.ceil(maxValue / 1000) * 1000);
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
    data: rounds,
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

  timeChart.setOption({
    ...baseChartOption,
    color: colors,
    yAxis: {
      ...baseChartOption.yAxis,
      name: "耗时 (ms)",
      min: 0,
      max: getAxisMax(runs, "timeMs"),
    },
    series: makeLineSeries(runs, "timeMs"),
  });

  charChart.setOption({
    ...baseChartOption,
    color: colors,
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
  });
};

const resizeCharts = () => {
  timeChart?.resize();
  charChart?.resize();
};

const addCurrentRunToComparison = () => {
  const meta = modeMeta[selectedMode.value];
  const profile = runProfiles[selectedMode.value];
  const datasetRuns = comparisonRuns.value[selectedDataset.value];
  const existingIndex = datasetRuns.findIndex((run) => run.mode === selectedMode.value);
  const nextRun: RunComparison = {
    mode: selectedMode.value,
    label: meta.label,
    color: meta.color,
    timeMs: profile.timeMs,
    textChars: profile.textChars,
  };

  if (existingIndex >= 0) {
    datasetRuns.splice(existingIndex, 1, nextRun);
  } else {
    datasetRuns.push(nextRun);
  }

  updateCharts();
};

const toggleRunning = () => {
  isRunning.value = true;
  addCurrentRunToComparison();
};

const stopRun = () => {
  isRunning.value = false;
};

onMounted(async () => {
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
