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
          <button class="user-button" type="button" @click="openImportDialog">⇪ 导入结果</button>
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

          <label class="field">
            <span>模型后端 <b>ⓘ</b></span>
            <select v-model="selectedLlmBackend" :disabled="!supportsLlmBackendSwitch">
              <option value="api">API</option>
              <option value="local">本地模型</option>
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
          <div class="timeline-workspace">
            <div class="timeline-matrix">
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
                <span class="task-state-head">状态</span>
              </div>
              <div class="timeline">
                <div
                  v-for="(task, taskIndex) in timelineTasks"
                  :key="task.id"
                  class="task-row"
                  :class="{ active: task.status === 'running' }"
                  :style="timelineGridStyle"
                >
                  <span class="rail-dot" :class="task.status"></span>
                  <span class="task-label">{{ task.label }}</span>
                  <button
                    v-for="agent in activeAgents"
                    :key="`${task.id}-${agent.key}`"
                    type="button"
                    class="agent-step"
                    :class="[
                      task.status,
                      {
                        selected: isSelectedTimelineCell(taskIndex, agent.key),
                        missing: task.status === 'done' && !hasAgentOutput(taskIndex, agent.key),
                      },
                    ]"
                    :disabled="task.status === 'waiting'"
                    :aria-label="`${task.label} ${agent.label} 输出`"
                    @click="selectTimelineCell(taskIndex, agent.key, task.status)"
                  >
                    <span v-if="task.status === 'done'" class="done-dot">✓</span>
                    <span v-else-if="task.status === 'running'" class="spinner"></span>
                    <span v-else class="empty-dot"></span>
                  </button>
                  <span class="task-state">{{ task.state }}</span>
                </div>
              </div>
            </div>

            <aside class="agent-detail-panel">
              <div class="detail-head">
                <div>
                  <span>{{ activeAgentDetail.taskLabel }}</span>
                  <strong>{{ activeAgentDetail.agentLabel }}</strong>
                </div>
                <span class="detail-status" :class="activeAgentDetail.status">
                  {{ activeAgentDetail.status === "done" ? "已完成" : activeAgentDetail.status === "running" ? "运行中" : "待运行" }}
                </span>
              </div>

              <div class="detail-meta">
                <span>exact_match: {{ formatExactMatch(activeAgentDetail.exactMatch) }}</span>
                <span v-if="activeAgentDetail.stateRef">state_ref: {{ activeAgentDetail.stateRef }}</span>
              </div>

              <div class="detail-scroll">
                <section class="detail-section">
                  <h3>问题</h3>
                  <p>{{ activeAgentDetail.question || "当前轮次暂无问题文本。" }}</p>
                </section>

                <section class="detail-section">
                  <h3>Agent 输出</h3>
                  <p>{{ activeAgentDetail.displayText || (activeAgentDetail.hasOutput ? "该 Agent 本轮未返回 display_text。" : "该 Agent 本轮暂无 agent_views 输出。") }}</p>
                </section>

                <section v-if="activeAgentDetail.structuredText" class="detail-section">
                  <h3>结构化内容</h3>
                  <pre>{{ activeAgentDetail.structuredText }}</pre>
                </section>

                <section
                  v-if="activeAgentDetail.agentKey === 'retriever' && activeAgentDetail.retrievalSummaryText"
                  class="detail-section"
                >
                  <h3>检索与记忆</h3>
                  <pre>{{ activeAgentDetail.retrievalSummaryText }}</pre>
                </section>

                <section
                  v-if="activeAgentDetail.agentKey === 'reasoner' && activeAgentDetail.executionResultText"
                  class="detail-section"
                >
                  <h3>推理结果</h3>
                  <pre>{{ activeAgentDetail.executionResultText }}</pre>
                </section>

                <section
                  v-if="activeAgentDetail.agentKey === 'summarizer' && activeAgentDetail.summaryResultText"
                  class="detail-section"
                >
                  <h3>答案与记忆写回</h3>
                  <pre>{{ activeAgentDetail.summaryResultText }}</pre>
                </section>

                <section
                  v-if="activeAgentDetail.agentKey === 'summarizer' && activeAgentDetail.memoryRefineText"
                  class="detail-section"
                >
                  <h3>经验提炼</h3>
                  <pre>{{ activeAgentDetail.memoryRefineText }}</pre>
                </section>

                <section
                  v-if="activeAgentDetail.agentKey === 'executor' && activeAgentDetail.executionResultText"
                  class="detail-section"
                >
                  <h3>执行结果</h3>
                  <pre>{{ activeAgentDetail.executionResultText }}</pre>
                </section>

                <section class="detail-section">
                  <h3>通信消息</h3>
                  <div v-if="activeAgentDetail.messages.length > 0" class="message-list">
                    <div v-for="message in activeAgentDetail.messages" :key="message.id" class="message-row">
                      <span>{{ message.from }} → {{ message.to }}</span>
                      <strong>{{ message.action }}</strong>
                      <small>{{ message.charCost ?? "N/A" }}</small>
                    </div>
                  </div>
                  <p v-else>暂无与该 Agent 相关的 messages。</p>
                </section>

              </div>
            </aside>
          </div>
        </article>

        <article class="panel memory-panel">
          <div class="section-title memory-card-title">
            <span>记忆检索模块</span>
            <div class="memory-card-controls">
              <select v-model="selectedMemoryRunKey" class="memory-run-select" aria-label="选择当前页面会话运行结果">
                <option value="" disabled>当前页面没有已完成运行</option>
                <option v-for="item in sessionMemoryRuns" :key="item.key" :value="item.key">
                  {{ item.label }}
                </option>
              </select>
              <div class="memory-task-controls">
                <button type="button" aria-label="上一任务" :disabled="memoryTaskIndex <= 0" @click="moveMemoryTask(-1)">‹</button>
                <select v-model.number="memoryTaskIndex" aria-label="选择任务问题" :disabled="memoryTaskOptions.length === 0">
                  <option v-for="(task, index) in memoryTaskOptions" :key="task.task_id ?? index" :value="index">
                    任务 {{ index + 1 }}
                  </option>
                </select>
                <button type="button" aria-label="下一任务" :disabled="memoryTaskIndex >= memoryTaskOptions.length - 1" @click="moveMemoryTask(1)">›</button>
              </div>
            </div>
          </div>
          <div class="memory-body">
            <div class="memory-question-row">
              <div class="memory-question">
                <span>当前问题</span>
                <p>{{ activeMemoryPanel.question }}</p>
              </div>
              <details class="retrieval-query">
                <summary>检索词</summary>
                <p>{{ activeMemoryPanel.query }}</p>
              </details>
            </div>
            <div class="memory-summary-grid">
              <div><span>返回候选</span><strong>{{ activeMemoryPanel.hitCount }}</strong></div>
              <div><span>实际使用</span><strong>{{ activeMemoryPanel.usedCount }}</strong></div>
              <div><span>证据记忆</span><strong>{{ activeMemoryPanel.evidenceCount }}</strong></div>
              <div><span>经验记忆</span><strong>{{ activeMemoryPanel.experienceCount }}</strong></div>
            </div>
            <div class="candidate-section">
              <div class="candidate-heading"><span>候选记忆</span><small>来源、类型、命中原因、分数与使用状态</small></div>
              <div v-if="activeMemoryPanel.candidates.length" class="candidate-scroll">
                <button
                  v-for="candidate in activeMemoryPanel.candidates"
                  :key="candidate.memoryId"
                  type="button"
                  class="memory-candidate"
                  :class="{ selected: candidate.memoryId === activeMemoryPanel.selected.memoryId, used: candidate.used }"
                  @click="selectedMemoryId = candidate.memoryId"
                >
                  <span class="candidate-id">{{ candidate.memoryId }}</span>
                  <span class="candidate-source">{{ candidate.source }} · {{ candidate.typeLabel }}</span>
                  <span class="candidate-signals">{{ candidate.signals.join(" · ") }}</span>
                  <strong>{{ candidate.score }}</strong>
                  <span class="candidate-status">{{ candidate.used ? "已使用" : "未使用" }}</span>
                </button>
              </div>
              <div v-else class="candidate-empty">本轮没有命中可展示的历史记忆。</div>
            </div>
            <div class="selected-memory-detail">
              <span>当前解释</span>
              <p>{{ activeMemoryPanel.selected.explanation }}</p>
            </div>
          </div>
        </article>

        <article class="panel memory-metric-panel">
          <div class="section-title memory-metric-title">
            <span>超图记忆指标</span>
            <small>{{ activeMemoryRunLabel }}</small>
          </div>
          <div class="memory-metric-body">
            <div class="memory-metric-grid">
              <div v-for="metric in memoryPerformance.metrics" :key="metric.label" class="memory-metric-tile">
                <span>{{ metric.label }}</span>
                <strong :style="{ color: metric.color }">{{ metric.value }}</strong>
                <small>{{ metric.caption }}</small>
              </div>
            </div>
            <div class="memory-quality-summary">
              <div class="quality-heading"><span>命中质量</span><small>有效 {{ memoryPerformance.relevantHits }} · 无关 {{ memoryPerformance.irrelevantHits }}</small></div>
              <div class="quality-bar"><i :style="{ width: `${memoryPerformance.relevantPercent}%` }"></i></div>
            </div>
            <div class="memory-type-summary">
              <div v-for="item in memoryPerformance.typeRows" :key="item.label">
                <span>{{ item.label }}</span>
                <small>命中 {{ item.hits }} · 使用 {{ item.used }} · 有效 {{ item.relevant }}</small>
              </div>
            </div>
            <p class="memory-metric-note">{{ memoryPerformance.note }}</p>
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
          <div class="updated-at">更新时间： {{ activeGainUpdatedAt }}  ↻</div>
        </article>
      </section>

      <div v-if="importDialogOpen" class="import-overlay" @click.self="closeImportDialog">
        <section class="import-dialog">
          <div class="import-head">
            <div>
              <h2>导入实验结果</h2>
              <p>{{ resultFileDirectory || "D:\\qq\\low-cost-memory-integrate\\run_result" }}</p>
            </div>
            <button type="button" class="icon-button" aria-label="关闭导入结果窗口" @click="closeImportDialog">×</button>
          </div>

          <div class="import-toolbar">
            <button type="button" class="ghost-button compact" :disabled="importLoading" @click="loadResultFiles">刷新</button>
            <button type="button" class="ghost-button compact" :disabled="resultFiles.length === 0" @click="selectAllResultFiles">全选</button>
            <button type="button" class="ghost-button compact" :disabled="selectedImportFiles.length === 0" @click="selectedImportFiles = []">取消全选</button>
            <span>已选择 {{ selectedImportFiles.length }} 个文件</span>
          </div>

          <div class="import-file-list">
            <section v-for="group in importFileGroups" :key="group.folder" class="import-folder">
              <label class="import-folder-row">
                <input
                  type="checkbox"
                  :checked="isImportFolderSelected(group)"
                  @change="toggleImportFolder(group)"
                />
                <span class="import-folder-icon">▾</span>
                <strong>{{ group.label }}</strong>
                <small>{{ group.files.length }} 个文件 · {{ formatFileSize(group.size) }}</small>
              </label>
              <label v-for="file in group.files" :key="file.path" class="import-file-row">
                <input v-model="selectedImportFiles" type="checkbox" :value="file.path" />
                <span class="import-tree-line"></span>
                <span class="import-file-name">{{ file.name }}</span>
                <span class="import-file-time">{{ file.modified_label }}</span>
                <span class="import-file-size">{{ formatFileSize(file.size) }}</span>
              </label>
            </section>
            <div v-if="!importLoading && resultFiles.length === 0" class="import-empty">
              run_result 目录下暂无可导入的 JSON 结果文件。
            </div>
            <div v-if="importLoading" class="import-empty">正在读取结果文件列表...</div>
          </div>

          <div class="import-actions">
            <button type="button" class="stop-button enabled" @click="closeImportDialog">取消</button>
            <button type="button" class="run-button" :disabled="selectedImportFiles.length === 0 || importLoading" @click="importSelectedResults">
              导入所选
            </button>
          </div>
        </section>
      </div>
    </main>
  </el-config-provider>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from "vue";
import * as echarts from "echarts";
import { ElMessage } from "element-plus";

type ExperimentMode =
  | "baseline_text"
  | "protocol_api"
  | "protocol_api_embedding"
  | "latent_hidden"
  | "codeact_tools";
type LlmBackend = "api" | "local";

type Dataset =
  | "hotpot_multihop"
  | "hotpot_continuous"
  | "hotpot_continuous_3turn"
  | "topiocqa_1_10turns"
  | "topiocqa_2_10turns"
  | "hidden_hotpot_5"
  | "hidden_continuous"
  | "codeact_1_10tasks"
  | "codeact_2_10tasks";

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
  task_id?: string;
  group_id?: string | null;
  turn_id?: string | null;
  dataset_type?: string;
  question?: string;
  expected_reuse?: unknown[];
  final_answer?: string;
  gold?: string;
  messages?: Array<{
    from?: string;
    to?: string;
    action?: string;
    char_cost?: number | string | null;
  }>;
  metrics?: BackendMetricBag;
  agent_views?: Record<string, AgentViewPayload>;
  agent_outputs?: Record<string, AgentViewPayload> | Array<Record<string, unknown>>;
  retrieval_debug?: Record<string, unknown>;
  memory_hits?: unknown[];
  memory_context?: unknown[];
  memory_refine_debug?: unknown;
  latest_evidence_write?: Record<string, unknown> | null;
  latest_experience_write?: Record<string, unknown> | null;
  latest_memory_write?: Record<string, unknown> | null;
  latent_states?: Record<string, unknown>;
  state_refs?: Record<string, string>;
  execution_result?: Record<string, unknown> | null;
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
  memory_evaluation?: MemoryEvaluation;
}

interface MemorySessionRun {
  key: string;
  dataset: Dataset;
  mode: ExperimentMode;
  run: BackendRunSummary;
  label: string;
}

interface MemoryEvaluationSummary {
  expected_reuse_task_count?: number;
  memory_hit_count?: number;
  relevant_memory_hit_count?: number;
  memory_precision?: number;
  memory_complete_retrieval_rate?: number;
  correct_reuse_count?: number;
  correct_reuse_rate?: number;
}

interface MemoryEvaluation {
  summary?: MemoryEvaluationSummary;
  rows?: Array<{
    hit_details?: Array<{
      memory_id?: string;
      memory_type?: string;
      relevant?: boolean;
      used?: boolean;
    }>;
  }>;
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

interface TimelineCell {
  taskIndex: number;
  agentKey: string;
}

interface AgentViewPayload {
  display_text?: string;
  structured?: unknown;
  state_ref?: string | null;
  [key: string]: unknown;
}

interface AgentDetailView {
  agentKey: string;
  taskLabel: string;
  agentLabel: string;
  status: "done" | "running" | "waiting";
  hasTask: boolean;
  hasOutput: boolean;
  question: string;
  finalAnswer: string;
  gold: string;
  exactMatch: boolean | number | null;
  displayText: string;
  structuredText: string;
  stateRef: string;
  retrievalQuery: string;
  plannerFocusText: string;
  memoryHitIds: string[];
  memoryContextText: string;
  memoryHitsText: string;
  retrievalSummaryText: string;
  summaryResultText: string;
  memoryRefineText: string;
  messages: Array<{
    id: string;
    from: string;
    to: string;
    action: string;
    charCost: number | string | null;
  }>;
  executionResultText: string;
}

interface GainMetric {
  label: string;
  value: string;
  caption: string;
  color: string;
}

interface GainSnapshot {
  metrics: GainMetric[];
  notice: string;
  updatedAt: string;
}

interface BackendRunPayload {
  ok?: boolean;
  error?: string;
  canceled?: boolean;
  result?: {
    runs?: BackendRunSummary[];
  };
  stdout_tail?: string;
}

interface BackendResultFile {
  name: string;
  path: string;
  folder?: string;
  modified_time: number;
  modified_label: string;
  size: number;
}

interface ImportFileGroup {
  folder: string;
  label: string;
  files: BackendResultFile[];
  size: number;
}

interface BackendResultFilesPayload {
  ok?: boolean;
  directory?: string;
  files?: BackendResultFile[];
  error?: string;
}

interface BackendImportedResult {
  file: string;
  experiment_mode: ExperimentMode;
  dataset: Dataset;
  result: {
    runs?: BackendRunSummary[];
  };
}

interface BackendImportResultsPayload {
  ok?: boolean;
  results?: BackendImportedResult[];
  errors?: Array<{ file?: string; error?: string }>;
  error?: string;
}

interface BackendStopPayload {
  ok?: boolean;
  stopped?: boolean;
  message?: string;
  error?: string;
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

const fallbackExperimentModes = [
  { value: "baseline_text", label: "baseline_text" },
  { value: "protocol_api", label: "protocol_api" },
  { value: "protocol_api_embedding", label: "protocol_api_embedding" },
  { value: "latent_hidden", label: "latent_hidden" },
  { value: "codeact_tools", label: "codeact_tools" },
] satisfies Array<{ value: ExperimentMode; label: string }>;

const visibleDatasetValues: Dataset[] = [
  "topiocqa_1_10turns",
  "topiocqa_2_10turns",
  "codeact_1_10tasks",
  "codeact_2_10tasks",
];

const isVisibleDataset = (value: Dataset) => visibleDatasetValues.includes(value);

const fallbackDatasets = [
  {
    value: "topiocqa_1_10turns",
    label: "TopiOCQA 第 1 组（10 轮）",
    default_limit: 1,
    preview_task_count: 10,
    script_dataset_type: "continuous",
    dataset_file: "examples/competition_agents/data/topiocqa_1_10turns.json",
  },
  {
    value: "topiocqa_2_10turns",
    label: "TopiOCQA 第 2 组（10 轮）",
    default_limit: 1,
    preview_task_count: 10,
    script_dataset_type: "continuous",
    dataset_file: "examples/competition_agents/data/topiocqa_2_10turns.json",
  },
  {
    value: "codeact_1_10tasks",
    label: "CodeAct 第 1 组（10 题）",
    default_limit: 10,
    preview_task_count: 10,
    script_dataset_type: "codeact",
    dataset_file: "examples/competition_agents/data/codeact_1_10tasks.json",
  },
  {
    value: "codeact_2_10tasks",
    label: "CodeAct 第 2 组（10 题）",
    default_limit: 10,
    preview_task_count: 10,
    script_dataset_type: "codeact",
    dataset_file: "examples/competition_agents/data/codeact_2_10tasks.json",
  },
] satisfies BackendDatasetOption[];

const fallbackDatasetMap: Record<ExperimentMode, Dataset[]> = {
  baseline_text: [...visibleDatasetValues],
  protocol_api: [...visibleDatasetValues],
  protocol_api_embedding: [...visibleDatasetValues],
  latent_hidden: [...visibleDatasetValues],
  codeact_tools: ["codeact_1_10tasks", "codeact_2_10tasks"],
};

const modeMeta: Record<ExperimentMode, { label: string; color: string }> = {
  baseline_text: { label: "baseline_text", color: "#2576f3" },
  protocol_api: { label: "protocol_api", color: "#ff9418" },
  protocol_api_embedding: { label: "protocol_api_embedding", color: "#8e5ad7" },
  latent_hidden: { label: "latent_hidden", color: "#18a957" },
  codeact_tools: { label: "codeact_tools", color: "#ef5a8a" },
};

const selectedMode = ref<ExperimentMode>("baseline_text");
const selectedDataset = ref<Dataset>("topiocqa_1_10turns");
const selectedLlmBackend = ref<LlmBackend>("api");
const apiBaseUrl = ref("");
const modelName = ref("");
const apiKey = ref("");
const showApiKey = ref(false);
const isRunning = ref(false);
const notRunLabel = "尚未运行";
const lastRunResult = ref<BackendRunSummary | null>(null);
const backendOptions = ref<BackendOptionsPayload | null>(null);
const activeRunController = ref<AbortController | null>(null);
const selectedTimelineCell = ref<TimelineCell | null>(null);
const importDialogOpen = ref(false);
const importLoading = ref(false);
const resultFiles = ref<BackendResultFile[]>([]);
const selectedImportFiles = ref<string[]>([]);
const resultFileDirectory = ref("");
let runMessage: { close: () => void } | null = null;
let stopMessageShown = false;
const runResults = ref<Record<Dataset, Partial<Record<ExperimentMode, BackendRunSummary>>>>({
  hotpot_multihop: {},
  hotpot_continuous: {},
  hotpot_continuous_3turn: {},
  topiocqa_1_10turns: {},
  topiocqa_2_10turns: {},
  hidden_hotpot_5: {},
  hidden_continuous: {},
  codeact_1_10tasks: {},
  codeact_2_10tasks: {},
});
const comparisonRuns = ref<Record<Dataset, RunComparison[]>>({
  hotpot_multihop: [],
  hotpot_continuous: [],
  hotpot_continuous_3turn: [],
  topiocqa_1_10turns: [],
  topiocqa_2_10turns: [],
  hidden_hotpot_5: [],
  hidden_continuous: [],
  codeact_1_10tasks: [],
  codeact_2_10tasks: [],
});
const gainSnapshots = ref<Record<Dataset, Partial<Record<ExperimentMode, GainSnapshot>>>>({
  hotpot_multihop: {},
  hotpot_continuous: {},
  hotpot_continuous_3turn: {},
  topiocqa_1_10turns: {},
  topiocqa_2_10turns: {},
  hidden_hotpot_5: {},
  hidden_continuous: {},
  codeact_1_10tasks: {},
  codeact_2_10tasks: {},
});

const experimentModes = computed(() =>
  backendOptions.value?.experiment_modes?.map((mode) => ({
    value: mode.value,
    label: mode.label,
  })) ?? fallbackExperimentModes,
);

const datasetOptions = computed(() =>
  (backendOptions.value?.datasets ?? fallbackDatasets).filter((dataset) => isVisibleDataset(dataset.value)),
);

const datasetMap = computed<Record<ExperimentMode, Dataset[]>>(() => {
  const options = backendOptions.value?.experiment_modes;
  if (!options || options.length === 0) {
    return fallbackDatasetMap;
  }
  const nextMap = { ...fallbackDatasetMap };
  options.forEach((mode) => {
    nextMap[mode.value] = mode.datasets.filter(isVisibleDataset);
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

const llmBackendSwitchModes = new Set<ExperimentMode>([
  "baseline_text",
  "protocol_api",
  "protocol_api_embedding",
]);

const supportsLlmBackendSwitch = computed(() => llmBackendSwitchModes.has(selectedMode.value));

const requestLimit = computed(() => {
  if (selectedDataset.value.startsWith("codeact_")) {
    return 10;
  }
  return 1;
});

watch(selectedMode, () => {
  if (!datasetMap.value[selectedMode.value].includes(selectedDataset.value)) {
    selectedDataset.value = datasetMap.value[selectedMode.value][0];
  }
});

watch(selectedDataset, () => {
  if (datasetMap.value[selectedMode.value].includes(selectedDataset.value)) {
    return;
  }
  const compatibleMode = experimentModes.value.find((mode) =>
    datasetMap.value[mode.value]?.includes(selectedDataset.value),
  );
  if (compatibleMode) {
    selectedMode.value = compatibleMode.value;
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

const timeChartRef = ref<HTMLDivElement | null>(null);
const charChartRef = ref<HTMLDivElement | null>(null);
let timeChart: echarts.ECharts | null = null;
let charChart: echarts.ECharts | null = null;

const FRONTEND_SERVER_URL = "http://127.0.0.1:8765";
const USE_DEVELOPER_MODE = false;
const RUN_TIMEOUT_SECONDS = 3600;
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

const selectedMemoryRunKey = ref("");
const memoryTaskIndex = ref(0);
const memoryRunKey = (dataset: Dataset, mode: ExperimentMode) => `${dataset}::${mode}`;
const currentMemoryRunKey = computed(() => memoryRunKey(selectedDataset.value, selectedMode.value));
const sessionMemoryRuns = computed<MemorySessionRun[]>(() => {
  const records: MemorySessionRun[] = [];

  (Object.keys(runResults.value) as Dataset[]).forEach((dataset) => {
    const runsForDataset = runResults.value[dataset];
    (Object.keys(runsForDataset) as ExperimentMode[]).forEach((mode) => {
      const run = runsForDataset[mode];
      if (!run) {
        return;
      }
      const datasetLabel = datasetInfoMap.value[dataset]?.label ?? dataset;
      records.push({
        key: memoryRunKey(dataset, mode),
        dataset,
        mode,
        run,
        label: `${datasetLabel} / ${modeMeta[mode].label}`,
      });
    });
  });

  return records;
});
const activeMemoryRun = computed(() =>
  sessionMemoryRuns.value.find((item) => item.key === selectedMemoryRunKey.value)?.run ?? null,
);
const activeMemoryRunLabel = computed(() => {
  const active = sessionMemoryRuns.value.find((item) => item.key === selectedMemoryRunKey.value);
  return active?.label ?? "当前页面会话暂无结果";
});

const resetMemoryCardSelection = () => {
  memoryTaskIndex.value = 0;
  selectedMemoryId.value = "";
};

watch([selectedDataset, selectedMode], () => {
  selectedMemoryRunKey.value = sessionMemoryRuns.value.some((item) => item.key === currentMemoryRunKey.value)
    ? currentMemoryRunKey.value
    : "";
  resetMemoryCardSelection();
});

watch(sessionMemoryRuns, () => {
  if (sessionMemoryRuns.value.some((item) => item.key === selectedMemoryRunKey.value)) {
    return;
  }
  selectedMemoryRunKey.value = sessionMemoryRuns.value.some((item) => item.key === currentMemoryRunKey.value)
    ? currentMemoryRunKey.value
    : "";
  resetMemoryCardSelection();
});

watch(selectedMemoryRunKey, resetMemoryCardSelection);

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

  const hasVisibleExecutorView = (view: unknown) => {
    if (!view || typeof view !== "object" || Array.isArray(view)) {
      return false;
    }

    const candidate = view as {
      display_text?: unknown;
      structured?: unknown;
      state_ref?: unknown;
    };
    const displayText = String(candidate.display_text ?? "").trim();
    const stateRef = String(candidate.state_ref ?? "").trim();
    const structured = candidate.structured;
    const hasStructuredObject =
      !!structured &&
      typeof structured === "object" &&
      !Array.isArray(structured) &&
      Object.keys(structured as Record<string, unknown>).length > 0;

    return Boolean(displayText || stateRef || hasStructuredObject);
  };

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

  const hasExecutorOutput = run?.task_results?.some((task) => {
    const views = task.agent_views;
    if (
      views &&
      typeof views === "object" &&
      !Array.isArray(views) &&
      hasVisibleExecutorView((views as Record<string, unknown>).executor)
    ) {
      return true;
    }

    const outputs = task.agent_outputs;
    if (outputs && typeof outputs === "object" && !Array.isArray(outputs) && outputs.executor) {
      return true;
    }

    if (Array.isArray(outputs)) {
      return outputs.some((output) => normalizeAgentOutputKey(output.agent) === "executor");
    }

    return false;
  });

  if (hasExecutorOutput && !keys.includes("executor")) {
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

const safeStringify = (value: unknown) => {
  if (value === null || value === undefined || value === "") {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
};

const getTaskResult = (taskIndex: number) => activeRunForSelection.value?.task_results?.[taskIndex] ?? null;

const normalizeAgentOutputKey = (value: unknown) =>
  String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/agent$/, "");

const agentOutputListItemToView = (item: Record<string, unknown>): AgentViewPayload => {
  const displayText =
    item.display_text ??
    item.raw_output ??
    item.answer_candidate ??
    item.reasoning_summary ??
    item.retrieval_query ??
    "";
  return {
    content_type: item.content_type ?? "structured",
    display_text: String(displayText ?? ""),
    structured: item,
    state_ref: typeof item.state_ref === "string" ? item.state_ref : "",
  };
};

const getAgentView = (task: BackendTaskResult | null, agentKey: string) => {
  const normalizedKey = agentKey.toLowerCase();
  const views = task?.agent_views;
  if (views && typeof views === "object" && !Array.isArray(views)) {
    const directView = views[normalizedKey];
    if (directView) {
      return directView;
    }
  }

  const outputs = task?.agent_outputs;
  if (outputs && typeof outputs === "object" && !Array.isArray(outputs)) {
    const directOutput = outputs[normalizedKey];
    if (directOutput) {
      return directOutput;
    }
  }

  if (Array.isArray(outputs)) {
    const item = outputs.find((output) => normalizeAgentOutputKey(output.agent) === normalizedKey);
    if (item) {
      return agentOutputListItemToView(item);
    }
  }

  return null;
};

const hasAgentOutput = (taskIndex: number, agentKey: string) => {
  const task = getTaskResult(taskIndex);
  const view = getAgentView(task, agentKey);
  return Boolean(view?.display_text || view?.structured || view?.state_ref);
};

const isSelectedTimelineCell = (taskIndex: number, agentKey: string) =>
  selectedTimelineCell.value?.taskIndex === taskIndex &&
  selectedTimelineCell.value?.agentKey === agentKey;

const selectTimelineCell = (taskIndex: number, agentKey: string, status: TimelineTask["status"]) => {
  if (status === "waiting") {
    return;
  }
  selectedTimelineCell.value = { taskIndex, agentKey };
};

const ensureSelectedTimelineCell = () => {
  const taskResults = activeRunForSelection.value?.task_results ?? [];
  if (taskResults.length > 0) {
    const current = selectedTimelineCell.value;
    const currentStillValid =
      current &&
      current.taskIndex < taskResults.length &&
      activeAgents.value.some((agent) => agent.key === current.agentKey);
    if (!currentStillValid) {
      selectedTimelineCell.value = {
        taskIndex: 0,
        agentKey: activeAgents.value[0]?.key ?? "planner",
      };
    }
    return;
  }

  if (isRunning.value) {
    selectedTimelineCell.value = {
      taskIndex: 0,
      agentKey: activeAgents.value[0]?.key ?? "planner",
    };
    return;
  }

  selectedTimelineCell.value = null;
};

const formatExactMatch = (value: boolean | number | null) => {
  if (value === null || value === undefined) {
    return "N/A";
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  return String(value);
};

const normalizeMessagesForAgent = (task: BackendTaskResult | null, agentKey: string) => {
  const normalizedKey = agentKey.toLowerCase();
  const messages = Array.isArray(task?.messages) ? task.messages : [];
  return messages
    .map((message, index) => ({
      id: `${index}-${message.from ?? ""}-${message.to ?? ""}-${message.action ?? ""}`,
      from: message.from ?? "",
      to: message.to ?? "",
      action: message.action ?? "",
      charCost: message.char_cost ?? null,
    }))
    .filter((message) =>
      [message.from, message.to]
        .map((value) => String(value).toLowerCase())
        .includes(normalizedKey),
    );
};

const getArrayField = (value: unknown) => (Array.isArray(value) ? value : []);

const activeAgentDetail = computed<AgentDetailView>(() => {
  const selectedCell = selectedTimelineCell.value;
  const taskIndex = selectedCell?.taskIndex ?? 0;
  const agentKey = selectedCell?.agentKey ?? activeAgents.value[0]?.key ?? "planner";
  const task = getTaskResult(taskIndex);
  const agent = activeAgents.value.find((item) => item.key === agentKey);
  const agentView = getAgentView(task, agentKey);
  const retrievalDebug = task?.retrieval_debug ?? {};
  const plannerFocus = getArrayField(retrievalDebug.planner_focus);
  const memoryHitIds = getArrayField(retrievalDebug.memory_hit_ids ?? retrievalDebug.context_hit_ids).map(String);
  const retrievalSummaryText = safeStringify({
    retrieval_query: retrievalDebug.retrieval_query ?? "",
    planner_focus: plannerFocus,
    context_source: retrievalDebug.context_source ?? "",
    context_hit_ids: retrievalDebug.context_hit_ids ?? [],
    context_hits: retrievalDebug.context_hits ?? [],
    memory_hit_ids: retrievalDebug.memory_hit_ids ?? [],
    memory_context: task?.memory_context ?? [],
    memory_hits: task?.memory_hits ?? [],
  });
  const summaryResultText = safeStringify({
    final_answer: task?.final_answer ?? "",
    latest_evidence_write: task?.latest_evidence_write ?? null,
    latest_experience_write: task?.latest_experience_write ?? null,
    execution_result: task?.execution_result ?? null,
  });
  const memoryRefineText = safeStringify(task?.memory_refine_debug ?? null);

  return {
    agentKey,
    taskLabel: `task ${taskIndex + 1}`,
    agentLabel: agent?.label ?? agentKey,
    status: task ? "done" : isRunning.value ? "running" : "waiting",
    hasTask: Boolean(task),
    hasOutput: Boolean(agentView),
    question: task?.question ?? "",
    finalAnswer: task?.final_answer ?? "",
    gold: task?.gold ?? "",
    exactMatch: task?.exact_match ?? null,
    displayText: agentView?.display_text ?? "",
    structuredText: safeStringify(agentView?.structured),
    stateRef: String(agentView?.state_ref ?? task?.state_refs?.[agentKey] ?? ""),
    retrievalQuery: String(retrievalDebug.retrieval_query ?? ""),
    plannerFocusText: safeStringify(plannerFocus),
    memoryHitIds,
    memoryContextText: safeStringify(task?.memory_context ?? []),
    memoryHitsText: safeStringify(task?.memory_hits ?? []),
    retrievalSummaryText,
    summaryResultText,
    memoryRefineText,
    messages: normalizeMessagesForAgent(task, agentKey),
    executionResultText: safeStringify(task?.execution_result),
  };
});

const selectedMemoryId = ref("");
const memoryTaskOptions = computed(() => activeMemoryRun.value?.task_results ?? []);
const selectedMemoryTask = computed(() => memoryTaskOptions.value[memoryTaskIndex.value] ?? null);

const moveMemoryTask = (offset: number) => {
  const nextIndex = memoryTaskIndex.value + offset;
  if (nextIndex >= 0 && nextIndex < memoryTaskOptions.value.length) {
    memoryTaskIndex.value = nextIndex;
  }
};

const asRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};

const formatMemoryScore = (value: unknown) => {
  const score = Number(value);
  return Number.isFinite(score) ? score.toFixed(2) : "—";
};

const formatMemoryPercent = (value: unknown) => {
  const ratio = Number(value);
  return Number.isFinite(ratio) ? `${(ratio * 100).toFixed(ratio === 1 ? 0 : 2)}%` : "待评估";
};

const shortNodeLabel = (value: unknown) => {
  const text = String(value ?? "").replace(/^[^:]+:/, "").replace(/-/g, " ");
  return text.length > 18 ? `${text.slice(0, 16)}…` : text;
};

const activeMemoryPanel = computed(() => {
  const task = selectedMemoryTask.value;
  const retrievalDebug = task?.retrieval_debug ?? {};
  const hitIds = getArrayField(retrievalDebug.memory_hit_ids ?? retrievalDebug.context_hit_ids).map(String);
  const hits = getArrayField(task?.memory_hits).map(asRecord).filter((item) => Object.keys(item).length > 0);
  const context = getArrayField(task?.memory_context).map(asRecord);
  const usedIds = new Set(context.map((item) => String(item.memory_id ?? "")).filter(Boolean));
  const candidates = hits.map((record, index) => {
    const memoryId = String(record.memory_id ?? hitIds[index] ?? `memory-${index + 1}`);
    const memoryType = String(record.memory_type ?? "evidence");
    const trace = asRecord(record.trace);
    const taskMeta = asRecord(trace.task_meta);
    const sourceCase = asRecord(record.source_case);
    const source = String(taskMeta.turn_id ?? sourceCase.turn_id ?? record.source_agent ?? "历史记录");
    const matchedNodes = getArrayField(record.matched_nodes).map(String);
    const artifact = matchedNodes.find((node) => node.startsWith("artifact:"));
    const signals = [
      artifact ? `产物：${shortNodeLabel(artifact)}` : "结构命中",
      Number.isFinite(Number(record.trigger_compatibility_score))
        ? `触发 ${formatMemoryScore(record.trigger_compatibility_score)}`
        : "触发未评估",
      Number.isFinite(Number(record.topic_compatibility_score))
        ? `主题 ${formatMemoryScore(record.topic_compatibility_score)}`
        : `结构 ${formatMemoryScore(record.structural_score)}`,
    ];
    const path = asRecord(getArrayField(record.reasoning_path)[0]);
    const pathText = path.query_node
      ? `${shortNodeLabel(path.query_node)} → ${shortNodeLabel(path.hyperedge_id)} → ${memoryId}`
      : getArrayField(record.candidate_gate_reasons).map(shortNodeLabel).join(" · ") || "命中说明暂不可用";

    return {
      memoryId,
      typeLabel: memoryType === "experience" ? "经验记忆" : "证据记忆",
      source,
      score: formatMemoryScore(record.retrieval_score ?? record.hypergraph_score ?? record.embedding_score),
      used: usedIds.has(memoryId),
      signals,
      explanation: `路径：${pathText}`,
      memoryType,
    };
  });
  const selected = candidates.find((item) => item.memoryId === selectedMemoryId.value) ?? candidates[0] ?? {
    memoryId: "",
    typeLabel: "",
    source: "",
    score: "—",
    used: false,
    signals: [],
    explanation: "选择已完成任务后，这里将显示实际命中记忆的解释路径。",
    memoryType: "",
  };

  return {
    question: String(task?.question ?? "请选择已完成任务。"),
    query: String(retrievalDebug.retrieval_query ?? "本轮未返回检索词。"),
    hitCount: candidates.length,
    usedCount: candidates.filter((item) => item.used).length,
    evidenceCount: candidates.filter((item) => item.memoryType !== "experience").length,
    experienceCount: candidates.filter((item) => item.memoryType === "experience").length,
    candidates,
    selected,
  };
});

const memoryPerformance = computed(() => {
  const run = activeMemoryRun.value;
  const evaluation = run?.memory_evaluation;
  const summary = evaluation?.summary;
  const tasks = run?.task_results ?? [];
  const exactMatches = tasks.filter((task) => task.exact_match === true).length;
  const answerRate = tasks.length ? exactMatches / tasks.length : Number.NaN;
  const details = (evaluation?.rows ?? []).flatMap((row) => row.hit_details ?? []);
  const totalHits = Number(summary?.memory_hit_count ?? details.length);
  const relevantHits = Number(summary?.relevant_memory_hit_count ?? details.filter((item) => item.relevant).length);
  const typeRow = (memoryType: string, label: string) => {
    const rows = details.filter((item) => item.memory_type === memoryType);
    return {
      label,
      hits: rows.length || "—",
      used: rows.length ? rows.filter((item) => item.used).length : "—",
      relevant: rows.length ? rows.filter((item) => item.relevant).length : "—",
    };
  };

  return {
    metrics: [
      {
        label: "正确复用率",
        value: formatMemoryPercent(summary?.correct_reuse_rate),
        caption: summary ? `${summary.correct_reuse_count} / ${summary.expected_reuse_task_count}` : "运行接口待评估",
        color: "#18a45e",
      },
      {
        label: "记忆准确率",
        value: formatMemoryPercent(summary?.memory_precision),
        caption: summary ? `${summary.relevant_memory_hit_count} / ${summary.memory_hit_count}` : "运行接口待评估",
        color: "#2377f7",
      },
      {
        label: "完整产物召回",
        value: formatMemoryPercent(summary?.memory_complete_retrieval_rate),
        caption: summary ? "需要复用的任务完整覆盖" : "运行接口待评估",
        color: "#8551d1",
      },
      {
        label: "回答正确率",
        value: formatMemoryPercent(answerRate),
        caption: tasks.length ? `${exactMatches} / ${tasks.length}` : "暂无任务结果",
        color: "#df8600",
      },
    ],
    relevantHits,
    irrelevantHits: Math.max(totalHits - relevantHits, 0),
    relevantPercent: totalHits ? Math.round((relevantHits / totalHits) * 100) : 0,
    typeRows: [typeRow("evidence", "证据记忆"), typeRow("experience", "经验记忆")],
    note: summary
      ? "指标依据：来源轮次、产物类型和实际使用状态。"
      : "完成一次新运行后，接口将返回正式超图记忆评估指标。",
  };
});

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
    10;

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
const gainSnapshotForSelection = computed(
  () => gainSnapshots.value[selectedDataset.value]?.[selectedMode.value] ?? null,
);
const activeGainUpdatedAt = computed(() => gainSnapshotForSelection.value?.updatedAt ?? notRunLabel);

const buildGainNotice = (
  dataset: Dataset,
  current: BackendRunSummary | null,
  baseline: BackendRunSummary | null,
) => {
  if (!datasetMap.value.baseline_text.includes(dataset)) {
    return "当前数据集不属于 baseline_text 可运行范围，收益统计仅展示当前实验指标。";
  }
  if (!baseline) {
    return "请先在当前数据集运行 baseline_text，再查看相对收益。";
  }
  if (!current) {
    return "请运行当前实验模式后查看收益统计。";
  }
  return "";
};

const buildGainMetrics = (
  current: BackendRunSummary | null,
  baseline: BackendRunSummary | null,
): GainMetric[] => {
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
};

const gainNotice = computed(() => {
  return (
    gainSnapshotForSelection.value?.notice ??
    buildGainNotice(selectedDataset.value, activeRunForSelection.value, baselineRunForSelection.value)
  );
});

const gainMetrics = computed<GainMetric[]>(() => {
  return (
    gainSnapshotForSelection.value?.metrics ??
    buildGainMetrics(activeRunForSelection.value, baselineRunForSelection.value)
  );
});

const createGainSnapshot = (
  dataset: Dataset,
  mode: ExperimentMode,
  updatedAt: string,
): GainSnapshot => {
  const current = runResults.value[dataset]?.[mode] ?? null;
  const baseline = runResults.value[dataset]?.baseline_text ?? null;
  return {
    metrics: buildGainMetrics(current, baseline),
    notice: buildGainNotice(dataset, current, baseline),
    updatedAt,
  };
};

const refreshGainSnapshot = (dataset: Dataset, mode: ExperimentMode, updatedAt: string) => {
  gainSnapshots.value[dataset][mode] = createGainSnapshot(dataset, mode, updatedAt);
};

const refreshDatasetGainSnapshots = (dataset: Dataset, updatedAt: string) => {
  (Object.keys(runResults.value[dataset]) as ExperimentMode[]).forEach((mode) => {
    refreshGainSnapshot(dataset, mode, updatedAt);
  });
};

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
  updatedAt: string,
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

  refreshGainSnapshot(dataset, mode, updatedAt);
  if (mode === "baseline_text") {
    refreshDatasetGainSnapshots(dataset, updatedAt);
  }
  updateCharts();
};

const formatFileSize = (size: number) => {
  if (!Number.isFinite(size) || size <= 0) {
    return "0 KB";
  }
  if (size < 1024 * 1024) {
    return `${Math.max(1, Math.round(size / 1024))} KB`;
  }
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
};

const importFileGroups = computed<ImportFileGroup[]>(() => {
  const groups = new Map<string, BackendResultFile[]>();
  resultFiles.value.forEach((file) => {
    const folder = file.folder || "未分组";
    groups.set(folder, [...(groups.get(folder) ?? []), file]);
  });
  return Array.from(groups.entries()).map(([folder, files]) => ({
    folder,
    label: folder,
    files,
    size: files.reduce((total, file) => total + file.size, 0),
  }));
});

const isImportFolderSelected = (group: ImportFileGroup) =>
  group.files.length > 0 && group.files.every((file) => selectedImportFiles.value.includes(file.path));

const toggleImportFolder = (group: ImportFileGroup) => {
  const selected = new Set(selectedImportFiles.value);
  const shouldSelect = !isImportFolderSelected(group);
  group.files.forEach((file) => {
    if (shouldSelect) {
      selected.add(file.path);
    } else {
      selected.delete(file.path);
    }
  });
  selectedImportFiles.value = Array.from(selected);
};

const loadResultFiles = async () => {
  importLoading.value = true;
  try {
    const response = await fetch(`${FRONTEND_SERVER_URL}/api/result-files`);
    const payload = (await response.json()) as BackendResultFilesPayload;
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || `读取结果文件失败，HTTP ${response.status}`);
    }
    resultFiles.value = payload.files ?? [];
    resultFileDirectory.value = payload.directory ?? "";
    selectedImportFiles.value = selectedImportFiles.value.filter((file) =>
      resultFiles.value.some((item) => item.path === file),
    );
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : String(error));
  } finally {
    importLoading.value = false;
  }
};

const openImportDialog = async () => {
  importDialogOpen.value = true;
  selectedImportFiles.value = [];
  await loadResultFiles();
};

const closeImportDialog = () => {
  importDialogOpen.value = false;
};

const selectAllResultFiles = () => {
  selectedImportFiles.value = resultFiles.value.map((file) => file.path);
};

const importSelectedResults = async () => {
  if (selectedImportFiles.value.length === 0) {
    return;
  }
  importLoading.value = true;
  try {
    const response = await fetch(`${FRONTEND_SERVER_URL}/api/import-results`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ files: selectedImportFiles.value }),
    });
    const payload = (await response.json()) as BackendImportResultsPayload;
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || `导入结果失败，HTTP ${response.status}`);
    }

    const updatedAt = new Date().toLocaleString();
    const imported = payload.results ?? [];
    imported.forEach((item) => {
      const run = item.result.runs?.[0];
      if (!run) {
        return;
      }
      addRunResultToComparison(run, item.experiment_mode, item.dataset, updatedAt);
    });

    const first = imported[0];
    if (first) {
      selectedMode.value = first.experiment_mode;
      selectedDataset.value = first.dataset;
      lastRunResult.value = first.result.runs?.[0] ?? null;
    }

    const failedCount = payload.errors?.length ?? 0;
    closeImportDialog();
    ElMessage.success(`已导入 ${imported.length} 个结果${failedCount ? `，${failedCount} 个文件失败` : ""}`);
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : String(error));
  } finally {
    importLoading.value = false;
  }
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
        limit: requestLimit.value,
        timeout_seconds: RUN_TIMEOUT_SECONDS,
        ...(supportsLlmBackendSwitch.value ? { llm_backend: selectedLlmBackend.value } : {}),
        ...(USE_DEVELOPER_MODE ? { developer_mode: true } : {}),
        base_url: apiBaseUrl.value,
        model: modelName.value,
        api_key: apiKey.value,
      }),
    });
    const payload = (await response.json()) as BackendRunPayload;

    if (!response.ok || !payload.ok) {
      if (payload.canceled) {
        throw new DOMException("实验已停止", "AbortError");
      }
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
    const updatedAt = new Date().toLocaleString();
    addRunResultToComparison(run, mode, dataset, updatedAt);
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

const stopRun = async () => {
  const controller = activeRunController.value;
  if (!controller) {
    return;
  }

  closeRunMessage();

  try {
    const response = await fetch(`${FRONTEND_SERVER_URL}/api/stop`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const payload = (await response.json()) as BackendStopPayload;
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || `停止实验失败，HTTP ${response.status}`);
    }
    if (!stopMessageShown) {
      stopMessageShown = true;
      ElMessage.warning(payload.message || "已请求停止后端实验进程");
    }
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : String(error));
  } finally {
    controller.abort();
    isRunning.value = false;
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
  ensureSelectedTimelineCell();
  nextTick(() => updateCharts());
});

watch([activeRunForSelection, activeAgents, isRunning], () => {
  ensureSelectedTimelineCell();
}, { immediate: true });

onUnmounted(() => {
  window.removeEventListener("resize", resizeCharts);
  timeChart?.dispose();
  charChart?.dispose();
});
</script>
