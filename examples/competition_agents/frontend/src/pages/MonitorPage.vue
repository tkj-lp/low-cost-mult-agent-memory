<template>
  <section class="page collab-page">
    <section class="monitor-result-bar" :class="{ locked: monitorSelectionLocked }">
      <div class="monitor-result-title">
        <strong>{{ monitorSelectionLocked ? '实时运行中' : '结果展示' }}</strong>
        <span>{{ monitorSelectionStatus }}</span>
      </div>
      <MetricsFilterBar
        class="monitor-shared-filter"
        :class="{ locked: monitorSelectionLocked }"
        :model-value="monitorFilters"
        :baseline-options="monitorBaselineOptions"
        @update:model-value="updateMonitorFilters"
        @refresh="refreshMonitorRecords"
      />
    </section>

    <div class="collab-summary compact-summary">
      <article v-for="item in summary" :key="item.label" class="collab-summary-card">
        <span :class="item.tone">{{ item.icon }}</span>
        <div>
          <em>{{ item.label }}</em>
          <strong>{{ item.value }}</strong>
        </div>
      </article>
    </div>

    <div class="collab-grid compact-collab-grid">
      <section class="collab-card agents-card">
        <header>
          <span></span>
          <h2>连续任务条</h2>
          <em>点击任务查看该轮记录</em>
        </header>
        <div v-if="tasks.length" class="task-strip-list compact-task-list">
          <button
            v-for="task in tasks"
            :key="task.id"
            class="task-strip-item"
            :class="[task.state, { active: task.id === selectedTaskId }]"
            type="button"
            @click="selectTask(task.id)"
          >
            <i></i>
            <div class="task-strip-main">
              <div class="task-strip-head">
                <strong>任务 {{ task.round }}</strong>
                <b>{{ task.status }}</b>
              </div>
              <p>{{ task.title }}</p>
              <div v-if="task.tags.length" class="chip-row">
                <span v-for="tag in task.tags" :key="tag">{{ tag }}</span>
              </div>
            </div>
            <em>{{ task.progress }}%</em>
          </button>
        </div>
        <div v-else class="runtime-empty-panel">
          <strong>暂无运行数据</strong>
          <span>启动实验后，这里会显示后端加载的真实任务队列。</span>
        </div>
      </section>

      <section class="collab-card sequence-card">
        <header>
          <span></span>
          <h2>运行时协作链路</h2>
          <em>{{ selectedTask.title }}</em>
        </header>
        <div v-if="selectedTask.messages.length" class="sequence-board flow-sequence-board">
          <div class="flow-agent-row" :style="flowGridStyle">
            <span v-for="agent in sequenceAgents" :key="agent" class="flow-agent">
              <b>{{ agent }}</b>
              <i></i>
            </span>
          </div>
          <div class="flow-message-list">
            <div
              v-for="(msg, index) in selectedTask.messages"
              :key="msg.id"
              class="flow-message-row"
              :class="{ selected: msg.id === selectedMessage?.id }"
              role="button"
              tabindex="0"
              @click="selectMessage(msg.id)"
              @keydown.enter.prevent="selectMessage(msg.id)"
              @keydown.space.prevent="selectMessage(msg.id)"
              :style="flowGridStyle"
            >
              <span v-for="agent in sequenceAgents" :key="`${msg.id}-${agent}`" class="flow-lifeline"></span>
              <div
                class="flow-edge seq-animated"
                :class="[msg.tone, { reverse: msg.reverse, running: selectedTask.state === 'running' && index === activeMessageIndex }]"
                :style="[msg.edgeStyle, { '--delay': `${index * 120}ms` }]"
              >
                <em>{{ msg.label }}</em>
                <b v-if="msg.memoryCount">{{ msg.memoryCount }} 条</b>
              </div>
            </div>
          </div>
          <div class="sequence-legend">
            <span><i class="blue"></i>结构化消息</span>
            <span><i class="green"></i>状态传递</span>
            <span><i class="orange"></i>记忆检索/复用</span>
            <span><i class="purple"></i>工具执行</span>
          </div>
        </div>
        <div v-else class="runtime-empty-panel">
          <strong>暂无协作链路</strong>
          <span>收到 Agent 消息、状态传递或记忆事件后自动绘制。</span>
        </div>
      </section>

      <section class="collab-card packet-detail-card protocol-card-live">
        <header>
          <span></span>
          <h2>协议信息</h2>
          <em>{{ selectedTask.roundText }}</em>
        </header>
        <div class="packet-tabs protocol-tabs">
          <button
            v-for="tab in protocolViewTabs"
            :key="tab.value"
            :class="{ active: protocolViewTab === tab.value }"
            type="button"
            @click="protocolViewTab = tab.value"
          >
            {{ tab.label }}
          </button>
        </div>
        <section v-if="activePacket.hasData && protocolViewTab === 'protocol'" class="protocol-detail-panel readable-protocol-panel">
          <div class="protocol-detail-head">
            <strong>{{ activePacket.title }}</strong>
            <span>{{ activePacket.badge }}</span>
          </div>
          <p>{{ activePacket.desc }}</p>
          <div class="protocol-readable-grid">
            <article v-for="item in activePacket.sections" :key="item.label">
              <b>{{ item.label }}</b>
              <span>{{ item.value }}</span>
            </article>
          </div>
          <article class="protocol-map-card">
            <b>协议映射</b>
            <span>{{ activePacket.mapping }}</span>
          </article>
          <details class="protocol-raw-detail">
            <summary>查看原始协议包</summary>
            <pre>{{ activePacket.payload }}</pre>
          </details>
        </section>
        <section v-else-if="protocolViewTab === 'transfer' && activeTransferDetail.hasData" class="protocol-detail-panel transfer-content-panel transfer-detail-panel">
          <article class="transfer-agent-detail">
            <div class="detail-head">
              <div>
                <span>{{ activeTransferDetail.taskLabel }}</span>
                <strong>{{ activeTransferDetail.agentLabel }}</strong>
              </div>
              <span class="detail-status" :class="activeTransferDetail.status">
                {{ transferStatusText(activeTransferDetail.status) }}
              </span>
            </div>

            <div class="detail-meta">
              <span>exact_match: {{ formatExactMatch(activeTransferDetail.exactMatch) }}</span>
              <span v-if="activeTransferDetail.stateRef">state_ref: {{ activeTransferDetail.stateRef }}</span>
            </div>

            <div class="detail-scroll">
              <section class="detail-section">
                <h3>&#38382;&#39064;</h3>
                <p>{{ activeTransferDetail.question || '&#24403;&#21069;&#36718;&#27425;&#26242;&#26080;&#38382;&#39064;&#25991;&#26412;&#12290;' }}</p>
              </section>

              <section class="detail-section">
                <h3>Agent &#36755;&#20986;</h3>
                <p>{{ activeTransferDetail.displayText || (activeTransferDetail.hasOutput ? '&#35813; Agent &#26412;&#36718;&#26410;&#36820;&#22238; display_text&#12290;' : '&#35813; Agent &#26412;&#36718;&#26242;&#26080; agent_views &#36755;&#20986;&#12290;') }}</p>
              </section>

              <section v-if="activeTransferDetail.structuredText" class="detail-section">
                <h3>&#32467;&#26500;&#21270;&#20869;&#23481;</h3>
                <pre>{{ activeTransferDetail.structuredText }}</pre>
              </section>

              <section v-if="activeTransferDetail.agentKey === 'retriever' && activeTransferDetail.retrievalSummaryText" class="detail-section">
                <h3>&#26816;&#32034;&#19982;&#35760;&#24518;</h3>
                <pre>{{ activeTransferDetail.retrievalSummaryText }}</pre>
              </section>

              <section v-if="activeTransferDetail.agentKey === 'reasoner' && activeTransferDetail.executionResultText" class="detail-section">
                <h3>&#25512;&#29702;&#32467;&#26524;</h3>
                <pre>{{ activeTransferDetail.executionResultText }}</pre>
              </section>

              <section v-if="activeTransferDetail.agentKey === 'summarizer' && activeTransferDetail.summaryResultText" class="detail-section">
                <h3>&#31572;&#26696;&#19982;&#35760;&#24518;&#20889;&#22238;</h3>
                <pre>{{ activeTransferDetail.summaryResultText }}</pre>
              </section>

              <section v-if="activeTransferDetail.agentKey === 'summarizer' && activeTransferDetail.memoryRefineText" class="detail-section">
                <h3>&#32463;&#39564;&#25552;&#28860;</h3>
                <pre>{{ activeTransferDetail.memoryRefineText }}</pre>
              </section>

              <section v-if="activeTransferDetail.agentKey === 'executor' && activeTransferDetail.executionResultText" class="detail-section">
                <h3>&#25191;&#34892;&#32467;&#26524;</h3>
                <pre>{{ activeTransferDetail.executionResultText }}</pre>
              </section>
            </div>
          </article>
        </section>
        <div v-else class="runtime-empty-panel">
          <strong>{{ activePacket.title }}</strong>
          <span>{{ activePacket.desc }}</span>
        </div>
      </section>

      <section class="collab-card event-log-card compact-event-card">
        <header>
          <span></span>
          <h2>实时日志</h2>
          <em>当前查看：{{ selectedTask.roundText }}</em>
        </header>
        <div v-if="selectedTask.events.length" class="collab-table-wrap">
          <table>
            <thead>
              <tr>
                <th>时间</th>
                <th>发送方</th>
                <th>接收方</th>
                <th>事件类型</th>
                <th>动作</th>
                <th>状态引用</th>
                <th>记忆引用</th>
                <th>文本开销</th>
                <th>耗时</th>
                <th>结果</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in selectedTask.events" :key="`${selectedTask.id}-${row.time}-${row.action}-${row.type}`">
                <td>{{ row.time }}</td>
                <td>{{ row.from }}</td>
                <td>{{ row.to }}</td>
                <td><b :class="row.tone">{{ row.type }}</b></td>
                <td>{{ row.action }}</td>
                <td><span>{{ row.state }}</span></td>
                <td><span>{{ row.memory }}</span></td>
                <td>{{ row.tokens }}</td>
                <td>{{ row.cost }}</td>
                <td>{{ row.result }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-else class="runtime-empty-panel">
          <strong>暂无日志</strong>
          <span>运行事件写入后，这里只展示后端记录的真实事件。</span>
        </div>
      </section>
    </div>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import {
  collaborationModes,
  datasets,
  defaultExperimentConfig,
  labelOf,
  memoryModes,
  normalizeExperimentConfig,
  retrievalOptionsFor,
  selectedCollaboration
} from '../experimentOptions'
import MetricsFilterBar from '../components/metrics/MetricsFilterBar.vue'

const CONFIG_STORAGE_KEY = 'competition-agent-console-config'
const FILTER_SYNC_EVENT = 'competition-agent-filter-sync'
const RUNNER_EVENTS_URL = 'http://127.0.0.1:8765/api/runner/events'
const RUNNER_RESULT_URL = 'http://127.0.0.1:8765/api/runner/result'
const RUNNER_STATUS_URL = 'http://127.0.0.1:8765/api/runner/status'
const RUNNER_METRICS_URL = 'http://127.0.0.1:8765/api/runner/metrics'

const props = defineProps({
  importedPayloads: {
    type: Array,
    default: () => []
  }
})

const selectedTaskId = ref('')
const selectedPacketTab = ref('message')
const protocolViewTab = ref('protocol')
const selectedMessageId = ref('')
const tick = ref(0)
const savedConfig = ref(readSavedConfig())
const runtimeEvents = ref([])
const latestRunResult = ref(null)
const runnerRunning = ref(false)
const monitorRecords = ref([])
const selectedMonitorRecordId = ref('')
const monitorLoadError = ref('')
const monitorFilters = ref(withFilterDefaults(readSavedConfig()))
let timer = null
let eventTimer = null

const protocolViewTabs = [
  { label: '\u534f\u8bae\u5b57\u6bb5', value: 'protocol' },
  { label: '\u4f20\u9012\u5185\u5bb9', value: 'transfer' }
]

const config = computed(() => normalizeExperimentConfig(savedConfig.value))
const collaboration = computed(() => selectedCollaboration(config.value))
const memoryLabel = computed(() => labelOf(memoryModes, config.value.memoryMode))
const retrievalLabel = computed(() => labelOf(retrievalOptionsFor(config.value.memoryMode), config.value.retrievalMode))
const monitorSelectionLocked = computed(() => runnerRunning.value)
const selectedMonitorRecord = computed(() =>
  monitorRecords.value.find((record) => record.recordId === selectedMonitorRecordId.value) ?? null
)
const importedRunPayload = computed(() => importedPayloadForCurrentConfig(props.importedPayloads, monitorFilters.value))
const selectedArchivePayload = computed(() => importedRunPayload.value ?? selectedMonitorRecord.value?.payload ?? null)
const effectiveRunResult = computed(() =>
  monitorSelectionLocked.value ? latestRunResult.value : selectedArchivePayload.value ?? latestRunResult.value
)
const effectiveRuntimeEvents = computed(() =>
  monitorSelectionLocked.value
    ? runtimeEvents.value
    : selectedArchivePayload.value
      ? eventsFromImportedPayload(selectedArchivePayload.value)
      : runtimeEvents.value
)
const hasRuntimeEvents = computed(() => effectiveRuntimeEvents.value.length > 0 || Boolean(latestPrimaryRun(effectiveRunResult.value)))
const monitorRecordOptions = computed(() =>
  monitorRecords.value
    .filter((record) =>
      record.dataset === monitorFilters.value.dataset &&
      record.collaborationMode === monitorFilters.value.collaborationMode &&
      record.memoryMode === monitorFilters.value.memoryMode &&
      record.retrievalMode === monitorFilters.value.retrievalMode
    )
    .map((record) => ({ label: monitorRecordLabel(record), value: record.recordId }))
)
const monitorBaselineOptions = computed(() =>
  monitorRecords.value.map((record) => ({
    label: monitorRecordLabel(record),
    value: record.recordId
  }))
)
const monitorSelectionStatus = computed(() => {
  if (monitorSelectionLocked.value) return '运行时锁定为实时协作链路，暂停手动切换展示结果。'
  if (monitorLoadError.value) return monitorLoadError.value
  if (selectedMonitorRecord.value) return `当前展示：${monitorRecordLabel(selectedMonitorRecord.value)}`
  if (selectedArchivePayload.value) return '当前展示导入或最近的历史结果。'
  return '未选择历史结果时展示最近一次运行快照。'
})
const runMeta = computed(() => effectiveRuntimeEvents.value.find((event) => event.event_type === 'run_start') ?? {})
const taskQueue = computed(() => {
  const queue = effectiveRuntimeEvents.value.find((event) => event.event_type === 'task_queue')
  return Array.isArray(queue?.tasks) ? queue.tasks : []
})
const runModeLabel = computed(() => {
  if (hasRuntimeEvents.value && runMeta.value.experiment_mode) {
    return `${modeText(runMeta.value.experiment_mode)} / ${runMeta.value.memory_organization || '--'} / ${runMeta.value.memory_retrieval_mode || '--'}`
  }
  return `${collaboration.value.label} / ${memoryLabel.value} / ${retrievalLabel.value}`
})

const tasks = computed(() => (hasRuntimeEvents.value ? buildTasksFromRuntimeEvents(effectiveRuntimeEvents.value, effectiveRunResult.value) : []))
const selectedTask = computed(() => tasks.value.find((task) => task.id === selectedTaskId.value) ?? tasks.value[0] ?? emptyTask())
const sequenceAgents = computed(() => selectedTask.value.agents || [])
const flowGridStyle = computed(() => ({
  gridTemplateColumns: `repeat(${Math.max(sequenceAgents.value.length, 1)}, minmax(0, 1fr))`
}))
const activeMessageIndex = computed(() => {
  const count = selectedTask.value.messages.length || 1
  return tick.value % count
})
const selectedMessage = computed(() => selectedTask.value.messages.find((message) => message.id === selectedMessageId.value) ?? selectedTask.value.messages[0] ?? null)
const activePacket = computed(() => packetForSelectedMessage(selectedTask.value, selectedMessage.value))
const activeTransferDetail = computed(() => transferAgentDetailForSelectedMessage(selectedTask.value, selectedMessage.value))

const summary = computed(() => [
  { label: '运行模式', value: runModeLabel.value, icon: '协', tone: 'green' },
  { label: '当前轮次', value: tasks.value.length ? `第 ${selectedTask.value.round} 轮 / 共 ${tasks.value.length} 轮` : '--', icon: '轮', tone: 'blue' },
  { label: '消息次数', value: `${selectedTask.value.metrics.messages} 次`, icon: '信', tone: 'purple' },
  { label: '文本开销', value: selectedTask.value.metrics.textCost, icon: '文', tone: 'blue' },
  { label: '状态传递', value: `${selectedTask.value.metrics.stateTransfers} 次`, icon: '态', tone: 'green' },
  { label: '记忆命中', value: `${selectedTask.value.metrics.memoryHits} 次`, icon: '记', tone: 'orange' },
  { label: '当前角色', value: selectedTask.value.currentAgent, icon: '角', tone: 'purple' }
])

watch(tasks, (nextTasks) => {
  if (!nextTasks.length) {
    selectedTaskId.value = ''
    return
  }
  if (!nextTasks.some((task) => task.id === selectedTaskId.value)) {
    const runningTask = nextTasks.find((task) => task.state === 'running')
    selectedTaskId.value = runningTask?.id ?? nextTasks[0].id
  }
})

onMounted(() => {
  refreshSavedConfig()
  refreshMonitorRecords()
  refreshRuntimeSnapshot()
  window.addEventListener('storage', refreshSavedConfig)
  window.addEventListener(FILTER_SYNC_EVENT, refreshSavedConfig)
  timer = window.setInterval(() => {
    tick.value += 1
  }, 900)
  eventTimer = window.setInterval(refreshRuntimeSnapshot, 1200)
})

onBeforeUnmount(() => {
  window.removeEventListener('storage', refreshSavedConfig)
  window.removeEventListener(FILTER_SYNC_EVENT, refreshSavedConfig)
  if (timer) window.clearInterval(timer)
  if (eventTimer) window.clearInterval(eventTimer)
})

function selectTask(taskId) {
  selectedTaskId.value = taskId
  selectedPacketTab.value = 'message'
  protocolViewTab.value = 'protocol'
  selectedMessageId.value = ''
}

function selectMessage(messageId) {
  selectedMessageId.value = messageId
  selectedPacketTab.value = 'message'
}

function readSavedConfig() {
  if (typeof window === 'undefined') return defaultExperimentConfig()
  try {
    const raw = window.localStorage.getItem(CONFIG_STORAGE_KEY)
    return raw ? normalizeExperimentConfig({ ...defaultExperimentConfig(), ...JSON.parse(raw) }) : defaultExperimentConfig()
  } catch {
    return defaultExperimentConfig()
  }
}

function refreshSavedConfig() {
  savedConfig.value = readSavedConfig()
  if (!monitorSelectionLocked.value) {
    monitorFilters.value = withFilterDefaults(savedConfig.value)
    selectedMonitorRecordId.value = monitorFilters.value.baselineRecordId || ''
    syncSelectedMonitorRecord()
  }
}

function withFilterDefaults(value) {
  return {
    ...normalizeExperimentConfig({ ...defaultExperimentConfig(), ...value }),
    baselineMode: value?.baselineMode ?? 'none',
    baselineRecordId: value?.baselineRecordId ?? ''
  }
}

function updateMonitorFilters(nextFilters) {
  if (monitorSelectionLocked.value) return
  const normalized = normalizeExperimentConfig({
    ...defaultExperimentConfig(),
    ...monitorFilters.value,
    ...nextFilters
  })
  monitorFilters.value = {
    ...normalized,
    baselineMode: nextFilters.baselineMode ?? monitorFilters.value.baselineMode ?? 'none',
    baselineRecordId: nextFilters.baselineRecordId ?? monitorFilters.value.baselineRecordId ?? ''
  }
  if (monitorFilters.value.baselineMode !== 'compare') {
    monitorFilters.value.baselineRecordId = ''
  }
  persistSharedFilters(monitorFilters.value)
  selectedMonitorRecordId.value = monitorFilters.value.baselineRecordId || ''
  syncSelectedMonitorRecord()
}

function persistSharedFilters(value) {
  try {
    window.localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(value))
    window.dispatchEvent(new CustomEvent(FILTER_SYNC_EVENT))
  } catch {
    // Ignore storage errors; the page-local selection still updates.
  }
}

async function refreshRuntimeSnapshot() {
  await refreshRunnerStatus()
  await Promise.all([refreshRuntimeEvents(), refreshRunResult()])
}

async function refreshRunnerStatus() {
  try {
    const response = await fetch(`${RUNNER_STATUS_URL}?t=${Date.now()}`, { cache: 'no-store' })
    if (!response.ok) {
      runnerRunning.value = false
      return
    }
    const payload = await response.json()
    runnerRunning.value = Boolean(payload.running || payload.status === 'running')
  } catch {
    runnerRunning.value = false
  }
}

async function refreshMonitorRecords() {
  if (monitorSelectionLocked.value) return
  try {
    const response = await fetch(`${RUNNER_METRICS_URL}?t=${Date.now()}`, { cache: 'no-store' })
    if (!response.ok) throw new Error('历史结果服务未返回有效数据')
    const payload = await response.json()
    monitorRecords.value = normalizeMonitorRecords(Array.isArray(payload.records) ? payload.records : [])
    monitorLoadError.value = monitorRecords.value.length ? '' : '暂无历史结果，请先完成一次运行。'
    syncSelectedMonitorRecord()
  } catch {
    monitorRecords.value = []
    monitorLoadError.value = '暂无历史结果，请先启动本地运行服务或完成一次运行。'
    syncSelectedMonitorRecord()
  }
}

function normalizeMonitorRecords(records) {
  return records
    .filter((record) => record?.payload)
    .map((record, index) => ({
      ...record,
      recordId: String(record.recordId || `${record.source || 'record'}-${index}`),
      dataset: datasetKey(record.dataset || record.payload?.dataset_file || record.source || '')
    }))
}

function syncSelectedMonitorRecord() {
  const options = monitorRecordOptions.value
  if (!options.length) {
    selectedMonitorRecordId.value = ''
    return
  }
  if (!options.some((option) => option.value === selectedMonitorRecordId.value)) {
    selectedMonitorRecordId.value = options[0].value
  }
  monitorFilters.value = {
    ...monitorFilters.value,
    baselineRecordId: selectedMonitorRecordId.value
  }
}

function applySelectedMonitorRecord() {
  const record = selectedMonitorRecord.value
  if (!record) return
  monitorFilters.value = normalizeExperimentConfig({
    ...monitorFilters.value,
    dataset: record.dataset,
    collaborationMode: record.collaborationMode,
    memoryMode: record.memoryMode,
    retrievalMode: record.retrievalMode
  })
}

function updateMonitorMemoryMode() {
  const options = retrievalOptionsFor(monitorFilters.value.memoryMode)
  if (!options.some((item) => item.value === monitorFilters.value.retrievalMode)) {
    monitorFilters.value = {
      ...monitorFilters.value,
      retrievalMode: options[0]?.value || monitorFilters.value.retrievalMode
    }
  }
  syncSelectedMonitorRecord()
}

async function refreshRuntimeEvents() {
  try {
    const response = await fetch(`${RUNNER_EVENTS_URL}?t=${Date.now()}`, { cache: 'no-store' })
    if (!response.ok) {
      runtimeEvents.value = []
      return
    }
    const payload = await response.json()
    runtimeEvents.value = Array.isArray(payload.events) ? payload.events : []
  } catch {
    runtimeEvents.value = []
  }
}

async function refreshRunResult() {
  try {
    const response = await fetch(`${RUNNER_RESULT_URL}?t=${Date.now()}`, { cache: 'no-store' })
    if (!response.ok) {
      latestRunResult.value = null
      return
    }
    const payload = await response.json()
    latestRunResult.value = payload?.payload || null
  } catch {
    latestRunResult.value = null
  }
}

function latestPrimaryRun(payload) {
  const runs = Array.isArray(payload?.runs) ? payload.runs : []
  return runs[0] || null
}

function importedPayloadForCurrentConfig(items, currentConfig) {
  const payloads = Array.isArray(items) ? items.map((item) => item?.payload).filter(Boolean) : []
  if (!payloads.length) return null
  return payloads.find((payload) => payloadMatchesCurrentConfig(payload, currentConfig)) ?? null
}

function datasetKey(file) {
  const text = String(file || '')
  if (text.includes('topiocqa_1_10turns')) return 'topiocqa-1'
  if (text.includes('topiocqa_2_10turns')) return 'topiocqa-2'
  if (text.includes('codeact_1_10tasks')) return 'codeact-1'
  if (text.includes('codeact_2_10tasks')) return 'codeact-2'
  return text
}

function payloadMatchesCurrentConfig(payload, currentConfig) {
  const run = latestPrimaryRun(payload) || {}
  const mode = payload?.experiment_mode || run.experiment_mode || run.mode
  const memory = payload?.memory_organization || run.memory_organization || payload?.memory_retrieval_mode || run.memory_retrieval_mode
  const retrieval = payload?.memory_retrieval_mode || run.memory_retrieval_mode
  return (
    datasetKey(payload?.dataset_file || run.dataset_file) === currentConfig.dataset &&
    collaborationKey(mode) === currentConfig.collaborationMode &&
    memoryKey(memory) === currentConfig.memoryMode &&
    String(retrieval || '') === currentConfig.retrievalMode
  )
}

function collaborationKey(mode) {
  return {
    baseline_text: 'plain',
    autogen_text: 'autogen',
    camel_text: 'camel',
    autogen_camel_text: 'autogen',
    autoform_text: 'autoform',
    llmlingua_text: 'llmlingua',
    text: 'plain',
    protocol_api: 'structured',
    protocol: 'structured',
    protocol_api_embedding: 'protocol-vector',
    hidden_api: 'latent',
    hidden: 'latent',
    latent_hidden: 'latent'
  }[String(mode || '').toLowerCase()] || 'plain'
}

function memoryKey(value) {
  return ['hypergraph', 'fusion'].includes(String(value || '').toLowerCase()) ? 'hypergraph' : 'text'
}

function monitorRecordLabel(record) {
  const dataset = labelOf(datasets, record.dataset, record.dataset || '-')
  const collaboration = labelOf(collaborationModes, record.collaborationMode, record.collaborationMode || '-')
  const memory = labelOf(memoryModes, record.memoryMode, record.memoryMode || '-')
  const retrieval = labelOf(retrievalOptionsFor(record.memoryMode), record.retrievalMode, record.retrievalMode || '-')
  const time = record.archivedAt ? formatArchiveTime(record.archivedAt) : record.source || 'latest'
  return `${dataset} / ${collaboration} / ${memory} / ${retrieval} / ${time}`
}

function formatArchiveTime(value) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return String(value)
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  const hour = String(date.getHours()).padStart(2, '0')
  const minute = String(date.getMinutes()).padStart(2, '0')
  return `${month}-${day} ${hour}:${minute}`
}

function eventsFromImportedPayload(payload) {
  const primaryRun = latestPrimaryRun(payload) || {}
  const taskResults = Array.isArray(primaryRun.task_results) ? primaryRun.task_results : []
  const timestamp = payload?.archived_at || new Date().toISOString()
  const events = [
    {
      event_type: 'run_start',
      timestamp,
      experiment_mode: payload?.experiment_mode || primaryRun.experiment_mode || primaryRun.mode,
      memory_organization: payload?.memory_organization || primaryRun.memory_organization,
      memory_retrieval_mode: payload?.memory_retrieval_mode || primaryRun.memory_retrieval_mode,
      dataset_file: payload?.dataset_file || primaryRun.dataset_file,
      task_count: taskResults.length || payload?.task_count || primaryRun.task_count || primaryRun.tasks
    },
    {
      event_type: 'task_queue',
      timestamp,
      tasks: taskResults.map((task, index) => ({
        task_index: index + 1,
        task_id: task.task_id || `task-${index + 1}`,
        task_title: task.task_title || task.question || task.task || `任务 ${index + 1}`,
        expected_reuse: task.expected_reuse || []
      }))
    }
  ]

  for (const [index, task] of taskResults.entries()) {
    const taskIndex = index + 1
    const taskId = task.task_id || `task-${taskIndex}`
    const taskTitle = task.task_title || task.question || task.task || `任务 ${taskIndex}`
    const metrics = task.metrics || {}
    const memoryRefs = normalizeList(task.memory_refs || task.memory_hit_ids || task.memory_ids)
    const stateRefs = normalizeList(task.state_refs)
    const memoryHits = importedMemoryHits(task)
    const contextHits = importedContextHits(task)
    const retrievalPolicy = importedRetrievalPolicy(task, memoryHits)
    events.push({
      event_type: 'task_start',
      timestamp,
      task_index: taskIndex,
      task_id: taskId,
      task_title: taskTitle,
      from_agent: 'runtime',
      to_agent: 'planner',
      action_type: 'start_task',
      return_result: { task_title: taskTitle }
    })
    events.push({
      event_type: 'agent_message',
      timestamp,
      task_index: taskIndex,
      task_id: taskId,
      task_title: taskTitle,
      from_agent: 'planner',
      to_agent: 'retriever',
      action_type: 'retrieve',
      state_refs: stateRefs,
      text_chars: Number(metrics.text_chars || 0),
      payload_summary: task.agent_views?.planner?.display_text || task.question || taskTitle,
      return_result: {
        summary: task.agent_views?.planner?.display_text || '',
        retrieval_policy: retrievalPolicy
      }
    })
    if (retrievalPolicy.need_memory) {
      pushImportedMemoryEvents(events, { timestamp, taskIndex, taskId, taskTitle, stateRefs, memoryRefs, memoryHits, task })
    }
    pushImportedKnowledgeEvents(events, { timestamp, taskIndex, taskId, taskTitle, stateRefs, contextHits, task })
    events.push({
      event_type: 'agent_message',
      timestamp,
      task_index: taskIndex,
      task_id: taskId,
      task_title: taskTitle,
      from_agent: 'retriever',
      to_agent: 'reasoner',
      action_type: 'reason',
      memory_refs: memoryRefs,
      state_refs: stateRefs,
      text_chars: Number(metrics.text_chars || 0),
      payload_summary: task.agent_views?.retriever?.display_text || task.memory_context || '',
      return_result: {
        summary: task.agent_views?.retriever?.display_text || '',
        memory_hit_count: memoryHits.length || memoryRefs.length,
        retrieval_query: task.retrieval_debug?.retrieval_query || '',
        context_hits: contextHits,
        memory_hits: memoryHits
      }
    })
    if (shouldShowExecutorForImportedTask(payload, primaryRun, task)) {
      events.push({
        event_type: 'agent_message',
        timestamp,
        task_index: taskIndex,
        task_id: taskId,
        task_title: taskTitle,
        from_agent: 'reasoner',
        to_agent: 'executor',
        action_type: 'execute',
        return_result: task.execution_result
      })
      events.push({
        event_type: 'agent_message',
        timestamp,
        task_index: taskIndex,
        task_id: taskId,
        task_title: taskTitle,
        from_agent: 'executor',
        to_agent: 'reasoner',
        action_type: 'execute_result',
        return_result: task.execution_result
      })
    }
    events.push({
      event_type: 'agent_message',
      timestamp,
      task_index: taskIndex,
      task_id: taskId,
      task_title: taskTitle,
      from_agent: 'reasoner',
      to_agent: 'summarizer',
      action_type: 'summarize',
      memory_refs: memoryRefs,
      state_refs: stateRefs,
      text_chars: Number(metrics.text_chars || 0),
      payload_summary: task.agent_views?.reasoner?.display_text || task.final_answer || '',
      return_result: { summary: task.final_answer || '', final_answer: task.final_answer || '' }
    })
    events.push({
      event_type: 'task_done',
      timestamp,
      task_index: taskIndex,
      task_id: taskId,
      task_title: taskTitle,
      from_agent: 'summarizer',
      to_agent: 'dashboard',
      action_type: 'done',
      memory_refs: memoryRefs,
      state_refs: stateRefs,
      text_chars: Number(metrics.text_chars || 0),
      elapsed_ms: Number(metrics.elapsed_ms || task.elapsed_ms || 0),
      return_result: {
        final_answer: task.final_answer || task.answer || '',
        exact_match: task.exact_match,
        memory_hit_count: Number(task.memory_hit_count || memoryRefs.length || metrics.memory_hit_count || 0)
      }
    })
  }
  events.push({ event_type: 'run_done', timestamp })
  return events
}

function normalizeList(value) {
  if (Array.isArray(value)) return value.filter(Boolean).map(String)
  if (value && typeof value === 'object') return Object.values(value).filter(Boolean).map(String)
  return value ? [String(value)] : []
}

function shouldShowExecutorForImportedTask(payload, run, task) {
  const datasetType = String(payload?.dataset_type || run?.dataset_type || task?.dataset_type || '').toLowerCase()
  const execution = task?.execution_result || {}
  const mode = String(execution.execution_mode || '').toLowerCase()
  return datasetType === 'codeact' || mode.includes('sandbox') || Boolean(execution.sandbox_status)
}

function importedRetrievalPolicy(task, memoryHits) {
  const policy =
    task.agent_views?.planner?.structured?.retrieval_policy ||
    task.agent_views?.retriever?.structured?.retrieval_policy ||
    task.retrieval_debug?.retrieval_policy
  if (policy && typeof policy === 'object') return policy
  const expectedReuse = normalizeList(task.expected_reuse)
  const turn = String(task.turn_id || task.task_id || '').toLowerCase()
  const needMemory = expectedReuse.length > 0 || memoryHits.length > 0 || (turn && !turn.includes('turn-1') && /\b(this|that|previous|earlier|same)\b/i.test(String(task.question || '')))
  return {
    need_memory: needMemory,
    need_knowledge: true,
    memory_first: true,
    reason: needMemory ? 'inferred: expected reuse or cross-turn reference' : 'inferred: first turn or no reuse signal',
    expected_reuse: expectedReuse
  }
}

function pushImportedMemoryEvents(events, context) {
  const { timestamp, taskIndex, taskId, taskTitle, stateRefs, memoryRefs, memoryHits, task } = context
  events.push({
    event_type: 'memory_query',
    timestamp,
    task_index: taskIndex,
    task_id: taskId,
    task_title: taskTitle,
    from_agent: 'retriever',
    to_agent: 'memory',
    action_type: 'query_memory',
    input_params: {
      query: task.retrieval_debug?.retrieval_query || task.question || taskTitle,
      top_k: memoryHits.length || memoryRefs.length
    },
    return_result: { hit_count: memoryHits.length || memoryRefs.length },
    state_refs: stateRefs,
    memory_refs: memoryRefs
  })
  for (const [hitIndex, hit] of memoryHits.entries()) {
    const memoryId = hit.memory_id || hit.id || memoryRefs[hitIndex] || `mem-${String(hitIndex + 1).padStart(4, '0')}`
    events.push({
      event_type: 'memory_hit',
      timestamp,
      task_index: taskIndex,
      task_id: taskId,
      task_title: taskTitle,
      from_agent: 'memory',
      to_agent: 'retriever',
      action_type: 'memory_hit',
      memory_refs: [memoryId],
      state_refs: stateRefs,
      return_result: {
        memory_id: memoryId,
        summary: hit.summary || hit.text || hit.content || '',
        retrieval_score: hit.score ?? hit.retrieval_score ?? hit.confidence,
        memory_type: hit.memory_type || hit.type || 'memory',
        selection_reason: hit.selection_reason || hit.reason || '',
        source_turn: hit.source_turn || hit.turn_id || ''
      }
    })
  }
}

function pushImportedKnowledgeEvents(events, context) {
  const { timestamp, taskIndex, taskId, taskTitle, stateRefs, contextHits, task } = context
  if (!contextHits.length) return
  events.push({
    event_type: 'knowledge_query',
    timestamp,
    task_index: taskIndex,
    task_id: taskId,
    task_title: taskTitle,
    from_agent: 'retriever',
    to_agent: 'knowledge',
    action_type: 'query_knowledge',
    state_refs: stateRefs,
    input_params: {
      query: task.retrieval_debug?.retrieval_query || task.question || taskTitle,
      top_k: contextHits.length,
      retrieval_mode: task.retrieval_debug?.context_source || 'hypergraph'
    },
    return_result: { hit_count: contextHits.length }
  })
  for (const [hitIndex, hit] of contextHits.entries()) {
    const contextId = hit.context_id || hit.id || `context-${String(hitIndex + 1).padStart(4, '0')}`
    events.push({
      event_type: 'knowledge_hit',
      timestamp,
      task_index: taskIndex,
      task_id: taskId,
      task_title: taskTitle,
      from_agent: 'knowledge',
      to_agent: 'retriever',
      action_type: 'return_knowledge',
      state_refs: stateRefs,
      memory_refs: [],
      return_result: {
        context_id: contextId,
        title: hit.title || hit.source_id || '',
        section: hit.section || '',
        summary: hit.text || hit.summary || hit.content || '',
        context_score: hit.context_score ?? hit.score ?? hit.retrieval_score,
        kind: hit.kind || 'context',
        matched_nodes: hit.matched_nodes || [],
        reasoning_path: hit.reasoning_path || []
      }
    })
  }
}

function importedMemoryHits(task) {
  const candidates = [
    task.memory_hits,
    task.retrieved_memories,
    task.retrieval_debug?.memory_hits
  ]
  for (const value of candidates) {
    if (Array.isArray(value) && value.length) return value
  }
  return normalizeList(task.memory_refs || task.memory_hit_ids || task.memory_ids).map((memoryId) => ({
    memory_id: memoryId,
    summary: ''
  }))
}

function importedContextHits(task) {
  const candidates = [
    task.context_hits,
    task.retrieval_debug?.context_hits,
    task.agent_views?.retriever?.structured?.context_hits
  ]
  for (const value of candidates) {
    if (Array.isArray(value) && value.length) return value
  }
  return []
}

function resultTaskForRound(taskResults, round) {
  const byIndex = taskResults.find((item) => Number(item?.task_index || item?.turn_index || 0) === round)
  return byIndex || taskResults[round - 1] || null
}

function buildTasksFromRuntimeEvents(events, runPayload) {
  const primaryRun = latestPrimaryRun(runPayload)
  const resultTasks = Array.isArray(primaryRun?.task_results) ? primaryRun.task_results : []
  const maxEventRound = Math.max(0, ...events.map((event) => Number(event.task_index) || 0))
  const total = Math.max(Number(runMeta.value.task_count) || 0, taskQueue.value.length, maxEventRound, resultTasks.length)
  const runDone = events.some((event) => event.event_type === 'run_done')

  return Array.from({ length: total }, (_, index) => {
    const round = index + 1
    const queueItem = taskQueue.value.find((item) => Number(item.task_index) === round) ?? {}
    const taskEvents = events.filter((event) => Number(event.task_index) === round)
    const taskResult = resultTaskForRound(resultTasks, round)
    const startEvent = taskEvents.find((event) => event.event_type === 'task_start')
    const doneEvent = [...taskEvents].reverse().find((event) => event.event_type === 'task_done')
    const state = taskResult || doneEvent ? 'done' : taskEvents.length && !runDone ? 'running' : 'waiting'
    const metrics = metricsFromRuntimeEvents(taskEvents, doneEvent, taskResult)
    const agents = agentsFromRuntimeEvents(taskEvents)
    return {
      id: `task${round}`,
      round,
      roundText: `任务 ${round}`,
      title: startEvent?.task_title || queueItem.task_title || doneEvent?.task_title || `任务 ${round}`,
      state,
      status: stateText(state),
      progress: state === 'done' ? 100 : state === 'running' ? progressFromEvents(taskEvents) : 0,
      tags: runtimeTags(taskEvents),
      currentAgent: runtimeCurrentAgent(taskEvents, state),
      metrics,
      agents,
      messages: messagesFromRuntimeEvents(taskEvents, agents),
      protocolEvents: taskEvents,
      taskResult,
      protocol: protocolFromRuntimeEvents(round, state, taskEvents),
      events: tableRowsFromRuntimeEvents(taskEvents)
    }
  })
}

function metricsFromRuntimeEvents(taskEvents, doneEvent, taskResult) {
  const messageEvents = taskEvents.filter((event) => event.event_type === 'agent_message')
  const stateEvents = taskEvents.filter((event) => event.event_type === 'state_transfer')
  const hitEvents = taskEvents.filter((event) => event.event_type === 'memory_hit')
  const doneResult = doneEvent?.return_result ?? {}
  const resultMetrics = taskResult?.metrics || {}
  const textChars =
    Number(doneEvent?.text_chars) ||
    Number(resultMetrics.text_chars) ||
    messageEvents.reduce((sum, event) => sum + (Number(event.text_chars) || 0), 0)
  return {
    messages: Number(doneResult.message_count) || Number(resultMetrics.message_count) || messageEvents.length,
    textCost: textChars ? `${textChars} ??` : '--',
    stateTransfers: Number(doneResult.state_transfer_count) || Number(resultMetrics.latent_transfer_count) || stateEvents.length,
    memoryHits: Number(doneResult.memory_hit_count) || Number(taskResult?.memory_hit_count) || hitEvents.length
  }
}

function progressFromEvents(taskEvents) {
  const executableEvents = taskEvents.filter((event) =>
    ['handshake', 'capability_discovery', 'protocol_mapping', 'agent_message', 'state_transfer', 'memory_query', 'memory_hit', 'memory_write'].includes(event.event_type)
  )
  return Math.min(95, Math.max(8, executableEvents.length * 12))
}

function runtimeTags(taskEvents) {
  if (!taskEvents.length) return []
  return Array.from(new Set(taskEvents.map((event) => eventLabel(event)).filter(Boolean))).slice(0, 3)
}

function runtimeCurrentAgent(taskEvents, state) {
  if (state === 'waiting') return '未开始'
  if (state === 'done') return '已完成'
  const last = [...taskEvents].reverse().find((event) => event.from_agent || event.to_agent)
  return normalizeAgentName(last?.to_agent || last?.from_agent || '运行中')
}

function agentsFromRuntimeEvents(taskEvents) {
  const ordered = ['Planner', 'Retriever', 'Memory', 'Knowledge', 'Reasoner', 'Executor', 'Summarizer']
  const names = new Set()
  flowEventsFromRuntimeEvents(taskEvents).forEach((event) => {
    const route = visualRouteForEvent(event)
    if (route.from && !isSystemAgent(route.from)) names.add(route.from)
    if (route.to && !isSystemAgent(route.to)) names.add(route.to)
  })
  const orderedNames = ordered.filter((name) => names.has(name))
  const extraNames = Array.from(names).filter((name) => !ordered.includes(name)).sort()
  return [...orderedNames, ...extraNames]
}

function flowEventsFromRuntimeEvents(taskEvents) {
  return taskEvents.flatMap((event) => {
    if (!['agent_message', 'state_transfer', 'memory_query', 'memory_hit', 'memory_write', 'knowledge_query', 'knowledge_hit'].includes(event.event_type)) {
      return []
    }
    return [event]
  })
}

function visualRouteForEvent(event) {
  if (event.flow_type === 'memory_query' || event.event_type === 'memory_query') {
    return { from: 'Retriever', to: 'Memory' }
  }
  if (event.flow_type === 'memory_result' || event.event_type === 'memory_hit') {
    return { from: 'Memory', to: 'Retriever' }
  }
  if (event.event_type === 'knowledge_query') {
    return { from: 'Retriever', to: 'Knowledge' }
  }
  if (event.event_type === 'knowledge_hit') {
    return { from: 'Knowledge', to: 'Retriever' }
  }
  if (event.event_type === 'memory_write') {
    return { from: 'Summarizer', to: 'Memory' }
  }
  if (event.event_type === 'state_transfer') {
    return {
      from: normalizeAgentName(event.from_agent || 'runtime'),
      to: normalizeAgentName(String(event.to_agent || '').split(',')[0] || event.from_agent || 'runtime')
    }
  }
  return {
    from: normalizeAgentName(event.from_agent || 'runtime'),
    to: normalizeAgentName(event.to_agent || event.from_agent || 'runtime')
  }
}

function messagesFromRuntimeEvents(taskEvents, agents) {
  const sourceEvents = flowEventsFromRuntimeEvents(taskEvents)
  if (!agents.length) return []
  return selectFlowEvents(sourceEvents).map((event, index) => {
    const route = visualRouteForEvent(event)
    const from = agentColumn(route.from, agents)
    const to = agentColumn(route.to, agents)
    return {
      id: event.synthetic_id || event.event_id || `${event.event_type}-${index}`,
      label: `${index + 1}. ${eventLabel(event)}`,
      tone: eventTone(event),
      reverse: from > to,
      edgeStyle: flowEdgeStyle(from, to, agents.length),
      memoryCount: eventMemoryCount(event),
      event,
      fromAgent: route.from,
      toAgent: route.to
    }
  })
}

function selectFlowEvents(events) {
  if (events.length <= 10) return events
  const rest = events.filter((event) => event.event_type !== 'memory_write')
  const latestMemoryWrite = [...events].reverse().find((event) => event.event_type === 'memory_write')
  const selected = rest.slice(0, latestMemoryWrite ? 11 : 12)
  return latestMemoryWrite ? [...selected, latestMemoryWrite] : selected
}

function flowEdgeStyle(from, to, count) {
  const columnCount = Math.max(count, 1)
  const fromCenter = ((from + 0.5) / columnCount) * 100
  const toCenter = ((to + 0.5) / columnCount) * 100
  const left = Math.min(fromCenter, toCenter)
  const width = Math.max(Math.abs(toCenter - fromCenter), 3)
  return {
    left: `${left}%`,
    width: `${width}%`
  }
}

function protocolFromRuntimeEvents(round, state, taskEvents) {
  const message = [...taskEvents].reverse().find((event) => isProtocolDetailEvent(event))
  return {
    message: packetFromEvent('\u4f20\u8f93\u8be6\u60c5', message, state === 'waiting' ? `\u4efb\u52a1 ${round} \u5c1a\u672a\u5f00\u59cb\u3002` : '\u70b9\u51fb\u8fd0\u884c\u65f6\u534f\u4f5c\u94fe\u8def\u4e2d\u7684\u6d88\u606f\u67e5\u770b\u4f20\u8f93\u8be6\u60c5\u3002')
  }
}

function isProtocolDetailEvent(event) {
  return ['agent_message', 'state_transfer', 'memory_query', 'memory_hit', 'memory_write', 'knowledge_query', 'knowledge_hit'].includes(event.event_type)
}

function packetForSelectedMessage(task, message) {
  if (!message?.event) {
    return emptyPacket('\u4f20\u8f93\u8be6\u60c5', task.state === 'waiting' ? '\u4efb\u52a1\u5c1a\u672a\u5f00\u59cb\u3002' : '\u70b9\u51fb\u534f\u4f5c\u94fe\u8def\u4e2d\u7684\u6d88\u606f\u67e5\u770b\u8be6\u60c5\u3002')
  }
  return packetFromEvent('\u4f20\u8f93\u8be6\u60c5', message.event, '\u70b9\u51fb\u534f\u4f5c\u94fe\u8def\u4e2d\u7684\u6d88\u606f\u67e5\u770b\u8be6\u60c5\u3002')
}

function packetFromEvent(title, event, emptyDesc) {
  if (!event) return emptyPacket(title, emptyDesc)
  const route = visualRouteForEvent(event)
  const action = event.action_type || event.event_type || '-'
  const stateRefs = listText(event.state_refs)
  const memoryRefs = listText(event.memory_refs)
  const result = resultText(event)
  return {
    title: `${normalizeAgentName(route.from)} \u2192 ${normalizeAgentName(route.to)}`,
    badge: actionText(action),
    desc: protocolReadableDesc(event, route),
    sections: [
      { label: '\u53d1\u9001\u65b9', value: normalizeAgentName(route.from) },
      { label: '\u63a5\u6536\u65b9', value: normalizeAgentName(route.to) },
      { label: '\u52a8\u4f5c\u7c7b\u578b', value: actionText(action) },
      { label: '\u4f20\u9012\u5185\u5bb9', value: protocolTransferContent(event) },
      { label: '\u72b6\u6001\u5f15\u7528', value: stateRefs },
      { label: protocolMemorySectionLabel(event), value: protocolMemorySectionValue(event, memoryRefs) },
      { label: '\u901a\u4fe1\u5f00\u9500', value: protocolCostText(event) },
      { label: '\u8fd4\u56de\u7ed3\u679c', value: result }
    ],
    mapping: protocolMappingText(event, route),
    payload: stringify(event),
    hasData: true
  }
}

function formatExactMatch(value) {
  if (value === null || typeof value === 'undefined') return 'N/A'
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  return String(value)
}

function transferStatusText(status) {
  if (status === 'done') return '\u5df2\u5b8c\u6210'
  if (status === 'running') return '\u8fd0\u884c\u4e2d'
  return '\u5f85\u8fd0\u884c'
}

function safeStringify(value) {
  if (value === null || typeof value === 'undefined' || value === '') return ''
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function getArrayField(value) {
  return Array.isArray(value) ? value : []
}

function normalizeAgentOutputKey(value) {
  return String(value || '').trim().toLowerCase().replace(/agent$/, '')
}

function agentOutputListItemToView(item) {
  const displayText = item.display_text ?? item.raw_output ?? item.answer_candidate ?? item.reasoning_summary ?? item.retrieval_query ?? ''
  return {
    display_text: String(displayText || ''),
    structured: item,
    state_ref: typeof item.state_ref === 'string' ? item.state_ref : ''
  }
}

function getAgentView(taskResult, agentKey) {
  const normalizedKey = String(agentKey || '').toLowerCase()
  const views = taskResult?.agent_views
  if (views && typeof views === 'object' && !Array.isArray(views) && views[normalizedKey]) {
    return views[normalizedKey]
  }
  const outputs = taskResult?.agent_outputs
  if (outputs && typeof outputs === 'object' && !Array.isArray(outputs) && outputs[normalizedKey]) {
    return outputs[normalizedKey]
  }
  if (Array.isArray(outputs)) {
    const item = outputs.find((output) => normalizeAgentOutputKey(output.agent) === normalizedKey)
    if (item) return agentOutputListItemToView(item)
  }
  return null
}

function transferAgentKeyFromMessage(message) {
  const event = message?.event || {}
  const action = String(event.action_type || '').toLowerCase()
  const from = String(event.from_agent || '').toLowerCase()
  const to = String(event.to_agent || '').toLowerCase()
  if (event.event_type === 'knowledge_query' || event.event_type === 'knowledge_hit') return 'retriever'
  if (event.event_type === 'memory_query' || event.event_type === 'memory_hit') return 'retriever'
  if (event.event_type === 'memory_write') return 'summarizer'
  if (action === 'retrieve') return 'planner'
  if (action === 'summarize') return 'summarizer'
  if (action === 'execute' || from === 'executor' || to === 'executor') return 'executor'
  if (action === 'reason') return from || 'retriever'
  return from || to || 'planner'
}

function buildFallbackTransferDetail(task, message, agentKey) {
  const event = message?.event || {}
  const payloadText = eventPayloadText(event)
  const eventResultText = safeStringify(event.return_result)
  return {
    hasData: true,
    agentKey,
    taskLabel: `task ${task.round}`,
    agentLabel: normalizeAgentName(agentKey),
    status: task.state || 'waiting',
    hasOutput: Boolean(payloadText || eventResultText),
    question: task.title || '',
    exactMatch: null,
    stateRef: listText(event.state_refs) === '-' ? '' : listText(event.state_refs),
    displayText: payloadText,
    structuredText: typeof event.payload === 'object' ? safeStringify(event.payload) : '',
    retrievalSummaryText: ['memory_query', 'memory_hit', 'knowledge_query', 'knowledge_hit'].includes(event.event_type) ? safeStringify(event.return_result || event.input_params) : '',
    summaryResultText: event.event_type === 'memory_write' || String(event.action_type || '').toLowerCase() === 'summarize' ? safeStringify({ payload: event.payload, return_result: event.return_result }) : '',
    memoryRefineText: '',
    executionResultText: agentKey === 'executor' ? safeStringify(event.return_result || event.payload) : ''
  }
}

function transferAgentDetailForSelectedMessage(task, message) {
  if (!message?.event) {
    return {
      hasData: false,
      taskLabel: '',
      agentLabel: '',
      status: 'waiting',
      hasOutput: false,
      question: '',
      exactMatch: null,
      stateRef: '',
      displayText: '',
      structuredText: '',
      retrievalSummaryText: '',
      summaryResultText: '',
      memoryRefineText: '',
      executionResultText: '',
      agentKey: ''
    }
  }

  const agentKey = transferAgentKeyFromMessage(message)
  const taskResult = task.taskResult
  if (!taskResult) {
    return buildFallbackTransferDetail(task, message, agentKey)
  }

  const agentView = getAgentView(taskResult, agentKey)
  const retrievalDebug = taskResult.retrieval_debug ?? {}
  const plannerFocus = getArrayField(retrievalDebug.planner_focus)
  const retrievalSummaryText = safeStringify({
    retrieval_query: retrievalDebug.retrieval_query ?? '',
    planner_focus: plannerFocus,
    context_source: retrievalDebug.context_source ?? '',
    context_hit_ids: retrievalDebug.context_hit_ids ?? [],
    context_hits: retrievalDebug.context_hits ?? [],
    memory_hit_ids: retrievalDebug.memory_hit_ids ?? [],
    memory_context: taskResult.memory_context ?? [],
    memory_hits: taskResult.memory_hits ?? []
  })
  const summaryResultText = safeStringify({
    final_answer: taskResult.final_answer ?? '',
    latest_evidence_write: taskResult.latest_evidence_write ?? null,
    latest_experience_write: taskResult.latest_experience_write ?? null,
    execution_result: taskResult.execution_result ?? null
  })

  return {
    hasData: true,
    agentKey,
    taskLabel: `task ${task.round}`,
    agentLabel: normalizeAgentName(agentKey),
    status: taskResult ? 'done' : task.state || 'waiting',
    hasOutput: Boolean(agentView),
    question: taskResult.question ?? task.title ?? '',
    exactMatch: taskResult.exact_match ?? null,
    stateRef: String(agentView?.state_ref ?? taskResult.state_refs?.[agentKey] ?? ''),
    displayText: agentView?.display_text ?? buildFallbackTransferDetail(task, message, agentKey).displayText,
    structuredText: safeStringify(agentView?.structured),
    retrievalSummaryText: agentKey === 'retriever' ? retrievalSummaryText : '',
    summaryResultText: agentKey === 'summarizer' ? summaryResultText : '',
    memoryRefineText: agentKey === 'summarizer' ? safeStringify(taskResult.memory_refine_debug ?? null) : '',
    executionResultText: (agentKey === 'executor' || agentKey === 'reasoner') ? safeStringify(taskResult.execution_result) : ''
  }
}

function eventPayloadText(event) {
  if (typeof event.payload === 'string' && event.payload.trim()) return event.payload.trim()
  if (event.payload && typeof event.payload === 'object') return safeStringify(event.payload)
  if (event.payload_summary) return String(event.payload_summary)
  if (event.input_params && Object.keys(event.input_params).length) return safeStringify(event.input_params)
  if (event.return_result && Object.keys(event.return_result).length) return safeStringify(event.return_result)
  return protocolContentSummary(event)
}

function protocolReadableDesc(event, route) {
  if (event.event_type === 'knowledge_query') return '\u0052\u0065\u0074\u0072\u0069\u0065\u0076\u0065\u0072 \u5411\u77e5\u8bc6\u8d85\u56fe\u53d1\u8d77\u4e0a\u4e0b\u6587\u8bc1\u636e\u68c0\u7d22\u3002'
  if (event.event_type === 'knowledge_hit') return '\u005b\u004b\u006e\u006f\u0077\u006c\u0065\u0064\u0067\u0065\u005d \u8fd4\u56de\u672c\u9898\u77e5\u8bc6\u8d85\u56fe\u53ec\u56de\u7684 supporting passage \u6216\u4e0a\u4e0b\u6587\u8bc1\u636e\u3002'
  if (event.event_type === 'memory_query') return 'Retriever \u5411\u5171\u4eab\u8bb0\u5fc6\u6a21\u5757\u53d1\u8d77\u68c0\u7d22\u8bf7\u6c42\u3002'
  if (event.event_type === 'memory_hit') return 'Memory \u8fd4\u56de\u53ef\u590d\u7528\u7684\u5386\u53f2\u8bb0\u5fc6\u6216\u8bc1\u636e\u547d\u4e2d\u3002'
  if (event.event_type === 'memory_write') return 'Summarizer \u5c06\u672c\u8f6e\u8bc1\u636e\u3001\u7ed3\u8bba\u6216\u7ecf\u9a8c\u5199\u56de\u5171\u4eab\u8bb0\u5fc6\u3002'
  if (event.event_type === 'state_transfer') return 'Agent \u4e4b\u95f4\u4f20\u9012\u975e\u6587\u672c\u72b6\u6001\u5f15\u7528\uff0c\u907f\u514d\u91cd\u590d\u4f20\u8f93\u957f\u6587\u672c\u3002'
  return `${normalizeAgentName(route.from)} \u5c06\u7ed3\u6784\u5316\u6d88\u606f\u53d1\u9001\u7ed9 ${normalizeAgentName(route.to)}\u3002`
}

function protocolTransferContent(event) {
  const result = event.return_result || {}
  if (event.action_type === 'retrieve') {
    return event.payload_summary || result.summary || eventPayloadText(event) || '-'
  }
  return protocolContentSummary(event)
}

function protocolMemorySectionLabel(event) {
  return event.action_type === 'retrieve' ? '\u68c0\u7d22\u89c4\u5212' : '\u8bb0\u5fc6\u5f15\u7528'
}

function protocolMemorySectionValue(event, memoryRefs) {
  const policy = (event.return_result || {}).retrieval_policy
  if (event.action_type === 'retrieve' && policy) {
    return `\u8bb0\u5fc6\u68c0\u7d22\uff1a${policy.need_memory ? '\u9700\u8981' : '\u8df3\u8fc7'}；\u77e5\u8bc6\u68c0\u7d22\uff1a${policy.need_knowledge === false ? '\u8df3\u8fc7' : '\u9700\u8981'}；\u7406\u7531\uff1a${policy.reason || '-'}`
  }
  return memoryRefs
}

function protocolContentSummary(event) {
  const result = event.return_result || {}
  if (event.event_type === 'knowledge_query') {
    return `\u77e5\u8bc6\u68c0\u7d22 query=${event.input_params?.query || '-'}`
  }
  if (event.event_type === 'knowledge_hit') {
    return result.summary || result.context_id || `\u547d\u4e2d ${eventMemoryCount(event)} \u6761\u77e5\u8bc6\u8bc1\u636e`
  }
  if (event.event_type === 'memory_hit') {
    return result.summary || result.memory_id || `\u547d\u4e2d ${eventMemoryCount(event)} \u6761\u8bb0\u5fc6`
  }
  if (event.event_type === 'memory_write') {
    return result.summary || result.memory_id || '\u5199\u56de\u65b0\u7684\u5171\u4eab\u8bb0\u5fc6\u5355\u5143'
  }
  if (event.event_type === 'state_transfer') {
    return event.summary || result.summary || `\u4f20\u9012 ${listText(event.state_refs)} \u72b6\u6001\u5f15\u7528`
  }
  if (result.summary) return result.summary
  if (result.answer_candidate) return result.answer_candidate
  if (result.evidence_hits || result.experience_hits) {
    return `\u8bc1\u636e\u547d\u4e2d ${result.evidence_hits || 0}\uff0c\u7ecf\u9a8c\u547d\u4e2d ${result.experience_hits || 0}`
  }
  return event.payload_summary || event.action_type || event.event_type || '-'
}

function protocolCostText(event) {
  const parts = []
  if (event.text_chars) parts.push(`${event.text_chars} \u5b57\u7b26`)
  if (event.protocol_bytes) parts.push(`\u534f\u8bae ${formatBytes(event.protocol_bytes)}`)
  if (event.communication_bytes) parts.push(`\u5b9e\u9645\u8f7d\u8377 ${formatBytes(event.communication_bytes)}`)
  if (event.bytes_size) parts.push(`\u72b6\u6001 ${formatBytes(event.bytes_size)}`)
  return parts.length ? parts.join(' / ') : '-'
}

function protocolMappingText(event, route) {
  const action = event.action_type || event.event_type
  if (event.event_type === 'knowledge_query') return '\u77e5\u8bc6\u68c0\u7d22\u610f\u56fe \u2192 knowledge_hypergraph.search(query, top_k, retrieval_mode)'
  if (event.event_type === 'knowledge_hit') return '\u77e5\u8bc6\u53ec\u56de \u2192 return_result(context_refs, passages, scores)'
  if (event.event_type === 'memory_query') return '\u68c0\u7d22\u610f\u56fe \u2192 memory.search(query, top_k, retrieval_mode)'
  if (event.event_type === 'memory_hit') return '\u5386\u53f2\u547d\u4e2d \u2192 return_result(memory_refs, summaries, scores)'
  if (event.event_type === 'memory_write') return '\u603b\u7ed3\u5199\u56de \u2192 memory.write(memory_type, summary, source_agent, task_topic)'
  if (event.event_type === 'state_transfer') return '\u4e2d\u95f4\u72b6\u6001 \u2192 state_ref(content_type, producer, consumers, bytes)'
  const map = {
    retrieve: '\u68c0\u7d22\u8bf7\u6c42 \u2192 retrieve(query, planner_focus, memory_scope)',
    reason: '\u8bc1\u636e\u63a8\u7406 \u2192 reason(evidence_refs, memory_refs, state_refs)',
    execute: '\u5de5\u5177\u6267\u884c \u2192 execute(tool_name, args, state_refs)',
    summarize: '\u7b54\u6848\u603b\u7ed3 \u2192 summarize(answer, evidence_refs, experience_refs)',
    start_task: '\u4efb\u52a1\u5206\u53d1 \u2192 start_task(task_title, dataset_type, required_roles)'
  }
  return map[action] || `${normalizeAgentName(route.from)} \u6d88\u606f \u2192 ${action || 'message'}(input_params, state_refs, memory_refs, return_result)`
}

function tableRowsFromRuntimeEvents(taskEvents) {
  return taskEvents.map((event) => ({
    time: formatEventTime(event.timestamp),
    from: normalizeAgentName(event.from_agent || 'runtime'),
    to: normalizeAgentName(event.to_agent || 'dashboard'),
    type: event.event_type || 'event',
    action: event.action_type || '-',
    state: listText(event.state_refs),
    memory: listText(event.memory_refs),
    tokens: event.text_chars ? `${event.text_chars} 字符` : '0 字符',
    cost: event.elapsed_ms ? `${Number(event.elapsed_ms).toFixed(2)} ms` : '0 ms',
    result: resultText(event),
    tone: eventTone(event)
  }))
}

function emptyTask() {
  return {
    id: '',
    round: '--',
    roundText: '未运行',
    title: '未运行',
    state: 'waiting',
    status: '未运行',
    progress: 0,
    tags: [],
    currentAgent: '未开始',
    metrics: { messages: 0, textCost: '--', stateTransfers: 0, memoryHits: 0 },
    agents: [],
    messages: [],
    protocol: {
      handshake: emptyPacket('握手', '启动实验后显示真实握手事件。'),
      capability: emptyPacket('能力发现', '启动实验后显示真实能力发现事件。'),
      mapping: emptyPacket('协议映射', '启动实验后显示真实协议映射事件。'),
      message: emptyPacket('消息详情', '启动实验后显示真实 Agent 消息。')
    },
    events: []
  }
}

function emptyPacket(title, desc = '暂无数据') {
  return {
    title,
    badge: '--',
    desc,
    payload: '',
    hasData: false
  }
}

function stateText(state) {
  if (state === 'done') return '已完成'
  if (state === 'running') return '运行中'
  return '未开始'
}

function modeText(mode) {
  const map = {
    baseline_text: '纯文本',
    autogen_text: 'autogen',
    camel_text: 'camel',
    autogen_camel_text: 'autogen',
    autoform_text: 'AutoForm',
    llmlingua_text: 'LLMLingua-2',
    protocol_api: '结构化协议',
    protocol_api_embedding: '协议+向量',
    hidden_api: '隐状态传递',
    text: '纯文本',
    protocol: '结构化协议',
    hidden: '隐状态传递',
    '<developer-modes>': '开发模式'
  }
  return map[mode] || mode
}

function agentColumn(name, agents) {
  const normalized = normalizeAgentName(name)
  const index = agents.indexOf(normalized)
  return index >= 0 ? index : 0
}

function normalizeAgentName(name) {
  const value = String(name || '')
  const map = {
    planner: 'Planner',
    retriever: 'Retriever',
    reasoner: 'Reasoner',
    executor: 'Executor',
    summarizer: 'Summarizer',
    memory: 'Memory',
    knowledge: 'Knowledge',
    runtime: 'Runtime',
    dashboard: 'Dashboard'
  }
  return map[value.toLowerCase()] || value || '-'
}

function isSystemAgent(name) {
  return ['Runtime', 'Dashboard', 'runtime', 'dashboard'].includes(String(name || ''))
}

function eventLabel(event) {
  if (event.event_type === 'knowledge_query') return '\u68c0\u7d22\u77e5\u8bc6'
  if (event.event_type === 'knowledge_hit') return `\u547d\u4e2d\u77e5\u8bc6 ${event.return_result?.context_id || ''}`.trim()
  if (event.flow_type === 'memory_query') return '检索记忆'
  if (event.flow_type === 'memory_result') return `命中记忆 ${eventMemoryCount(event)}`
  const map = {
    run_start: '启动',
    task_queue: '任务队列',
    task_start: '开始任务',
    handshake: '握手',
    capability_discovery: '能力发现',
    protocol_mapping: '协议映射',
    agent_message: actionText(event.action_type),
    state_transfer: `STATE ${listText(event.state_refs)}`,
    memory_query: '检索记忆',
    memory_hit: `命中记忆 ${listText(event.memory_refs)}`,
    memory_write: `写回记忆 ${listText(event.memory_refs)}`,
    task_done: '任务完成',
    run_done: '运行完成'
  }
  return map[event.event_type] || String(event.action_type || event.event_type || 'event').toUpperCase()
}

function actionText(action) {
  const map = {
    retrieve: '检索',
    reason: '推理',
    execute: '执行',
    summarize: '总结',
    start_task: '分发任务',
    open_session: '建立会话',
    discover_capability: '能力发现',
    map_protocol: '协议映射'
  }
  return map[action] || String(action || '消息').toUpperCase()
}

function eventTone(event) {
  if (event.event_type === 'state_transfer') return 'green'
  if (event.flow_type || event.event_type === 'memory_query' || event.event_type === 'memory_hit' || event.event_type === 'memory_write') return 'orange'
  if (event.event_type === 'protocol_mapping' || String(event.to_agent || '').toLowerCase().includes('executor')) return 'purple'
  return 'blue'
}

function eventMemoryCount(event) {
  const refs = Array.isArray(event.memory_refs) ? event.memory_refs.filter(Boolean) : event.memory_refs ? [event.memory_refs] : []
  const result = event.return_result || {}
  return Number(result.memory_hit_count || result.hit_count || refs.length || 0)
}

function listText(value) {
  const values = Array.isArray(value) ? value.filter(Boolean) : value ? [value] : []
  return values.length ? values.join(', ') : '-'
}

function resultText(event) {
  const result = event.return_result
  if (!result) return event.event_type === 'task_start' ? '开始' : '-'
  if (typeof result === 'string') return result
  if (result.memory_id) return result.summary ? `${result.memory_id}：${result.summary}` : result.memory_id
  if (typeof result.message_count !== 'undefined') return `消息 ${result.message_count}，命中 ${result.memory_hit_count || 0}`
  if (Array.isArray(result.result_refs)) return result.result_refs.join(', ')
  return JSON.stringify(result)
}

function formatEventTime(value) {
  if (!value) return '--'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return String(value)
  const hh = String(date.getHours()).padStart(2, '0')
  const mm = String(date.getMinutes()).padStart(2, '0')
  const ss = String(date.getSeconds()).padStart(2, '0')
  const ms = String(date.getMilliseconds()).padStart(3, '0')
  return `${hh}:${mm}:${ss}.${ms}`
}

function formatBytes(value) {
  const bytes = Number(value) || 0
  if (bytes <= 0) return '0 B'
  if (bytes < 1024) return `${bytes} B`
  return `${(bytes / 1024).toFixed(1)} KB`
}

function stringify(payload) {
  return JSON.stringify(payload, null, 2)
}
</script>
