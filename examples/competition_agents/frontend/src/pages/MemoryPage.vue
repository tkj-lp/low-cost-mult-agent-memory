<template>
  <section class="page memory-page memory-flow-page">
    <section class="memory-shared-filter" :class="{ locked: memorySelectionLocked }">
      <MetricsFilterBar
        :model-value="sharedFilters"
        :baseline-options="memoryBaselineOptions"
        @update:model-value="updateSharedFilters"
        @refresh="refreshMemoryRecords"
      />
    </section>

    <section class="memory-card memory-task-flow-card">
      <header>
        <span></span>
        <h2>连续任务记忆流转</h2>
        <div class="memory-run-badges">
          <b>{{ collaborationLabel }}</b>
          <b>{{ datasetLabel }}</b>
          <b>{{ memoryRetrievalLabel }}</b>
        </div>
      </header>

      <div v-if="!tasks.length" class="runtime-empty-panel">
        <strong>暂无真实记忆流转数据</strong>
        <span>启动实验后，这里会按任务展示检索、命中、复用和写回链路。</span>
      </div>

      <template v-else>
        <div class="memory-task-tabs">
          <button
            v-for="task in tasks"
            :key="task.key"
            type="button"
            :class="[task.status, { active: task.key === selectedTaskKey }]"
            @click="selectTask(task.key)"
          >
            <strong>任务 {{ task.index }}</strong>
            <span>{{ task.shortTitle }}</span>
          </button>
        </div>

        <div class="memory-flow-timeline" :key="selectedTaskKey">
          <article
            v-for="(node, index) in timelineNodes"
            :key="node.id"
            class="memory-flow-node"
            :class="node.tone"
            :style="{ '--step-index': index }"
          >
            <div class="memory-flow-marker">
              <i>{{ node.step }}</i>
              <em v-if="node.edge">{{ node.edge }}</em>
            </div>
            <div class="memory-flow-content">
              <strong>{{ node.title }}</strong>
              <p>{{ node.detail }}</p>
              <span>{{ node.meta }}</span>
            </div>
          </article>
        </div>
      </template>
    </section>

    <div class="memory-workspace">
      <section class="memory-card memory-retrieval-panel">
        <header>
          <span></span>
          <h2>检索情况</h2>
        </header>

        <div v-if="!selectedTask" class="runtime-empty-panel">
          <strong>请选择任务</strong>
          <span>选择上方任务后展示该任务的检索查询和候选记忆。</span>
        </div>

        <template v-else>
          <div class="retrieval-question">
            <span>当前任务问题</span>
            <strong>{{ selectedTask.title }}</strong>
          </div>

          <div class="retrieval-list" v-if="retrievalRows.length">
            <button
              v-for="memory in retrievalRows"
              :key="memory.id"
              type="button"
              class="retrieval-memory-row"
              :class="{ active: memory.id === selectedMemoryId, adopted: memory.adopted }"
              @click="selectMemory(memory.id)"
            >
              <div>
                <strong>{{ memory.id }}</strong>
                <em>{{ memory.adopted ? '已采用' : '候选' }}</em>
              </div>
              <div class="retrieval-memory-main">
                <p>{{ memory.summary }}</p>
                <span>选择理由：{{ memory.reason }}</span>
              </div>
              <dl>
                <div><dt>命中方式</dt><dd>{{ memory.method }}</dd></div>
                <div><dt>分数</dt><dd>{{ memory.score }}</dd></div>
                <div><dt>采用状态</dt><dd>{{ memory.adopted ? '参与回答' : '候选排序' }}</dd></div>
              </dl>
            </button>
          </div>

          <div v-else class="runtime-empty-panel compact-empty-panel">
            <strong>当前任务未命中记忆</strong>
            <span>如果后续任务命中历史记忆，这里会展示候选列表和采用状态。</span>
          </div>
        </template>
      </section>

      <aside class="memory-side-stack">
        <section class="memory-card memory-detail-panel">
          <header>
            <span></span>
            <h2>记忆详情</h2>
          </header>

          <div v-if="!selectedMemory" class="runtime-empty-panel">
            <strong>暂无选中记忆</strong>
            <span>点击左侧候选记忆后展示详情。</span>
          </div>

          <dl v-else class="memory-detail-list">
            <div><dt>记忆编号</dt><dd>{{ selectedMemory.id }}</dd></div>
            <div><dt>来源 Agent</dt><dd>{{ selectedMemory.source }}</dd></div>
            <div><dt>创建时间</dt><dd>{{ selectedMemory.time }}</dd></div>
            <div><dt>任务主题</dt><dd>{{ selectedMemory.topic }}</dd></div>
            <div><dt>摘要</dt><dd>{{ selectedMemory.summary }}</dd></div>
            <div><dt>标签</dt><dd>{{ selectedMemory.tags }}</dd></div>
            <div><dt>命中方式</dt><dd>{{ selectedMemory.method }}</dd></div>
            <div><dt>命中分数</dt><dd>{{ selectedMemory.score }}</dd></div>
            <div><dt>复用状态</dt><dd>{{ selectedMemory.adopted ? '已参与本轮回答' : '候选命中' }}</dd></div>
            <div><dt>状态引用</dt><dd>{{ stateRefText }}</dd></div>
          </dl>
        </section>

        <section class="memory-card evidence-detail-panel">
          <header>
            <span></span>
            <h2>证据链</h2>
          </header>

          <div v-if="!selectedTask" class="runtime-empty-panel">
            <strong>暂无证据链</strong>
            <span>任务运行完成后会展示记忆如何参与回答。</span>
          </div>

          <div v-else class="evidence-node-chain">
            <article
              v-for="(item, index) in evidenceChain"
              :key="item.label"
              class="evidence-node"
              :class="item.tone"
              :style="{ '--step-index': index }"
            >
              <i>{{ index + 1 }}</i>
              <div>
                <strong>{{ item.label }}</strong>
                <p>{{ item.value }}</p>
              </div>
            </article>
          </div>
        </section>
      </aside>
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
  retrievalOptionsFor
} from '../experimentOptions'
import MetricsFilterBar from '../components/metrics/MetricsFilterBar.vue'

const RUNNER_EVENTS_URL = 'http://127.0.0.1:8765/api/runner/events'
const RUNNER_STATUS_URL = 'http://127.0.0.1:8765/api/runner/status'
const RUNNER_METRICS_URL = 'http://127.0.0.1:8765/api/runner/metrics'
const CONFIG_STORAGE_KEY = 'competition-agent-console-config'
const FILTER_SYNC_EVENT = 'competition-agent-filter-sync'
const props = defineProps({
  importedPayloads: {
    type: Array,
    default: () => []
  }
})

const selectedTaskKey = ref('')
const selectedMemoryId = ref('')
const runtimeEvents = ref([])
const savedConfig = ref(readSavedConfig())
const sharedFilters = ref(withFilterDefaults(savedConfig.value))
const runnerRunning = ref(false)
const memoryRecords = ref([])
const memoryLoadError = ref('')
let timer = null

const memorySelectionLocked = computed(() => runnerRunning.value)
const selectedMemoryRecord = computed(() =>
  memoryRecords.value.find((record) =>
    record.dataset === sharedFilters.value.dataset &&
    record.collaborationMode === sharedFilters.value.collaborationMode &&
    record.memoryMode === sharedFilters.value.memoryMode &&
    record.retrievalMode === sharedFilters.value.retrievalMode
  ) ?? null
)
const memoryBaselineOptions = computed(() =>
  memoryRecords.value.map((record) => ({
    label: memoryRecordLabel(record),
    value: record.recordId
  }))
)
const importedRunPayload = computed(() => importedPayloadForCurrentConfig(props.importedPayloads, sharedFilters.value))
const selectedArchivePayload = computed(() => importedRunPayload.value ?? selectedMemoryRecord.value?.payload ?? null)
const displayRuntimeEvents = computed(() =>
  memorySelectionLocked.value
    ? runtimeEvents.value
    : selectedArchivePayload.value
      ? eventsFromImportedPayload(selectedArchivePayload.value)
      : runtimeEvents.value
)
const runMeta = computed(() => displayRuntimeEvents.value.find((event) => event.event_type === 'run_start') ?? {})
const taskQueue = computed(() => {
  const queue = displayRuntimeEvents.value.find((event) => event.event_type === 'task_queue')
  return Array.isArray(queue?.tasks) ? queue.tasks : []
})

const retrievalModeLabel = computed(() => runMeta.value.memory_retrieval_mode || '检索方式待运行')
const collaborationLabel = computed(() => `协作模式：${modeLabel(runMeta.value.experiment_mode || runMeta.value.mode)}`)
const datasetLabel = computed(() => `数据集：${datasetLabelFromMeta(runMeta.value)}`)
const memoryRetrievalLabel = computed(() => {
  const memory = memoryModeLabel(runMeta.value.memory_organization || runMeta.value.memory_retrieval_mode)
  const retrieval = retrievalModeLabelText(runMeta.value.memory_retrieval_mode)
  return `记忆组织：${memory}｜检索方式：${retrieval}`
})

const tasks = computed(() => {
  const byIndex = new Map()
  for (const item of taskQueue.value) {
    const index = Number(item.task_index || byIndex.size + 1)
    byIndex.set(index, {
      key: String(index),
      index,
      id: item.task_id || String(index),
      title: item.task_title || item.question || item.task_id || `任务 ${index}`,
      shortTitle: shorten(item.task_title || item.question || item.task_id || `任务 ${index}`, 34),
      expectedReuse: item.expected_reuse || []
    })
  }
  for (const event of displayRuntimeEvents.value) {
    const index = Number(event.task_index || 0)
    if (!index) continue
    if (!byIndex.has(index)) {
      byIndex.set(index, {
        key: String(index),
        index,
        id: event.task_id || String(index),
        title: event.task_title || event.task_id || `任务 ${index}`,
        shortTitle: shorten(event.task_title || event.task_id || `任务 ${index}`, 34),
        expectedReuse: []
      })
    }
  }
  return [...byIndex.values()]
    .sort((a, b) => a.index - b.index)
    .map((task) => ({ ...task, status: taskStatus(task.index) }))
})

const selectedTask = computed(() => tasks.value.find((task) => task.key === selectedTaskKey.value) ?? null)
const selectedTaskEvents = computed(() => {
  if (!selectedTask.value) return []
  return displayRuntimeEvents.value.filter((event) => Number(event.task_index || 0) === selectedTask.value.index)
})

const taskDoneEvent = computed(() => selectedTaskEvents.value.find((event) => event.event_type === 'task_done') ?? null)
const memoryWriteEvents = computed(() => selectedTaskEvents.value.filter((event) => event.event_type === 'memory_write'))
const stateTransferEvents = computed(() => selectedTaskEvents.value.filter((event) => event.event_type === 'state_transfer'))

const retrievalQuery = computed(() => {
  const retrieveEvent = selectedTaskEvents.value.find((event) =>
    event.event_type === 'agent_message' &&
    String(event.action_type || '').toLowerCase().includes('retrieve')
  )
  return shorten(eventPayloadText(retrieveEvent) || selectedTask.value?.title || '暂无检索查询', 260)
})

const retrievalRows = computed(() =>
  selectedTaskEvents.value
    .filter((event) => event.event_type === 'memory_hit')
    .map((event, index) => {
      const result = event.return_result || {}
      const id = result.memory_id || event.memory_refs?.[0] || `memory-hit-${index + 1}`
      return {
        id,
        summary: result.summary || '该记忆命中事件未提供摘要',
        score: scoreText(result.retrieval_score),
        scoreRaw: Number(result.retrieval_score || 0),
        type: result.memory_type || 'memory',
        method: retrievalModeLabel.value,
        reason: selectionReason(result),
        scoreComponents: scoreComponentText(result.score_components),
        source: normalizeAgent(event.from_agent || 'Memory'),
        time: formatTime(event.timestamp),
        topic: event.task_title || selectedTask.value?.title || '--',
        tags: tagText(result.tags),
        adopted: adoptedMemoryIds.value.has(String(id)),
        stateRefs: event.state_refs || []
      }
    })
)

const adoptedMemoryIds = computed(() => new Set((taskDoneEvent.value?.memory_refs || []).map((item) => String(item))))

const selectedMemory = computed(() => {
  const selected = retrievalRows.value.find((memory) => memory.id === selectedMemoryId.value)
  if (selected) return selected
  const firstAdopted = retrievalRows.value.find((memory) => memory.adopted)
  if (firstAdopted) return firstAdopted
  if (retrievalRows.value[0]) return retrievalRows.value[0]
  const write = memoryWriteEvents.value[0]
  if (!write) return null
  const result = write.return_result || {}
  return {
    id: result.memory_id || write.memory_refs?.[0] || 'memory-write',
    summary: result.summary || '该记忆写入事件未提供摘要',
    score: '--',
    type: result.memory_type || 'memory',
    method: '写入',
    source: normalizeAgent(write.from_agent || 'Summarizer'),
    time: formatTime(write.timestamp),
    topic: write.task_title || selectedTask.value?.title || '--',
    tags: '--',
    adopted: true,
    stateRefs: write.state_refs || []
  }
})

const stateRefText = computed(() => {
  const directRefs = selectedMemory.value?.stateRefs || []
  if (directRefs.length) return directRefs.join('，')
  const state = stateTransferEvents.value[0]?.state
  if (!state) return '本任务未记录关联状态引用'
  const shape = Array.isArray(state.shape) && state.shape.length ? `${state.shape.join('×')} 维` : '维度未记录'
  return `${state.state_ref || 'state_ref'} · ${state.content_type || 'state'} · ${shape} · ${byteText(state.bytes)}`
})

const timelineNodes = computed(() => {
  if (!selectedTask.value) return []
  const nodes = [
    {
      id: 'question',
      step: '1',
      title: '任务问题',
      detail: selectedTask.value.title,
      meta: selectedTask.value.expectedReuse.length
        ? `期望复用：${selectedTask.value.expectedReuse.join('，')}`
        : '首轮或无显式期望复用',
      tone: 'blue',
      edge: '生成查询'
    },
    {
      id: 'query',
      step: '2',
      title: '检索查询',
      detail: retrievalQuery.value,
      meta: `检索方式：${retrievalModeLabel.value}`,
      tone: 'green',
      edge: '检索记忆库'
    }
  ]

  if (retrievalRows.value.length) {
    nodes.push({
      id: 'hit',
      step: '3',
      title: '命中记忆',
      detail: retrievalRows.value.map((memory) => `${memory.id}（${memory.score}）`).join('，'),
      meta: `${retrievalRows.value.length} 条候选，${retrievalRows.value.filter((memory) => memory.adopted).length} 条被采用`,
      tone: 'orange',
      edge: '参与回答'
    })
  } else {
    nodes.push({
      id: 'hit-empty',
      step: '3',
      title: '命中记忆',
      detail: '当前任务没有记录 memory_hit 事件',
      meta: '等待后续任务或更高召回检索',
      tone: 'gray',
      edge: '参与回答'
    })
  }

  nodes.push({
    id: 'reuse',
    step: '4',
    title: '复用到回答',
    detail: taskDoneEvent.value
      ? shorten(taskDoneEvent.value.return_result?.final_answer || '任务已完成，但未记录最终回答', 180)
      : '任务尚未完成',
    meta: adoptedMemoryIds.value.size ? `复用记忆：${[...adoptedMemoryIds.value].join('，')}` : '未记录被复用记忆',
    tone: taskDoneEvent.value ? 'purple' : 'gray',
    edge: '写回共享记忆'
  })

  nodes.push({
    id: 'write',
    step: '5',
    title: '写入新记忆',
    detail: memoryWriteEvents.value.length
      ? memoryWriteEvents.value.map((event) => event.return_result?.memory_id || event.memory_refs?.[0]).filter(Boolean).join('，')
      : '当前任务未记录 memory_write 事件',
    meta: memoryWriteEvents.value.length ? '写回共享记忆库' : '无新增记忆',
    tone: memoryWriteEvents.value.length ? 'blue' : 'gray'
  })
  return nodes
})

const evidenceChain = computed(() => [
  { label: '原问题', value: selectedTask.value?.title || '--', tone: 'blue' },
  { label: '生成查询', value: retrievalQuery.value, tone: 'green' },
  {
    label: '选中记忆',
    value: selectedMemory.value ? `${selectedMemory.value.id}：${selectedMemory.value.summary}` : '未命中记忆',
    tone: selectedMemory.value ? 'orange' : 'gray'
  },
  {
    label: '支撑依据',
    value: selectedMemory.value
      ? `${selectedMemory.value.reason}${selectedMemory.value.scoreComponents ? `；${selectedMemory.value.scoreComponents}` : ''}`
      : '没有可用记忆支撑本轮回答',
    tone: adoptedMemoryIds.value.size ? 'purple' : 'gray'
  },
  {
    label: '最终回答',
    value: taskDoneEvent.value?.return_result?.final_answer || '任务尚未完成或未记录最终回答',
    tone: taskDoneEvent.value ? 'blue' : 'gray'
  }
])

watch(tasks, (nextTasks) => {
  if (!nextTasks.length) {
    selectedTaskKey.value = ''
    selectedMemoryId.value = ''
    return
  }
  if (!nextTasks.some((task) => task.key === selectedTaskKey.value)) {
    const runningTask = nextTasks.find((task) => task.status === 'running')
    const firstHitTask = nextTasks.find((task) =>
      displayRuntimeEvents.value.some((event) => event.event_type === 'memory_hit' && Number(event.task_index || 0) === task.index)
    )
    selectedTaskKey.value = runningTask?.key ?? firstHitTask?.key ?? nextTasks[0].key
  }
}, { immediate: true })

watch(retrievalRows, (rows) => {
  if (!rows.length) {
    selectedMemoryId.value = ''
    return
  }
  if (!rows.some((memory) => memory.id === selectedMemoryId.value)) {
    selectedMemoryId.value = (rows.find((memory) => memory.adopted) ?? rows[0]).id
  }
}, { immediate: true })

onMounted(() => {
  refreshSavedConfig()
  refreshMemoryRecords()
  refreshRuntimeEvents()
  window.addEventListener('storage', refreshSavedConfig)
  window.addEventListener(FILTER_SYNC_EVENT, refreshSavedConfig)
  timer = window.setInterval(refreshRuntimeEvents, 1200)
})

onBeforeUnmount(() => {
  window.removeEventListener('storage', refreshSavedConfig)
  window.removeEventListener(FILTER_SYNC_EVENT, refreshSavedConfig)
  if (timer) window.clearInterval(timer)
})

function selectTask(taskKey) {
  selectedTaskKey.value = taskKey
}

function selectMemory(memoryId) {
  selectedMemoryId.value = memoryId
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
  if (!memorySelectionLocked.value) {
    sharedFilters.value = withFilterDefaults(savedConfig.value)
  }
}

async function refreshRuntimeEvents() {
  await refreshRunnerStatus()
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

async function refreshMemoryRecords() {
  if (memorySelectionLocked.value) return
  try {
    const response = await fetch(`${RUNNER_METRICS_URL}?t=${Date.now()}`, { cache: 'no-store' })
    if (!response.ok) throw new Error('历史结果服务未返回有效数据')
    const payload = await response.json()
    memoryRecords.value = normalizeMemoryRecords(Array.isArray(payload.records) ? payload.records : [])
    memoryLoadError.value = memoryRecords.value.length ? '' : '暂无历史结果，请先完成一次运行。'
  } catch {
    memoryRecords.value = []
    memoryLoadError.value = '暂无历史结果，请先启动本地运行服务或完成一次运行。'
  }
}

function updateSharedFilters(nextFilters) {
  if (memorySelectionLocked.value) return
  const normalized = withFilterDefaults({
    ...sharedFilters.value,
    ...nextFilters
  })
  if (normalized.baselineMode !== 'compare') {
    normalized.baselineRecordId = ''
  }
  sharedFilters.value = normalized
  persistSharedFilters(normalized)
}

function persistSharedFilters(value) {
  try {
    window.localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(value))
    window.dispatchEvent(new CustomEvent(FILTER_SYNC_EVENT))
  } catch {
    // Page-level filters still update even if storage is unavailable.
  }
}

function withFilterDefaults(value) {
  return {
    ...normalizeExperimentConfig({ ...defaultExperimentConfig(), ...value }),
    baselineMode: value?.baselineMode ?? 'none',
    baselineRecordId: value?.baselineRecordId ?? ''
  }
}

function normalizeMemoryRecords(records) {
  return records
    .filter((record) => record?.payload)
    .map((record, index) => ({
      ...record,
      recordId: String(record.recordId || `${record.source || 'record'}-${index}`),
      dataset: datasetKey(record.dataset || record.payload?.dataset_file || record.source || '')
    }))
}

function memoryRecordLabel(record) {
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

function taskStatus(index) {
  const events = displayRuntimeEvents.value.filter((event) => Number(event.task_index || 0) === index)
  if (events.some((event) => event.event_type === 'task_done')) return 'done'
  if (events.some((event) => event.event_type === 'task_start')) return 'running'
  return 'waiting'
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
    const memoryHits = importedMemoryHits(task)
    const memoryRefs = normalizeList(task.memory_refs || task.memory_hit_ids || task.memory_ids)
    const stateRefs = normalizeList(task.state_refs)
    events.push({
      event_type: 'task_start',
      timestamp,
      task_index: taskIndex,
      task_id: taskId,
      task_title: taskTitle,
      from_agent: 'runtime',
      to_agent: 'planner',
      action_type: 'start_task'
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
      input_params: {
        question: task.question || taskTitle,
        planner_focus: task.retrieval_debug?.planner_focus || []
      },
      state_refs: stateRefs
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
          summary: hit.summary || hit.text || hit.content || '导入结果记录了记忆命中',
          retrieval_score: hit.score ?? hit.retrieval_score ?? hit.confidence,
          memory_type: hit.memory_type || hit.type || 'memory',
          selection_reason: hit.selection_reason || hit.reason || '由导入实验结果恢复的记忆命中',
          tags: hit.tags || []
        }
      })
    }
    for (const write of importedMemoryWrites(task)) {
      const memoryId = write.memory_id || write.id || write.memory_ref || 'memory-write'
      events.push({
        event_type: 'memory_write',
        timestamp,
        task_index: taskIndex,
        task_id: taskId,
        task_title: taskTitle,
        from_agent: 'summarizer',
        to_agent: 'memory',
        action_type: 'write_memory',
        memory_refs: [memoryId],
        state_refs: stateRefs,
        return_result: {
          memory_id: memoryId,
          memory_type: write.memory_type || write.type || 'memory',
          summary: write.summary || write.text || write.content || '导入结果记录了记忆写回'
        }
      })
    }
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
        memory_hit_count: Number(task.memory_hit_count || memoryHits.length || metrics.memory_hit_count || 0)
      }
    })
  }
  events.push({ event_type: 'run_done', timestamp })
  return events
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

function importedMemoryHits(task) {
  const candidates = [
    task.memory_hits,
    task.retrieved_memories,
    task.retrieval_debug?.memory_hits,
    task.retrieval_debug?.context_hits
  ]
  for (const value of candidates) {
    if (Array.isArray(value) && value.length) return value
  }
  const refs = normalizeList(task.memory_refs || task.memory_hit_ids || task.memory_ids)
  return refs.map((memoryId) => ({ memory_id: memoryId, summary: '导入结果记录了该记忆参与本轮任务' }))
}

function importedMemoryWrites(task) {
  const writes = []
  for (const value of [task.latest_evidence_write, task.latest_experience_write, task.memory_write, task.memory_writes]) {
    if (Array.isArray(value)) writes.push(...value)
    else if (value && typeof value === 'object') writes.push(value)
  }
  return writes
}

function normalizeList(value) {
  if (Array.isArray(value)) return value.filter(Boolean).map(String)
  if (value && typeof value === 'object') return Object.values(value).filter(Boolean).map(String)
  return value ? [String(value)] : []
}

function eventPayloadText(event) {
  if (!event) return ''
  return stringifyShort(event.input_params || event.payload || event.return_result || event.action_type || '', 300)
}

function stringifyShort(value, limit = 240) {
  if (value == null) return ''
  if (typeof value === 'string') return shorten(value, limit)
  try {
    return shorten(JSON.stringify(value, null, 0), limit)
  } catch {
    return shorten(String(value), limit)
  }
}

function shorten(value, limit = 80) {
  const text = String(value || '').replace(/\s+/g, ' ').trim()
  if (text.length <= limit) return text || '--'
  return `${text.slice(0, limit)}...`
}

function scoreText(value) {
  const number = Number(value)
  return Number.isFinite(number) && value !== null && value !== undefined ? number.toFixed(3) : '--'
}

function formatTime(value) {
  if (!value) return '--'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '--'
  return date.toLocaleTimeString('zh-CN', { hour12: false })
}

function normalizeAgent(value) {
  const text = String(value || '')
  const map = {
    planner: 'Planner',
    retriever: 'Retriever',
    reasoner: 'Reasoner',
    executor: 'Executor',
    summarizer: 'Summarizer',
    memory: 'Memory',
    runtime: 'Runtime'
  }
  return map[text.toLowerCase()] || text || '--'
}

function selectionReason(result) {
  if (result.selection_reason) return shorten(result.selection_reason, 140)
  const terms = Array.isArray(result.matched_terms) ? result.matched_terms.filter(Boolean).slice(0, 5) : []
  if (terms.length) return `匹配查询词：${terms.join('、')}`
  const path = Array.isArray(result.reasoning_path) ? result.reasoning_path[0] : null
  if (path && typeof path === 'object') {
    const parts = [path.query_node, path.relation || path.hyperedge_id, path.memory_id || path.target_id || path.context_id]
      .filter(Boolean)
      .map(String)
    if (parts.length) return `检索路径：${parts.join(' → ')}`
  }
  if (result.retrieval_score != null) return `综合检索分数靠前：${result.retrieval_score}`
  return '由当前检索器排序返回'
}

function scoreComponentText(components) {
  if (!components || typeof components !== 'object') return ''
  const labels = {
    text_score: '文本',
    semantic_score: '语义',
    embedding_score: '向量',
    hypergraph_score: '超图',
    fusion_score: '融合',
    lexical_overlap_score: '词项',
    tag_overlap_score: '标签',
    graph_expansion_score: '图扩展',
    evidence_chain_query_score: '证据链'
  }
  return Object.entries(components)
    .filter(([key, value]) => key !== 'retrieval_score' && value != null)
    .slice(0, 3)
    .map(([key, value]) => `${labels[key] || key}=${value}`)
    .join('，')
}

function tagText(value) {
  if (Array.isArray(value)) {
    const tags = value.filter(Boolean).map(String)
    return tags.length ? tags.join('，') : '--'
  }
  return value ? String(value) : '--'
}

function modeLabel(value) {
  const map = {
    baseline_text: '纯文本',
    autogen_text: 'autogen',
    camel_text: 'camel',
    autogen_camel_text: 'autogen',
    autoform_text: 'AutoForm',
    llmlingua_text: 'LLMLingua-2',
    text: '纯文本',
    protocol_api: '结构化协议',
    protocol: '结构化协议',
    protocol_api_embedding: '协议 + 向量',
    hidden_api: '隐状态传递',
    hidden: '隐状态传递',
    latent_hidden: '隐状态传递'
  }
  return map[String(value || '').toLowerCase()] || value || '等待运行'
}

function datasetLabelFromMeta(meta) {
  const file = String(meta.dataset_file || '')
  if (file.includes('topiocqa_1_10turns')) return 'TopiOCQA 第 1 组（10轮）'
  if (file.includes('topiocqa_2_10turns')) return 'TopiOCQA 第 2 组（10轮）'
  if (file.includes('codeact_1_10tasks')) return 'CodeAct 第 1 组（10题）'
  if (file.includes('codeact_2_10tasks')) return 'CodeAct 第 2 组（10题）'
  return meta.dataset_type || '等待运行'
}

function memoryModeLabel(value) {
  const normalized = String(value || '').toLowerCase()
  if (['hypergraph', 'fusion'].includes(normalized)) return '超图记忆'
  if (['text', 'keyword', 'tag', 'semantic'].includes(normalized)) return '普通文本记忆'
  return value || '等待运行'
}

function retrievalModeLabelText(value) {
  const map = {
    keyword: '关键词',
    tag: '标签',
    semantic: '语义相似度',
    hypergraph: '超图跳跃',
    fusion: '融合检索'
  }
  return map[String(value || '').toLowerCase()] || value || '等待运行'
}

function byteText(value) {
  const bytes = Number(value || 0)
  if (!bytes) return '大小未记录'
  if (bytes < 1024) return `${bytes} B`
  return `${(bytes / 1024).toFixed(1)} KB`
}
</script>
