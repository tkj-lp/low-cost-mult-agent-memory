<template>
  <section class="page metrics-page">
    <MetricsFilterBar
      :model-value="filters"
      :baseline-options="baselineOptions"
      @update:model-value="updateFilters"
      @refresh="fetchMetrics"
    />
    <div class="metrics-content">
      <div class="metrics-status-line">{{ statusLine }}</div>

      <div class="metrics-kpi-grid">
        <MetricKpiCard
          v-for="item in kpis"
          :key="item.title"
          v-bind="item"
        />
      </div>

      <div class="metrics-chart-grid">
        <LineChartCard
          title="文本通信开销趋势"
          unit="字符数"
          :series="charSeries"
          :labels="chartLabels"
        />
        <LineChartCard
          title="总任务耗时趋势"
          unit="秒"
          :series="timeSeries"
          :labels="chartLabels"
        />
        <StateTransferCard :rows="chartRecords" />
      </div>

      <div class="metrics-lower-grid">
        <MethodComparisonTable :rows="methodRows" />
      </div>
    </div>

    <div v-if="props.recordDialogOpen" class="metrics-record-backdrop" @click.self="closeRecordDialog">
      <section class="metrics-record-dialog" role="dialog" aria-modal="true" aria-label="实验记录">
        <header>
          <div>
            <strong>实验记录</strong>
            <span>选择要展示到图表中的记录；基线仍由上方“基线记录”单独决定。</span>
          </div>
          <button type="button" class="metrics-dialog-close" @click="closeRecordDialog">关闭</button>
        </header>
        <div v-if="!records.length" class="metrics-record-empty">暂无实验记录，完整运行一次后会自动保存。</div>
        <div v-else class="metrics-record-list dialog-list">
          <article
            v-for="record in records"
            :key="record.recordId"
            class="metrics-record-item"
            :class="{ selected: selectedRecordIds.includes(record.recordId) }"
          >
            <label>
              <input
                type="checkbox"
                :checked="selectedRecordIds.includes(record.recordId)"
                @change="toggleRecord(record.recordId, $event.target.checked)"
              />
              <span>
                <strong>{{ fullMethodLabel(record) }}</strong>
                <em>{{ recordMetaLabel(record) }}</em>
                <small class="metrics-record-time">{{ recordTimeLabel(record) }}</small>
              </span>
            </label>
            <button type="button" @click="deleteRecord(record)">删除</button>
          </article>
        </div>
      </section>
    </div>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import MetricsFilterBar from '../components/metrics/MetricsFilterBar.vue'
import MetricKpiCard from '../components/metrics/MetricKpiCard.vue'
import LineChartCard from '../components/metrics/LineChartCard.vue'
import StateTransferCard from '../components/metrics/StateTransferCard.vue'
import MethodComparisonTable from '../components/metrics/MethodComparisonTable.vue'
import {
  collaborationModes,
  defaultExperimentConfig,
  labelOf,
  memoryModes,
  normalizeExperimentConfig,
  retrievalOptionsFor
} from '../experimentOptions'

const CONFIG_STORAGE_KEY = 'competition-agent-console-config'
const RUNNER_METRICS_URL = 'http://127.0.0.1:8765/api/runner/metrics'
const LATEST_STDOUT_URL = '/runtime/latest_run_stdout.log'
const RUN_RESULT_MARKER = '=== run_result_json ==='

const props = defineProps({
  recordDialogOpen: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['close-records'])

const savedConfig = readSavedConfig()
const filters = reactive({
  dataset: savedConfig.dataset,
  collaborationMode: savedConfig.collaborationMode,
  memoryMode: savedConfig.memoryMode,
  retrievalMode: savedConfig.retrievalMode,
  baselineMode: 'none',
  baselineRecordId: ''
})
const records = ref([])
const selectedRecordIds = ref([])
const loadError = ref('')
let timer = null

const selectedRecord = computed(() =>
  records.value.find((record) =>
    record.dataset === filters.dataset &&
    record.collaborationMode === filters.collaborationMode &&
    record.memoryMode === filters.memoryMode &&
    record.retrievalMode === filters.retrievalMode
  ) ?? null
)

const baselineRecord = computed(() => {
  if (filters.baselineMode !== 'compare' || !filters.baselineRecordId) return null
  return records.value.find((record) => record.recordId === filters.baselineRecordId) ?? null
})

const baselineOptions = computed(() =>
  records.value.map((record) => ({
    label: recordOptionLabel(record),
    value: record.recordId
  }))
)

const baselineSelected = computed(() => filters.baselineMode === 'compare')
const hasBaseline = computed(() => Boolean(baselineSelected.value && baselineRecord.value))
const statusLine = computed(() => {
  if (loadError.value) return loadError.value
  if (!selectedRecord.value) return '当前筛选组合还没有真实运行结果。'
  if (baselineSelected.value && !baselineRecord.value) return '请选择一条历史实验记录作为基线。'
  return `当前方法：${fullMethodLabel(selectedRecord.value)}`
})

const chartRecords = computed(() => {
  const list = []
  if (hasBaseline.value) list.push(enrichRecord(baselineRecord.value, 'blue'))
  for (const record of selectedChartRecords.value) {
    if (hasBaseline.value && record.recordId === baselineRecord.value.recordId) continue
    list.push(enrichRecord(record, toneForMode(record.collaborationMode)))
  }
  return list
})

const selectedChartRecords = computed(() => {
  const selected = records.value.filter((record) => selectedRecordIds.value.includes(record.recordId))
  if (selected.length) return selected
  return selectedRecord.value ? [selectedRecord.value] : []
})

const chartLabels = computed(() => {
  const maxLength = Math.max(...chartRecords.value.map((record) => record.series.textChars.length), 0)
  return Array.from({ length: maxLength }, (_, index) => `任务${index + 1}`)
})

const charSeries = computed(() => buildSeries('textChars'))
const timeSeries = computed(() => buildSeries('elapsedSeconds'))

const kpis = computed(() => {
  const record = selectedRecord.value
  if (baselineSelected.value) {
    const baseline = baselineRecord.value
    return [
      {
        title: '字符节省率',
        value: improvementText(baseline?.textChars, record?.textChars),
        desc: baseline && record ? `当前 ${integerText(record.textChars)} 字 / 基线 ${integerText(baseline.textChars)} 字` : '缺少匹配基线',
        tone: 'blue',
        icon: '▣'
      },
      {
        title: '单任务耗时下降',
        value: improvementText(avgTaskMs(baseline), avgTaskMs(record)),
        desc: baseline && record ? `当前 ${secondsText(avgTaskMs(record))} / 基线 ${secondsText(avgTaskMs(baseline))}` : '缺少匹配基线',
        tone: 'green',
        icon: '◷'
      },
      {
        title: '通信开销',
        value: communicationKpiValueText(record),
        desc: baseline && record ? communicationBaselineDesc(record, baseline) : '缺少匹配基线',
        tone: 'purple',
        icon: '↔'
      },
      qualityKpi('记忆复用', record?.memoryHitRate, '旧口径 memory_hit_rate：有记忆命中的任务占比', 'orange', '▰'),
      qualityKpi('回答准确率', record?.answerAccuracy, '旧口径 exact_match：由 run_demo 的 answers_match 汇总', 'blue', '◎')
    ]
  }

  return [
    absoluteKpi('文本字符数', integerText(record?.textChars), record ? '累计文本通信字符' : '当前筛选组合暂无真实结果', 'blue', '▣'),
    absoluteKpi('单任务耗时', secondsText(avgTaskMs(record)), record ? '平均每个任务端到端耗时' : '当前筛选组合暂无真实结果', 'green', '◷'),
    communicationKpi(record),
    qualityKpi('记忆复用', record?.memoryHitRate, '旧口径 memory_hit_rate：有记忆命中的任务占比', 'orange', '▰'),
    qualityKpi('回答准确率', record?.answerAccuracy, '旧口径 exact_match：由 run_demo 的 answers_match 汇总', 'blue', '◎')
  ]
})

const methodRows = computed(() => {
  const baseline = baselineSelected.value ? baselineRecord.value : null
  return chartRecords.value.map((record) => ({
    id: record.recordId || `${record.dataset}|${record.collaborationMode}|${record.memoryMode}|${record.retrievalMode}|${record.source}`,
    method: fullMethodLabel(record),
    messages: integerText(record.messageCount),
    chars: integerText(record.textChars),
    stateCount: integerText(record.stateTransferCount),
    stateSize: byteText(record.stateBytes),
    duration: secondsText(record.elapsedMs),
    avgDuration: secondsText(avgTaskMs(record)),
    memoryHit: percentText(record.memoryHitRate),
    expectedReuse: percentText(record.expectedReuseHitRate),
    memoryReuse: percentText(record.memoryHitRate),
    precision: percentText(record.memoryPrecisionRate),
    communication: communicationCostText(record),
    accuracy: percentText(record.answerAccuracy),
    recall: percentText(record.memoryRecallRate),
    improvement: baseline && record !== baseline ? overallImprovementText(baseline, record) : '--',
    tone: record.tone
  }))
})

onMounted(() => {
  fetchMetrics()
  timer = window.setInterval(fetchMetrics, 3000)
})

onBeforeUnmount(() => {
  if (timer) window.clearInterval(timer)
})

function updateFilters(nextFilters) {
  const normalized = normalizeExperimentConfig({
    ...defaultExperimentConfig(),
    ...filters,
    ...nextFilters
  })
  filters.dataset = normalized.dataset
  filters.collaborationMode = normalized.collaborationMode
  filters.memoryMode = normalized.memoryMode
  filters.retrievalMode = normalized.retrievalMode
  filters.baselineMode = nextFilters.baselineMode ?? filters.baselineMode
  filters.baselineRecordId = nextFilters.baselineRecordId ?? filters.baselineRecordId
  if (filters.baselineMode !== 'compare') {
    filters.baselineRecordId = ''
  }
}

async function fetchMetrics() {
  try {
    const response = await fetch(`${RUNNER_METRICS_URL}?t=${Date.now()}`, { cache: 'no-store' })
    if (!response.ok) throw new Error('指标服务未返回有效结果')
    const payload = await response.json()
    records.value = normalizeRecords(Array.isArray(payload.records) ? payload.records : [])
    syncSelectedRecords()
    loadError.value = records.value.length ? '' : '暂无真实指标数据：请先完成一次实验运行。'
  } catch {
    records.value = []
    syncSelectedRecords()
    loadError.value = '暂无真实指标数据：请先启动本地运行服务并完成一次实验。'
  }
}

function syncSelectedRecords() {
  const validIds = new Set(records.value.map((record) => record.recordId))
  selectedRecordIds.value = selectedRecordIds.value.filter((id) => validIds.has(id))
  if (!selectedRecordIds.value.length && selectedRecord.value?.recordId) {
    selectedRecordIds.value = [selectedRecord.value.recordId]
  }
  if (filters.baselineRecordId && !validIds.has(filters.baselineRecordId)) {
    filters.baselineRecordId = ''
  }
}

function toggleRecord(recordId, checked) {
  const current = new Set(selectedRecordIds.value)
  if (checked) current.add(recordId)
  else current.delete(recordId)
  selectedRecordIds.value = [...current]
}

function closeRecordDialog() {
  emit('close-records')
}

async function deleteRecord(record) {
  try {
    const response = await fetch('http://127.0.0.1:8765/api/runner/delete-record', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ recordId: record.sourceRecordId || record.recordId })
    })
    if (!response.ok) throw new Error('删除记录失败')
    const payload = await response.json()
    records.value = Array.isArray(payload.records)
      ? normalizeRecords(payload.records)
      : records.value.filter((item) => item.recordId !== record.recordId)
    selectedRecordIds.value = selectedRecordIds.value.filter((id) => id !== record.recordId)
    if (filters.baselineRecordId === record.recordId) filters.baselineRecordId = ''
    syncSelectedRecords()
  } catch (error) {
    loadError.value = error instanceof Error ? error.message : String(error)
  }
}

function normalizeRecords(rawRecords) {
  const seen = new Map()
  return rawRecords.map((record, index) => {
    const sourceRecordId = String(record.recordId || `record-${index}`)
    const base = [
      sourceRecordId,
      record.fingerprint || '',
      record.archivedAt || '',
      record.source || '',
      index
    ].filter(Boolean).join('|')
    const count = seen.get(base) || 0
    seen.set(base, count + 1)
    return {
      ...record,
      sourceRecordId,
      recordId: count ? `${base}|${count}` : base
    }
  })
}

async function readLatestStdoutMetrics() {
  try {
    const response = await fetch(`${LATEST_STDOUT_URL}?t=${Date.now()}`, { cache: 'no-store' })
    if (!response.ok) return []
    const text = await response.text()
    const index = text.lastIndexOf(RUN_RESULT_MARKER)
    if (index < 0) return []
    const payload = JSON.parse(text.slice(index + RUN_RESULT_MARKER.length).trim().split(/\r?\n/)[0])
    return (payload.runs || []).map((run, index) => fallbackRecordFromRun(payload, run, index))
  } catch {
    return []
  }
}

function readSavedConfig() {
  try {
    const raw = window.localStorage.getItem(CONFIG_STORAGE_KEY)
    return raw ? normalizeExperimentConfig({ ...defaultExperimentConfig(), ...JSON.parse(raw) }) : defaultExperimentConfig()
  } catch {
    return defaultExperimentConfig()
  }
}

function buildSeries(key) {
  return chartRecords.value
    .map((record) => ({
      name: record.shortMethod,
      color: record.color,
      values: record.series[key] || []
    }))
    .filter((line) => line.values.length)
}

function enrichRecord(record, tone) {
  return {
    ...record,
    tone,
    color: colorForTone(tone),
    shortMethod: shortMethodLabel(record)
  }
}

watch([records, () => filters.baselineMode], () => {
  if (filters.baselineMode !== 'compare') return
  if (!records.value.length) {
    filters.baselineRecordId = ''
    return
  }
  if (!records.value.some((record) => record.recordId === filters.baselineRecordId)) {
    const plainMatch = records.value.find((record) =>
      record.dataset === filters.dataset &&
      record.collaborationMode === 'plain' &&
      record.memoryMode === filters.memoryMode &&
      record.retrievalMode === filters.retrievalMode
    )
    filters.baselineRecordId = (plainMatch ?? records.value[0]).recordId
  }
}, { immediate: true })

watch(() => props.recordDialogOpen, (open) => {
  if (open) fetchMetrics()
})

function qualityKpi(title, value, desc, tone, icon) {
  return {
    title,
    value: percentText(value),
    desc: value == null ? '真实结果暂未提供该字段' : desc,
    tone,
    icon
  }
}

function absoluteKpi(title, value, desc, tone, icon) {
  return {
    title,
    value: value ?? '--',
    desc,
    tone,
    icon
  }
}

function avgTaskMs(record) {
  if (!record) return null
  const taskCount = Number(record.taskCount || 0)
  const elapsedMs = Number(record.elapsedMs || 0)
  if (!taskCount || !elapsedMs) return null
  return elapsedMs / taskCount
}

function communicationKpi(record) {
  return {
    title: '通信开销',
    value: communicationKpiValueText(record),
    desc: communicationKpiDescText(record),
    tone: 'purple',
    icon: '↔'
  }
}

function communicationKpiValueText(record) {
  if (!record) return '--'
  const messages = Number(record.messageCount || 0)
  const stateBytes = Number(record.stateBytes || 0)
  if (!messages && !stateBytes) return '--'
  return `${integerText(messages)} 次`
}

function communicationKpiDescText(record) {
  if (!record) return '当前筛选组合暂无真实结果'
  const chars = Number(record.textChars || 0)
  const stateCount = Number(record.stateTransferCount || 0)
  const stateBytes = Number(record.stateBytes || 0)
  if (!chars && !stateCount && !stateBytes) return '当前筛选组合暂无真实结果'
  const parts = [`文本 ${integerText(chars)} 字`]
  if (stateCount || stateBytes) {
    parts.push(`状态 ${integerText(stateCount)} 次 / ${byteText(stateBytes)}`)
  }
  return parts.join(' · ')
}

function communicationBaselineDesc(record, baseline) {
  const saving = improvementText(baseline?.communicationCost, record?.communicationCost)
  const current = communicationKpiDescText(record)
  return saving === '--' ? current : `文本通信较基线下降 ${saving} · ${current}`
}

function communicationCostText(record) {
  if (!record) return '--'
  const messages = Number(record.messageCount || 0)
  const chars = Number(record.textChars || 0)
  const stateCount = Number(record.stateTransferCount || 0)
  const stateBytes = Number(record.stateBytes || 0)
  if (!messages && !chars && !stateBytes) return '--'
  const parts = [`${integerText(messages)} 次`, `${integerText(chars)} 字`]
  if (stateCount || stateBytes) {
    parts.push(`状态 ${integerText(stateCount)} 次 / ${byteText(stateBytes)}`)
  }
  return parts.join(' / ')
}

function fullMethodLabel(record) {
  return `${labelOf(collaborationModes, record.collaborationMode)} + ${labelOf(memoryModes, record.memoryMode)} + ${retrievalLabel(record)}`
}

function recordOptionLabel(record) {
  const dataset = labelOf(datasetOptions(), record.dataset, record.dataset)
  const method = fullMethodLabel(record)
  const taskCount = record.taskCount ? `${record.taskCount}轮` : '任务数未知'
  const time = recordTimeText(record)
  return `${dataset}｜${method}｜${taskCount}｜${time}`
}

function recordMetaLabel(record) {
  const dataset = labelOf(datasetOptions(), record.dataset, record.dataset)
  const taskCount = record.taskCount ? `${record.taskCount}轮` : '任务数未知'
  return `${dataset}｜${taskCount}`
}

function recordTimeLabel(record) {
  return `运行时间：${recordTimeText(record)}`
}

function recordTimeText(record) {
  if (record.archivedAt) {
    const time = new Date(record.archivedAt)
    if (!Number.isNaN(time.getTime())) {
      return time.toLocaleString('zh-CN', { hour12: false })
    }
  }
  return record.source || '未记录'
}

function datasetOptions() {
  return [
    { label: 'TopiOCQA 第 1 组（10轮）', value: 'topiocqa-1' },
    { label: 'TopiOCQA 第 2 组（10轮）', value: 'topiocqa-2' },
    { label: 'Hotpot 连续任务（3轮）', value: 'hotpot-3' },
    { label: 'CodeAct 第 1 组（10题）', value: 'codeact-1' }
  ]
}

function shortMethodLabel(record) {
  return `${labelOf(collaborationModes, record.collaborationMode)} + ${retrievalLabel(record)}`
}

function retrievalLabel(record) {
  return labelOf(retrievalOptionsFor(record.memoryMode), record.retrievalMode, record.retrievalMode)
}

function toneForMode(mode) {
  return {
    plain: 'blue',
    structured: 'green',
    'protocol-vector': 'orange',
    latent: 'purple'
  }[mode] || 'blue'
}

function colorForTone(tone) {
  return {
    blue: '#1677ff',
    green: '#10a96b',
    orange: '#ff7a00',
    purple: '#7651e8'
  }[tone] || '#1677ff'
}

function percentText(value) {
  return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '--'
}

function improvementText(base, current) {
  if (!base || current == null) return '--'
  return `${(((base - current) / base) * 100).toFixed(1)}%`
}

function overallImprovementText(base, current) {
  const gains = [
    base.textChars ? (base.textChars - current.textChars) / base.textChars : null,
    base.elapsedMs ? (base.elapsedMs - current.elapsedMs) / base.elapsedMs : null
  ].filter((item) => typeof item === 'number' && Number.isFinite(item))
  if (!gains.length) return '--'
  const score = gains.reduce((sum, item) => sum + item, 0) / gains.length
  return `${score >= 0 ? '↑' : '↓'} ${Math.abs(score * 100).toFixed(1)}%`
}

function integerText(value) {
  return Number(value || 0).toLocaleString('zh-CN')
}

function compactNumber(value) {
  const number = Number(value || 0)
  if (number >= 1000000) return `${(number / 1000000).toFixed(1)}M`
  if (number >= 1000) return `${(number / 1000).toFixed(1)}k`
  return integerText(number)
}

function byteText(value) {
  const bytes = Number(value || 0)
  if (!bytes) return '--'
  return `${(bytes / 1024).toFixed(1)} KB`
}

function secondsText(ms) {
  const value = Number(ms || 0) / 1000
  return value ? `${value.toFixed(2)} 秒` : '--'
}

function fallbackRecordFromRun(payload, run, index = 0) {
  const metrics = run.metrics || {}
  const taskResults = run.task_results || []
  const collaborationMode = collaborationKey(run.experiment_mode || payload.experiment_mode || run.mode)
  const reuseSummary = run.reuse_report?.summary || {}
  const expectedReuseTaskCount = Number(reuseSummary.expected_reuse_task_count || 0)
  const elapsedMs = Number(run.elapsed_ms || metrics.elapsed_ms || 0)
  const stateGenerationMs = collaborationMode === 'plain' ? 0 : Number(metrics.state_generation_ms || 0)
  const comparableElapsedMs = Number(metrics.adjusted_elapsed_ms ?? Math.max(0, elapsedMs - stateGenerationMs))
  const dataset = datasetKey(payload.dataset_file || run.dataset_file)
  const retrievalMode = run.memory_retrieval_mode || payload.memory_retrieval_mode
  const memoryHitRate = numberOrNull(run.memory_hit_rate ?? metrics.memory_hit_rate)
  const expectedReuseHitRate = expectedReuseTaskCount
    ? numberOrNull(reuseSummary.memory_retrieval_accuracy)
    : null
  const answerAccuracy = exactMatchRate(run, metrics, taskResults)
  return {
    recordId: [
      payload.archive_id || 'latest',
      index,
      dataset,
      run.experiment_mode || payload.experiment_mode || run.mode,
      retrievalMode
    ].join('|'),
    archivedAt: payload.archived_at || null,
    source: 'latest',
    dataset,
    collaborationMode,
    experimentMode: run.experiment_mode || payload.experiment_mode || run.mode,
    memoryMode: memoryKey(run.memory_organization || payload.memory_organization || run.memory_retrieval_mode),
    retrievalMode,
    taskCount: Number(run.task_count || run.tasks || metrics.task_count || payload.task_count || 0),
    messageCount: Number(metrics.message_count || 0),
    textChars: Number(metrics.text_chars || 0),
    stateTransferCount: collaborationMode === 'plain' ? 0 : Number(metrics.latent_transfer_count || 0),
    stateBytes: collaborationMode === 'plain' ? 0 : Number(metrics.latent_bytes || 0),
    communicationCost: Number(metrics.text_chars || 0),
    stateGenerationMs,
    elapsedMs,
    comparableElapsedMs: collaborationMode === 'plain' ? elapsedMs : comparableElapsedMs,
    memoryHitRate,
    expectedReuseHitRate,
    memoryReuseRate: expectedReuseTaskCount ? numberOrNull(reuseSummary.correct_reuse_rate) : null,
    effectiveMemoryReuseRate: Object.prototype.hasOwnProperty.call(metrics, 'reuse_success_count')
      ? Math.min(1, Number(metrics.reuse_success_count || 0) / Number(run.task_count || run.tasks || 1))
      : null,
    answerAccuracy,
    memoryRecallRate: expectedReuseTaskCount ? numberOrNull(reuseSummary.expected_artifact_recall) : null,
    memoryPrecisionRate: expectedReuseTaskCount ? numberOrNull(reuseSummary.memory_precision) : null,
    memoryCompleteRetrievalRate: expectedReuseTaskCount ? numberOrNull(reuseSummary.memory_complete_retrieval_rate) : null,
    expectedReuseTaskCount,
    series: {
      textChars: taskResults.map((task) => Number(task.metrics?.text_chars || 0)),
      elapsedSeconds: taskResults.map((task) => Number(task.metrics?.elapsed_ms || 0) / 1000),
      memoryReuse: taskResults.map((task) => Number(task.metrics?.memory_hit_rate || (task.memory_hit_count ? 1 : 0)) * 100)
    }
  }
}

function exactMatchRate(run, metrics, taskResults) {
  const provided = numberOrNull(run.exact_match ?? metrics.exact_match)
  if (provided != null) return provided
  const comparable = taskResults
    .map((task) => task.exact_match)
    .filter((value) => typeof value === 'boolean')
  if (!comparable.length) return null
  return comparable.filter(Boolean).length / comparable.length
}

function numberOrNull(value) {
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

function datasetKey(file) {
  const text = String(file || '')
  if (text.includes('topiocqa_1_10turns')) return 'topiocqa-1'
  if (text.includes('topiocqa_2_10turns')) return 'topiocqa-2'
  if (text.includes('hotpot_continuous_tasks_3turn')) return 'hotpot-3'
  if (text.includes('codeact_1_10tasks')) return 'codeact-1'
  return text
}

function collaborationKey(mode) {
  return {
    baseline_text: 'plain',
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
</script>
