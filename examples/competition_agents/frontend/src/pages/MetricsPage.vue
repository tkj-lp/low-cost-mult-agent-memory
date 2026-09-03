<template>
  <section class="page metrics-page">
    <MetricsFilterBar
      :model-value="filters"
      :baseline-options="baselineOptions"
      @update:model-value="updateFilters"
      @refresh="refreshMetricsFromArchive"
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
      <section class="metrics-record-dialog" role="dialog" aria-modal="true" aria-label="导入实验结果">
        <header>
          <div>
            <strong>导入实验结果</strong>
            <span>{{ resultFileDirectory || 'run_result' }}</span>
          </div>
          <button type="button" class="metrics-dialog-close" @click="closeRecordDialog">关闭</button>
        </header>
        <div class="metrics-import-toolbar">
          <button type="button" @click="loadResultFiles">刷新</button>
          <button type="button" @click="selectAllResultFiles">全选</button>
          <button type="button" @click="clearSelectedResultFiles">取消全选</button>
          <span>已选择 {{ selectedImportFiles.length }} 个文件</span>
        </div>
        <div v-if="!resultFiles.length" class="metrics-record-empty">暂无可导入的 JSON 结果文件。</div>
        <div v-else class="metrics-import-tree">
          <section v-for="group in importFileGroups" :key="group.folder" class="metrics-import-folder">
            <label class="metrics-import-folder-row">
              <input
                type="checkbox"
                :checked="isImportFolderSelected(group)"
                @change="toggleImportFolder(group, $event.target.checked)"
              />
              <strong>{{ group.label }}</strong>
              <span>{{ group.files.length }} 个文件 / {{ formatFileSize(group.size) }}</span>
            </label>
            <label
              v-for="file in group.files"
              :key="file.path"
              class="metrics-import-file-row"
            >
              <input
                type="checkbox"
                :value="file.path"
                v-model="selectedImportFiles"
              />
              <span>{{ file.name }}</span>
              <em>{{ file.modifiedLabel }}</em>
              <small>{{ formatFileSize(file.size) }}</small>
            </label>
          </section>
        </div>
        <div class="metrics-import-footer">
          <button type="button" class="metrics-import-cancel" @click="closeRecordDialog">取消</button>
          <button
            type="button"
            class="metrics-import-submit"
            :disabled="!selectedImportFiles.length || importLoading"
            @click="importSelectedResults"
          >
            {{ importLoading ? '导入中...' : '导入所选' }}
          </button>
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
const FILTER_SYNC_EVENT = 'competition-agent-filter-sync'
const RUNNER_METRICS_URL = 'http://127.0.0.1:8765/api/runner/metrics'
const RUNNER_RESULT_FILES_URL = 'http://127.0.0.1:8765/api/runner/result-files'
const RUNNER_IMPORT_RESULTS_URL = 'http://127.0.0.1:8765/api/runner/import-results'
const LATEST_STDOUT_URL = '/runtime/latest_run_stdout.log'
const RUN_RESULT_MARKER = '=== run_result_json ==='

const props = defineProps({
  recordDialogOpen: {
    type: Boolean,
    default: false
  },
  importedRecords: {
    type: Array,
    default: () => []
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
const resultFiles = ref([])
const selectedImportFiles = ref([])
const resultFileDirectory = ref('')
const importLoading = ref(false)
const importedRecordsActive = ref(false)
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

const importFileGroups = computed(() => {
  const groups = new Map()
  for (const file of resultFiles.value) {
    const folder = file.folder || '未分组'
    if (!groups.has(folder)) groups.set(folder, [])
    groups.get(folder).push(file)
  }
  return [...groups.entries()].map(([folder, files]) => ({
    folder,
    label: folder,
    files,
    size: files.reduce((sum, file) => sum + Number(file.size || 0), 0)
  }))
})

const baselineSelected = computed(() => filters.baselineMode === 'compare')
const hasBaseline = computed(() => Boolean(baselineSelected.value && baselineRecord.value))
const statusLine = computed(() => {
  if (loadError.value) return loadError.value
  if (!selectedRecord.value && selectedChartRecords.value.length) return '当前数据集已有导入记录；当前方法组合还没有真实运行结果。'
  if (!selectedRecord.value) return '当前筛选组合还没有真实运行结果。'
  if (baselineSelected.value && !baselineRecord.value) return '请选择一条历史实验记录作为基线。'
  return `当前方法：${fullMethodLabel(selectedRecord.value)}`
})

const chartRecords = computed(() => {
  const sourceRecords = []
  if (hasBaseline.value) sourceRecords.push(baselineRecord.value)
  for (const record of selectedChartRecords.value) {
    if (hasBaseline.value && record.recordId === baselineRecord.value.recordId) continue
    sourceRecords.push(record)
  }
  return sourceRecords.map((record, index) => enrichRecord(record, index, sourceRecords.length))
})

const selectedChartRecords = computed(() => {
  const selected = records.value.filter((record) =>
    selectedRecordIds.value.includes(record.recordId) &&
    matchesCurrentDataset(record)
  )
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
        value: improvementText(baseline?.communicationCost, record?.communicationCost),
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
  const previousDataset = filters.dataset
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
  if (previousDataset !== filters.dataset) {
    selectCurrentDatasetRecords()
  }
  persistSharedFilters(filters)
}

function persistSharedFilters(value) {
  try {
    window.localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify({ ...value }))
    window.dispatchEvent(new CustomEvent(FILTER_SYNC_EVENT))
  } catch {
    // Page-level filters still update even if storage is unavailable.
  }
}

async function refreshMetricsFromArchive() {
  importedRecordsActive.value = false
  await fetchMetrics({ force: true })
}

async function fetchMetrics(options = {}) {
  if (importedRecordsActive.value && !options.force) return
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
  if (!selectedChartRecords.value.length && selectedRecord.value?.recordId) {
    selectedRecordIds.value = [selectedRecord.value.recordId]
  }
  if (filters.baselineRecordId && !validIds.has(filters.baselineRecordId)) {
    filters.baselineRecordId = ''
  }
}

function selectCurrentDatasetRecords() {
  selectedRecordIds.value = records.value
    .filter((record) => matchesCurrentDataset(record))
    .map((record) => record.recordId)
  loadError.value = ''
  syncSelectedRecords()
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

async function loadResultFiles() {
  try {
    const response = await fetch(`${RUNNER_RESULT_FILES_URL}?t=${Date.now()}`, { cache: 'no-store' })
    if (!response.ok) throw new Error('结果文件列表读取失败')
    const payload = await response.json()
    resultFiles.value = Array.isArray(payload.files) ? payload.files : []
    resultFileDirectory.value = payload.directory || ''
    const validPaths = new Set(resultFiles.value.map((file) => file.path))
    selectedImportFiles.value = selectedImportFiles.value.filter((path) => validPaths.has(path))
  } catch (error) {
    resultFiles.value = []
    selectedImportFiles.value = []
    loadError.value = error instanceof Error ? error.message : String(error)
  }
}

function selectAllResultFiles() {
  selectedImportFiles.value = resultFiles.value.map((file) => file.path)
}

function clearSelectedResultFiles() {
  selectedImportFiles.value = []
}

function isImportFolderSelected(group) {
  return group.files.length > 0 && group.files.every((file) => selectedImportFiles.value.includes(file.path))
}

function toggleImportFolder(group, checked) {
  const selected = new Set(selectedImportFiles.value)
  for (const file of group.files) {
    if (checked) selected.add(file.path)
    else selected.delete(file.path)
  }
  selectedImportFiles.value = [...selected]
}

async function importSelectedResults() {
  if (!selectedImportFiles.value.length || importLoading.value) return
  importLoading.value = true
  try {
    const response = await fetch(RUNNER_IMPORT_RESULTS_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ files: selectedImportFiles.value })
    })
    if (!response.ok) throw new Error('导入结果失败')
    const payload = await response.json()
    records.value = normalizeRecords(Array.isArray(payload.records) ? payload.records : [])
    importedRecordsActive.value = true
    selectCurrentDatasetRecords()
    loadError.value = selectedRecordIds.value.length ? '' : '所选文件没有当前数据集的实验记录。'
    closeRecordDialog()
  } catch (error) {
    loadError.value = error instanceof Error ? error.message : String(error)
  } finally {
    importLoading.value = false
  }
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
      dataset: datasetKey(record.dataset || record.source || ''),
      sourceRecordId,
      recordId: count ? `${base}|${count}` : base
    }
  })
}

function matchesCurrentDataset(record) {
  return record?.dataset === filters.dataset
}

function latestRecordPerMethod(sourceRecords) {
  const byMethod = new Map()
  for (const record of sourceRecords) {
    const key = [
      record.dataset,
      record.collaborationMode,
      record.memoryMode,
      record.retrievalMode,
      record.experimentMode || ''
    ].join('|')
    const previous = byMethod.get(key)
    if (!previous || recordTimestamp(record) >= recordTimestamp(previous)) {
      byMethod.set(key, record)
    }
  }
  return [...byMethod.values()]
}

function recordTimestamp(record) {
  const time = record?.archivedAt ? new Date(record.archivedAt).getTime() : 0
  return Number.isFinite(time) ? time : 0
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

function enrichRecord(record, index = 0, total = 1) {
  const tone = toneForRecord(record)
  return {
    ...record,
    tone,
    color: colorForRecord(record, index),
    shortMethod: shortMethodLabel(record, index, total)
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
  if (open) {
    selectedImportFiles.value = []
    loadResultFiles()
  }
})

watch(() => props.importedRecords, (nextRecords) => {
  if (!Array.isArray(nextRecords) || !nextRecords.length) return
  records.value = normalizeRecords(nextRecords)
  importedRecordsActive.value = true
  selectCurrentDatasetRecords()
  loadError.value = selectedRecordIds.value.length ? '' : '所选文件没有当前数据集的实验记录。'
}, { deep: true, immediate: true })

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
  const cost = Number(record.communicationCost || 0)
  if (!cost) return '--'
  return byteText(cost)
}

function communicationKpiDescText(record) {
  if (!record) return '当前筛选组合暂无真实结果'
  const chars = Number(record.textChars || 0)
  const protocolBytes = Number(record.protocolBytes || 0)
  const stateCount = Number(record.stateTransferCount || 0)
  const stateBytes = Number(record.stateBytes || 0)
  const communicationCost = Number(record.communicationCost || 0)
  if (!chars && !protocolBytes && !stateCount && !stateBytes && !communicationCost) return '当前筛选组合暂无真实结果'
  const parts = [`文本 ${integerText(chars)} 字`]
  if (protocolBytes) {
    parts.push(`协议 ${byteText(protocolBytes)}`)
  }
  if (stateCount || stateBytes) {
    parts.push(`状态 ${integerText(stateCount)} 次 / ${byteText(stateBytes)}`)
  }
  if (communicationCost) {
    parts.push(`载荷 ${byteText(communicationCost)}`)
  }
  return parts.join(' · ')
}

function communicationBaselineDesc(record, baseline) {
  const saving = improvementText(baseline?.communicationCost, record?.communicationCost)
  const current = communicationKpiDescText(record)
  return saving === '--' ? current : `通信载荷较基线下降 ${saving} · ${current}`
}

function communicationCostText(record) {
  if (!record) return '--'
  const messages = Number(record.messageCount || 0)
  const chars = Number(record.textChars || 0)
  const protocolBytes = Number(record.protocolBytes || 0)
  const stateCount = Number(record.stateTransferCount || 0)
  const stateBytes = Number(record.stateBytes || 0)
  const communicationCost = Number(record.communicationCost || 0)
  if (!messages && !chars && !protocolBytes && !stateBytes && !communicationCost) return '--'
  const parts = [`${integerText(messages)} 次`, `${integerText(chars)} 字`]
  if (protocolBytes) {
    parts.push(`协议 ${byteText(protocolBytes)}`)
  }
  if (stateCount || stateBytes) {
    parts.push(`状态 ${integerText(stateCount)} 次 / ${byteText(stateBytes)}`)
  }
  if (communicationCost) {
    parts.push(`载荷 ${byteText(communicationCost)}`)
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
    { label: 'CodeAct 第 1 组（10题）', value: 'codeact-1' },
    { label: 'CodeAct 第 2 组（10题）', value: 'codeact-2' }
  ]
}

function shortMethodLabel(record, index = 0, total = 1) {
  const variant = recordVariantLabel(record)
  const label = `${labelOf(collaborationModes, record.collaborationMode)}${variant ? ` ${variant}` : ''} + ${retrievalLabel(record)}`
  return total > 1 ? `${label} #${index + 1}` : label
}

function recordVariantLabel(record) {
  const sourceText = `${record?.source || ''} ${record?.sourceRecordId || ''} ${record?.recordId || ''}`
  const rateMatch = sourceText.match(/llmlingua_text_([0-9.]+)_/i)
  if (rateMatch) return `rate=${rateMatch[1]}`
  const archiveText = String(record?.archiveId || record?.sourceRecordId || record?.recordId || '')
  const archiveRateMatch = archiveText.match(/llmlingua_text_([0-9.]+)/i)
  if (archiveRateMatch) return `rate=${archiveRateMatch[1]}`
  return ''
}

function retrievalLabel(record) {
  return labelOf(retrievalOptionsFor(record.memoryMode), record.retrievalMode, record.retrievalMode)
}

function toneForMode(mode) {
  return {
    autogen: 'blue',
    camel: 'purple',
    autoform: 'cyan',
    llmlingua: 'orange',
    plain: 'blue',
    structured: 'green',
    'protocol-vector': 'orange',
    latent: 'purple'
  }[mode] || 'blue'
}

function toneForRecord(record) {
  return toneForMode(record?.collaborationMode)
}

function colorForTone(tone) {
  return {
    blue: '#1677ff',
    green: '#10a96b',
    orange: '#ff7a00',
    purple: '#7651e8'
  }[tone] || '#1677ff'
}

function colorForRecord(record, index = 0) {
  const palette = [
    '#1677ff',
    '#7651e8',
    '#10a96b',
    '#ff7a00',
    '#e5484d',
    '#08979c',
    '#9c27b0',
    '#5c6bc0',
    '#c2410c',
    '#0f766e',
    '#be185d',
    '#7c3aed',
    '#2563eb',
    '#65a30d',
    '#ca8a04',
    '#475569'
  ]
  return palette[index % palette.length] || colorForTone(toneForRecord(record))
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

function formatFileSize(value) {
  const bytes = Number(value || 0)
  if (!bytes) return '0 B'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
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
    protocolBytes: Number(metrics.protocol_bytes || metrics.text_chars || 0),
    stateTransferCount: collaborationMode === 'plain' ? 0 : Number(metrics.latent_transfer_count || 0),
    stateBytes: collaborationMode === 'plain' ? 0 : Number(metrics.latent_bytes || 0),
    communicationCost: Number(metrics.communication_bytes || Number(metrics.protocol_bytes || metrics.text_chars || 0)),
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
  if (text.includes('codeact_1_10tasks')) return 'codeact-1'
  if (text.includes('codeact_2_10tasks')) return 'codeact-2'
  if (text === 'codeact_2_10tasks.json') return 'codeact-2'
  return text
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
</script>
