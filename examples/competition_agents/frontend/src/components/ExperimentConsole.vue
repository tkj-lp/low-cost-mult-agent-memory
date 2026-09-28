<template>
  <div class="console-layout">
    <section class="top-grid">
      <ConfigCard :model-value="config" @update:config="updateConfig" />
      <RunControlCard
        :config="config"
        :runtime="runtime"
        @start="startExperiment()"
        @stop="stopExperiment"
        @import-result="importResult"
        @show-records="showRecords"
        @clear-experience="clearExperienceMemory"
      />
    </section>
    <ExperimentMatrix
      :rows="matrixRows"
      :config="config"
      :running-row-id="runtime.runningRowId"
      @run-record="startExperiment"
    />
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import ConfigCard from './ConfigCard.vue'
import RunControlCard from './RunControlCard.vue'
import ExperimentMatrix from './ExperimentMatrix.vue'
import {
  buildMatrixRows,
  configFromRunResult,
  decodeRunResultBuffer,
  defaultExperimentConfig,
  datasetKeyForFile,
  datasets,
  normalizeExperimentConfig,
  parseRunResultText,
  selectedCollaboration,
  selectedDataset
} from '../experimentOptions'

const CONFIG_STORAGE_KEY = 'competition-agent-console-config'
const RUNNER_API = 'http://127.0.0.1:8765/api/runner'

const config = reactive(loadInitialConfig())
const statusTimer = ref(null)

const runtime = reactive({
  status: 'idle',
  statusText: '未开始',
  currentTask: 0,
  completedTasks: 0,
  failedTasks: 0,
  totalTasks: config.taskCount,
  progress: 0,
  currentAgent: '未开始',
  startedAt: '--',
  elapsedText: '--',
  importedName: '',
  importedSummary: null,
  runningRowId: '',
  lastError: '',
  messageCount: 0,
  textChars: 0,
  stateTransfers: 0,
  memoryHits: 0,
  clearingMemory: false,
  memoryClearMessage: ''
})

const matrixRows = computed(() => buildMatrixRows(config))

async function clearExperienceMemory() {
  if (runtime.clearingMemory) return
  const selectedConfig = { ...config }
  const model = selectedConfig.backend === 'local' ? selectedConfig.localModel : selectedConfig.remoteModel
  if (!window.confirm(`清空当前模型 ${model} / ${selectedCollaboration(selectedConfig).label} 的经验记忆？范围包含当前记忆组织和检索方式。`)) return
  runtime.clearingMemory = true
  runtime.memoryClearMessage = ''
  try {
    const response = await fetch(`${RUNNER_API}/clear-experience`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ confirm: true, config: selectedConfig })
    })
    const payload = await response.json()
    if (!response.ok) throw new Error(payload.error || '清空经验缓存失败')
    runtime.memoryClearMessage = `已清空 ${model} / ${selectedCollaboration(selectedConfig).label} 的 ${payload.deletedCount} 条经验记忆`
  } catch (error) {
    runtime.memoryClearMessage = error instanceof Error ? error.message : String(error)
  } finally {
    runtime.clearingMemory = false
  }
}

function applyConfig(nextConfig) {
  const normalized = normalizeExperimentConfig(nextConfig)
  Object.assign(config, normalized)
  runtime.totalTasks = normalized.taskCount
  persistConfig(normalized)
}

function updateConfig(patch) {
  applyConfig({ ...config, ...patch })
}

function rowIdFor(nextConfig) {
  return [
    nextConfig.dataset,
    nextConfig.collaborationMode,
    nextConfig.memoryMode,
    nextConfig.retrievalMode
  ].join('|')
}

function clearStatusTimer() {
  if (statusTimer.value) {
    window.clearInterval(statusTimer.value)
    statusTimer.value = null
  }
}

function nowText() {
  return new Date().toLocaleTimeString('zh-CN', { hour12: false })
}

async function startExperiment(overrideConfig) {
  const nextConfig = normalizeExperimentConfig(overrideConfig ? { ...config, ...overrideConfig } : config)
  applyConfig(nextConfig)
  clearStatusTimer()

  runtime.status = 'starting'
  runtime.statusText = '启动中'
  runtime.currentTask = 0
  runtime.completedTasks = 0
  runtime.failedTasks = 0
  runtime.totalTasks = nextConfig.taskCount
  runtime.progress = 0
  runtime.currentAgent = '等待后端事件'
  runtime.startedAt = nowText()
  runtime.elapsedText = '00:00'
  runtime.importedName = ''
  runtime.importedSummary = null
  runtime.runningRowId = rowIdFor(nextConfig)
  runtime.lastError = ''
  runtime.messageCount = 0
  runtime.textChars = 0
  runtime.stateTransfers = 0
  runtime.memoryHits = 0

  try {
    const payload = await requestRunner('/start', { config: nextConfig })
    applyRunnerStatus(payload)
    startStatusPolling()
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    if (message.includes('已有实验正在运行')) {
      // The runner can still be flushing an earlier run after emitting its
      // final event. Restore that real status instead of leaving a false 0/10.
      await refreshRunnerStatus()
      if (runtime.status === 'running' || runtime.status === 'starting') {
        startStatusPolling()
        return
      }
    }
    runtime.status = 'error'
    runtime.statusText = '启动失败'
    runtime.currentAgent = '未运行'
    runtime.runningRowId = ''
    runtime.lastError = message
  }
}

async function stopExperiment() {
  try {
    const payload = await requestRunner('/stop', {})
    applyRunnerStatus(payload)
    runtime.status = 'stopped'
    runtime.statusText = '已停止'
    runtime.currentAgent = '已停止'
    runtime.runningRowId = ''
    clearStatusTimer()
  } catch (error) {
    runtime.lastError = error instanceof Error ? error.message : String(error)
  }
}

async function importResult(file) {
  if (!file) return
  try {
    const text = decodeRunResultBuffer(await file.arrayBuffer())
    const payload = parseRunResultText(text)
    const importedConfig = configFromRunResult(payload, config)
    applyConfig(importedConfig)
    clearStatusTimer()

    runtime.status = 'imported'
    runtime.statusText = '已导入'
    runtime.currentTask = payload.task_count || importedConfig.taskCount
    runtime.totalTasks = payload.task_count || importedConfig.taskCount
    runtime.progress = 100
    runtime.currentAgent = '历史结果'
    runtime.startedAt = '--'
    runtime.elapsedText = '--'
    runtime.importedName = file.name
    runtime.runningRowId = ''
    runtime.lastError = ''
    runtime.importedSummary = {
      experimentMode: payload.experiment_mode,
      dataset: datasets.find(item => item.value === datasetKeyForFile(payload.dataset_file))?.label || payload.dataset_file || '历史数据集',
      collaboration: selectedCollaboration(importedConfig).label,
      taskCount: payload.task_count
    }
  } catch (error) {
    runtime.status = 'error'
    runtime.statusText = '导入失败'
    runtime.lastError = error instanceof Error ? error.message : String(error)
  }
}

function showRecords() {
  runtime.statusText = runtime.importedName ? '已导入历史结果' : '请选择结果文件'
}

function loadInitialConfig() {
  try {
    const raw = window.localStorage.getItem(CONFIG_STORAGE_KEY)
    return raw ? normalizeExperimentConfig({ ...defaultExperimentConfig(), ...JSON.parse(raw) }) : defaultExperimentConfig()
  } catch {
    return defaultExperimentConfig()
  }
}

function persistConfig(nextConfig) {
  try {
    window.localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(nextConfig))
  } catch {
    // 本地存储不可用时不影响实验控制台运行。
  }
}

async function requestRunner(path, body) {
  const response = await fetch(`${RUNNER_API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(payload.error || '运行服务请求失败')
  }
  return payload
}

async function refreshRunnerStatus() {
  try {
    const response = await fetch(`${RUNNER_API}/status?t=${Date.now()}`, { cache: 'no-store' })
    if (!response.ok) return
    applyRunnerStatus(await response.json())
  } catch {
    if (runtime.status === 'running' || runtime.status === 'starting') {
      runtime.lastError = '本地运行服务未启动：http://127.0.0.1:8765'
    }
  }
}

function startStatusPolling() {
  clearStatusTimer()
  statusTimer.value = window.setInterval(refreshRunnerStatus, 1200)
}

function applyRunnerStatus(payload) {
  if (payload.config?.dataset && (rowIdFor(payload.config) !== rowIdFor(config) || payload.config.backend !== config.backend || payload.config.taskCount !== config.taskCount)) {
    runtime.status = 'idle'
    runtime.statusText = payload.running ? '其他配置正在运行' : '当前配置未运行'
    runtime.progress = 0
    runtime.currentTask = 0
    runtime.currentAgent = '未开始'
    return
  }
  const summary = payload.summary || {}
  const status = payload.status || 'idle'
  const taskCount = Number(summary.taskCount || runtime.totalTasks || config.taskCount)
  const completedTasks = Number(summary.completedTasks || 0)
  const currentTask = Number(summary.currentTask || completedTasks || 0)

  runtime.completedTasks = completedTasks
  runtime.failedTasks = Number(summary.failedTasks || 0)
  runtime.status = status
  runtime.statusText = statusText(status)
  runtime.totalTasks = taskCount
  runtime.currentTask = Math.min(taskCount, Math.max(0, currentTask))
  runtime.progress = taskCount ? Math.round((completedTasks / taskCount) * 100) : 0
  if (status === 'running' && runtime.progress === 0 && currentTask > 0) {
    runtime.progress = Math.max(1, Math.round(((currentTask - 1) / taskCount) * 100))
  }
  if (status === 'completed') {
    runtime.progress = 100
    runtime.runningRowId = ''
  }
  if (status === 'failed') {
    runtime.runningRowId = ''
  }
  if ((status === 'running' || status === 'starting') && payload.config) {
    runtime.runningRowId = rowIdFor(normalizeExperimentConfig({ ...config, ...payload.config }))
  }
  if (status === 'completed' || status === 'failed' || status === 'stopped' || (status === 'idle' && runtime.runningRowId)) {
    runtime.runningRowId = ''
    clearStatusTimer()
  }
  runtime.currentAgent = summary.currentAgent || (status === 'idle' ? '未开始' : '等待后端事件')
  runtime.elapsedText = secondsToClock(Number(payload.elapsedSeconds || 0))
  runtime.messageCount = Number(summary.messageCount || 0)
  runtime.textChars = Number(summary.textChars || 0)
  runtime.stateTransfers = Number(summary.stateTransfers || 0)
  runtime.memoryHits = Number(summary.memoryHits || 0)
  runtime.lastError = payload.lastError || ''
}

function statusText(status) {
  const map = {
    idle: '未开始',
    starting: '启动中',
    running: '运行中',
    completed: '已完成',
    failed: '运行失败',
    stopped: '已停止',
    imported: '已导入',
    error: '错误'
  }
  return map[status] || status
}

function secondsToClock(seconds) {
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`
}

onMounted(() => {
  refreshRunnerStatus()
})

onBeforeUnmount(() => {
  clearStatusTimer()
})
</script>
