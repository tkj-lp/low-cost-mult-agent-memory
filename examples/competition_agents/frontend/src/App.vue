<template>
  <div class="app-shell">
    <AppHeader
      :active-tab="activeTab"
      @change-tab="changeTab"
      @open-records="openRecordDialog"
    />
    <main class="app-main">
      <ExperimentConsole v-if="activeTab === '实验控制台'" />
      <MonitorPage v-else-if="activeTab === '协作机制'" />
      <MemoryPage v-else-if="activeTab === '记忆复用'" />
      <MetricsPage
        v-else
        :imported-records="importedRecords"
      />
    </main>
    <ResultImportDialog
      v-if="recordDialogOpen"
      @close="recordDialogOpen = false"
      @imported="handleImportedResults"
    />
    <footer class="app-footer">
      <div>版本：v1.0.0 <span class="footer-sep">|</span> © 2025 多智能体低开销通信实验系统</div>
      <div>开源社区版，仅用于学术研究与比赛演示</div>
    </footer>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import AppHeader from './components/AppHeader.vue'
import ExperimentConsole from './components/ExperimentConsole.vue'
import MonitorPage from './pages/MonitorPage.vue'
import MemoryPage from './pages/MemoryPage.vue'
import MetricsPage from './pages/MetricsPage.vue'
import ResultImportDialog from './components/ResultImportDialog.vue'

const IMPORTED_RESULTS_STORAGE_KEY = 'competition-agent-imported-results'
const activeTab = ref('实验控制台')
const recordDialogOpen = ref(false)
const savedImportedResults = readSavedImportedResults()
const importedRecords = ref(savedImportedResults.records)
const importedPayloads = ref(savedImportedResults.payloads)

function changeTab(tab) {
  activeTab.value = tab
}

function openRecordDialog() {
  recordDialogOpen.value = true
}

function handleImportedResults(payload) {
  importedRecords.value = Array.isArray(payload.records) ? payload.records : []
  importedPayloads.value = Array.isArray(payload.payloads) ? payload.payloads : []
  persistImportedResults()
}

function readSavedImportedResults() {
  try {
    const raw = window.localStorage.getItem(IMPORTED_RESULTS_STORAGE_KEY)
    if (!raw) return { records: [], payloads: [] }
    const parsed = JSON.parse(raw)
    return {
      records: Array.isArray(parsed.records) ? parsed.records : [],
      payloads: Array.isArray(parsed.payloads) ? parsed.payloads : []
    }
  } catch {
    return { records: [], payloads: [] }
  }
}

function persistImportedResults() {
  try {
    window.localStorage.setItem(IMPORTED_RESULTS_STORAGE_KEY, JSON.stringify({
      records: importedRecords.value,
      payloads: importedPayloads.value
    }))
  } catch {
    // 导入结果只用于前端展示，持久化失败时不影响本次页面内查看。
  }
}
</script>
