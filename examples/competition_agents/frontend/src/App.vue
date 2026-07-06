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
        :record-dialog-open="recordDialogOpen"
        @close-records="recordDialogOpen = false"
      />
    </main>
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

const activeTab = ref('实验控制台')
const recordDialogOpen = ref(false)

function changeTab(tab) {
  activeTab.value = tab
  if (tab !== '指标评估') {
    recordDialogOpen.value = false
  }
}

function openRecordDialog() {
  if (activeTab.value === '指标评估') {
    recordDialogOpen.value = true
  }
}
</script>
