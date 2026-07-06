<template>
  <div class="monitor-summary">
    <article v-for="item in items" :key="item.title" class="summary-card">
      <span class="summary-icon" :class="item.tone">{{ item.icon }}</span>
      <div class="summary-text">
        <span>{{ item.title }}</span>
        <strong :class="item.tone">{{ item.value }}</strong>
      </div>
      <ProgressBar v-if="item.progress" :value="70" />
    </article>
  </div>
</template>

<script setup>
import ProgressBar from '../ProgressBar.vue'

const items = [
  { title: '运行批次', value: 'TopiOCQA 连续任务 A', icon: '▰', tone: 'blue' },
  { title: '协作模式', value: '结构化协议 + 向量态', icon: '▣', tone: 'green' },
  { title: '协议会话', value: 'session-2025-06-30-07', icon: '↔', tone: 'purple' },
  { title: '当前任务', value: 'task7 / 10', icon: '◴', tone: 'blue', progress: true },
  { title: '活动角色', value: '检索 Agent', icon: '⌕', tone: 'green' },
  { title: '状态交换', value: 'embedding + state_ref', icon: '◇', tone: 'orange' },
  { title: '共享记忆', value: 'mem-021 已复用', icon: '⌘', tone: 'purple' },
  { title: '运行状态', value: '执行中 · 可回放', icon: '●', tone: 'blue' }
]
</script>
