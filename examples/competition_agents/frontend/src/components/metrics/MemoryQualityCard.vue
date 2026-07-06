<template>
  <BaseCard title="记忆检索质量（10轮平均）" class="metrics-card memory-quality-card">
    <template #icon><span class="info-dot">ⓘ</span></template>
    <div class="quality-grid">
      <article v-for="item in items" :key="item.title" class="quality-card" :class="item.tone">
        <span class="quality-icon">{{ item.icon }}</span>
        <h3>{{ item.title }}</h3>
        <strong>{{ item.value }}%</strong>
        <p>{{ item.desc }}</p>
        <div class="quality-progress">
          <ProgressBar :value="item.value" :tone="item.tone" />
          <em>{{ item.value }}%</em>
        </div>
      </article>
    </div>
  </BaseCard>
</template>

<script setup>
import BaseCard from '../BaseCard.vue'
import ProgressBar from '../ProgressBar.vue'

const items = [
  { title: '记忆召回率', value: 76.3, desc: '相关记忆被成功找回的比例', tone: 'blue', icon: '▣' },
  { title: '记忆准确率', value: 68.5, desc: '检索结果中相关记忆的比例', tone: 'green', icon: '▣' },
  { title: '采用率', value: 72.0, desc: '命中记忆被 Agent 采用的比例', tone: 'orange', icon: '◎' },
  { title: '有效支撑率', value: 58.0, desc: '被采用且有效支撑的比例', tone: 'purple', icon: '✓' }
]
</script>
