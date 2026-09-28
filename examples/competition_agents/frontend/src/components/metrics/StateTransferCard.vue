<template>
  <BaseCard title="计算状态复用" class="metrics-card state-transfer-card">
    <template #icon><span class="info-dot">ⓘ</span></template>
    <div v-if="!bars.length" class="runtime-empty-panel metrics-empty-panel">
      <strong>暂无计算状态复用记录</strong>
      <span>隐状态显示同进程续接次数与缓存峰值，跨机器传输量另行测量。</span>
    </div>
    <div v-else class="bar-chart-panel compact">
      <h3>续接次数与缓存峰值</h3>
      <div class="bar-chart">
        <div v-for="bar in bars" :key="bar.name" class="bar-item">
          <strong>{{ bar.value }}</strong>
          <span :style="{ height: `${bar.height}%`, background: bar.color }"></span>
          <em>{{ bar.name }}</em>
        </div>
      </div>
    </div>
  </BaseCard>
</template>

<script setup>
import { computed } from 'vue'
import BaseCard from '../BaseCard.vue'

const props = defineProps({
  rows: {
    type: Array,
    default: () => []
  }
})

const colors = ['#2f7df6', '#10a96b', '#ff7a00', '#7651e8']

const bars = computed(() => {
  const rawBars = props.rows
    .filter((row) => Number(row.stateReuseCount || row.stateTransferCount || 0) > 0)
    .map((row, index) => ({
      name: row.shortMethod,
      value: Number(row.stateReuseCount || row.stateTransferCount || 0),
      sizeKb: Number(row.statePeakBytes || row.stateBytes || 0) / 1024,
      color: colors[index % colors.length]
    }))
  const maxValue = Math.max(...rawBars.map((bar) => bar.value), 1)
  return rawBars.map((bar) => ({
    ...bar,
    value: bar.sizeKb ? `${bar.value} / ${bar.sizeKb.toFixed(1)}KB` : String(bar.value),
    height: Math.max(10, Math.round((bar.value / maxValue) * 90))
  }))
})
</script>
