<template>
  <BaseCard :title="title" class="metrics-card line-chart-card">
    <template #icon><span class="info-dot">ⓘ</span></template>
    <div v-if="!series.length" class="runtime-empty-panel metrics-empty-panel">
      <strong>暂无可绘制数据</strong>
      <span>完成一次真实运行后会显示对应任务的趋势图。</span>
    </div>
    <div v-else class="chart-shell">
      <div class="chart-scroll">
      <svg class="line-chart-svg" viewBox="0 0 760 260" preserveAspectRatio="xMidYMid meet" aria-label="趋势图">
        <g class="chart-grid-lines">
          <line v-for="tick in yTicks" :key="tick.value" :x1="plot.left" :y1="tick.y" :x2="plot.right" :y2="tick.y" />
        </g>
        <line :x1="plot.left" :y1="plot.top" :x2="plot.left" :y2="plot.bottom" class="axis-line" />
        <line :x1="plot.left" :y1="plot.bottom" :x2="plot.right" :y2="plot.bottom" class="axis-line" />
        <text x="8" y="18" class="axis-title">{{ unit }}</text>
        <g v-for="tick in yTicks" :key="`label-${tick.value}`">
          <line :x1="plot.left - 5" :x2="plot.left" :y1="tick.y" :y2="tick.y" class="axis-tick" />
          <text :x="plot.left - 10" :y="tick.y + 4" class="axis-label y-label" text-anchor="end">{{ tick.value }}</text>
        </g>
        <g v-for="(label, index) in labels" :key="label">
          <line :x1="x(index)" :x2="x(index)" :y1="plot.bottom" :y2="plot.bottom + 5" class="axis-tick" />
          <text :x="x(index)" y="248" class="axis-label" text-anchor="middle">{{ label }}</text>
        </g>
        <g
          v-for="line in normalized"
          :key="line.name"
          class="chart-line-group"
          :class="{ muted: activeLine && activeLine !== line.name, active: activeLine === line.name }"
          @mouseenter="activeLine = line.name"
          @mouseleave="activeLine = ''"
        >
          <polyline
            :points="line.points"
            fill="none"
            :stroke="line.color"
            :stroke-dasharray="line.dash"
            :stroke-width="activeLine === line.name ? 4 : 2.5"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
          <circle
            v-for="point in line.pointList"
            :key="`${line.name}-${point.x}`"
            :cx="point.x"
            :cy="point.y"
            r="4"
            fill="#fff"
            :stroke="line.color"
            stroke-width="2.2"
          />
        </g>
      </svg>
      </div>
      <div class="chart-legend-list">
        <button
          v-for="line in legendItems"
          :key="line.name"
          type="button"
          class="chart-legend-item"
          :class="{ active: activeLine === line.name, muted: activeLine && activeLine !== line.name }"
          :title="line.name"
          @mouseenter="activeLine = line.name"
          @mouseleave="activeLine = ''"
        >
          <svg class="legend-line" viewBox="0 0 34 8" aria-hidden="true">
            <line
              x1="1"
              y1="4"
              x2="33"
              y2="4"
              :stroke="line.color"
              :stroke-dasharray="line.dash"
              stroke-width="3"
              stroke-linecap="round"
            />
          </svg>
          <span>{{ line.label }}</span>
        </button>
      </div>
    </div>
  </BaseCard>
</template>

<script setup>
import { computed, ref } from 'vue'
import BaseCard from '../BaseCard.vue'

const props = defineProps({
  title: String,
  unit: String,
  series: {
    type: Array,
    required: true
  },
  labels: {
    type: Array,
    default: () => []
  }
})

const labels = computed(() => {
  const maxLength = Math.max(...props.series.map((line) => line.values.length), 0)
  return props.labels.length
    ? props.labels
    : Array.from({ length: maxLength }, (_, index) => `任务${index + 1}`)
})
const plot = {
  left: 58,
  right: 735,
  top: 62,
  bottom: 222
}
const activeLine = ref('')
const dashPatterns = ['', '8 5', '3 4', '12 4 3 4', '2 6', '10 3', '6 3 2 3', '14 5']
const allValues = computed(() => props.series.flatMap((line) => line.values))
const minValue = computed(() => Math.min(0, ...allValues.value))
const maxValue = computed(() => (allValues.value.length ? Math.max(...allValues.value) : 1))
const niceMax = computed(() => {
  if (maxValue.value <= 10) return Math.ceil(maxValue.value)
  return Math.ceil(maxValue.value / 20) * 20
})
const span = computed(() => Math.max(1, niceMax.value - minValue.value))
const yTicks = computed(() =>
  Array.from({ length: 6 }, (_, index) => {
    const value = Number(((niceMax.value / 5) * index).toFixed(niceMax.value <= 10 ? 1 : 0))
    return { value, y: y(value) }
  }).reverse()
)

const x = (index) => {
  if (labels.value.length <= 1) return (plot.left + plot.right) / 2
  return plot.left + index * ((plot.right - plot.left) / (labels.value.length - 1))
}
const y = (value) => plot.bottom - ((value - minValue.value) / span.value) * (plot.bottom - plot.top)

const legendItems = computed(() =>
  normalized.value.map((line) => ({
    ...line,
    label: shortenLabel(line.name)
  }))
)

const normalized = computed(() =>
  props.series.map((line, lineIndex) => {
    const pointList = line.values.map((value, index) => ({ x: x(index), y: y(value) }))
    return {
      ...line,
      dash: line.dash || dashPatterns[lineIndex % dashPatterns.length],
      pointList,
      points: pointList.map((point) => `${point.x},${point.y}`).join(' ')
    }
  })
)

function shortenLabel(label) {
  const text = String(label || '')
    .replace(/\s*\+\s*/g, ' + ')
    .replace(/\s*#\d+$/, '')
    .trim()
  return text.length > 30 ? `${text.slice(0, 29)}?` : text
}
</script>
