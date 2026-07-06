<template>
  <BaseCard title="对照实验矩阵" class="matrix-card">
    <template #icon>
      <svg viewBox="0 0 20 20" aria-hidden="true">
        <path d="M3 3h5v5H3zM12 3h5v5h-5zM3 12h5v5H3zM12 12h5v5h-5z" />
      </svg>
    </template>

    <div class="matrix-content">
      <div class="matrix-toolbar">
        <div class="filter-field">
          <label>数据集筛选</label>
          <BaseSelect v-model="filters.dataset" :options="datasetOptions" />
        </div>
        <div class="filter-field">
          <label>协作方式筛选</label>
          <BaseSelect v-model="filters.collaborationMode" :options="collaborationOptions" />
        </div>
        <div class="filter-field">
          <label>记忆组织方式筛选</label>
          <BaseSelect v-model="filters.memoryMode" :options="memoryOptions" />
        </div>
        <div class="filter-field">
          <label>检索方式筛选</label>
          <BaseSelect v-model="filters.retrievalMode" :options="retrievalOptions" />
        </div>
        <BaseButton variant="outline" compact class="refresh-button" @click="resetFilters">
          <template #icon>⟳</template>
          刷新
        </BaseButton>
      </div>

      <div class="matrix-table-wrap">
        <table>
          <colgroup>
            <col class="col-dataset" />
            <col class="col-count" />
            <col class="col-mode" />
            <col class="col-memory" />
            <col class="col-retrieval" />
            <col class="col-status" />
            <col class="col-cost" />
            <col class="col-duration" />
            <col class="col-reuse" />
            <col class="col-action" />
          </colgroup>
          <thead>
            <tr>
              <th>数据集</th>
              <th>任务数量</th>
              <th>协作方式</th>
              <th>记忆组织方式</th>
              <th>检索方式</th>
              <th>状态</th>
              <th>字符开销（累计）</th>
              <th>耗时（累计）</th>
              <th>记忆复用率（%）</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="record in visibleRows" :key="record.id">
              <td>
                <span class="cell-ellipsis">{{ record.dataset }}</span>
              </td>
              <td>{{ record.taskCount }}</td>
              <td>{{ record.collaborationMode }}</td>
              <td>{{ record.memoryMode }}</td>
              <td>{{ record.retrievalMode }}</td>
              <td>
                <span class="status-pill" :class="{ active: record.isSelected, running: runningRowId === record.id }">
                  <span class="pill-dot"></span>
                  {{ statusText(record) }}
                </span>
              </td>
              <td>{{ record.charCost }}</td>
              <td>{{ record.duration }}</td>
              <td>{{ record.reuseRate }}</td>
              <td>
                <button class="run-action" type="button" @click="$emit('run-record', record.config)">
                  <span>▷</span>
                  运行
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </BaseCard>
</template>

<script setup>
import { computed, reactive, watch } from 'vue'
import BaseCard from './BaseCard.vue'
import BaseSelect from './BaseSelect.vue'
import BaseButton from './BaseButton.vue'
import {
  collaborationModes,
  datasets,
  hypergraphRetrievalModes,
  memoryModes,
  retrievalOptionsFor,
  textRetrievalModes
} from '../experimentOptions'

const props = defineProps({
  rows: {
    type: Array,
    required: true
  },
  config: {
    type: Object,
    required: true
  },
  runningRowId: {
    type: String,
    default: ''
  }
})

defineEmits(['run-record'])

const filters = reactive({
  dataset: props.config.dataset,
  collaborationMode: props.config.collaborationMode,
  memoryMode: props.config.memoryMode,
  retrievalMode: 'all'
})

const datasetOptions = computed(() => [
  { label: '全部', value: 'all' },
  ...datasets.map(({ label, value }) => ({ label, value }))
])
const collaborationOptions = computed(() => [
  { label: '全部', value: 'all' },
  ...collaborationModes.map(({ label, value }) => ({ label, value }))
])
const memoryOptions = computed(() => [
  { label: '全部', value: 'all' },
  ...memoryModes.map(({ label, value }) => ({ label, value }))
])
const retrievalOptions = computed(() => [
  { label: '全部', value: 'all' },
  ...textRetrievalModes.map(({ label, value }) => ({ label, value })),
  ...hypergraphRetrievalModes.map(({ label, value }) => ({ label, value }))
])

const visibleRows = computed(() => {
  const datasetCandidates = filters.dataset === 'all'
    ? datasets
    : datasets.filter((item) => item.value === filters.dataset)
  const collaborationCandidates = filters.collaborationMode === 'all'
    ? collaborationModes
    : collaborationModes.filter((item) => item.value === filters.collaborationMode)
  const memoryCandidates = filters.memoryMode === 'all'
    ? memoryModes
    : memoryModes.filter((item) => item.value === filters.memoryMode)
  const rows = []

  for (const dataset of datasetCandidates) {
    for (const collaboration of collaborationCandidates) {
      for (const memory of memoryCandidates) {
        const retrievalCandidates = retrievalOptionsFor(memory.value).filter((retrieval) =>
          filters.retrievalMode === 'all' || retrieval.value === filters.retrievalMode
        )

        for (const retrieval of retrievalCandidates) {
          const rowConfig = {
            ...props.config,
            dataset: dataset.value,
            taskCount: dataset.value === props.config.dataset ? props.config.taskCount : dataset.taskCount,
            collaborationMode: collaboration.value,
            memoryMode: memory.value,
            retrievalMode: retrieval.value
          }
          const isSelected =
            rowConfig.dataset === props.config.dataset &&
            rowConfig.collaborationMode === props.config.collaborationMode &&
            rowConfig.memoryMode === props.config.memoryMode &&
            rowConfig.retrievalMode === props.config.retrievalMode

          rows.push({
            id: [rowConfig.dataset, rowConfig.collaborationMode, rowConfig.memoryMode, rowConfig.retrievalMode].join('|'),
            dataset: dataset.label,
            taskCount: rowConfig.taskCount,
            collaborationMode: collaboration.label,
            memoryMode: memory.label,
            retrievalMode: retrieval.label,
            status: isSelected ? '当前配置' : '待运行',
            charCost: '--',
            duration: '--',
            reuseRate: '--',
            isSelected,
            config: rowConfig
          })
        }
      }
    }
  }

  return rows
})

watch(
  () => [props.config.dataset, props.config.collaborationMode, props.config.memoryMode],
  () => {
    filters.dataset = props.config.dataset
    filters.collaborationMode = props.config.collaborationMode
    filters.memoryMode = props.config.memoryMode
    filters.retrievalMode = 'all'
  }
)

function resetFilters() {
  filters.dataset = props.config.dataset
  filters.collaborationMode = props.config.collaborationMode
  filters.memoryMode = props.config.memoryMode
  filters.retrievalMode = 'all'
}

function statusText(record) {
  if (props.runningRowId === record.id) return '运行中'
  return record.isSelected ? '当前配置' : '待运行'
}
</script>
