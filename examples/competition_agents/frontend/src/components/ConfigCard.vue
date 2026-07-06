<template>
  <BaseCard title="实验配置" class="config-card">
    <template #icon>
      <svg viewBox="0 0 20 20" aria-hidden="true">
        <path d="M8.7 2.4h2.6l.5 2a6 6 0 0 1 1.2.5l1.8-1 1.8 1.9-1 1.7c.2.4.4.8.5 1.3l2 .5v2.6l-2 .5c-.1.4-.3.9-.5 1.2l1 1.8-1.8 1.8-1.8-1a6 6 0 0 1-1.2.5l-.5 2H8.7l-.5-2a6 6 0 0 1-1.2-.5l-1.8 1-1.8-1.8 1-1.8a6 6 0 0 1-.5-1.2l-2-.5V9.3l2-.5c.1-.5.3-.9.5-1.3l-1-1.7 1.8-1.9 1.8 1c.4-.2.8-.4 1.2-.5l.5-2Z" />
        <circle cx="10" cy="10.6" r="2.6" />
      </svg>
    </template>

    <div class="config-stack">
      <div class="config-row dataset-row">
        <div class="field dataset-field">
          <label>数据集</label>
          <BaseSelect
            :model-value="modelValue.dataset"
            :options="datasetOptions"
            @update:model-value="updateDataset"
          />
        </div>
        <div class="field count-field">
          <label>任务数量</label>
          <div class="stepper" aria-label="任务数量">
            <button type="button" @click="updateTaskCount(-1)">−</button>
            <span>{{ modelValue.taskCount }}</span>
            <button type="button" @click="updateTaskCount(1)">＋</button>
          </div>
        </div>
        <p class="dataset-note">{{ datasetNote }}</p>
      </div>

      <div class="option-section">
        <label>协作方式</label>
        <div class="choice-group four">
          <button
            v-for="mode in collaborationModes"
            :key="mode.value"
            class="choice-button"
            :class="{ selected: modelValue.collaborationMode === mode.value }"
            type="button"
            @click="emitPatch({ collaborationMode: mode.value })"
          >
            <span class="choice-icon" v-html="mode.icon"></span>
            <span>{{ mode.label }}</span>
          </button>
        </div>
      </div>

      <div class="option-section">
        <label>记忆组织方式</label>
        <div class="choice-group memory">
          <button
            v-for="mode in memoryModes"
            :key="mode.value"
            class="choice-button"
            :class="{ selected: modelValue.memoryMode === mode.value }"
            type="button"
            @click="updateMemoryMode(mode.value)"
          >
            <span class="choice-icon" v-html="mode.icon"></span>
            <span>{{ mode.label }}</span>
          </button>
        </div>
      </div>

      <div class="option-section">
        <label>记忆检索方式</label>
        <div class="choice-group retrieval">
          <button
            v-for="mode in retrievalOptions"
            :key="mode.value"
            class="choice-button"
            :class="{ selected: modelValue.retrievalMode === mode.value }"
            type="button"
            @click="emitPatch({ retrievalMode: mode.value })"
          >
            <span class="choice-icon" v-html="mode.icon"></span>
            <span>{{ mode.label }}</span>
          </button>
        </div>
      </div>

      <div class="backend-grid" :class="{ external: modelValue.backend === 'openai-compatible' }">
        <div class="field">
          <label>模型后端</label>
          <BaseSelect
            :model-value="modelValue.backend"
            :options="backends"
            @update:model-value="emitPatch({ backend: $event })"
          />
        </div>
        <div v-if="modelValue.backend === 'local'" class="field">
          <label>model</label>
          <BaseSelect
            :model-value="modelValue.localModel"
            :options="localModels"
            @update:model-value="emitPatch({ localModel: $event })"
          />
        </div>
        <template v-else>
          <div class="field">
            <label>base_url</label>
            <input
              type="text"
              :value="modelValue.baseUrl"
              placeholder="例如：https://dashscope.aliyuncs.com/compatible-mode/v1"
              @input="emitPatch({ baseUrl: $event.target.value })"
            />
          </div>
          <div class="field">
            <label>model</label>
            <input
              type="text"
              :value="modelValue.remoteModel"
              placeholder="例如：qwen-plus"
              @input="emitPatch({ remoteModel: $event.target.value })"
            />
          </div>
          <div class="field">
            <label>api_key</label>
            <input
              class="api-key-input"
              type="password"
              :value="modelValue.apiKey"
              autocomplete="off"
              @input="emitPatch({ apiKey: $event.target.value })"
            />
          </div>
        </template>
      </div>
    </div>
  </BaseCard>
</template>

<script setup>
import { computed } from 'vue'
import BaseCard from './BaseCard.vue'
import BaseSelect from './BaseSelect.vue'
import {
  backends,
  collaborationModes,
  datasets,
  defaultRetrievalFor,
  localModels,
  memoryModes,
  retrievalOptionsFor,
  selectedDataset
} from '../experimentOptions'

const props = defineProps({
  modelValue: {
    type: Object,
    required: true
  }
})

const emit = defineEmits(['update:config'])

const datasetOptions = datasets.map(({ label, value }) => ({ label, value }))

const retrievalOptions = computed(() => retrievalOptionsFor(props.modelValue.memoryMode))
const datasetNote = computed(() => {
  const dataset = selectedDataset(props.modelValue)
  return `${dataset.taskCount} 轮任务，来源：${dataset.file.split('/').pop()}`
})

function emitPatch(patch) {
  emit('update:config', patch)
}

function updateDataset(datasetValue) {
  const dataset = datasets.find((item) => item.value === datasetValue)
  emitPatch({
    dataset: datasetValue,
    taskCount: dataset?.taskCount ?? props.modelValue.taskCount
  })
}

function updateTaskCount(delta) {
  const nextCount = Math.max(1, Number(props.modelValue.taskCount || 1) + delta)
  emitPatch({ taskCount: nextCount })
}

function updateMemoryMode(memoryMode) {
  emitPatch({
    memoryMode,
    retrievalMode: defaultRetrievalFor(memoryMode)
  })
}
</script>
