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
          <label>本次任务数量</label>
          <div class="stepper fixed-count" aria-label="固定任务数量">
            <span>{{ modelValue.taskCount }}</span>
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
            :disabled="modelValue.dataset === 'codeact' && !['plain', 'structured'].includes(mode.value)"
            :title="modelValue.dataset === 'codeact' && !['plain', 'structured'].includes(mode.value) ? '该方式尚未适配代码执行工具' : ''"
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
            v-for="mode in availableMemoryModes"
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
            :options="availableBackends"
            @update:model-value="emitPatch({ backend: $event })"
          />
        </div>
        <div v-if="modelValue.backend === 'local'" class="field">
          <label>模型</label>
          <BaseSelect
            :model-value="modelValue.localModel"
            :options="availableLocalModels"
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
            <label>模型</label>
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
  selectedDataset,
  selectedCollaboration
} from '../experimentOptions'

const props = defineProps({
  modelValue: {
    type: Object,
    required: true
  }
})

const emit = defineEmits(['update:config'])

const datasetOptions = datasets.map(({ label, value }) => ({ label, value }))

const chosenMethod = computed(() => selectedCollaboration(props.modelValue))
const availableMemoryModes = computed(() => memoryModes.filter((item) => item.value === chosenMethod.value.memory))
const availableBackends = computed(() => backends.filter((item) => chosenMethod.value.backends.includes(item.value)))
const availableLocalModels = computed(() => localModels.filter((item) => !chosenMethod.value.models || chosenMethod.value.models.includes(item.value)))
const retrievalOptions = computed(() => retrievalOptionsFor(props.modelValue.memoryMode))
const datasetNote = computed(() => {
  const dataset = selectedDataset(props.modelValue)
  return dataset.datasetType === 'codeact' ? '按原顺序运行前10项功能任务；原库20项全部保留，非官方连续问答基准。' : '官方第153组完整10轮，按原顺序连续执行；使用同一份固定预备知识库。'
})

function emitPatch(patch) {
  emit('update:config', patch)
}

function updateDataset(datasetValue) {
  const dataset = datasets.find((item) => item.value === datasetValue)
  emitPatch({
    dataset: datasetValue,
    taskCount: dataset?.runTaskCount ?? dataset?.taskCount ?? props.modelValue.taskCount
  })
}

function updateMemoryMode(memoryMode) {
  emitPatch({
    memoryMode,
    retrievalMode: defaultRetrievalFor(memoryMode)
  })
}
</script>

<style scoped>
.fixed-count { grid-template-columns: 1fr; }
.fixed-count span { border: 0; }
</style>
