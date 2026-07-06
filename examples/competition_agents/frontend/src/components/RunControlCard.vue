<template>
  <BaseCard title="运行控制" class="run-card">
    <template #icon>
      <svg viewBox="0 0 20 20" aria-hidden="true">
        <circle cx="10" cy="10" r="9" />
        <path d="M8 6.6v6.8l5.3-3.4L8 6.6Z" />
      </svg>
    </template>

    <div class="run-stack">
      <div class="control-buttons">
        <BaseButton variant="primary" :disabled="isBusy" @click="$emit('start')">
          <template #icon>▶</template>
          启动实验
        </BaseButton>
        <BaseButton variant="danger" :disabled="!isBusy" @click="$emit('stop')">
          <template #icon>■</template>
          停止
        </BaseButton>
        <BaseButton variant="outline" @click="chooseResultFile">
          <template #icon>
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <path d="M10 13.5V4.8" />
              <path d="m6.5 8.3 3.5-3.5 3.5 3.5" />
              <path d="M5 11.4a3.4 3.4 0 0 0 0 6.8h10a3.4 3.4 0 0 0 .7-6.7" />
            </svg>
          </template>
          导入结果
        </BaseButton>
        <BaseButton variant="outline" @click="$emit('show-records')">
          <template #icon>
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <path d="M6 3.3h6.5L16 6.8v9.9H6z" />
              <path d="M12.5 3.3v3.5H16" />
              <path d="M8 10h5.7M8 13h4.5" />
            </svg>
          </template>
          实验记录
        </BaseButton>
        <input
          ref="fileInput"
          class="hidden-file-input"
          type="file"
          accept=".txt,.log,.json"
          @change="handleResultFile"
        />
      </div>

      <section class="status-panel">
        <h3>当前运行状态</h3>
        <div class="status-grid">
          <div class="status-list">
            <div v-for="item in statusItems" :key="item.label" class="status-item">
              <span class="mini-icon" v-html="item.icon"></span>
              <span class="status-label">{{ item.label }}：</span>
              <span class="status-value">{{ item.value }}</span>
            </div>
          </div>
          <div class="progress-list">
            <div class="progress-block">
              <div class="progress-label">
                <span class="mini-icon">◴</span>
                <span>任务进度：</span>
              </div>
              <div class="progress-line">
                <strong>{{ runtime.currentTask }} / {{ runtime.totalTasks }}</strong>
                <div class="progress-bar"><span :style="{ width: `${runtime.progress}%` }"></span></div>
                <em>{{ runtime.progress }}%</em>
              </div>
            </div>
            <div class="progress-block compact">
              <div class="progress-label">
                <span class="mini-icon">▣</span>
                <span>当前任务：</span>
                <strong>{{ currentTaskText }}</strong>
              </div>
            </div>
            <div class="progress-block compact">
              <div class="progress-label">
                <span class="mini-icon">♙</span>
                <span>当前智能体：</span>
                <strong class="orange">{{ runtime.currentAgent }}</strong>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section class="estimate-section">
        <h3>运行信息</h3>
        <div class="estimate-grid">
          <div v-for="item in estimateItems" :key="item.label" class="estimate-card">
            <span class="estimate-icon" v-html="item.icon"></span>
            <div>
              <span>{{ item.label }}</span>
              <strong>{{ item.value }}</strong>
            </div>
          </div>
        </div>
        <p v-if="runtime.lastError" class="error-text">{{ runtime.lastError }}</p>
      </section>
    </div>
  </BaseCard>
</template>

<script setup>
import { computed, ref } from 'vue'
import BaseCard from './BaseCard.vue'
import BaseButton from './BaseButton.vue'
import {
  labelOf,
  memoryModes,
  retrievalOptionsFor,
  selectedCollaboration,
  selectedDataset
} from '../experimentOptions'

const props = defineProps({
  config: {
    type: Object,
    required: true
  },
  runtime: {
    type: Object,
    required: true
  }
})

const emit = defineEmits(['start', 'stop', 'import-result', 'show-records'])
const fileInput = ref(null)
const isBusy = computed(() => props.runtime.status === 'running' || props.runtime.status === 'starting')

const statusItems = computed(() => [
  { label: '数据集', icon: '▦', value: selectedDataset(props.config).label },
  { label: '协作方式', icon: '◌', value: selectedCollaboration(props.config).label },
  { label: '记忆组织方式', icon: '⌘', value: labelOf(memoryModes, props.config.memoryMode) },
  {
    label: '检索方式',
    icon: '⌕',
    value: labelOf(retrievalOptionsFor(props.config.memoryMode), props.config.retrievalMode)
  }
])

const currentTaskText = computed(() => {
  if (props.runtime.status === 'idle') return `-- / 共 ${props.runtime.totalTasks} 轮`
  if (props.runtime.status === 'imported') return '历史结果'
  if (!props.runtime.currentTask) return `-- / 共 ${props.runtime.totalTasks} 轮`
  return `第 ${props.runtime.currentTask} 轮 / 共 ${props.runtime.totalTasks} 轮`
})

const estimateItems = computed(() => [
  { label: '已运行', icon: '◷', value: isBusy.value || props.runtime.status === 'completed' ? props.runtime.elapsedText : '--' },
  { label: '字符开销', icon: '▣', value: props.runtime.textChars ? `${props.runtime.textChars} 字符` : '--' },
  { label: '消息次数', icon: '▥', value: props.runtime.messageCount ? String(props.runtime.messageCount) : '--' }
])

function chooseResultFile() {
  fileInput.value?.click()
}

function handleResultFile(event) {
  const file = event.target.files?.[0]
  if (file) {
    emit('import-result', file)
  }
  event.target.value = ''
}
</script>
