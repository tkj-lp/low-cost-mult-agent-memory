<template>
  <section class="metrics-filter-bar">
    <div v-for="field in fields" :key="field.key" class="metrics-filter-field" :class="{ wide: field.wide }">
      <label>{{ field.label }}</label>
      <BaseSelect
        :model-value="modelValue[field.key]"
        :options="field.options"
        @update:model-value="updateField(field.key, $event)"
      />
    </div>
    <BaseButton variant="outline" compact class="metrics-refresh" @click="$emit('refresh')">
      <template #icon>⟳</template>
      刷新
    </BaseButton>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import BaseSelect from '../BaseSelect.vue'
import BaseButton from '../BaseButton.vue'
import {
  collaborationModes,
  datasets,
  memoryModes,
  retrievalOptionsFor
} from '../../experimentOptions'

const props = defineProps({
  modelValue: {
    type: Object,
    required: true
  },
  baselineOptions: {
    type: Array,
    default: () => []
  }
})

const emit = defineEmits(['update:modelValue', 'refresh'])

const fields = computed(() => {
  const items = [
    { key: 'dataset', label: '数据集', options: datasets.map(({ label, value }) => ({ label, value })) },
    { key: 'collaborationMode', label: '协作方式', options: collaborationModes.map(({ label, value }) => ({ label, value })) },
    { key: 'memoryMode', label: '记忆组织', options: memoryModes.filter((item) => item.value === collaborationModes.find((method) => method.value === props.modelValue.collaborationMode)?.memory).map(({ label, value }) => ({ label, value })) },
    {
      key: 'retrievalMode',
      label: '检索方式',
      wide: true,
      options: retrievalOptionsFor(props.modelValue.memoryMode).map(({ label, value }) => ({ label, value }))
    },
    {
      key: 'baselineMode',
      label: '对比基线',
      options: [
        { label: '自动同条件对比', value: 'none' },
        { label: '指定基线记录', value: 'compare' }
      ]
    }
  ]
  if (props.modelValue.baselineMode === 'compare') {
    items.push({
      key: 'baselineRecordId',
      label: '基线记录',
      wide: true,
      options: props.baselineOptions.length
        ? props.baselineOptions
        : [{ label: '暂无可用实验记录', value: '' }]
    })
  }
  return items
})

function updateField(key, value) {
  const next = { ...props.modelValue, [key]: value }
  const method = collaborationModes.find((item) => item.value === next.collaborationMode)
  if (method) Object.assign(next, { memoryMode: method.memory, retrievalMode: method.retrieval })
  emit('update:modelValue', next)
}
</script>
