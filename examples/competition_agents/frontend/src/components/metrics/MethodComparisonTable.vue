<template>
  <BaseCard title="方法对比汇总" class="metrics-card method-table-card">
    <div v-if="!rows.length" class="runtime-empty-panel"><strong>暂无运行结果</strong></div>
    <ScrollTable v-else variant="method-table">
      <thead><tr><th>协作方案</th><th v-for="column in columns" :key="column.key">{{ column.label }}</th></tr></thead>
      <tbody><tr v-for="row in rows" :key="row.id"><td><span :class="['method-dot', row.tone]"></span>{{ row.method }}</td><td v-for="column in columns" :key="column.key">{{ format(row[column.key], column.kind) }}</td></tr></tbody>
    </ScrollTable>
    <p class="metrics-definition">只显示结果实际记录的指标。知识库检索耗时为工具耗时；文本内容的阅读计入模型耗时。状态快照大小与网络传输实测分开解释。</p>
  </BaseCard>
</template>
<script setup>
import { computed } from 'vue'
import BaseCard from '../BaseCard.vue'
import ScrollTable from '../ScrollTable.vue'
const props = defineProps({ rows: { type: Array, default: () => [] } })
const specifications = [
  ['handoffPayloadBytes','交接内容规模','bytes'], ['elapsedMs','完整耗时','seconds'],
  ['modelCalls','模型调用','number'], ['inputTokens','输入词元','number'], ['outputTokens','输出词元','number'],
  ['modelServiceMs','模型服务耗时','seconds'], ['knowledgeSearchMs','知识库检索','milliseconds'],
  ['knowledgeSearches','知识库检索次数','number'], ['memorySearchMs','记忆检索','milliseconds'],
  ['memoryHitRate','记忆命中任务占比','percent'], ['answerAccuracy','答案／执行覆盖度','percent'],
  ['improvement','耗时下降','text']
]
const columns = computed(() => specifications.filter(([key]) => props.rows.some(row => row[key] != null))
  .map(([key,label,kind]) => ({key,label,kind})))
function format(value, kind) {
  if (value == null) return '—'
  if (kind === 'text') return value
  if (kind === 'percent') return `${(value*100).toFixed(1)}%`
  if (kind === 'seconds') return `${(value/1000).toFixed(2)}秒`
  if (kind === 'milliseconds') return `${Number(value).toFixed(2)}毫秒`
  if (kind === 'bytes') return value >= 1048576 ? `${(value/1048576).toFixed(2)} MiB` : value >= 1024 ? `${(value/1024).toFixed(2)} KiB` : `${value} 字节`
  return Number(value).toLocaleString('zh-CN')
}
</script>
