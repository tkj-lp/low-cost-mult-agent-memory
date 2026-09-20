<template>
  <BaseCard title="方法对比汇总" class="metrics-card method-table-card">
    <template #icon><span class="info-dot">ⓘ</span></template>
    <div v-if="!rows.length" class="runtime-empty-panel metrics-empty-panel">
      <strong>暂无方法数据</strong>
      <span>请先完成一次真实运行，或选择已有运行结果对应的方法。</span>
    </div>
    <ScrollTable v-else variant="method-table">
      <thead>
        <tr>
          <th>完整方法</th>
          <th>真实消息载荷<br><small>实际 / 引用</small></th>
          <th>单任务耗时<br><small>完整 / 状态生成</small></th>
          <th>非文本状态<br><small>次数 / 规模</small></th>
          <th>记忆复用<br><small>命中任务占比 / QA召回 / QA精度</small></th>
          <th>回答正确率<br><small>严格 EM / Token F1</small></th>
          <th>消息次数</th>
          <th>效率提升</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in rows" :key="row.id">
          <td><span :class="['method-dot', row.tone]"></span>{{ row.method }}</td>
          <td>{{ row.actualBytes }} / {{ row.referenceBytes }}</td>
          <td>{{ row.avgDuration }} / {{ row.stateGeneration }}</td>
          <td>{{ row.stateCount }} / {{ row.stateSize }}</td>
          <td :title="`命中任务占比只表示“召回过记忆”，不判断对错。QA召回/精度需 expected_reuse 标注或 .reuse-evaluation 侧车。有效QA复用（相关记忆被采用且 EM 正确）：${row.effectiveReuse}`">{{ row.memoryReuse }} / {{ row.recall }} / {{ row.precision }}</td>
          <td>{{ row.accuracy }} / {{ row.tokenF1 }}</td>
          <td>{{ row.messages }}</td>
          <td class="green-text">{{ row.improvement }}</td>
        </tr>
      </tbody>
    </ScrollTable>
  </BaseCard>
</template>

<script setup>
import BaseCard from '../BaseCard.vue'
import ScrollTable from '../ScrollTable.vue'

defineProps({
  rows: {
    type: Array,
    default: () => []
  }
})
</script>
