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
          <th>文本字符</th>
          <th>单任务耗时</th>
          <th>通信开销</th>
          <th>记忆复用</th>
          <th>回答准确率</th>
          <th>消息次数</th>
          <th>非文本状态次数</th>
          <th>非文本数据规模</th>
          <th>记忆召回率</th>
          <th>记忆精度</th>
          <th>整体提升</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in rows" :key="row.id">
          <td><span :class="['method-dot', row.tone]"></span>{{ row.method }}</td>
          <td>{{ row.chars }}</td>
          <td>{{ row.avgDuration }}</td>
          <td>{{ row.communication }}</td>
          <td>{{ row.memoryReuse }}</td>
          <td>{{ row.accuracy }}</td>
          <td>{{ row.messages }}</td>
          <td>{{ row.stateCount }}</td>
          <td>{{ row.stateSize }}</td>
          <td>{{ row.recall }}</td>
          <td>{{ row.precision }}</td>
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
