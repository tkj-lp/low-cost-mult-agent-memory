<template>
  <BaseCard title="运行事件流与协议回放" class="monitor-card event-stream-card runtime-card">
    <template #icon><span class="title-mark"></span></template>
    <ScrollTable variant="event-table">
      <thead>
        <tr>
          <th>时间</th>
          <th>模块</th>
          <th>事件</th>
          <th>运行内容</th>
          <th>引用</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="event in events" :key="event.time">
          <td>{{ event.time }}</td>
          <td><span :class="['event-agent', event.tone]">{{ event.module }}</span></td>
          <td>{{ event.type }}</td>
          <td><span class="ellipsis">{{ event.content }}</span></td>
          <td><StatBadge :tone="event.tone">{{ event.ref }}</StatBadge></td>
        </tr>
      </tbody>
    </ScrollTable>
  </BaseCard>
</template>

<script setup>
import BaseCard from '../BaseCard.vue'
import ScrollTable from '../ScrollTable.vue'
import StatBadge from '../StatBadge.vue'

const events = [
  { time: '12:14:02.128', module: '协议调度器', type: '握手', content: 'client_hello → capability_manifest，确认 schema:v2', ref: 'sess-007', tone: 'blue' },
  { time: '12:14:02.542', module: '检索 Agent', type: '能力发现', content: '发现 memory.search.fusion，支持关键词、标签和语义向量检索', ref: 'cap-013', tone: 'green' },
  { time: '12:14:03.331', module: '检索 Agent', type: '记忆命中', content: '命中 mem-021，摘要：长距离纠缠传输受损耗和退相干限制', ref: 'mem-021', tone: 'green' },
  { time: '12:14:03.508', module: '状态交换', type: '向量态传递', content: '生成 embedding_ref=emb-7a2，并传递给工具执行 Agent 用于证据重排', ref: 'emb-7a2', tone: 'orange' },
  { time: '12:14:03.821', module: '工具执行 Agent', type: '工具调用', content: '执行 qa_parser.normalize，将证据片段整理为限制因素候选结构', ref: 'tool-007', tone: 'purple' },
  { time: '12:14:04.652', module: '总结生成 Agent', type: '状态消费', content: '读取 state_ref=tool-state-007 与 result_ref=tool-007，生成回答草稿', ref: 'state-007', tone: 'orange' },
  { time: '12:14:05.912', module: '共享记忆', type: '写回', content: '写回新记忆单元 mem-030，记录主题、摘要、来源 Agent 和创建时间', ref: 'mem-030', tone: 'blue' }
]
</script>
