<template>
  <BaseCard title="共享记忆复用与证据链" class="monitor-card memory-evidence-card runtime-card">
    <template #icon><span class="title-mark"></span></template>
    <div class="evidence-content runtime-memory-content">
      <div class="evidence-controls">
        <span>检索策略：<StatBadge tone="orange">关键词 + 标签 + 语义相似度</StatBadge></span>
        <span>当前主题：<StatBadge tone="blue">量子纠缠安全通信</StatBadge></span>
      </div>
      <div class="runtime-memory-grid">
        <section class="memory-unit-list">
          <article v-for="memory in memories" :key="memory.id" class="memory-unit-card">
            <header>
              <strong>{{ memory.id }}</strong>
              <StatBadge :tone="memory.tone">{{ memory.type }}</StatBadge>
            </header>
            <p>{{ memory.summary }}</p>
            <dl>
              <div><dt>来源</dt><dd>{{ memory.source }}</dd></div>
              <div><dt>时间</dt><dd>{{ memory.time }}</dd></div>
              <div><dt>主题</dt><dd>{{ memory.topic }}</dd></div>
            </dl>
          </article>
        </section>
        <section class="chain-panel">
          <h3>当前证据链</h3>
          <div class="chain-scroll runtime-chain">
            <article v-for="node in chain" :key="node.title" class="chain-node" :class="node.tone">
              <strong>{{ node.title }}</strong>
              <span>{{ node.text }}</span>
            </article>
          </div>
        </section>
      </div>
    </div>
  </BaseCard>
</template>

<script setup>
import BaseCard from '../BaseCard.vue'
import StatBadge from '../StatBadge.vue'

const memories = [
  {
    id: 'mem-021',
    type: 'evidence',
    tone: 'green',
    source: 'Retriever',
    time: '12:14:03',
    topic: '安全通信限制',
    summary: '量子纠缠传输在长距离信道中受损耗、退相干和中继稳定性影响。'
  },
  {
    id: 'exp-018',
    type: 'experience',
    tone: 'orange',
    source: 'ToolExecutor',
    time: '12:14:04',
    topic: '证据重排策略',
    summary: '优先保留与限制条件、实验规模和信道噪声相关的证据片段。'
  },
  {
    id: 'mem-009',
    type: 'summary',
    tone: 'blue',
    source: 'Summarizer',
    time: '11:52:40',
    topic: '纠缠分发',
    summary: '上一轮总结指出纠缠分发效率是大尺度部署的核心瓶颈。'
  }
]

const chain = [
  { title: 'Query', text: '量子纠缠传输能否支撑大尺度安全通信', tone: 'blue' },
  { title: '实体节点', text: '纠缠分发 / 安全通信 / 大尺度部署', tone: 'green' },
  { title: '证据记忆', text: 'mem-021 + mem-009', tone: 'orange' },
  { title: '经验片段', text: 'exp-018 用于重排限制因素', tone: 'purple' },
  { title: '下游使用', text: '交给总结 Agent 生成结论并写回新记忆', tone: 'gray' }
]
</script>
