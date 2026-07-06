<template>
  <BaseCard title="连续任务队列与当前处理步骤" class="monitor-card task-pipeline-card runtime-card">
    <template #icon><span class="title-mark"></span></template>
    <div class="task-pipeline runtime-task-panel">
      <section class="task-list-panel runtime-queue">
        <h3>任务队列 <small>10 轮连续任务</small></h3>
        <div class="task-list-scroll">
          <div
            v-for="task in tasks"
            :key="task.name"
            class="task-step"
            :class="task.status"
          >
            <span class="timeline-dot"></span>
            <strong>{{ task.name }}</strong>
            <span class="task-state">{{ task.topic }}</span>
          </div>
        </div>
      </section>

      <section class="agent-flow-panel runtime-steps">
        <h3>task7 处理链 <small>检索 → 工具执行 → 总结生成</small></h3>
        <div class="runtime-step-scroll">
          <article v-for="step in steps" :key="step.name" class="runtime-step" :class="step.tone">
            <div class="runtime-step-head">
              <span>{{ step.icon }}</span>
              <strong>{{ step.name }}</strong>
              <em>{{ step.status }}</em>
            </div>
            <p>{{ step.desc }}</p>
            <div class="tag-group">
              <StatBadge v-for="tag in step.tags" :key="tag" :tone="step.tone">{{ tag }}</StatBadge>
            </div>
          </article>
        </div>
      </section>
    </div>
  </BaseCard>
</template>

<script setup>
import BaseCard from '../BaseCard.vue'
import StatBadge from '../StatBadge.vue'

const tasks = Array.from({ length: 10 }, (_, index) => {
  const num = index + 1
  const topics = ['链式问答', '证据补全', '限定条件', '跨轮引用', '经验复用', '语义聚合', '安全通信', '方案对比', '结论校验', '记忆写回']
  if (num <= 6) return { name: `task${num}`, status: 'done', topic: topics[index] }
  if (num === 7) return { name: `task${num}`, status: 'active', topic: `${topics[index]} · 运行中` }
  return { name: `task${num}`, status: 'waiting', topic: topics[index] }
})

const steps = [
  {
    name: '检索 Agent',
    icon: '⌕',
    status: '正在检索',
    tone: 'green',
    desc: '根据关键词、标签和语义向量检索共享记忆，返回证据片段与引用 ID。',
    tags: ['action: retrieve', 'mem-021', 'exp-018']
  },
  {
    name: '工具执行 Agent',
    icon: '⚙',
    status: '等待输入',
    tone: 'blue',
    desc: '接收结构化参数，执行解析、归一化或外部工具调用，并输出 result_ref。',
    tags: ['tool: qa_parser', 'cap: normalize']
  },
  {
    name: '总结生成 Agent',
    icon: '◇',
    status: '排队中',
    tone: 'orange',
    desc: '汇总证据链和中间状态，生成最终回答并写回新的记忆单元。',
    tags: ['write: mem-030', 'state_ref']
  }
]
</script>
