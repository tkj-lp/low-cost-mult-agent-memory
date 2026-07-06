<template>
  <BaseCard title="当前协议包与非文本状态交换" class="monitor-card agent-output-card runtime-card">
    <template #icon><span class="title-mark"></span></template>
    <div class="agent-output-scroll runtime-detail-scroll">
      <section class="runtime-detail-section">
        <h3>当前动作</h3>
        <div class="action-row">
          <StatBadge tone="green">retrieve_result</StatBadge>
          <span>检索 Agent → 工具执行 Agent</span>
        </div>
      </section>
      <section class="runtime-detail-section">
        <h3>结构化通信内容</h3>
        <pre class="code-wrap">{
  "action_type": "retrieve_result",
  "capability": "memory.search.fusion",
  "input_params": {
    "query": "量子纠缠 安全通信 限制因素",
    "top_k": 5,
    "filters": ["evidence", "experience"]
  },
  "return_result": {
    "result_refs": ["mem-021", "exp-018", "mem-009"],
    "confidence": 0.83
  },
  "next_action": "tool_call"
}</pre>
      </section>
      <section class="runtime-detail-section">
        <h3>非文本中间状态</h3>
        <div class="state-transfer-list">
          <article v-for="item in states" :key="item.id">
            <strong>{{ item.id }}</strong>
            <span>{{ item.desc }}</span>
            <em>{{ item.route }}</em>
          </article>
        </div>
      </section>
    </div>
  </BaseCard>
</template>

<script setup>
import BaseCard from '../BaseCard.vue'
import StatBadge from '../StatBadge.vue'

const states = [
  { id: 'emb-7a2', desc: '查询语义向量，768 维，用于相似度检索和下游重排。', route: '生成：Retriever · 接收：ToolExecutor' },
  { id: 'state-007', desc: '证据链隐藏摘要，包含实体、约束、上下文窗口引用。', route: '生成：Retriever · 接收：Summarizer' },
  { id: 'tool-state-007', desc: '工具解析后的结构化候选答案与限制条件标注。', route: '生成：ToolExecutor · 接收：Summarizer' }
]
</script>
