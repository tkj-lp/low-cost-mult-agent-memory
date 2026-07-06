<template>
  <BaseCard title="多 Agent 运行时与结构化通信链路" class="monitor-card communication-card runtime-card">
    <template #icon><span class="title-mark"></span></template>
    <div class="runtime-link">
      <section class="runtime-agents">
        <article v-for="agent in agents" :key="agent.name" class="runtime-agent" :class="agent.tone">
          <div class="runtime-agent-head">
            <span>{{ agent.icon }}</span>
            <strong>{{ agent.name }}</strong>
            <em>{{ agent.status }}</em>
          </div>
          <p>{{ agent.capability }}</p>
          <div class="tag-group">
            <StatBadge v-for="tag in agent.tags" :key="tag" :tone="agent.tone">{{ tag }}</StatBadge>
          </div>
        </article>
      </section>

      <section class="protocol-lane">
        <article v-for="packet in packets" :key="packet.id" class="protocol-packet" :class="packet.tone">
          <header>
            <span>{{ packet.from }}</span>
            <strong>→</strong>
            <span>{{ packet.to }}</span>
            <em>{{ packet.type }}</em>
          </header>
          <dl>
            <div><dt>action</dt><dd>{{ packet.action }}</dd></div>
            <div><dt>params</dt><dd>{{ packet.params }}</dd></div>
            <div><dt>result</dt><dd>{{ packet.result }}</dd></div>
            <div><dt>state</dt><dd>{{ packet.state }}</dd></div>
          </dl>
        </article>
      </section>

      <section class="runtime-subgrid">
        <article class="runtime-mini-panel">
          <h3>握手与能力发现</h3>
          <div class="handshake-list">
            <span v-for="item in handshakes" :key="item">{{ item }}</span>
          </div>
        </article>
        <article class="runtime-mini-panel">
          <h3>协议映射</h3>
          <div class="mapping-grid">
            <span>自然语言意图</span><strong>retrieve(query, top_k)</strong>
            <span>工具执行请求</span><strong>tool_call(name, args)</strong>
            <span>总结写回</span><strong>memory.write(unit)</strong>
          </div>
        </article>
      </section>
    </div>
  </BaseCard>
</template>

<script setup>
import BaseCard from '../BaseCard.vue'
import StatBadge from '../StatBadge.vue'

const agents = [
  {
    name: '协议调度器',
    icon: '↔',
    status: '路由中',
    tone: 'blue',
    capability: '解析动作类型、参数约束和能力描述，完成协议映射与消息路由。',
    tags: ['handshake', 'schema:v2']
  },
  {
    name: '检索 Agent',
    icon: '⌕',
    status: '运行中',
    tone: 'green',
    capability: '执行关键词、标签和语义相似度检索，返回记忆引用与证据链。',
    tags: ['retrieve', 'memory.search']
  },
  {
    name: '工具执行 Agent',
    icon: '⚙',
    status: '待调用',
    tone: 'purple',
    capability: '根据结构化输入调用解析、归一化、校验等工具，输出 result_ref。',
    tags: ['tool_call', 'validate']
  },
  {
    name: '总结生成 Agent',
    icon: '◇',
    status: '排队中',
    tone: 'orange',
    capability: '消费证据、工具结果和非文本状态，生成答案并写回共享记忆。',
    tags: ['summarize', 'memory.write']
  }
]

const packets = [
  {
    id: 'p1',
    from: '调度器',
    to: '检索',
    type: '能力匹配',
    action: 'discover_capability',
    params: 'role=retrieval, modality=vector',
    result: 'cap_id: retriever.fusion',
    state: 'session_ref=sess-007'
  },
  {
    id: 'p2',
    from: '检索',
    to: '工具',
    type: '结构化请求',
    action: 'tool_call',
    params: 'query_terms + evidence_refs',
    result: 'pending',
    state: 'embedding_ref=emb-7a2'
  },
  {
    id: 'p3',
    from: '工具',
    to: '总结',
    type: '结果传递',
    action: 'return_result',
    params: 'result_ref=tool-007',
    result: 'normalized evidence',
    state: 'state_ref=tool-state-007'
  }
]

const handshakes = [
  'client_hello',
  'capability_manifest',
  'schema_negotiation',
  'session_ack'
]
</script>
