<template>
  <section class="page collab-page">
    <div class="collab-summary compact-summary">
      <article v-for="item in summary" :key="item.label" class="collab-summary-card">
        <span :class="item.tone">{{ item.icon }}</span>
        <div>
          <em>{{ item.label }}</em>
          <strong>{{ item.value }}</strong>
        </div>
      </article>
    </div>

    <div class="collab-grid compact-collab-grid">
      <section class="collab-card agents-card">
        <header>
          <span></span>
          <h2>连续任务条</h2>
          <em>点击任务查看该轮记录</em>
        </header>
        <div v-if="tasks.length" class="task-strip-list compact-task-list">
          <button
            v-for="task in tasks"
            :key="task.id"
            class="task-strip-item"
            :class="[task.state, { active: task.id === selectedTaskId }]"
            type="button"
            @click="selectTask(task.id)"
          >
            <i></i>
            <div class="task-strip-main">
              <div class="task-strip-head">
                <strong>任务 {{ task.round }}</strong>
                <b>{{ task.status }}</b>
              </div>
              <p>{{ task.title }}</p>
              <div v-if="task.tags.length" class="chip-row">
                <span v-for="tag in task.tags" :key="tag">{{ tag }}</span>
              </div>
            </div>
            <em>{{ task.progress }}%</em>
          </button>
        </div>
        <div v-else class="runtime-empty-panel">
          <strong>暂无运行数据</strong>
          <span>启动实验后，这里会显示后端加载的真实任务队列。</span>
        </div>
      </section>

      <section class="collab-card sequence-card">
        <header>
          <span></span>
          <h2>运行时协作链路</h2>
          <em>{{ selectedTask.title }}</em>
        </header>
        <div v-if="selectedTask.messages.length" class="sequence-board flow-sequence-board">
          <div class="flow-agent-row" :style="flowGridStyle">
            <span v-for="agent in sequenceAgents" :key="agent" class="flow-agent">
              <b>{{ agent }}</b>
              <i></i>
            </span>
          </div>
          <div class="flow-message-list">
            <div
              v-for="(msg, index) in selectedTask.messages"
              :key="msg.id"
              class="flow-message-row"
              :style="flowGridStyle"
            >
              <span v-for="agent in sequenceAgents" :key="`${msg.id}-${agent}`" class="flow-lifeline"></span>
              <div
                class="flow-edge seq-animated"
                :class="[msg.tone, { reverse: msg.reverse, running: selectedTask.state === 'running' && index === activeMessageIndex }]"
                :style="[msg.edgeStyle, { '--delay': `${index * 120}ms` }]"
              >
                <em>{{ msg.label }}</em>
                <b v-if="msg.memoryCount">{{ msg.memoryCount }} 条</b>
              </div>
            </div>
          </div>
          <div class="sequence-legend">
            <span><i class="blue"></i>结构化消息</span>
            <span><i class="green"></i>状态传递</span>
            <span><i class="orange"></i>记忆检索/复用</span>
            <span><i class="purple"></i>工具执行</span>
          </div>
        </div>
        <div v-else class="runtime-empty-panel">
          <strong>暂无协作链路</strong>
          <span>收到 Agent 消息、状态传递或记忆事件后自动绘制。</span>
        </div>
      </section>

      <section class="collab-card packet-detail-card protocol-card-live">
        <header>
          <span></span>
          <h2>协议信息</h2>
          <em>{{ selectedTask.roundText }}</em>
        </header>
        <div class="packet-tabs protocol-tabs">
          <button
            v-for="tab in packetTabs"
            :key="tab.value"
            :class="{ active: selectedPacketTab === tab.value }"
            type="button"
            @click="selectedPacketTab = tab.value"
          >
            {{ tab.label }}
          </button>
        </div>
        <section v-if="activePacket.hasData" class="protocol-detail-panel">
          <div class="protocol-detail-head">
            <strong>{{ activePacket.title }}</strong>
            <span>{{ activePacket.badge }}</span>
          </div>
          <p>{{ activePacket.desc }}</p>
          <pre>{{ activePacket.payload }}</pre>
        </section>
        <div v-else class="runtime-empty-panel">
          <strong>{{ activePacket.title }}</strong>
          <span>{{ activePacket.desc }}</span>
        </div>
      </section>

      <section class="collab-card event-log-card compact-event-card">
        <header>
          <span></span>
          <h2>实时日志</h2>
          <em>当前查看：{{ selectedTask.roundText }}</em>
        </header>
        <div v-if="selectedTask.events.length" class="collab-table-wrap">
          <table>
            <thead>
              <tr>
                <th>时间</th>
                <th>发送方</th>
                <th>接收方</th>
                <th>事件类型</th>
                <th>动作</th>
                <th>状态引用</th>
                <th>记忆引用</th>
                <th>文本开销</th>
                <th>耗时</th>
                <th>结果</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in selectedTask.events" :key="`${selectedTask.id}-${row.time}-${row.action}-${row.type}`">
                <td>{{ row.time }}</td>
                <td>{{ row.from }}</td>
                <td>{{ row.to }}</td>
                <td><b :class="row.tone">{{ row.type }}</b></td>
                <td>{{ row.action }}</td>
                <td><span>{{ row.state }}</span></td>
                <td><span>{{ row.memory }}</span></td>
                <td>{{ row.tokens }}</td>
                <td>{{ row.cost }}</td>
                <td>{{ row.result }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-else class="runtime-empty-panel">
          <strong>暂无日志</strong>
          <span>运行事件写入后，这里只展示后端记录的真实事件。</span>
        </div>
      </section>
    </div>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import {
  defaultExperimentConfig,
  labelOf,
  memoryModes,
  normalizeExperimentConfig,
  retrievalOptionsFor,
  selectedCollaboration
} from '../experimentOptions'

const CONFIG_STORAGE_KEY = 'competition-agent-console-config'
const RUNNER_EVENTS_URL = 'http://127.0.0.1:8765/api/runner/events'

const selectedTaskId = ref('')
const selectedPacketTab = ref('message')
const tick = ref(0)
const savedConfig = ref(readSavedConfig())
const runtimeEvents = ref([])
let timer = null
let eventTimer = null

const packetTabs = [
  { label: '握手', value: 'handshake' },
  { label: '能力发现', value: 'capability' },
  { label: '协议映射', value: 'mapping' },
  { label: '消息详情', value: 'message' }
]

const config = computed(() => normalizeExperimentConfig(savedConfig.value))
const collaboration = computed(() => selectedCollaboration(config.value))
const memoryLabel = computed(() => labelOf(memoryModes, config.value.memoryMode))
const retrievalLabel = computed(() => labelOf(retrievalOptionsFor(config.value.memoryMode), config.value.retrievalMode))
const hasRuntimeEvents = computed(() => runtimeEvents.value.length > 0)
const runMeta = computed(() => runtimeEvents.value.find((event) => event.event_type === 'run_start') ?? {})
const taskQueue = computed(() => {
  const queue = runtimeEvents.value.find((event) => event.event_type === 'task_queue')
  return Array.isArray(queue?.tasks) ? queue.tasks : []
})
const runModeLabel = computed(() => {
  if (hasRuntimeEvents.value && runMeta.value.experiment_mode) {
    return `${modeText(runMeta.value.experiment_mode)} / ${runMeta.value.memory_organization || '--'} / ${runMeta.value.memory_retrieval_mode || '--'}`
  }
  return `${collaboration.value.label} / ${memoryLabel.value} / ${retrievalLabel.value}`
})

const tasks = computed(() => (hasRuntimeEvents.value ? buildTasksFromRuntimeEvents(runtimeEvents.value) : []))
const selectedTask = computed(() => tasks.value.find((task) => task.id === selectedTaskId.value) ?? tasks.value[0] ?? emptyTask())
const sequenceAgents = computed(() => selectedTask.value.agents || [])
const flowGridStyle = computed(() => ({
  gridTemplateColumns: `repeat(${Math.max(sequenceAgents.value.length, 1)}, minmax(0, 1fr))`
}))
const activeMessageIndex = computed(() => {
  const count = selectedTask.value.messages.length || 1
  return tick.value % count
})
const activePacket = computed(() => selectedTask.value.protocol[selectedPacketTab.value] ?? emptyPacket(selectedPacketTab.value))

const summary = computed(() => [
  { label: '运行模式', value: runModeLabel.value, icon: '协', tone: 'green' },
  { label: '当前轮次', value: tasks.value.length ? `第 ${selectedTask.value.round} 轮 / 共 ${tasks.value.length} 轮` : '--', icon: '轮', tone: 'blue' },
  { label: '消息次数', value: `${selectedTask.value.metrics.messages} 次`, icon: '信', tone: 'purple' },
  { label: '文本开销', value: selectedTask.value.metrics.textCost, icon: '文', tone: 'blue' },
  { label: '状态传递', value: `${selectedTask.value.metrics.stateTransfers} 次`, icon: '态', tone: 'green' },
  { label: '记忆命中', value: `${selectedTask.value.metrics.memoryHits} 次`, icon: '记', tone: 'orange' },
  { label: '当前角色', value: selectedTask.value.currentAgent, icon: '角', tone: 'purple' }
])

watch(tasks, (nextTasks) => {
  if (!nextTasks.length) {
    selectedTaskId.value = ''
    return
  }
  if (!nextTasks.some((task) => task.id === selectedTaskId.value)) {
    const runningTask = nextTasks.find((task) => task.state === 'running')
    selectedTaskId.value = runningTask?.id ?? nextTasks[0].id
  }
})

onMounted(() => {
  refreshSavedConfig()
  refreshRuntimeEvents()
  window.addEventListener('storage', refreshSavedConfig)
  timer = window.setInterval(() => {
    tick.value += 1
  }, 900)
  eventTimer = window.setInterval(refreshRuntimeEvents, 1200)
})

onBeforeUnmount(() => {
  window.removeEventListener('storage', refreshSavedConfig)
  if (timer) window.clearInterval(timer)
  if (eventTimer) window.clearInterval(eventTimer)
})

function selectTask(taskId) {
  selectedTaskId.value = taskId
  selectedPacketTab.value = 'message'
}

function readSavedConfig() {
  if (typeof window === 'undefined') return defaultExperimentConfig()
  try {
    const raw = window.localStorage.getItem(CONFIG_STORAGE_KEY)
    return raw ? normalizeExperimentConfig({ ...defaultExperimentConfig(), ...JSON.parse(raw) }) : defaultExperimentConfig()
  } catch {
    return defaultExperimentConfig()
  }
}

function refreshSavedConfig() {
  savedConfig.value = readSavedConfig()
}

async function refreshRuntimeEvents() {
  try {
    const response = await fetch(`${RUNNER_EVENTS_URL}?t=${Date.now()}`, { cache: 'no-store' })
    if (!response.ok) {
      runtimeEvents.value = []
      return
    }
    const payload = await response.json()
    runtimeEvents.value = Array.isArray(payload.events) ? payload.events : []
  } catch {
    runtimeEvents.value = []
  }
}

function buildTasksFromRuntimeEvents(events) {
  const maxEventRound = Math.max(0, ...events.map((event) => Number(event.task_index) || 0))
  const total = Math.max(Number(runMeta.value.task_count) || 0, taskQueue.value.length, maxEventRound)
  const runDone = events.some((event) => event.event_type === 'run_done')

  return Array.from({ length: total }, (_, index) => {
    const round = index + 1
    const queueItem = taskQueue.value.find((item) => Number(item.task_index) === round) ?? {}
    const taskEvents = events.filter((event) => Number(event.task_index) === round)
    const startEvent = taskEvents.find((event) => event.event_type === 'task_start')
    const doneEvent = [...taskEvents].reverse().find((event) => event.event_type === 'task_done')
    const state = doneEvent ? 'done' : taskEvents.length && !runDone ? 'running' : 'waiting'
    const metrics = metricsFromRuntimeEvents(taskEvents, doneEvent)
    const agents = agentsFromRuntimeEvents(taskEvents)
    return {
      id: `task${round}`,
      round,
      roundText: `任务 ${round}`,
      title: startEvent?.task_title || queueItem.task_title || doneEvent?.task_title || `任务 ${round}`,
      state,
      status: stateText(state),
      progress: state === 'done' ? 100 : state === 'running' ? progressFromEvents(taskEvents) : 0,
      tags: runtimeTags(taskEvents),
      currentAgent: runtimeCurrentAgent(taskEvents, state),
      metrics,
      agents,
      messages: messagesFromRuntimeEvents(taskEvents, agents),
      protocol: protocolFromRuntimeEvents(round, state, taskEvents),
      events: tableRowsFromRuntimeEvents(taskEvents)
    }
  })
}

function metricsFromRuntimeEvents(taskEvents, doneEvent) {
  const messageEvents = taskEvents.filter((event) => event.event_type === 'agent_message')
  const stateEvents = taskEvents.filter((event) => event.event_type === 'state_transfer')
  const hitEvents = taskEvents.filter((event) => event.event_type === 'memory_hit')
  const doneResult = doneEvent?.return_result ?? {}
  const textChars =
    Number(doneEvent?.text_chars) ||
    messageEvents.reduce((sum, event) => sum + (Number(event.text_chars) || 0), 0)
  return {
    messages: Number(doneResult.message_count) || messageEvents.length,
    textCost: textChars ? `${textChars} 字符` : '--',
    stateTransfers: Number(doneResult.state_transfer_count) || stateEvents.length,
    memoryHits: Number(doneResult.memory_hit_count) || hitEvents.length
  }
}

function progressFromEvents(taskEvents) {
  const executableEvents = taskEvents.filter((event) =>
    ['handshake', 'capability_discovery', 'protocol_mapping', 'agent_message', 'state_transfer', 'memory_query', 'memory_hit', 'memory_write'].includes(event.event_type)
  )
  return Math.min(95, Math.max(8, executableEvents.length * 12))
}

function runtimeTags(taskEvents) {
  if (!taskEvents.length) return []
  return Array.from(new Set(taskEvents.map((event) => eventLabel(event)).filter(Boolean))).slice(0, 3)
}

function runtimeCurrentAgent(taskEvents, state) {
  if (state === 'waiting') return '未开始'
  if (state === 'done') return '已完成'
  const last = [...taskEvents].reverse().find((event) => event.from_agent || event.to_agent)
  return normalizeAgentName(last?.to_agent || last?.from_agent || '运行中')
}

function agentsFromRuntimeEvents(taskEvents) {
  const ordered = ['Planner', 'Retriever', 'Memory', 'Reasoner', 'Executor', 'Summarizer']
  const names = new Set()
  flowEventsFromRuntimeEvents(taskEvents).forEach((event) => {
    const route = visualRouteForEvent(event)
    if (route.from && !isSystemAgent(route.from)) names.add(route.from)
    if (route.to && !isSystemAgent(route.to)) names.add(route.to)
  })
  const orderedNames = ordered.filter((name) => names.has(name))
  const extraNames = Array.from(names).filter((name) => !ordered.includes(name)).sort()
  return [...orderedNames, ...extraNames]
}

function flowEventsFromRuntimeEvents(taskEvents) {
  return taskEvents.flatMap((event) => {
    if (!['agent_message', 'state_transfer', 'memory_query', 'memory_hit', 'memory_write'].includes(event.event_type)) {
      return []
    }
    return [event]
  })
}

function visualRouteForEvent(event) {
  if (event.flow_type === 'memory_query' || event.event_type === 'memory_query') {
    return { from: 'Retriever', to: 'Memory' }
  }
  if (event.flow_type === 'memory_result' || event.event_type === 'memory_hit') {
    return { from: 'Memory', to: 'Retriever' }
  }
  if (event.event_type === 'memory_write') {
    return { from: 'Summarizer', to: 'Memory' }
  }
  if (event.event_type === 'state_transfer') {
    return {
      from: normalizeAgentName(event.from_agent || 'runtime'),
      to: normalizeAgentName(String(event.to_agent || '').split(',')[0] || event.from_agent || 'runtime')
    }
  }
  return {
    from: normalizeAgentName(event.from_agent || 'runtime'),
    to: normalizeAgentName(event.to_agent || event.from_agent || 'runtime')
  }
}

function messagesFromRuntimeEvents(taskEvents, agents) {
  const sourceEvents = flowEventsFromRuntimeEvents(taskEvents)
  if (!agents.length) return []
  return selectFlowEvents(sourceEvents).map((event, index) => {
    const route = visualRouteForEvent(event)
    const from = agentColumn(route.from, agents)
    const to = agentColumn(route.to, agents)
    return {
      id: event.synthetic_id || event.event_id || `${event.event_type}-${index}`,
      label: `${index + 1}. ${eventLabel(event)}`,
      tone: eventTone(event),
      reverse: from > to,
      edgeStyle: flowEdgeStyle(from, to, agents.length),
      memoryCount: eventMemoryCount(event)
    }
  })
}

function selectFlowEvents(events) {
  if (events.length <= 10) return events
  const rest = events.filter((event) => event.event_type !== 'memory_write')
  const latestMemoryWrite = [...events].reverse().find((event) => event.event_type === 'memory_write')
  const selected = rest.slice(0, latestMemoryWrite ? 11 : 12)
  return latestMemoryWrite ? [...selected, latestMemoryWrite] : selected
}

function flowEdgeStyle(from, to, count) {
  const columnCount = Math.max(count, 1)
  const fromCenter = ((from + 0.5) / columnCount) * 100
  const toCenter = ((to + 0.5) / columnCount) * 100
  const left = Math.min(fromCenter, toCenter)
  const width = Math.max(Math.abs(toCenter - fromCenter), 3)
  return {
    left: `${left}%`,
    width: `${width}%`
  }
}

function protocolFromRuntimeEvents(round, state, taskEvents) {
  const handshake = taskEvents.find((event) => event.event_type === 'handshake')
  const capability = taskEvents.find((event) => event.event_type === 'capability_discovery')
  const mapping = [...taskEvents].reverse().find((event) => event.event_type === 'protocol_mapping')
  const message = [...taskEvents].reverse().find((event) => event.event_type === 'agent_message')
  return {
    handshake: packetFromEvent('握手', handshake, '等待后端握手事件。'),
    capability: packetFromEvent('能力发现', capability, '等待后端能力发现事件。'),
    mapping: packetFromEvent('协议映射', mapping, '等待后端协议映射事件。'),
    message: packetFromEvent('消息详情', message, state === 'waiting' ? `任务 ${round} 尚未开始。` : '等待 Agent 消息事件。')
  }
}

function packetFromEvent(title, event, emptyDesc) {
  if (!event) return emptyPacket(title, emptyDesc)
  return {
    title,
    badge: event.action_type || event.event_type,
    desc: `${normalizeAgentName(event.from_agent)} -> ${normalizeAgentName(event.to_agent)}`,
    payload: stringify(event),
    hasData: true
  }
}

function tableRowsFromRuntimeEvents(taskEvents) {
  return taskEvents.map((event) => ({
    time: formatEventTime(event.timestamp),
    from: normalizeAgentName(event.from_agent || 'runtime'),
    to: normalizeAgentName(event.to_agent || 'dashboard'),
    type: event.event_type || 'event',
    action: event.action_type || '-',
    state: listText(event.state_refs),
    memory: listText(event.memory_refs),
    tokens: event.text_chars ? `${event.text_chars} 字符` : '0 字符',
    cost: event.elapsed_ms ? `${Number(event.elapsed_ms).toFixed(2)} ms` : '0 ms',
    result: resultText(event),
    tone: eventTone(event)
  }))
}

function emptyTask() {
  return {
    id: '',
    round: '--',
    roundText: '未运行',
    title: '未运行',
    state: 'waiting',
    status: '未运行',
    progress: 0,
    tags: [],
    currentAgent: '未开始',
    metrics: { messages: 0, textCost: '--', stateTransfers: 0, memoryHits: 0 },
    agents: [],
    messages: [],
    protocol: {
      handshake: emptyPacket('握手', '启动实验后显示真实握手事件。'),
      capability: emptyPacket('能力发现', '启动实验后显示真实能力发现事件。'),
      mapping: emptyPacket('协议映射', '启动实验后显示真实协议映射事件。'),
      message: emptyPacket('消息详情', '启动实验后显示真实 Agent 消息。')
    },
    events: []
  }
}

function emptyPacket(title, desc = '暂无数据') {
  return {
    title,
    badge: '--',
    desc,
    payload: '',
    hasData: false
  }
}

function stateText(state) {
  if (state === 'done') return '已完成'
  if (state === 'running') return '运行中'
  return '未开始'
}

function modeText(mode) {
  const map = {
    baseline_text: '纯文本',
    protocol_api: '结构化协议',
    protocol_api_embedding: '协议+向量',
    hidden_api: '隐状态传递',
    text: '纯文本',
    protocol: '结构化协议',
    hidden: '隐状态传递',
    '<developer-modes>': '开发模式'
  }
  return map[mode] || mode
}

function agentColumn(name, agents) {
  const normalized = normalizeAgentName(name)
  const index = agents.indexOf(normalized)
  return index >= 0 ? index : 0
}

function normalizeAgentName(name) {
  const value = String(name || '')
  const map = {
    planner: 'Planner',
    retriever: 'Retriever',
    reasoner: 'Reasoner',
    executor: 'Executor',
    summarizer: 'Summarizer',
    memory: 'Memory',
    runtime: 'Runtime',
    dashboard: 'Dashboard'
  }
  return map[value.toLowerCase()] || value || '-'
}

function isSystemAgent(name) {
  return ['Runtime', 'Dashboard', 'runtime', 'dashboard'].includes(String(name || ''))
}

function eventLabel(event) {
  if (event.flow_type === 'memory_query') return '检索记忆'
  if (event.flow_type === 'memory_result') return `命中记忆 ${eventMemoryCount(event)}`
  const map = {
    run_start: '启动',
    task_queue: '任务队列',
    task_start: '开始任务',
    handshake: '握手',
    capability_discovery: '能力发现',
    protocol_mapping: '协议映射',
    agent_message: actionText(event.action_type),
    state_transfer: `STATE ${listText(event.state_refs)}`,
    memory_query: '检索记忆',
    memory_hit: `命中记忆 ${listText(event.memory_refs)}`,
    memory_write: `写回记忆 ${listText(event.memory_refs)}`,
    task_done: '任务完成',
    run_done: '运行完成'
  }
  return map[event.event_type] || String(event.action_type || event.event_type || 'event').toUpperCase()
}

function actionText(action) {
  const map = {
    retrieve: '检索',
    reason: '推理',
    execute: '执行',
    summarize: '总结',
    start_task: '分发任务',
    open_session: '建立会话',
    discover_capability: '能力发现',
    map_protocol: '协议映射'
  }
  return map[action] || String(action || '消息').toUpperCase()
}

function eventTone(event) {
  if (event.event_type === 'state_transfer') return 'green'
  if (event.flow_type || event.event_type === 'memory_query' || event.event_type === 'memory_hit' || event.event_type === 'memory_write') return 'orange'
  if (event.event_type === 'protocol_mapping' || String(event.to_agent || '').toLowerCase().includes('executor')) return 'purple'
  return 'blue'
}

function eventMemoryCount(event) {
  const refs = Array.isArray(event.memory_refs) ? event.memory_refs.filter(Boolean) : event.memory_refs ? [event.memory_refs] : []
  const result = event.return_result || {}
  return Number(result.memory_hit_count || result.hit_count || refs.length || 0)
}

function listText(value) {
  const values = Array.isArray(value) ? value.filter(Boolean) : value ? [value] : []
  return values.length ? values.join(', ') : '-'
}

function resultText(event) {
  const result = event.return_result
  if (!result) return event.event_type === 'task_start' ? '开始' : '-'
  if (typeof result === 'string') return result
  if (result.memory_id) return result.summary ? `${result.memory_id}：${result.summary}` : result.memory_id
  if (typeof result.message_count !== 'undefined') return `消息 ${result.message_count}，命中 ${result.memory_hit_count || 0}`
  if (Array.isArray(result.result_refs)) return result.result_refs.join(', ')
  return JSON.stringify(result)
}

function formatEventTime(value) {
  if (!value) return '--'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return String(value)
  const hh = String(date.getHours()).padStart(2, '0')
  const mm = String(date.getMinutes()).padStart(2, '0')
  const ss = String(date.getSeconds()).padStart(2, '0')
  const ms = String(date.getMilliseconds()).padStart(3, '0')
  return `${hh}:${mm}:${ss}.${ms}`
}

function stringify(payload) {
  return JSON.stringify(payload, null, 2)
}
</script>
