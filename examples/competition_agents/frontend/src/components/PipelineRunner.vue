<template>
  <div class="runner-page">
    <section class="page-block">
      <div class="block-head">
        <div>
          <h3 class="block-title">任务状态</h3>
          <p class="block-desc">根据模拟日志更新执行阶段与进度。</p>
        </div>
        <div class="kv-inline">
          <span>轮次：{{ currentTurn }}/3</span>
          <span>状态：执行中</span>
          <span>节点：{{ pipelineNodes[activeNodeIndex].name }}</span>
        </div>
      </div>
      <div class="block-body">
        <el-progress
          :percentage="progressPercentage"
          :show-text="false"
          color="#262626"
        />
      </div>
    </section>

    <section class="page-block">
      <div class="block-head">
        <div>
          <h3 class="block-title">执行拓扑</h3>
          <p class="block-desc">突出当前节点，其他节点保持静态呈现。</p>
        </div>
      </div>
      <div class="block-body">
        <div class="topology">
          <template v-for="(node, index) in pipelineNodes" :key="node.name">
            <div class="node-card" :class="{ active: activeNodeIndex === index }">
              <span class="node-index">{{ index + 1 }}</span>
              <strong>{{ node.name }}</strong>
              <span class="node-code">{{ node.stateId }}</span>
              <span class="node-meta">动作：{{ node.action }}</span>
            </div>
            <div v-if="index < pipelineNodes.length - 1" class="node-link">
              <span>ref: {{ node.ref }}</span>
            </div>
          </template>
        </div>
      </div>
    </section>

    <section class="page-block">
      <div class="block-head">
        <div>
          <h3 class="block-title">日志输出</h3>
          <p class="block-desc">展示模拟执行过程中的标准输出流。</p>
        </div>
      </div>
      <div class="block-body">
        <div class="terminal-box" ref="terminalRef">
          <div v-for="(log, index) in logs" :key="index" class="log-line">
            {{ log }}
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const currentTurn = ref(1)
const progressPercentage = ref(20)
const activeNodeIndex = ref(0)
const terminalRef = ref<HTMLElement | null>(null)

const pipelineNodes = [
  { name: '规划器', action: 'plan', stateId: 'st-001', ref: 'null' },
  { name: '检索器', action: 'retrieve', stateId: 'st-002', ref: 'hidden-09x' },
  { name: '推理器', action: 'reason', stateId: 'st-003', ref: 'hidden-10x' },
  { name: '汇总器', action: 'summarize', stateId: 'st-004', ref: 'hidden-11x' }
]

const logs = ref([
  '[INFO] 启动 run_demo.py',
  '[INFO] 正在载入共享记忆模块',
  '[OK] 张量内存池分配成功，容量 512MB',
  '[RUNNING] 开始处理数据链 group-01'
])

let timer: ReturnType<typeof setInterval> | undefined

onMounted(() => {
  let step = 0
  timer = setInterval(() => {
    step += 1

    if (step === 2) {
      activeNodeIndex.value = 1
      logs.value.push('[AGENT] 检索器开始读取图索引。')
      progressPercentage.value = 45
    } else if (step === 4) {
      activeNodeIndex.value = 2
      currentTurn.value = 2
      logs.value.push('[LATENT] 推理器命中隐藏态引用 hidden-10x。')
      progressPercentage.value = 75
    } else if (step === 6) {
      activeNodeIndex.value = 3
      logs.value.push('[OK] 汇总完成，exact_match = true。')
      progressPercentage.value = 100
    } else if (step >= 8) {
      if (timer) {
        clearInterval(timer)
      }
      router.push('/metrics')
    }

    if (terminalRef.value) {
      terminalRef.value.scrollTop = terminalRef.value.scrollHeight
    }
  }, 1000)
})

onUnmounted(() => {
  if (timer) {
    clearInterval(timer)
  }
})
</script>

<style scoped>
.runner-page {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.topology {
  display: grid;
  grid-template-columns: repeat(7, minmax(0, 1fr));
  gap: 10px;
  align-items: center;
}

.node-card,
.node-link {
  min-height: 96px;
  border: 1px solid var(--ui-border);
  background: #fff;
}

.node-card {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 12px;
}

.node-card.active {
  border-color: var(--ui-text);
  background: #fafafa;
}

.node-index,
.node-code,
.node-meta,
.node-link {
  font-size: 12px;
  color: var(--ui-muted);
}

.node-link {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 1px;
  padding: 0 8px;
  border-left: none;
  border-right: none;
  background: transparent;
}

.terminal-box {
  height: 240px;
  overflow-y: auto;
  border: 1px solid var(--ui-border);
  background: #fafafa;
  padding: 12px;
}

.log-line {
  padding: 6px 0;
  border-bottom: 1px solid #ededed;
  color: var(--ui-text);
}

.log-line:last-child {
  border-bottom: none;
}

@media (max-width: 1100px) {
  .topology {
    grid-template-columns: 1fr;
  }

  .node-link {
    min-height: 40px;
    border: 1px solid var(--ui-border);
  }
}
</style>
