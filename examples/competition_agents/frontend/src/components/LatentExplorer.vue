<template>
  <div class="latent-page">
    <section class="page-block">
      <div class="block-head">
        <div>
          <h3 class="block-title">追踪范围</h3>
          <p class="block-desc">当前查看的执行单元来自全局效能大盘的明细表。</p>
        </div>
        <el-button @click="goBackDashboard">返回效能大盘</el-button>
      </div>
      <div class="block-body">
        <div class="focus-box">当前单元：{{ turnId }}</div>
      </div>
    </section>

    <section class="page-block">
      <div class="block-head">
        <div>
          <h3 class="block-title">局部拓扑</h3>
          <p class="block-desc">简要展示该轮内部的角色流转关系。</p>
        </div>
      </div>
      <div class="block-body">
        <div class="trace-row">
          <div class="trace-card">
            <strong>规划器</strong>
            <span>char_cost: 2.1k</span>
          </div>
          <div class="trace-link">refs: hidden-0001 / hidden-0002</div>
          <div class="trace-card active">
            <strong>检索器</strong>
            <span>char_cost: 0</span>
          </div>
        </div>
      </div>
    </section>

    <section class="page-block">
      <div class="block-head">
        <div>
          <h3 class="block-title">隐藏态结构</h3>
          <p class="block-desc">展示张量尺寸、来源模块与校验摘要。</p>
        </div>
      </div>
      <div class="block-body">
        <el-tabs type="border-card">
          <el-tab-pane label="结构信息">
            <table class="kv-table">
              <tbody>
                <tr>
                  <th>STATE_IDENTIFIER</th>
                  <td>state_ref_hidden_002a_t2</td>
                </tr>
                <tr>
                  <th>PRODUCER_AGENT</th>
                  <td>Reasoner_Module_Core</td>
                </tr>
                <tr>
                  <th>TRANSFORM_LAYER_INDEX</th>
                  <td>Layer #32</td>
                </tr>
                <tr>
                  <th>TENSOR_SHAPE_DIMENSIONS</th>
                  <td>[1, 45, 4096]</td>
                </tr>
                <tr>
                  <th>POOLED_COMPRESSION_SHAPE</th>
                  <td>[1, 4096]</td>
                </tr>
                <tr>
                  <th>SHA-256_CHECKSUM</th>
                  <td>8f2a1b9c3e5d7f1a6c4b9d8e...</td>
                </tr>
              </tbody>
            </table>
          </el-tab-pane>

          <el-tab-pane label="召回路径">
            <div class="memory-panel">
              <div class="memory-row">
                <span>MEMORY_ID: mem-0921-x7</span>
                <span>RETRIEVAL_SCORE: 0.9482</span>
              </div>
              <p class="memory-summary">
                文档锚点记录了 “Kiss and Tell” 与 Shirley Temple 之间的
                关系线索。
              </p>
              <div class="memory-path">
                [Query Node] -> [Relational Hyperedge: Starring_In] -> [Target]
              </div>
            </div>
          </el-tab-pane>
        </el-tabs>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'

const route = useRoute()
const router = useRouter()
const turnId = ref('turn-1-2')

onMounted(() => {
  if (route.query.turnId) {
    turnId.value = route.query.turnId as string
  }
})

const goBackDashboard = () => {
  router.push('/metrics')
}
</script>

<style scoped>
.latent-page {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.focus-box {
  padding: 12px;
  border: 1px solid var(--ui-border);
  background: #fafafa;
  color: var(--ui-text);
}

.trace-row {
  display: grid;
  grid-template-columns: 1fr 220px 1fr;
  gap: 12px;
  align-items: center;
}

.trace-card,
.trace-link {
  min-height: 88px;
  padding: 12px;
  border: 1px solid var(--ui-border);
  background: #fff;
}

.trace-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.trace-card.active {
  background: #fafafa;
  border-color: var(--ui-text);
}

.trace-link {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  color: var(--ui-muted);
}

.kv-table {
  width: 100%;
  border-collapse: collapse;
}

.kv-table th,
.kv-table td {
  padding: 10px 12px;
  border: 1px solid var(--ui-border);
  text-align: left;
  vertical-align: top;
}

.kv-table th {
  width: 280px;
  background: #fafafa;
  color: var(--ui-muted);
  font-weight: 400;
}

.memory-panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.memory-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  padding: 10px 12px;
  border: 1px solid var(--ui-border);
  background: #fafafa;
}

.memory-summary {
  margin: 0;
  padding: 12px;
  border: 1px solid var(--ui-border);
  background: #fff;
  color: var(--ui-muted);
}

.memory-path {
  padding: 12px;
  border: 1px solid var(--ui-border);
  background: #fff;
  color: var(--ui-text);
}

@media (max-width: 900px) {
  .trace-row {
    grid-template-columns: 1fr;
  }
}
</style>
