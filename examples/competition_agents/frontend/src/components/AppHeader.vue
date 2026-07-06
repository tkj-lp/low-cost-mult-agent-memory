<template>
  <header class="app-header">
    <div class="brand">
      <svg class="brand-logo" viewBox="0 0 42 42" aria-hidden="true">
        <circle cx="21" cy="8" r="5" />
        <circle cx="8" cy="31" r="5" />
        <circle cx="34" cy="31" r="5" />
        <path d="M18.7 12.7 10.5 26.4M23.3 12.7l8.2 13.7M13.2 31h15.6" />
      </svg>
      <span class="brand-title">多智能体低开销通信实验系统</span>
    </div>

    <nav class="header-tabs" aria-label="主导航">
      <button
        v-for="tab in tabs"
        :key="tab"
        class="header-tab"
        :class="{ active: tab === activeTab }"
        type="button"
        @click="$emit('change-tab', tab)"
      >
        {{ tab }}
      </button>
    </nav>

    <div v-if="activeTab === '实验控制台'" class="header-actions">
      <button class="help-button" type="button">
        <svg viewBox="0 0 20 20" aria-hidden="true">
          <circle cx="10" cy="10" r="8" />
          <path d="M8.1 7.8a2 2 0 1 1 3.2 1.6c-.9.6-1.3 1.1-1.3 2.1" />
          <path d="M10 14.2h.01" />
        </svg>
        <span>使用说明</span>
      </button>
      <div class="user-menu">
        <span class="avatar">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="12" cy="8" r="4" />
            <path d="M4.8 20c1.5-4 4-6 7.2-6s5.7 2 7.2 6" />
          </svg>
        </span>
        <span class="user-name">admin</span>
        <span class="chevron">⌄</span>
      </div>
    </div>
    <div v-else-if="activeTab === '协作机制'" class="header-actions page-actions">
      <button class="top-action" type="button">⇧ 导入结果</button>
      <button class="top-action" type="button">▦ 实验记录</button>
      <button class="icon-action" type="button" aria-label="设置">⚙</button>
    </div>
    <div v-else-if="activeTab === '记忆复用'" class="header-actions page-actions">
      <button class="top-action" type="button">⟳ 刷新记忆</button>
      <button class="top-action" type="button">▦ 实验记录</button>
    </div>
    <div v-else class="header-actions page-actions">
      <button class="top-action" type="button" @click="$emit('open-records')">▦ 实验记录</button>
      <button class="top-action" type="button">⇩ 导出报告⌄</button>
    </div>
  </header>
</template>

<script setup>
defineProps({
  activeTab: {
    type: String,
    required: true
  }
})

defineEmits(['change-tab', 'open-records'])

const tabs = ['实验控制台', '协作机制', '记忆复用', '指标评估']
</script>
