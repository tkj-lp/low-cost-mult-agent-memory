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

    <div class="header-actions page-actions">
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
