<template>
  <article class="kpi-card" :class="tone">
    <span class="kpi-icon">{{ icon }}</span>
    <div class="kpi-main">
      <h3>{{ title }} <small>ⓘ</small></h3>
      <strong>{{ value }}</strong>
      <p>{{ desc }}</p>
    </div>
  </article>
</template>

<script setup>
defineProps({
  title: String,
  value: String,
  desc: String,
  tone: String,
  icon: String
})
</script>
