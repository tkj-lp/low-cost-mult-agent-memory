<template>
  <span class="stat-badge" :class="tone">
    <slot></slot>
  </span>
</template>

<script setup>
defineProps({
  tone: {
    type: String,
    default: 'blue'
  }
})
</script>
