<template>
  <div class="progress-track" :class="tone">
    <span :style="{ width: `${value}%` }"></span>
  </div>
</template>

<script setup>
defineProps({
  value: {
    type: Number,
    default: 0
  },
  tone: {
    type: String,
    default: 'blue'
  }
})
</script>
