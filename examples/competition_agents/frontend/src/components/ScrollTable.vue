<template>
  <div class="scroll-table-wrap" :class="variant">
    <table>
      <slot></slot>
    </table>
  </div>
</template>

<script setup>
defineProps({
  variant: {
    type: String,
    default: ''
  }
})
</script>
