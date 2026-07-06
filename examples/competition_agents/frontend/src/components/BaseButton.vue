<template>
  <button class="base-button" :class="[`button-${variant}`, { 'button-compact': compact }]" type="button">
    <span v-if="$slots.icon" class="button-icon">
      <slot name="icon"></slot>
    </span>
    <span class="button-label">
      <slot></slot>
    </span>
  </button>
</template>

<script setup>
defineProps({
  variant: {
    type: String,
    default: 'outline'
  },
  compact: {
    type: Boolean,
    default: false
  }
})
</script>
