<template>
  <section class="base-card" :class="$attrs.class">
    <header class="card-header">
      <span class="card-icon">
        <slot name="icon"></slot>
      </span>
      <h2 class="card-title">{{ title }}</h2>
      <div class="card-action" v-if="$slots.action">
        <slot name="action"></slot>
      </div>
    </header>
    <div class="card-body">
      <slot></slot>
    </div>
  </section>
</template>

<script setup>
defineProps({
  title: {
    type: String,
    required: true
  }
})
</script>
