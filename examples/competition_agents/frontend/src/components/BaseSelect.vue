<template>
  <label class="base-select">
    <select :value="modelValue" @change="$emit('update:modelValue', $event.target.value)">
      <option v-for="item in options" :key="item.value" :value="item.value">
        {{ item.label }}
      </option>
    </select>
    <span class="select-arrow">⌄</span>
  </label>
</template>

<script setup>
defineProps({
  modelValue: {
    type: [String, Number],
    default: ''
  },
  options: {
    type: Array,
    required: true
  }
})

defineEmits(['update:modelValue'])
</script>
