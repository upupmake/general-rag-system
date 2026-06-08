<script setup>
import {computed} from "vue"
import {kbs, selectedKb, kbGroupLabels, findKbById} from "@/vars.js"

const props = defineProps({
  width: {
    type: String,
    default: '280px'
  },
  size: {
    type: String,
    default: 'middle'
  },
  disabled: {
    type: Boolean,
    default: false
  }
})

const currentKb = computed(() => findKbById(selectedKb.value))

const adaptiveWidth = computed(() => {
  const kbName = currentKb.value?.name || ''
  if (!kbName || !props.width.endsWith('px')) return props.width

  const baseWidth = Number.parseFloat(props.width)
  const textWidth = [...kbName].reduce((width, char) => width + (/[^\x00-\xff]/.test(char) ? 14 : 8), 0)
  return `${Math.max(baseWidth, textWidth + 64)}px`
})

// 知识库搜索过滤
const filterKbOption = (input, option) => {
  if (option.value?.toString().startsWith('empty-')) return false
  const kb = [...kbs.value.private, ...kbs.value.shared, ...kbs.value.public, ...kbs.value.invited]
      .find(k => k.id === option.value)
  return kb?.name?.toLowerCase().includes(input.toLowerCase())
}
</script>

<template>
  <div class="kb-selector-wrapper">
    <a-select
        v-model:value="selectedKb"
        :style="{ width: adaptiveWidth, maxWidth: '100%' }"
        :size="size"
        :disabled="disabled"
        placeholder="知识库（可选）"
        allowClear
        show-search
        :filter-option="filterKbOption"
        class="kb-select">
      <a-select-opt-group
          v-for="(list, group) in kbs"
          :key="group"
          :label="kbGroupLabels[group]">
        <a-select-option
            v-for="kb in list"
            :key="kb.id"
            :value="kb.id">
          {{ kb.name }}
        </a-select-option>
        <a-select-option v-if="list.length === 0" disabled :value="'empty-' + group">
          暂无知识库
        </a-select-option>
      </a-select-opt-group>
    </a-select>
  </div>
</template>

<style scoped>
.kb-selector-wrapper {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  max-width: 100%;
}

.kb-select {
  min-width: 120px;
  flex: 0 1 auto;
}
</style>
