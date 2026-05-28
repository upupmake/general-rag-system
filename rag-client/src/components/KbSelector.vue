<script setup>
import {kbs, selectedKb, kbGroupLabels} from "@/vars.js"

defineProps({
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
        :style="{ width }"
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
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.kb-select {
  min-width: 120px;
  flex: 1;
}
</style>
