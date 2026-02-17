<script setup>
import {kbs, selectedKb, kbGroupLabels, ragMode} from "@/vars.js"

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
        placeholder="选择知识库（可选）"
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

    <!-- RAG模式切换按钮 -->
    <a-segmented
        v-if="selectedKb"
        v-model:value="ragMode"
        :size="size"
        :options="[
          { label: 'Fast', value: 'fast' },
          { label: 'Agentic', value: 'agentic' }
        ]"
        class="rag-mode-toggle"
    />
  </div>
</template>

<style scoped>
.kb-selector-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  flex-wrap: wrap;
}

.kb-select {
  min-width: 120px;
  flex: 1;
}

.rag-mode-toggle {
  flex-shrink: 0;
}

/* 增强 Segmented 样式 - 使用渐变和阴影效果 */
.rag-mode-toggle :deep(.ant-segmented) {
  display: flex;
  background: linear-gradient(145deg, #f0f2f5, #e8eaed);
  box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.08);
  border: 1px solid rgba(0, 0, 0, 0.06);
}

.rag-mode-toggle :deep(.ant-segmented-item) {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  color: rgba(0, 0, 0, 0.65);
  font-weight: 500;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.rag-mode-toggle :deep(.ant-segmented-item:hover:not(.ant-segmented-item-selected)) {
  color: #1890ff;
}

.rag-mode-toggle :deep(.ant-segmented-item-selected) {
  background: linear-gradient(135deg, #40a9ff 0%, #1890ff 100%);
  color: #fff;
  font-weight: 600;
  box-shadow: 0 2px 8px rgba(24, 144, 255, 0.3);
}

.rag-mode-toggle :deep(.ant-segmented-thumb) {
  background: linear-gradient(135deg, #40a9ff 0%, #1890ff 100%);
  box-shadow: 0 2px 8px rgba(24, 144, 255, 0.3);
}

/* 移动端适配 */
@media (max-width: 768px) {
  .kb-selector-wrapper {
    flex-direction: column;
    align-items: stretch;
    gap: 8px;
    width: 100%;
  }
  
  .kb-select {
    width: 100% !important;
  }
  
  .rag-mode-toggle {
    width: 100%;
    align-self: center;
  }
  
  .rag-mode-toggle :deep(.ant-segmented) {
    display: flex;
    width: 100%;
  }
  
  .rag-mode-toggle :deep(.ant-segmented-item) {
    flex: 1;
    justify-content: center;
    text-align: center;
  }
}

@media (max-width: 480px) {
  .rag-mode-toggle {
    max-width: 280px;
  }
  
  .rag-mode-toggle :deep(.ant-segmented-item) {
    font-size: 12px;
    padding: 4px 8px;
  }
}
</style>
