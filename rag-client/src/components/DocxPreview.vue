<script setup>
import {onBeforeUnmount, onMounted, ref} from 'vue'
import {renderAsync} from 'docx-preview'

const props = defineProps({
  blob: {
    type: Blob,
    required: true
  }
})

const container = ref(null)
const loading = ref(true)
const error = ref('')

onMounted(async () => {
  try {
    await renderAsync(props.blob, container.value, null, {
      className: 'docx-preview',
      inWrapper: true,
      breakPages: true,
      ignoreLastRenderedPageBreak: true,
      useBase64URL: true,
      renderAltChunks: false
    })
    sanitizeLinks()
  } catch (e) {
    console.error('Docx preview failed', e)
    error.value = 'Word 文档渲染失败'
  } finally {
    loading.value = false
  }
})

const sanitizeLinks = () => {
  container.value?.querySelectorAll('a[href]').forEach(link => {
    const href = link.getAttribute('href') || ''
    if (/^\s*(javascript|vbscript|data:text\/html)/i.test(href)) {
      link.removeAttribute('href')
      return
    }
    if (/^https?:\/\//i.test(href)) {
      link.setAttribute('target', '_blank')
      link.setAttribute('rel', 'noopener noreferrer')
    }
  })
}

onBeforeUnmount(() => {
  if (container.value) container.value.innerHTML = ''
})
</script>

<template>
  <div class="docx-preview-wrap">
    <div v-if="error" class="docx-preview-error">{{ error }}</div>
    <a-spin v-else :spinning="loading">
      <div ref="container" class="docx-preview-container"/>
    </a-spin>
  </div>
</template>

<style scoped>
.docx-preview-wrap {
  max-height: 72vh;
  overflow: auto;
}

.docx-preview-container :deep(.docx-preview-wrapper) {
  padding: 0;
  background: transparent;
}

.docx-preview-container :deep(.docx-preview-wrapper > section.docx-preview) {
  margin-bottom: 16px;
  box-shadow: 0 1px 4px rgb(0 0 0 / 12%);
}

.docx-preview-error {
  padding: 24px;
  color: #cf1322;
  text-align: center;
}

@media (max-width: 768px) {
  .docx-preview-wrap {
    height: calc(100vh - 80px);
    max-height: none;
  }
}
</style>
