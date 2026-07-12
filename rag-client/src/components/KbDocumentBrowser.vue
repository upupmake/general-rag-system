<script setup>
import {computed, onBeforeUnmount, ref, watch} from 'vue'
import {message} from 'ant-design-vue'
import {
  DeleteOutlined,
  FileOutlined,
  FolderOpenOutlined,
  FolderOutlined,
  LoadingOutlined,
  ReloadOutlined,
} from '@ant-design/icons-vue'
import {deleteDocument, listDocuments, previewDocument} from '@/api/kbApi.js'
import {findKbById} from '@/vars.js'
import {useUserStore} from '@/stores/user.js'
import md from '@/utils/markdown.js'

const props = defineProps({
  visible: Boolean,
  kbId: [Number, String],
})

const emit = defineEmits(['update:visible'])
const userStore = useUserStore()
const documents = ref([])
const currentPath = ref([])
const loading = ref(false)
const deleting = ref(false)
const previewLoading = ref(false)
const previewVisible = ref(false)
const previewTitle = ref('')
const previewType = ref('')
const previewContent = ref('')
const isMobile = ref(window.innerWidth <= 768)
let pollTimer = null

const currentKb = computed(() => findKbById(props.kbId))
const currentPathString = computed(() => currentPath.value.join('/'))
const isOwner = computed(() => currentKb.value?.ownerUserId === userStore.userId)

const canDeleteFile = (file) => isOwner.value || file.uploaderId === userStore.userId

const displayList = computed(() => {
  const list = []
  const folders = new Set()
  const pathPrefix = currentPathString.value ? `${currentPathString.value}/` : ''
  const sorted = [...documents.value].sort((a, b) => (a.fileName || '').localeCompare(b.fileName || ''))

  sorted.forEach(file => {
    const fullName = (file.fileName || '').replace(/\\/g, '/')
    if (!fullName.startsWith(pathPrefix)) return
    const relativeName = fullName.slice(pathPrefix.length)
    if (!relativeName) return

    const parts = relativeName.split('/')
    if (parts.length === 1) {
      list.push({...file, fileName: parts[0], isFolder: false})
      return
    }

    const folderName = parts[0]
    if (folders.has(folderName)) return
    folders.add(folderName)

    const folderPrefix = `${pathPrefix}${folderName}/`
    const files = documents.value.filter(item => (item.fileName || '').replace(/\\/g, '/').startsWith(folderPrefix))
    const hasFailed = files.some(item => item.status === 'failed')
    const hasProcessing = files.some(item => item.status === 'processing')

    list.push({
      id: `folder-${folderPrefix}`,
      fileName: folderName,
      isFolder: true,
      fileSize: files.reduce((total, item) => total + (item.fileSize || 0), 0),
      status: hasFailed ? 'failed' : (hasProcessing ? 'processing' : 'ready'),
      files,
      canDelete: files.length > 0 && files.every(canDeleteFile),
    })
  })

  return list.sort((a, b) => {
    if (a.isFolder !== b.isFolder) return a.isFolder ? -1 : 1
    return a.fileName.localeCompare(b.fileName)
  })
})

const stopPolling = () => {
  if (pollTimer) {
    clearTimeout(pollTimer)
    pollTimer = null
  }
}

const schedulePolling = () => {
  stopPolling()
  if (!props.visible || !documents.value.some(item => item.status === 'processing')) return
  pollTimer = setTimeout(fetchDocuments, 3000)
}

const fetchDocuments = async () => {
  if (!props.visible || !props.kbId) return
  loading.value = true
  try {
    documents.value = await listDocuments(props.kbId)
  } finally {
    loading.value = false
    schedulePolling()
  }
}

const close = () => emit('update:visible', false)
const enterFolder = (name) => currentPath.value.push(name)
const navigateTo = (index) => {
  currentPath.value = index < 0 ? [] : currentPath.value.slice(0, index + 1)
}

const formatSize = (size = 0) => {
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  return `${(size / 1024 / 1024).toFixed(2)} MB`
}

const clearPreview = () => {
  if (['pdf', 'image'].includes(previewType.value) && previewContent.value) {
    URL.revokeObjectURL(previewContent.value)
  }
  previewContent.value = ''
  previewType.value = ''
}

const previewFile = async (file) => {
  clearPreview()
  previewLoading.value = true
  previewTitle.value = file.fileName || '文件预览'
  try {
    const blob = await previewDocument(props.kbId, file.id)
    const fileName = (file.fileName || '').toLowerCase()
    if (fileName.endsWith('.pdf')) {
      previewType.value = 'pdf'
      previewContent.value = URL.createObjectURL(blob)
    } else if (/\.(jpeg|jpg|png|gif|bmp|webp)$/.test(fileName)) {
      previewType.value = 'image'
      previewContent.value = URL.createObjectURL(blob)
    } else if (fileName.endsWith('.md')) {
      previewType.value = 'markdown'
      previewContent.value = await blob.text()
    } else {
      previewType.value = 'text'
      previewContent.value = await blob.text()
    }
    previewVisible.value = true
  } finally {
    previewLoading.value = false
  }
}

const closePreview = () => {
  previewVisible.value = false
  clearPreview()
}

const removeFile = async (file) => {
  deleting.value = true
  try {
    await deleteDocument(props.kbId, file.id)
    message.success('文件删除成功')
    await fetchDocuments()
  } finally {
    deleting.value = false
  }
}

const removeFolder = async (folder) => {
  deleting.value = true
  const hide = message.loading(`正在删除 ${folder.files.length} 个文件...`, 0)
  try {
    for (const file of folder.files) {
      await deleteDocument(props.kbId, file.id)
    }
    message.success('文件夹删除成功')
  } finally {
    hide()
    deleting.value = false
    await fetchDocuments()
  }
}

const handleResize = () => {
  isMobile.value = window.innerWidth <= 768
}

watch(() => props.visible, visible => {
  if (visible) {
    window.addEventListener('resize', handleResize)
    currentPath.value = []
    fetchDocuments()
  } else {
    window.removeEventListener('resize', handleResize)
    stopPolling()
  }
})

watch(() => props.kbId, () => {
  currentPath.value = []
  if (props.visible) fetchDocuments()
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize)
  stopPolling()
  clearPreview()
})
</script>

<template>
  <a-drawer
      :visible="visible"
      :width="isMobile ? '100%' : 620"
      title="知识库文件"
      placement="right"
      class="kb-document-browser"
      @close="close">
    <template #extra>
      <a-button type="text" :loading="loading" title="刷新" @click="fetchDocuments">
        <ReloadOutlined/>
      </a-button>
    </template>

    <div class="browser-header">
      <div class="kb-name">{{ currentKb?.name || '未选择知识库' }}</div>
      <a-breadcrumb>
        <a-breadcrumb-item><a @click="navigateTo(-1)">根目录</a></a-breadcrumb-item>
        <a-breadcrumb-item v-for="(folder, index) in currentPath" :key="`${folder}-${index}`">
          <a @click="navigateTo(index)">{{ folder }}</a>
        </a-breadcrumb-item>
      </a-breadcrumb>
    </div>

    <a-spin :spinning="loading">
      <a-empty v-if="!loading && displayList.length === 0" description="当前目录暂无文件"/>
      <div v-else class="document-list">
        <div v-for="item in displayList" :key="item.id" class="document-row">
          <button v-if="item.isFolder" type="button" class="document-main folder-button" @click="enterFolder(item.fileName)">
            <FolderOutlined class="document-icon folder-icon"/>
            <span class="document-info">
              <span class="document-name">{{ item.fileName }}</span>
              <span class="document-meta">{{ item.files.length }} 个文件 · {{ formatSize(item.fileSize) }}</span>
            </span>
            <FolderOpenOutlined class="open-icon"/>
          </button>
          <button v-else type="button" class="document-main file-button" :disabled="previewLoading" @click="previewFile(item)">
            <FileOutlined class="document-icon"/>
            <span class="document-info">
              <span class="document-name" :title="item.fileName">{{ item.fileName }}</span>
              <span class="document-meta">{{ formatSize(item.fileSize) }} · 点击预览</span>
            </span>
          </button>

          <div class="document-actions">
            <a-tag v-if="item.status === 'processing'" color="blue"><LoadingOutlined/> 处理中</a-tag>
            <a-tag v-else-if="item.status === 'ready'" color="green">完成</a-tag>
            <a-tag v-else-if="item.status === 'failed'" color="red">失败</a-tag>
            <a-tag v-else>{{ item.status }}</a-tag>

            <a-popconfirm
                v-if="item.isFolder ? item.canDelete : canDeleteFile(item)"
                :title="item.isFolder ? '确定删除该目录及其全部文件吗？' : '确定删除该文件吗？'"
                ok-text="删除"
                cancel-text="取消"
                @confirm="item.isFolder ? removeFolder(item) : removeFile(item)">
              <a-button type="text" danger size="small" :disabled="deleting" title="删除">
                <DeleteOutlined/>
              </a-button>
            </a-popconfirm>
          </div>
        </div>
      </div>
    </a-spin>

    <a-modal
        :visible="previewVisible"
        :title="previewTitle"
        :width="isMobile ? '100%' : 860"
        :footer="null"
        :style="isMobile ? { top: 0, margin: 0, maxWidth: '100%' } : { top: '6vh' }"
        :body-style="isMobile ? { padding: '10px', height: 'calc(100vh - 55px)' } : {}"
        @cancel="closePreview">
      <iframe v-if="previewType === 'pdf'" :src="previewContent" class="preview-pdf" title="PDF 预览"/>
      <div v-else-if="previewType === 'markdown'" class="markdown-body preview-text" v-html="md.render(previewContent)"/>
      <div v-else-if="previewType === 'image'" class="preview-image-wrap">
        <img :src="previewContent" class="preview-image" alt="文件预览"/>
      </div>
      <pre v-else class="preview-text">{{ previewContent }}</pre>
    </a-modal>
  </a-drawer>
</template>

<style scoped>
.browser-header {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 16px;
}

.kb-name {
  color: #1f1f1f;
  font-size: 15px;
  font-weight: 600;
}

.document-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.document-row {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
  padding: 10px 12px;
  border: 1px solid #f0f0f0;
  border-radius: 8px;
}

.document-main {
  display: flex;
  flex: 1;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.folder-button,
.file-button {
  padding: 0;
  color: inherit;
  text-align: left;
  background: none;
  border: 0;
  cursor: pointer;
}

.file-button:hover .document-name {
  color: #1890ff;
}

.file-button:disabled {
  cursor: wait;
}

.document-icon {
  flex-shrink: 0;
  color: #8c8c8c;
  font-size: 18px;
}

.folder-icon,
.open-icon {
  color: #1890ff;
}

.open-icon {
  flex-shrink: 0;
}

.document-info {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-width: 0;
}

.document-name {
  overflow: hidden;
  font-weight: 500;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.document-meta {
  color: #8c8c8c;
  font-size: 12px;
}

.document-actions {
  display: flex;
  flex-shrink: 0;
  align-items: center;
  gap: 4px;
}

.preview-pdf {
  width: 100%;
  height: 72vh;
  border: 0;
}

.preview-image-wrap {
  text-align: center;
}

.preview-image {
  max-width: 100%;
  max-height: 72vh;
}

.preview-text {
  max-height: 72vh;
  margin: 0;
  overflow: auto;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

:global(.is-dark) .kb-name {
  color: #e8e8e8;
}

:global(.is-dark) .document-row {
  border-color: #303030;
}

@media (max-width: 768px) {
  .document-row {
    align-items: flex-start;
    gap: 8px;
    padding: 10px;
  }

  .document-actions {
    flex-direction: column;
    align-items: flex-end;
  }

  .document-actions :deep(.ant-tag) {
    margin-inline-end: 0;
  }

  .preview-pdf,
  .preview-text {
    height: calc(100vh - 80px);
    max-height: none;
  }

  .preview-image {
    max-height: calc(100vh - 80px);
  }
}
</style>
