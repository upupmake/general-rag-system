<script setup>
import {ref, onMounted, computed, h, watch, onUnmounted} from 'vue'
import {useRouter, useRoute} from 'vue-router'
import {message, Button, Typography, Modal, Input} from 'ant-design-vue'
import {Conversations} from 'ant-design-x-vue'
import {
  CommentOutlined,
  ClockCircleOutlined,
  DeleteOutlined,
  DownloadOutlined,
  EditOutlined,
  SearchOutlined
} from '@ant-design/icons-vue'
import {deleteSession, fetchSessions, fetchSessionMessages, renameSession} from '@/api/chatApi'
import {events} from '@/events.js';

const router = useRouter()
const route = useRoute()

const loading = ref(false)
const groups = ref([])
const cursor = ref(null)
const hasMore = ref(true)
const activeKey = ref(route.params.sessionId)
const renameState = ref({visible: false, sessionId: null, title: ''})
const renameLoading = ref(false)
const searchQuery = ref('')
let searchTimer = null
let searchRequestId = 0

const loadData = async ({reset = false, keyword = searchQuery.value} = {}) => {
  if ((loading.value && !reset) || (!hasMore.value && !reset)) return
  if (reset) {
    cursor.value = null
    hasMore.value = true
  }
  loading.value = true
  const requestId = ++searchRequestId

  try {
    const res = await fetchSessions({
      lastActiveAt: reset ? undefined : cursor.value?.lastActiveAt,
      lastId: reset ? undefined : cursor.value?.lastId,
      pageSize: 20,
      keyword: keyword.trim() || undefined
    })

    if (requestId !== searchRequestId) return
    if (reset) {
      groups.value = []
    }
    groups.value.push(...(res.groups || []))
    cursor.value = res.nextCursor
    hasMore.value = res.hasMore
  } catch (error) {
    if (requestId === searchRequestId) {
      message.error('加载历史会话失败，请重试')
    }
  } finally {
    if (requestId === searchRequestId) {
      loading.value = false
    }
  }
}

const handleSearch = value => {
  searchQuery.value = value
  clearTimeout(searchTimer)
  searchTimer = setTimeout(() => {
    loadData({reset: true, keyword: value})
  }, 300)
}

const conversationItems = computed(() =>
    groups.value.flatMap(g =>
        g.items.map(i => ({
          key: i.id.toString(),
          label: i.title,
          timestamp: i.timestamp,
          group: g.group,
          icon: h(CommentOutlined),
        }))
    )
)

const exportSession = async (sessionId) => {
  try {
    const messages = await fetchSessionMessages(sessionId)
    if (!messages || messages.length === 0) {
      message.warning('该会话没有消息可导出')
      return
    }

    // 格式化为 Markdown
    let mdContent = `# 会话 ${sessionId}\n\n`
    messages.forEach(msg => {
      const role = msg.role === 'user' ? '用户' : 'AI'
      const content = msg.content || ''
      mdContent += `## ${role}\n\n`
      // 思考内容（引用格式，支持多行）
      if (msg.thinking) {
        mdContent += `> 思考内容：\n`
        const lines = msg.thinking.split('\n')
        for (const line of lines) {
          mdContent += `> ${line}\n`
        }
        mdContent += '\n'
      }
      mdContent += `${content}\n\n---\n\n`
    })

    // 创建下载链接
    const blob = new Blob([mdContent], {type: 'text/markdown;charset=utf-8'})
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${sessionId}.md`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    message.success('导出成功')
  } catch (e) {
    console.error('导出失败:', e)
    message.error('导出失败，请重试')
  }
}

const menu = (conversation) => ({
  items: [
    {
      label: '导出 Markdown',
      key: 'export',
      icon: h(DownloadOutlined),
    },
    {
      label: '重命名',
      key: 'rename',
      icon: h(EditOutlined),
    },
    {
      type: 'divider',
    },
    {
      label: '删除会话',
      key: 'delete',
      icon: h(DeleteOutlined),
      danger: true,
    },
  ],
  onClick: (menuInfo) => {
    if (menuInfo.key === 'delete') {
      deleteSession(conversation.key).then(() => {
        const group = groups.value.find(g => g.group === conversation.group)
        if (group) {
          group.items = group.items.filter(i => i.id.toString() !== conversation.key)
          // 如果该组没有更多会话，移除该组
          if (group.items.length === 0) {
            groups.value = groups.value.filter(g => g.group !== conversation.group)
          }
        }
        if (activeKey.value === conversation.key) {
          router.push('/chat/new')
        }
      })
    } else if (menuInfo.key === 'export') {
      exportSession(conversation.key)
    } else if (menuInfo.key === 'rename') {
      renameState.value = {visible: true, sessionId: conversation.key, title: conversation.label}
    }
  },
});

const confirmRename = async () => {
  const newTitle = renameState.value.title.trim()
  if (!newTitle) {
    message.warning('会话标题不能为空')
    return
  }
  renameLoading.value = true
  try {
    await renameSession(renameState.value.sessionId, newTitle)
    for (const g of groups.value) {
      const item = g.items.find(i => String(i.id) === String(renameState.value.sessionId))
      if (item) {
        item.title = newTitle
        break
      }
    }
    events.emit('sessionTitleUpdated', {sessionId: renameState.value.sessionId, title: newTitle})
    message.success('重命名成功')
    renameState.value.visible = false
  } catch (e) {
    message.error('重命名失败，请重试')
  } finally {
    renameLoading.value = false
  }
}

const cancelRename = () => {
  renameState.value.visible = false
}

const groupable = {
  sort(a, b) {
    const order = {TODAY: 0, YESTERDAY: 1, EARLIER: 2}
    return (order[a] ?? 99) - (order[b] ?? 99)
  },
  title: (group, {components: {GroupTitle}}) => {
    const map = {
      TODAY: '今天',
      YESTERDAY: '昨天',
      EARLIER: '更早'
    }
    return h(GroupTitle, null, () => map[group] || group)
  }
}

const handleActiveChange = (key) => {
  const isOnChatSession = route.path.startsWith('/chat/') && !route.path.startsWith('/chat/new')
  if (isOnChatSession && String(route.params.sessionId) !== String(key)) {
    window.open(`/chat/${key}`, '_blank')
  } else {
    router.push(`/chat/${key}`)
  }
}

watch(
    () => route.params.sessionId,
    (val) => {
      activeKey.value = val
    }
)

// 当活跃会话或列表数据变化时，通知 MainLayout 更新页面标题
watch(
    [activeKey, conversationItems],
    () => {
      if (!activeKey.value) return
      const item = conversationItems.value.find(i => i.key === String(activeKey.value))
      if (item) {
        events.emit('sessionTitleUpdated', {
          sessionId: activeKey.value,
          title: item.label || '对话'
        })
      }
    },
    {immediate: true}
)

onMounted(() => {
  loadData()
  events.on('sessionListRefresh', () => {
    loadData({reset: true})
  })
  events.on('sessionTitleUpdated', ({sessionId, title}) => {
    for (const g of groups.value) {
      const item = g.items.find(i => String(i.id) === String(sessionId))
      if (item) {
        item.title = title
        break
      }
    }
  })
})
onUnmounted(() => {
  clearTimeout(searchTimer)
  events.off('sessionListRefresh')
  events.off('sessionTitleUpdated')
})
</script>

<template>
  <div class="session-list" style='display: flex; flex-direction: column; height: 100%;'>
    <div class="session-search">
      <Input
          :value="searchQuery"
          class="session-search-input"
          allow-clear
          :bordered="false"
          placeholder="搜索聊天"
          aria-label="搜索全部聊天"
          @update:value="handleSearch"
      >
        <template #prefix>
          <SearchOutlined class="session-search-icon" />
        </template>
      </Input>
    </div>
    <div style='flex: 1; overflow-y: auto;'>
      <Conversations
          :items='conversationItems'
          :activeKey='activeKey'
          :menu='menu'
          :groupable='groupable'
          :onActiveChange='handleActiveChange'
      />
      <Button
          block
          :loading='loading'
          @click='loadData'>
        <ClockCircleOutlined/>
        查看更多历史
      </Button>
    </div>
    <Modal
        :open='renameState.visible'
        title='重命名会话'
        :confirm-loading='renameLoading'
        ok-text='保存'
        cancel-text='取消'
        @ok='confirmRename'
        @cancel='cancelRename'
    >
      <Input
          v-model:value='renameState.title'
          placeholder='请输入新的会话标题'
          :maxlength='100'
          allow-clear
          @press-enter='confirmRename'
      />
    </Modal>
  </div>
</template>

<style scoped>
.session-search {
  padding: 3px 6px 9px;
}

.session-search-input {
  height: 32px;
  padding-inline: 10px;
  background: rgba(0, 0, 0, 0.035);
  border: 1px solid transparent;
  border-radius: 9px;
  box-shadow: none;
  transition: background-color 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}

.session-search-input:hover {
  background: rgba(0, 0, 0, 0.055);
}

.session-search-input:focus-within {
  background: #fff;
  border-color: rgba(22, 119, 255, 0.45);
  box-shadow: 0 0 0 2px rgba(22, 119, 255, 0.08);
}

.session-search-icon {
  color: rgba(0, 0, 0, 0.35);
  font-size: 13px;
  transition: color 0.2s ease;
}

.session-search-input:focus-within .session-search-icon {
  color: #1677ff;
}

.session-search-input :deep(.ant-input) {
  color: rgba(0, 0, 0, 0.78);
  background: transparent;
  font-size: 13px;
}

.session-search-input :deep(.ant-input::placeholder) {
  color: rgba(0, 0, 0, 0.38);
}

:global(body[data-theme='dark']) .session-search-input {
  background: rgba(255, 255, 255, 0.06);
}

:global(body[data-theme='dark']) .session-search-input:hover {
  background: rgba(255, 255, 255, 0.09);
}

:global(body[data-theme='dark']) .session-search-input:focus-within {
  background: rgba(255, 255, 255, 0.08);
  border-color: rgba(64, 169, 255, 0.55);
  box-shadow: 0 0 0 2px rgba(64, 169, 255, 0.1);
}

:global(body[data-theme='dark']) .session-search-icon {
  color: rgba(255, 255, 255, 0.42);
}

:global(body[data-theme='dark']) .session-search-input:focus-within .session-search-icon {
  color: #40a9ff;
}

:global(body[data-theme='dark']) .session-search-input :deep(.ant-input) {
  color: rgba(255, 255, 255, 0.85);
}

:global(body[data-theme='dark']) .session-search-input :deep(.ant-input::placeholder) {
  color: rgba(255, 255, 255, 0.38);
}

:deep(.ant-conversations) {
  .ant-conversations-group-title {
    padding-inline-start: 0 !important;
  }

  .ant-conversations-item {
    padding-inline-start: 5px !important;
  }
}

@media (hover: none), (pointer: coarse) {
  :deep(.ant-conversations .ant-conversations-menu-icon) {
    opacity: 1;
  }
}
</style>
