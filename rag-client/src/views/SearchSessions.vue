<script setup>
import {ref, onMounted, nextTick} from 'vue'
import {useRouter, onBeforeRouteLeave} from 'vue-router'
import {message, Button, List} from 'ant-design-vue'
import {DeleteOutlined} from '@ant-design/icons-vue'
import {deleteSession, searchSessions} from '@/api/chatApi'
import {useSearchStore} from '@/stores/search'
import {storeToRefs} from 'pinia'

const router = useRouter()
const searchStore = useSearchStore()
const { keyword, results, offset, hasMore, scrollPosition } = storeToRefs(searchStore)

// Search state
const searchLoading = ref(false)

// Search logic
const handleSearch = async () => {
  const kw = keyword.value.trim()
  if (!kw) {
    searchStore.clearState()
    return
  }
  
  searchLoading.value = true
  searchStore.setOffset(0)
  searchStore.setResults([])
  searchStore.setHasMore(true)
  
  await loadMoreSearchResults()
}

const loadMoreSearchResults = async () => {
  if (!hasMore.value) return
  
  searchLoading.value = true
  try {
    const page = await searchSessions({
      keyword: keyword.value,
      limit: 20,
      offset: offset.value
    })
    const items = page?.items || []
    if (items.length > 0) searchStore.appendResults(items)
    searchStore.setOffset(page?.nextOffset ?? (offset.value + items.length))
    searchStore.setHasMore(!!page?.hasMore)
  } catch (e) {
    message.error('搜索失败')
  } finally {
    searchLoading.value = false
  }
}

const handleJump = (sessionId) => {
  router.push(`/chat/${sessionId}`)
}

const handleDelete = async (sessionId) => {
   try {
     await deleteSession(sessionId)
     searchStore.removeItem(sessionId)
     message.success('删除成功')
   } catch (e) {
     message.error('删除失败')
   }
}

const highlightText = (text) => {
  const escapeHtml = (value) => value.replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[char])
  const safeText = escapeHtml(text || '')
  if (!keyword.value || !safeText) return safeText
  const kw = keyword.value.trim()
  if (!kw) return safeText

  const escapedKw = kw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const regex = new RegExp(`(${escapedKw})`, 'gi');
  return safeText.replace(regex, '<span class="highlight-text">$1</span>');
}

// Restore scroll position
onMounted(() => {
  if (results.value.length > 0) {
     nextTick(() => {
        window.scrollTo(0, scrollPosition.value)
     })
  }
})

// Save scroll position
onBeforeRouteLeave((to, from, next) => {
  searchStore.setScrollPosition(window.scrollY)
  next()
})
</script>

<template>
  <div class="search-page-container">
    <div class="search-header">
        <h2>搜索历史对话</h2>
        <a-input-search
            v-model:value="keyword"
            placeholder="输入关键词搜索..."
            allowClear
            enter-button="搜索"
            size="large"
            @search="handleSearch"
        />
    </div>

    <div class="search-results-list">
         <List
            item-layout="horizontal"
            :data-source="results"
            :loading="searchLoading"
            :locale="{ emptyText: '暂无搜索结果' }"
         >
            <template #renderItem="{ item }">
              <List.Item>
                 <template #actions>
                    <a-popconfirm
                    title="确定要删除这个会话吗？"
                    ok-text="确定"
                    cancel-text="取消"
                    @confirm="handleDelete(item.sessionId)"
                  >
                    <Button type="text" danger size="small" class="delete-btn">
                      <DeleteOutlined /> <span class="delete-text">删除</span>
                    </Button>
                  </a-popconfirm>
                 </template>
                 <List.Item.Meta>
                    <template #title>
                       <a @click="handleJump(item.sessionId)" class="session-link">{{ item.sessionTitle || `会话 ${item.sessionId}` }}</a>
                    </template>
                    <template #description>
                       <div class="search-content-snippet">
                          <div v-for="(snippet, idx) in item.contentList" :key="idx" class="snippet-item">
                             <div class="snippet-header">
                                <span v-if="snippet.matchType === 'title'" class="snippet-match-title">标题匹配</span>
                                <span class="snippet-time">{{ snippet.createdAt }}</span>
                             </div>
                             <div class="snippet-content" v-html="highlightText(snippet.content)"></div>
                          </div>
                       </div>
                    </template>
                 </List.Item.Meta>
              </List.Item>
            </template>
            
            <template #loadMore>
               <div v-if="hasMore && results.length > 0 && !searchLoading" style="text-align: center; margin-top: 12px; margin-bottom: 20px;">
                 <Button @click="loadMoreSearchResults">加载更多</Button>
               </div>
            </template>
         </List>
    </div>
  </div>
</template>

<style scoped>
.search-page-container {
  padding: 24px;
  max-width: 800px;
  margin: 0 auto;
}

.search-header {
    margin-bottom: 24px;
    text-align: center;
}

.search-header h2 {
    margin-bottom: 20px;
}

.search-results-list {
  background: #fff;
  border-radius: 8px;
  padding: 0 16px;
}

.session-link {
    font-size: 16px;
    font-weight: 500;
}

.session-time {
    margin-left: 10px;
    font-size: 12px;
    color: #999;
}

.search-content-snippet {
  margin-top: 8px;
}

.snippet-item {
  background: #f9f9f9;
  padding: 8px;
  border-radius: 4px;
  margin-bottom: 8px;
  font-size: 14px;
  color: #555;
  border-left: 3px solid #1890ff;
}

.snippet-header {
  margin-bottom: 4px;
  font-size: 12px;
  color: #999;
}

.snippet-match-title {
  display: inline-block;
  margin-right: 8px;
  padding: 0 6px;
  color: #0958d9;
  background: #e6f4ff;
  border-radius: 4px;
}

.snippet-content {
  word-break: break-word;
}

:deep(.highlight-text) {
  background-color: #ffd591; /* orange-3 for a soft highlight */
  color: #d4380d; /* dust-red-7 for text contrast */
  font-weight: bold;
  border-radius: 2px;
  padding: 0 2px;
}

@media (max-width: 768px) {
  .search-page-container {
    padding: 12px;
  }
  
  .search-results-list {
    padding: 0 12px;
  }

  :deep(.ant-list-item) {
    flex-direction: column;
    align-items: flex-start;
    position: relative;
    padding-bottom: 32px;
  }

  :deep(.ant-list-item-action) {
    margin-left: 0;
    margin-top: 8px;
    position: absolute;
    bottom: 8px;
    right: 0;
    width: auto;
  }

  :deep(.ant-list-item-meta) {
    width: 100%;
  }

  .delete-text {
    display: none;
  }
}
</style>