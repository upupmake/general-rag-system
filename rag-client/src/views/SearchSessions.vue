<script setup>
import {ref, onMounted, nextTick, computed} from 'vue'
import {useRouter, onBeforeRouteLeave} from 'vue-router'
import {message, theme} from 'ant-design-vue'
import {DeleteOutlined, MessageOutlined, HistoryOutlined} from '@ant-design/icons-vue'
import {deleteSession, searchSessions} from '@/api/chatApi'
import {useSearchStore} from '@/stores/search'
import {storeToRefs} from 'pinia'

const {useToken} = theme
const {token} = useToken()
const router = useRouter()
const searchStore = useSearchStore()
const { keyword, results, offset, hasMore, scrollPosition } = storeToRefs(searchStore)

// Search state
const searchLoading = ref(false)

const hasSearched = computed(() => !!keyword.value.trim() || results.value.length > 0)

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
      <div class="search-heading">
        <h2>搜索历史对话</h2>
        <p>按关键词检索历史会话内容，点击结果直接跳转</p>
      </div>
      <a-input-search
          v-model:value="keyword"
          class="search-input"
          placeholder="输入关键词搜索历史对话..."
          allowClear
          enter-button="搜索"
          size="large"
          @search="handleSearch"
      />
    </div>

    <div class="search-results">
      <!-- 初始状态 -->
      <div v-if="!hasSearched && !searchLoading" class="search-empty">
        <a-empty description="输入关键词开始搜索历史对话">
          <template #image>
            <span class="search-empty-icon"><history-outlined/></span>
          </template>
        </a-empty>
      </div>

      <!-- 首次加载骨架 -->
      <div v-else-if="searchLoading && !results.length" class="search-skeleton-list">
        <div v-for="n in 3" :key="n" class="search-skeleton">
          <span class="search-skeleton-line w-40"></span>
          <span class="search-skeleton-line w-90"></span>
          <span class="search-skeleton-line w-70"></span>
        </div>
      </div>

      <!-- 无结果 -->
      <a-empty v-else-if="!results.length" class="search-empty" description="暂无搜索结果"/>

      <!-- 结果列表 -->
      <template v-else>
        <div class="search-result-count">
          找到 {{ results.length }} 条结果<span v-if="hasMore">，继续加载可查看更多</span>
        </div>

        <div class="search-result-list">
          <a-card
              v-for="item in results"
              :key="item.sessionId"
              :bordered="false"
              class="search-result-card"
          >
            <div class="result-head">
              <a class="session-link" @click="handleJump(item.sessionId)">
                <message-outlined class="session-icon"/>
                <span class="session-title">{{ item.sessionTitle || `会话 ${item.sessionId}` }}</span>
              </a>
              <a-popconfirm
                  title="确定要删除这个会话吗？"
                  ok-text="确定"
                  cancel-text="取消"
                  @confirm="handleDelete(item.sessionId)"
              >
                <a-button type="text" danger size="small" class="delete-btn" @click.stop>
                  <template #icon><delete-outlined/></template>
                  <span class="delete-text">删除</span>
                </a-button>
              </a-popconfirm>
            </div>

            <div class="search-content-snippet">
              <div v-for="(snippet, idx) in item.contentList" :key="idx" class="snippet-item">
                <div class="snippet-header">
                  <span v-if="snippet.matchType === 'title'" class="snippet-match-title">标题匹配</span>
                  <span class="snippet-time">{{ snippet.createdAt }}</span>
                </div>
                <div class="snippet-content" v-html="highlightText(snippet.content)"></div>
              </div>
            </div>
          </a-card>
        </div>

        <div v-if="hasMore" class="load-more-wrap">
          <a-button :loading="searchLoading" @click="loadMoreSearchResults">加载更多</a-button>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.search-page-container {
  padding: 24px;
  max-width: 860px;
  margin: 0 auto;
}

.search-header {
  margin-bottom: 20px;
}

.search-heading h2 {
  margin: 0 0 4px;
  font-size: 22px;
  font-weight: 600;
  color: v-bind('token.colorTextHeading');
}

.search-heading p {
  margin: 0 0 16px;
  font-size: 13px;
  color: v-bind('token.colorTextDescription');
}

.search-input {
  max-width: 560px;
}

.search-result-count {
  margin-bottom: 12px;
  font-size: 12px;
  color: v-bind('token.colorTextTertiary');
}

.search-result-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.search-result-card {
  border: 1px solid v-bind('token.colorSplit');
  border-radius: 12px;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.search-result-card :deep(.ant-card-body) {
  padding: 16px 18px;
}

.search-result-card:hover {
  border-color: v-bind('token.colorPrimaryBorder');
  box-shadow: v-bind('token.boxShadowSecondary');
}

.result-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.session-link {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  font-size: 15px;
  font-weight: 600;
  color: v-bind('token.colorText');
  transition: color 0.2s ease;
}

.session-link:hover {
  color: v-bind('token.colorPrimary');
}

.session-icon {
  flex: none;
  font-size: 16px;
  color: v-bind('token.colorPrimary');
}

.session-title {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.delete-btn {
  flex: none;
  opacity: 0;
  transition: opacity 0.2s ease;
}

.search-result-card:hover .delete-btn,
.search-result-card:focus-within .delete-btn {
  opacity: 1;
}

.search-content-snippet {
  margin-top: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.snippet-item {
  padding: 10px 12px;
  border-radius: 8px;
  font-size: 13px;
  line-height: 1.6;
  color: v-bind('token.colorTextSecondary');
  background: v-bind('token.colorFillQuaternary');
  border-left: 3px solid v-bind('token.colorPrimaryBorder');
}

.snippet-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
  font-size: 12px;
  color: v-bind('token.colorTextTertiary');
}

.snippet-match-title {
  display: inline-block;
  padding: 0 6px;
  color: v-bind('token.colorPrimary');
  background: v-bind('token.colorPrimaryBg');
  border-radius: 4px;
}

.snippet-content {
  word-break: break-word;
}

:deep(.highlight-text) {
  background-color: v-bind('token.colorWarningBg');
  color: v-bind('token.colorWarningText');
  font-weight: bold;
  border-radius: 3px;
  padding: 0 3px;
}

.search-empty {
  padding: 48px 0;
}

.search-empty-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 64px;
  height: 64px;
  border-radius: 50%;
  font-size: 28px;
  color: v-bind('token.colorPrimary');
  background: v-bind('token.colorPrimaryBg');
}

.search-skeleton-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.search-skeleton {
  padding: 16px;
  border: 1px solid v-bind('token.colorSplit');
  border-radius: 12px;
}

.search-skeleton-line {
  display: block;
  height: 12px;
  margin-top: 10px;
  border-radius: 6px;
  background: v-bind('token.colorFillSecondary');
  animation: search-shimmer 1.2s ease-in-out infinite;
}

.search-skeleton-line:first-child {
  margin-top: 0;
}

.w-40 { width: 40%; }
.w-90 { width: 90%; }
.w-70 { width: 70%; }

@keyframes search-shimmer {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.45; }
}

.load-more-wrap {
  margin-top: 16px;
  text-align: center;
}

@media (hover: none) {
  /* 触屏设备无悬停态，删除按钮常驻 */
  .delete-btn {
    opacity: 1;
  }
}

@media (max-width: 768px) {
  .search-page-container {
    padding: 16px;
  }

  .search-input {
    width: 100%;
    max-width: 100%;
  }

  .search-result-card :deep(.ant-card-body) {
    padding: 14px;
  }

  .delete-text {
    display: none;
  }
}
</style>