import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useSearchStore = defineStore('search', () => {
  const keyword = ref('')
  const results = ref([])
  const offset = ref(0)
  const hasMore = ref(true)
  const scrollPosition = ref(0)

  // Actions
  const setKeyword = (val) => {
    keyword.value = val
  }

  const setResults = (val) => {
    results.value = val
  }
  
  const appendResults = (newResults) => {
    newResults.forEach(item => {
      const existing = results.value.find(result => result.sessionId === item.sessionId)
      if (!existing) {
        results.value.push(item)
        return
      }
      const snippets = existing.contentList || (existing.contentList = [])
      const seen = new Set(snippets.map(snippet =>
          `${snippet.matchType || 'content'}\u0000${snippet.createdAt || ''}\u0000${snippet.content || ''}`))
      ;(item.contentList || []).forEach(snippet => {
        const key = `${snippet.matchType || 'content'}\u0000${snippet.createdAt || ''}\u0000${snippet.content || ''}`
        if (!seen.has(key)) {
          seen.add(key)
          snippets.push(snippet)
        }
      })
    })
  }

  const setOffset = (val) => {
    offset.value = val
  }

  const setHasMore = (val) => {
    hasMore.value = val
  }
  
  const setScrollPosition = (val) => {
    scrollPosition.value = val
  }
  
  const clearState = () => {
    keyword.value = ''
    results.value = []
    offset.value = 0
    hasMore.value = true
    scrollPosition.value = 0
  }

  const removeItem = (sessionId) => {
    results.value = results.value.filter(item => item.sessionId !== sessionId)
  }

  return {
    keyword,
    results,
    offset,
    hasMore,
    scrollPosition,
    setKeyword,
    setResults,
    appendResults,
    setOffset,
    setHasMore,
    setScrollPosition,
    clearState,
    removeItem
  }
})
