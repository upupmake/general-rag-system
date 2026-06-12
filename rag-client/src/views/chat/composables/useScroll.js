import {ref, nextTick} from 'vue'

export function useScroll() {
  const messagesContainer = ref(null)
  const userScrolledUp = ref(false)
  const isAutoScrolling = ref(false)
  const isUserScrolling = ref(false)
  let userScrollTimer = null

  const markUserScrolling = () => {
    isUserScrolling.value = true
    isAutoScrolling.value = false
    if (userScrollTimer) clearTimeout(userScrollTimer)
    userScrollTimer = setTimeout(() => {
      isUserScrolling.value = false
    }, 120)
  }

  const updateUserScrolledUp = () => {
    if (!messagesContainer.value) return

    const container = messagesContainer.value
    const threshold = 50
    const distanceToBottom = container.scrollHeight - container.scrollTop - container.clientHeight
    userScrolledUp.value = distanceToBottom >= threshold
  }

  const scrollToBottom = (behavior = 'smooth', force = false) => {
    if (!messagesContainer.value || isUserScrolling.value || (!force && userScrolledUp.value)) return

    isAutoScrolling.value = true
    nextTick(() => {
      const container = messagesContainer.value
      if (container) {
        container.scrollTo({
          top: container.scrollHeight,
          behavior: behavior
        })
      }
      setTimeout(() => {
        isAutoScrolling.value = false
      }, behavior === 'smooth' ? 300 : 0)
    })
  }

  const handleScroll = () => {
    if (!messagesContainer.value || (isAutoScrolling.value && !isUserScrolling.value)) return
    updateUserScrolledUp()
  }

  return {
    messagesContainer,
    userScrolledUp,
    isAutoScrolling,
    scrollToBottom,
    handleScroll,
    markUserScrolling
  }
}
