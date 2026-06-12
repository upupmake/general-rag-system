import {ref, nextTick} from 'vue'

export function useScroll() {
  const messagesContainer = ref(null)
  const userScrolledUp = ref(false)
  const isAutoScrolling = ref(false)
  const isTouchHolding = ref(false)

  const nearBottomThreshold = 50

  const updateUserScrolledUp = () => {
    if (!messagesContainer.value) return

    const container = messagesContainer.value
    const distanceToBottom = container.scrollHeight - container.scrollTop - container.clientHeight
    userScrolledUp.value = distanceToBottom > nearBottomThreshold
  }

  const handleWheel = () => {
    isAutoScrolling.value = false
    userScrolledUp.value = true
  }

  const startTouchScrolling = () => {
    isTouchHolding.value = true
    isAutoScrolling.value = false
  }

  const endTouchScrolling = () => {
    isTouchHolding.value = false
    updateUserScrolledUp()
  }

  const scrollToBottom = (behavior = 'smooth', force = false) => {
    if (!messagesContainer.value || isTouchHolding.value || (!force && userScrolledUp.value)) return

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
    if (!messagesContainer.value || (isAutoScrolling.value && !isTouchHolding.value)) return
    updateUserScrolledUp()
  }

  return {
    messagesContainer,
    userScrolledUp,
    isAutoScrolling,
    scrollToBottom,
    handleScroll,
    handleWheel,
    startTouchScrolling,
    endTouchScrolling
  }
}
