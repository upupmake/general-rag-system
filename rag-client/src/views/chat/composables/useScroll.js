import {ref, nextTick, onScopeDispose} from 'vue'

export function useScroll() {
  const messagesContainer = ref(null)
  const userScrolledUp = ref(false)
  const isAutoScrolling = ref(false)
  const isTouchHolding = ref(false)
  let scrollQueued = false
  let pendingBehavior = 'auto'
  let pendingForce = false
  let resetTimer = null
  let disposed = false

  onScopeDispose(() => {
    disposed = true
    clearTimeout(resetTimer)
  })

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

    pendingBehavior = behavior
    pendingForce = pendingForce || force
    if (scrollQueued) return
    scrollQueued = true
    nextTick(() => {
      scrollQueued = false
      const forceScroll = pendingForce
      pendingForce = false
      const container = messagesContainer.value
      if (disposed || !container || isTouchHolding.value || (!forceScroll && userScrolledUp.value)) return
      isAutoScrolling.value = true
      container.scrollTo({
        top: container.scrollHeight,
        behavior: pendingBehavior
      })
      clearTimeout(resetTimer)
      resetTimer = setTimeout(() => {
        isAutoScrolling.value = false
      }, pendingBehavior === 'smooth' ? 300 : 0)
    })
  }

  const handleScroll = () => {
    if (!messagesContainer.value || (isAutoScrolling.value && !isTouchHolding.value)) return
    updateUserScrolledUp()
  }

  // 用户主动点击"回到底部"：重置跟随状态并强制平滑滚到底部
  const jumpToBottom = () => {
    if (!messagesContainer.value) return
    userScrolledUp.value = false
    isTouchHolding.value = false
    scrollToBottom('smooth', true)
  }

  return {
    messagesContainer,
    userScrolledUp,
    isAutoScrolling,
    scrollToBottom,
    handleScroll,
    handleWheel,
    startTouchScrolling,
    endTouchScrolling,
    jumpToBottom
  }
}
