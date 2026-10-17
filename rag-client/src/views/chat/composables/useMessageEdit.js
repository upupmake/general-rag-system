import {ref, reactive} from 'vue'
import {message as antMessage} from 'ant-design-vue'
import {editMessageStream, retryMessageStream, startChatStream} from '@/api/chatApi'

export function useMessageEdit(
    messages,
    sessionId,
    selectedModel,
    selectedKb,
    isKbSupported,
    selectedTools,
    thinkingEnabled,
    currentModel,
    userScrolledUp,
    scrollToBottom,
    handleStreamCallbacks,
    abortController,
    prepareGeneration
) {
    const editingIndex = ref(-1)
    const editingContent = ref('')

    const startEditByItem = (item) => {
        let index = messages.value.indexOf(item)

        // 如果通过引用找不到（可能是Proxy对象），尝试通过ID查找
        if (index === -1 && item.id) {
            index = messages.value.findIndex(m => m.id === item.id)
        }

        if (index === -1 && item.key != null) {
            index = messages.value.findIndex(m => m.key === item.key)
        }

        // 如果还是找不到，尝试查找最后一个用户消息
        if (index === -1) {
            for (let i = messages.value.length - 1; i >= 0; i--) {
                if (messages.value[i].role === 'user') {
                    if (messages.value[i].content === item.content) {
                        index = i
                    }
                    break
                }
            }
        }

        if (index === -1) {
            console.error('Cannot find message index for edit', item)
            return
        }

        editingIndex.value = index
        editingContent.value = item.content
    }

    const cancelEdit = () => {
        editingIndex.value = -1
        editingContent.value = ''
    }

    const confirmEdit = async () => {
        if (!editingContent.value.trim()) {
            antMessage.warning('问题内容不能为空')
            return
        }

        const ready = prepareGeneration()
        if (ready !== true && !await ready) return
        const index = messages.value.findLastIndex(msg => msg.role === 'user')
        const userMsg = messages.value[index]

        userMsg.content = editingContent.value
        userMsg.status = 'pending'

        if (index < messages.value.length - 1 && messages.value[index + 1].role === 'assistant') {
            messages.value.splice(index + 1, 1)
        }

        messages.value.push(reactive({
            role: 'assistant',
            content: '',
            loading: true,
            ragProcess: [],
            latencyMs: 0,
            totalTokens: 0,
            thinkingCollapseKeys: []
        }))
        const assistant = messages.value[messages.value.length - 1]

        const options = {}
        if (selectedTools.value.includes('webSearch')) {
            options.webSearch = true
        }

        if (currentModel.value?.metadata?.thinking && thinkingEnabled.value) {
            options.thinking = true
        }

        if (selectedKb.value) {
            options.agenticRag = true
        }

        const {onOpen, onMessage, onError, onClose} = handleStreamCallbacks(assistant, userMsg, options)
        // 首条响应前取消时，问题可能还未落库；此时使用普通发送入口。
        abortController.value = !userMsg.id ? startChatStream(
            sessionId.value, selectedModel.value, userMsg.content,
            isKbSupported.value ? (selectedKb.value || undefined) : undefined,
            options, onOpen, onMessage, onError, onClose
        ) : editMessageStream(
            userMsg.id,
            sessionId.value,
            selectedModel.value,
            isKbSupported.value ? (selectedKb.value || undefined) : undefined,
            editingContent.value,
            options,
            onOpen,
            onMessage,
            onError,
            onClose
        )

        editingIndex.value = -1
        editingContent.value = ''
        userScrolledUp.value = false
        scrollToBottom('auto', true)
    }

    const onRetry = async () => {
        const ready = prepareGeneration()
        if (ready !== true && !await ready) return
        const userMsgIndex = messages.value.findLastIndex(msg => msg.role === 'user')
        const userMsg = messages.value[userMsgIndex]
        userMsg.status = 'pending'

        if (userMsgIndex < messages.value.length - 1 && messages.value[userMsgIndex + 1].role === 'assistant') {
            messages.value.splice(userMsgIndex + 1, 1)
        }

        messages.value.push(reactive({
            role: 'assistant',
            content: '',
            loading: true,
            ragProcess: [],
            latencyMs: 0,
            totalTokens: 0,
            thinkingCollapseKeys: []
        }))
        const assistant = messages.value[messages.value.length - 1]

        const options = {}
        if (selectedTools.value.includes('webSearch')) {
            options.webSearch = true
        }

        if (currentModel.value?.metadata?.thinking && thinkingEnabled.value) {
            options.thinking = true
        }

        if (selectedKb.value) {
            options.agenticRag = true
        }

        const {onOpen, onMessage, onError, onClose} = handleStreamCallbacks(assistant, userMsg, options)
        abortController.value = !userMsg.id ? startChatStream(
            sessionId.value, selectedModel.value, userMsg.content,
            isKbSupported.value ? (selectedKb.value || undefined) : undefined,
            options, onOpen, onMessage, onError, onClose
        ) : retryMessageStream(
            userMsg.id,
            sessionId.value,
            selectedModel.value,
            isKbSupported.value ? (selectedKb.value || undefined) : undefined,
            options,
            onOpen,
            onMessage,
            onError,
            onClose
        )

        userScrolledUp.value = false
        scrollToBottom('auto', true)
    }

    const onRetryFromAssistant = () => {
        for (let i = messages.value.length - 1; i >= 0; i--) {
            if (messages.value[i].role === 'user') {
                return onRetry(i)
            }
        }
        antMessage.warning('没有找到可重试的用户消息')
    }

    return {
        editingIndex,
        editingContent,
        startEditByItem,
        cancelEdit,
        confirmEdit,
        onRetry,
        onRetryFromAssistant
    }
}
