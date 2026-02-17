import {ref, reactive} from 'vue'
import {message as antMessage} from 'ant-design-vue'
import {editMessageStream, retryMessageStream} from '@/api/chatApi'
import {ragMode} from '@/vars.js'

export function useMessageEdit(
    messages,
    sessionId,
    selectedModel,
    selectedKb,
    isKbSupported,
    selectedTools,
    thinkingEnabled,
    currentModel,
    isGenerating,
    userScrolledUp,
    handleStreamCallbacks
) {
    const editingIndex = ref(-1)
    const editingContent = ref('')

    const startEditByItem = (item) => {
        let index = messages.value.indexOf(item)

        // 如果通过引用找不到（可能是Proxy对象），尝试通过ID查找
        if (index === -1 && item.id) {
            index = messages.value.findIndex(m => m.id === item.id)
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

    const confirmEdit = () => {
        if (!editingContent.value.trim()) {
            antMessage.warning('问题内容不能为空')
            return
        }

        const index = editingIndex.value
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
            completionTokens: 0,
            thinkingCollapseKeys: []
        }))
        const assistant = messages.value[messages.value.length - 1]
        const {onOpen, onMessage, onError, onClose} = handleStreamCallbacks(assistant, userMsg)

        isGenerating.value = true
        const options = {}
        if (selectedTools.value.includes('webSearch')) {
            options.webSearch = true
        }

        if (currentModel.value?.metadata?.thinking && thinkingEnabled.value) {
            options.thinking = true
        }

        // 添加RAG模式选项
        if (selectedKb.value) {
            if (ragMode.value === 'agentic') {
                options.agenticRag = true
            } else if (ragMode.value === 'fast') {
                options.agenticRag = false
            }
        }

        editMessageStream(
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
    }

    const onRetry = (userMsgIndex) => {
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
            completionTokens: 0,
            thinkingCollapseKeys: []
        }))
        const assistant = messages.value[messages.value.length - 1]
        const {onOpen, onMessage, onError, onClose} = handleStreamCallbacks(assistant, userMsg)

        isGenerating.value = true
        const options = {}
        if (selectedTools.value.includes('webSearch')) {
            options.webSearch = true
        }

        if (currentModel.value?.metadata?.thinking && thinkingEnabled.value) {
            options.thinking = true
        }

        // 添加RAG模式选项
        if (selectedKb.value) {
            if (ragMode.value === 'agentic') {
                options.agenticRag = true
            } else if (ragMode.value === 'fast') {
                options.agenticRag = false
            }
        }

        retryMessageStream(
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
    }

    const onRetryFromAssistant = () => {
        for (let i = messages.value.length - 1; i >= 0; i--) {
            if (messages.value[i].role === 'user') {
                onRetry(i)
                return
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
