import {ref, shallowRef, reactive, computed, nextTick, onScopeDispose} from 'vue'
import {message as antMessage} from 'ant-design-vue'
import {
    fetchSessionMessages,
    startChatStream
} from '@/api/chatApi'
import {models, selectedModel, selectedKb, contextMultiplier, findKbById, loadChatResources} from '@/vars.js'
import {useUserStore} from '@/stores/user.js'
import {buildMessages, getChatSession, releaseChatSession, createStreamCallbacks} from './chatSessions.js'

export function useChat(
    sessionId,
    selectedTools,
    thinkingEnabled,
    currentModel,
    isKbSupported,
    isRestoring,
    isUserUncheckedWebSearch,
    availableTools,
    userScrolledUp,
    scrollToBottom
) {
    const userStore = useUserStore()
    const activeSession = shallowRef(getChatSession(sessionId.value))
    const messages = computed({
        get: () => activeSession.value.messages.value,
        set: value => { activeSession.value.messages.value = value }
    })
    const loading = ref(false)
    const loadError = ref('')
    const isGenerating = computed(() => activeSession.value.isGenerating.value)
    const isSynchronizing = computed(() => activeSession.value.synchronizing.value)
    const question = ref('')
    const abortController = computed({
        get: () => activeSession.value.abortController.value,
        set: value => { activeSession.value.abortController.value = value }
    })
    const streamStarted = computed(() => activeSession.value.streamStarted.value)
    let loadVersion = 0
    let disposed = false

    onScopeDispose(() => {
        disposed = true
        loadVersion++
        releaseChatSession(activeSession.value)
    })

    const handleStreamCallbacks = (assistantMsg, userMsg = null, options = null) => {
        if (userMsg) {
            userMsg.modelId = selectedModel.value
            userMsg.kbId = isKbSupported.value ? selectedKb.value : null
            userMsg.options = options
        }
        return createStreamCallbacks(activeSession.value, assistantMsg, userMsg)
    }

    const restoreSession = (data) => {
        if (data.length > 0) {
            const lastMsg = data.findLast(msg => msg.role === 'user') || data[data.length - 1]
            // 恢复模型选择
            if (lastMsg.modelId && models.value.find(m => m.modelId === lastMsg.modelId)) {
                selectedModel.value = lastMsg.modelId
            } else {
                let len = models.value.length
                selectedModel.value = len > 0 ? models.value[len - 1].modelId : null
            }
            // 恢复知识库选择
            if (isKbSupported.value && lastMsg.kbId && findKbById(lastMsg.kbId)) {
                selectedKb.value = lastMsg.kbId
            } else {
                selectedKb.value = null
            }
        } else {
            selectedModel.value = selectedModel.value || null
            selectedKb.value = null
        }

        // 恢复工具和思考状态
        const lastUserMsg = messages.value.filter(m => m.role === 'user').pop()
        if (lastUserMsg && lastUserMsg.options) {
            const opts = lastUserMsg.options
            selectedTools.value = []
            if (opts.webSearch) {
                selectedTools.value.push('webSearch')
                isUserUncheckedWebSearch.value = false
            } else {
                if (currentModel.value?.metadata?.tools?.includes('webSearch')) {
                    isUserUncheckedWebSearch.value = true
                }
            }

            if (opts.thinking) {
                thinkingEnabled.value = true
            } else {
                thinkingEnabled.value = false
            }

            contextMultiplier.value = opts.contextMultiplier !== undefined ? opts.contextMultiplier : null

            if (currentModel.value?.metadata?.thinking?.editable === false) {
                thinkingEnabled.value = currentModel.value.metadata.thinking.default
            }
        } else {
            selectedTools.value = []
            if (availableTools.value.includes('webSearch')) {
                selectedTools.value.push('webSearch')
                isUserUncheckedWebSearch.value = false
            } else {
                isUserUncheckedWebSearch.value = false
            }

            if (currentModel.value?.metadata?.thinking) {
                thinkingEnabled.value = currentModel.value.metadata.thinking.default
            } else {
                thinkingEnabled.value = false
            }

            if (lastUserMsg) {
                thinkingEnabled.value = false
            }
        }
    }

    const loadSession = async (newSessionId) => {
        const version = ++loadVersion
        if (activeSession.value.id !== String(newSessionId)) releaseChatSession(activeSession.value)
        const session = getChatSession(newSessionId)
        activeSession.value = session
        const isCurrent = () => !disposed && version === loadVersion
            && activeSession.value === session && userStore.token === session.token
        session.onUpdate = () => {
            if (isCurrent() && !loading.value) scrollToBottom('auto')
        }
        loading.value = true
        loadError.value = ''
        isRestoring.value = true
        try {
            const [, data] = await Promise.all([
                loadChatResources(),
                session.hasLocalStream ? Promise.resolve(null) : fetchSessionMessages(newSessionId)
            ])
            if (!isCurrent()) return
            if (data) session.messages.value = buildMessages(data)
            restoreSession(messages.value)

            const lastMsg = messages.value[messages.value.length - 1]
            if (data && lastMsg?.role === 'user' && lastMsg.status === 'pending') {
                const userMsg = lastMsg
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
                const options = userMsg.options
                const {onOpen, onMessage, onError, onClose} = handleStreamCallbacks(assistant, userMsg, options)
                abortController.value = startChatStream(newSessionId, selectedModel.value, null, selectedKb.value || undefined, options, onOpen, onMessage, onError, onClose)
            }
            userScrolledUp.value = false
            session.flush?.()
        } catch (e) {
            if (isCurrent()) loadError.value = e?.message || '会话加载失败，请重试'
        } finally {
            if (isCurrent()) {
                loading.value = false
                await nextTick()
                if (isCurrent()) {
                    scrollToBottom('auto', true)
                    isRestoring.value = false
                }
            }
        }
    }

    const prepareGeneration = () => {
        if (loading.value || loadError.value || isGenerating.value || isSynchronizing.value) return false
        const session = activeSession.value
        const version = loadVersion
        if (!session.synchronize) return true
        return session.synchronize().then(ready => {
            if (disposed || session !== activeSession.value || version !== loadVersion) return false
            if (!ready) antMessage.warning('上一轮结果尚未同步，请稍后重试；已生成内容已保留')
            return ready
        })
    }

    const onSend = async (text) => {
        const ready = prepareGeneration()
        if (ready !== true && !await ready) return false
        if (question.value === text) question.value = ''
        userScrolledUp.value = false
        const unpersisted = activeSession.value.unpersistedRound
        if (unpersisted) {
            messages.value.splice(messages.value.indexOf(unpersisted.userMsg), 2)
            activeSession.value.unpersistedRound = null
        }
        const lastMsg = messages.value[messages.value.length - 1]
        const userMsg = lastMsg?.role === 'user' ? lastMsg : reactive({role: 'user', content: text, status: 'pending'})
        userMsg.content = text
        if (userMsg !== lastMsg) messages.value.push(userMsg)
        messages.value.push(reactive({
            role: 'assistant',
            content: '',
            loading: true,
            ragProcess: [],
            latencyMs: 0,
            totalTokens: 0,
            thinkingCollapseKeys: []
        }))
        scrollToBottom('auto', true)
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

        if (contextMultiplier.value !== null) {
            options.contextMultiplier = contextMultiplier.value
        }

        const {onOpen, onMessage, onError, onClose} = handleStreamCallbacks(assistant, userMsg, options)
        abortController.value = startChatStream(sessionId.value, selectedModel.value, text, isKbSupported.value ? (selectedKb.value || undefined) : undefined, options, onOpen, onMessage, onError, onClose)
        return true
    }

    const stopGeneration = () => {
        activeSession.value.stop?.()
    }

    const onCopy = (textToCopy) => {
        if (!textToCopy) {
            antMessage.warning('消息内容为空')
            return
        }
        navigator.clipboard.writeText(textToCopy).then(() => {
            antMessage.success('复制成功')
        }).catch(() => {
            antMessage.error('复制失败')
        })
    }

    const lastUserMessage = computed(() => {
        const msgs = messages.value
        for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].role === 'user') {
                return msgs[i]
            }
        }
        return null
    })

    const isLastUserMsgGenerating = computed(() => {
        return lastUserMessage.value?.status === 'generating'
    })

    const lastAssistantMessage = computed(() => {
        const msgs = messages.value
        for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].role === 'assistant') {
                return msgs[i]
            }
        }
        return null
    })

    const lastUserHasAssistant = computed(() => {
        const msgs = messages.value
        for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].role === 'user') {
                return msgs[i + 1]?.role === 'assistant'
            }
        }
        return false
    })

    const canEditOrRetry = computed(() => {
        if (loading.value || loadError.value || isGenerating.value || isSynchronizing.value) return false
        const msg = lastUserMessage.value
        if (!msg) return false
        // 上一轮中断/失败（user 后没有 assistant）时也允许编辑/重试
        if (!lastUserHasAssistant.value) return true
        return ['completed', 'error'].includes(msg.status) || !msg.status
    })

    const isLastUserMessage = (item) => {
        const last = lastUserMessage.value
        if (!last) return false
        return item === last || (item.id && last.id && item.id === last.id)
            || (item.key != null && item.key === last.key)
    }

    const isLastAssistantMessage = (item) => {
        const last = lastAssistantMessage.value
        if (!last) return false
        return item === last || (item.id && last.id && item.id === last.id)
            || (item.key != null && item.key === last.key)
    }

    return {
        messages,
        loading,
        loadError,
        isGenerating,
        isSynchronizing,
        streamStarted,
        question,
        handleStreamCallbacks,
        prepareGeneration,
        loadSession,
        onSend,
        stopGeneration,
        abortController,
        onCopy,
        lastUserMessage,
        isLastUserMsgGenerating,
        lastAssistantMessage,
        lastUserHasAssistant,
        canEditOrRetry,
        isLastUserMessage,
        isLastAssistantMessage
    }
}
