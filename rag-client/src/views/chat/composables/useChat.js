import {ref, reactive, computed, nextTick} from 'vue'
import {message as antMessage} from 'ant-design-vue'
import {
    fetchAvailableModels,
    fetchSessionMessages,
    startChatStream
} from '@/api/chatApi'
import {models, selectedModel, selectedKb, ragMode, findKbById, loadKbs} from '@/vars.js'

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
    const messages = ref([])
    const loading = ref(false)
    const isGenerating = ref(false)
    const question = ref('')

    const handleStreamCallbacks = (assistantMsg, userMsg = null) => {
        if (userMsg) {
            userMsg.status = 'generating'
        }
        return {
            onOpen: () => {
            },
            onMessage: (data) => {
                if (data.type === 'content') {
                    if (assistantMsg.loading) assistantMsg.loading = false
                    if (assistantMsg.thinking && assistantMsg.thinkingCollapseKeys?.length) {
                        nextTick(() => {
                            assistantMsg.thinkingCollapseKeys = []
                        })
                    }
                    assistantMsg.content += data.content
                } else if (data.type === 'thinking') {
                    if (assistantMsg.loading) assistantMsg.loading = false
                    // 收到思考内容时，自动展开
                    if (!assistantMsg.thinkingCollapseKeys || assistantMsg.thinkingCollapseKeys.length === 0) {
                        nextTick(() => {
                            assistantMsg.thinkingCollapseKeys = ['thinking-panel']
                        })
                    }
                    if (typeof assistantMsg.thinking === "string") {
                        assistantMsg.thinking += data.content
                    } else {
                        assistantMsg.thinking = data.content
                    }
                } else if (data.type === 'process') {
                    if (assistantMsg.loading) assistantMsg.loading = false

                    if (!assistantMsg.ragProcess) {
                        assistantMsg.ragProcess = []
                    }
                    const processInfo = data.payload

                    const existingIndex = assistantMsg.ragProcess.findIndex(
                        p => p.step === processInfo.step && p.status === processInfo.status
                    )

                    if (existingIndex !== -1) {
                        assistantMsg.ragProcess[existingIndex] = processInfo
                    } else {
                        assistantMsg.ragProcess.push(processInfo)
                    }
                } else if (data.type === 'done') {
                    if (userMsg) {
                        if (data.userMessageId) {
                            userMsg.id = data.userMessageId
                        }
                        userMsg.status = 'completed'
                    }
                    if (data.assistantMessageId) {
                        assistantMsg.id = data.assistantMessageId
                    }
                    assistantMsg.status = 'completed'
                    assistantMsg.loading = false
                    // 思考结束，立即收起面板
                    if (assistantMsg.thinking) {
                        nextTick(() => {
                            assistantMsg.thinkingCollapseKeys = []
                        })
                    }
                } else if (data.type === 'usage') {
                    if (data.payload) {
                        if (data.payload.latency_ms != null) {
                            assistantMsg.latencyMs = data.payload.latency_ms
                        }
                        if (data.payload.completion_tokens != null) {
                            assistantMsg.completionTokens = data.payload.completion_tokens
                        }
                        if (data.payload.total_tokens != null) {
                            assistantMsg.totalTokens = data.payload.total_tokens
                        }
                    }
                }
            },
            onError: (err) => {
                assistantMsg.content += `\n[Error: 请求发起失败！]`
                assistantMsg.loading = false
                isGenerating.value = false
                if (userMsg) {
                    userMsg.status = 'pending'
                }
            },
            onClose: () => {
                assistantMsg.loading = false
                assistantMsg.status = 'completed'

                // 思考结束，确保收起面板
                if (assistantMsg.thinking) {
                    nextTick(() => {
                        assistantMsg.thinkingCollapseKeys = []
                    })
                }

                isGenerating.value = false
                if (userMsg) {
                    userMsg.status = 'completed'
                }
            }
        }
    }

    const loadSession = async (newSessionId) => {
        loading.value = true
        isRestoring.value = true
        messages.value = []
        models.value = await fetchAvailableModels()
        await loadKbs()
        let data = await fetchSessionMessages(newSessionId)

        if (data.length > 0) {
            const lastMsg = data[data.length - 1]
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

        messages.value = data.map(msg => {
            let ragProcess = null
            if (msg.ragContext && msg.role === 'assistant') {
                try {
                    ragProcess = typeof msg.ragContext === 'string' ? JSON.parse(msg.ragContext) : msg.ragContext
                } catch (e) {
                    console.error('Failed to parse ragContext:', e)
                }
            }

            let options = null
            if (msg.options) {
                try {
                    options = typeof msg.options === 'string' ? JSON.parse(msg.options) : msg.options
                } catch (e) {
                    console.error('Failed to parse options:', e)
                }
            }
            return reactive({
                id: msg.id,
                role: msg.role,
                content: msg.content,
                status: msg.status,
                loading: msg.role === 'assistant' && msg.status === 'generating',
                ragProcess: ragProcess,
                latencyMs: msg.latencyMs,
                totalTokens: msg.totalTokens,
                options: options,
                thinking: msg.thinking,
                thinkingCollapseKeys: []
            })
        })

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

        if (data.length > 0) {
            const lastMsg = data[data.length - 1]
            if (lastMsg.role === 'user' && lastMsg.status === 'pending') {
                const userMsg = messages.value[messages.value.length - 1]
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
                const {onOpen, onMessage, onError, onClose} = handleStreamCallbacks(assistant, userMsg)
                isGenerating.value = true

                let options = null
                if (lastMsg.options) {
                    try {
                        options = typeof lastMsg.options === 'string' ? JSON.parse(lastMsg.options) : lastMsg.options
                    } catch (e) {
                        console.error("Parse options failed", e)
                    }
                }

                startChatStream(newSessionId, selectedModel.value, null, selectedKb.value || undefined, options, onOpen, onMessage, onError, onClose)
            }
        }
        loading.value = false
        userScrolledUp.value = false
        nextTick(() => {
            scrollToBottom('auto')
            setTimeout(() => {
                isRestoring.value = false
            }, 0)
        })
    }

    const onSend = (text) => {
        if (loading.value || isGenerating.value) return
        question.value = ''
        userScrolledUp.value = false
        const userMsg = reactive({role: 'user', content: text, status: 'pending'})
        messages.value.push(userMsg)
        messages.value.push(reactive({
            role: 'assistant',
            content: '',
            loading: true,
            ragProcess: [],
            latencyMs: 0,
            totalTokens: 0,
            thinkingCollapseKeys: []
        }))
        scrollToBottom()
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
                options.agenticRag = false  // Fast RAG 模式
            }
        }

        startChatStream(sessionId.value, selectedModel.value, text, isKbSupported.value ? (selectedKb.value || undefined) : undefined, options, onOpen, onMessage, onError, onClose)
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

    const canEditOrRetry = computed(() => {
        const msg = lastUserMessage.value
        if (!msg) return false
        return ['completed', 'error'].includes(msg.status) || !msg.status
    })

    const isLastUserMessage = (item) => {
        const last = lastUserMessage.value
        if (!last) return false
        return item === last || (item.id && last.id && item.id === last.id)
    }

    const isLastAssistantMessage = (item) => {
        const last = lastAssistantMessage.value
        if (!last) return false
        return item === last || (item.id && last.id && item.id === last.id)
    }

    return {
        messages,
        loading,
        isGenerating,
        question,
        handleStreamCallbacks,
        loadSession,
        onSend,
        onCopy,
        lastUserMessage,
        isLastUserMsgGenerating,
        lastAssistantMessage,
        canEditOrRetry,
        isLastUserMessage,
        isLastAssistantMessage
    }
}
