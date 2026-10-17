import {ref, shallowRef} from 'vue'
import {fetchSessionMessages} from '@/api/chatApi'
import {useUserStore} from '@/stores/user.js'
import {events} from '@/events.js'

// 流的生命周期属于会话，而不是当前页面；刷新页面或断网不做续流。
const sessions = new Map()
let sessionToken = null

events.on('chatSessionDeleted', id => {
    const session = sessions.get(String(id))
    if (!session) return
    session.generation++
    session.dispose?.()
    session.abortController.value = null
    session.isGenerating.value = false
    session.synchronizing.value = false
    session.messages.value = []
    sessions.delete(String(id))
})

export function getChatSession(sessionId) {
    const token = useUserStore().token
    if (sessionToken !== token) {
        sessions.forEach(session => session.dispose?.())
        sessions.clear()
        sessionToken = token
    }
    const id = String(sessionId)
    if (!sessions.has(id)) {
        sessions.set(id, {
            id,
            token,
            messages: ref([]),
            isGenerating: ref(false),
            streamStarted: ref(false),
            abortController: shallowRef(null),
            generation: 0,
            hasLocalStream: false,
            keepOnLeave: false,
            synchronizing: ref(false),
            synchronize: null,
            unpersistedRound: null,
            syncTimer: null,
            onUpdate: null,
            flush: null,
            stop: null,
            dispose: null
        })
    }
    return sessions.get(id)
}

export function releaseChatSession(session) {
    session.onUpdate = null
    session.flush?.()
    if (!session.isGenerating.value && !session.keepOnLeave && !session.syncTimer
        && sessions.get(session.id) === session) {
        sessions.delete(session.id)
    }
}

export const buildMessages = (data) => data.map(msg => {
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
    return {
        id: msg.id,
        role: msg.role,
        content: msg.content,
        status: msg.status,
        loading: msg.role === 'assistant' && msg.status === 'generating',
        renderKey: {},
        ragProcess,
        modelId: msg.modelId,
        kbId: msg.kbId,
        latencyMs: msg.latencyMs,
        totalTokens: msg.totalTokens,
        options,
        thinking: msg.thinking,
        thinkingCollapseKeys: []
    }
})

export function createStreamCallbacks(session, assistantMsg, userMsg = null) {
    clearTimeout(session.syncTimer)
    session.syncTimer = null
    const generation = ++session.generation
    session.hasLocalStream = true
    session.keepOnLeave = false
    session.synchronizing.value = false
    session.synchronize = null
    session.unpersistedRound = null
    session.isGenerating.value = true
    session.streamStarted.value = false
    if (userMsg) {
        userMsg.status = 'generating'
        userMsg.key ??= `${session.id}:${generation}:user`
    }
    assistantMsg.key = `${session.id}:${generation}:assistant`
    assistantMsg.renderKey = {}
    // Bubble 的默认 loading 会遮住 message 插槽，等待阶段使用自己的进度提示。
    assistantMsg.loading = false
    assistantMsg.streamPhase = 'connecting'
    assistantMsg.elapsedSeconds = 0
    assistantMsg.idleSeconds = 0

    let finished = false
    let contentBuffer = ''
    let thinkingBuffer = ''
    let flushTimer = null
    let scrollPending = false
    let receivedOutput = false
    let syncRequest = null
    const local = session.messages.value
    const userIndex = local.indexOf(userMsg)
    const startedAt = Date.now()
    let lastEventAt = startedAt
    const isCurrent = () => session.generation === generation && session.token === useUserStore().token
    const isLive = () => isCurrent() && !finished

    const flush = () => {
        clearTimeout(flushTimer)
        flushTimer = null
        if (thinkingBuffer) {
            assistantMsg.thinking = (assistantMsg.thinking || '') + thinkingBuffer
            thinkingBuffer = ''
            if (assistantMsg.streamPhase === 'thinking' && !assistantMsg.thinkingCollapseKeys?.length) {
                assistantMsg.thinkingCollapseKeys = ['thinking-panel']
            }
        }
        if (contentBuffer) {
            assistantMsg.content += contentBuffer
            contentBuffer = ''
            assistantMsg.thinkingCollapseKeys = []
        }
        if (scrollPending) {
            scrollPending = false
            session.onUpdate?.()
        }
    }
    session.flush = flush

    const scheduleFlush = () => {
        scrollPending = true
        // 后台只接收数据，切回或结束时冲刷，不触发页面渲染和滚动。
        if (session.onUpdate && flushTimer === null) {
            flushTimer = setTimeout(flush, 80)
        }
    }

    const waitTimer = setInterval(() => {
        if (!isLive()) return
        assistantMsg.elapsedSeconds = Math.floor((Date.now() - startedAt) / 1000)
        assistantMsg.idleSeconds = Math.floor((Date.now() - lastEventAt) / 1000)
    }, 1000)

    const finish = (status = 'completed', error = null) => {
        if (!isLive()) return
        flush()
        finished = true
        clearInterval(waitTimer)
        if (error) assistantMsg.content += `\n[Error: ${error}]`
        assistantMsg.loading = false
        assistantMsg.status = status
        assistantMsg.streamPhase = null
        assistantMsg.thinkingCollapseKeys = []
        if (userMsg) userMsg.status = 'completed'
        session.isGenerating.value = false
        session.streamStarted.value = false
        session.abortController.value = null
        session.flush = null
        session.stop = null
        session.onUpdate?.()
    }

    const synchronize = () => {
        if (syncRequest) return syncRequest
        clearTimeout(session.syncTimer)
        session.syncTimer = null
        session.synchronizing.value = true
        syncRequest = (async () => {
            try {
                const data = await fetchSessionMessages(session.id, {timeout: 10_000})
                if (!isCurrent() || session.messages.value !== local) return false
                const serverUserIndex = userMsg.id ? data.findIndex(msg => msg.id === userMsg.id) : userIndex
                const serverUser = data[serverUserIndex]
                const serverAssistant = data[serverUserIndex + 1]
                if (serverUser?.role === 'user' && serverUser.content === userMsg.content) {
                    userMsg.id = serverUser.id
                    if (['completed', 'error'].includes(serverUser.status)
                        && serverAssistant?.role === 'assistant'
                        && ['completed', 'error'].includes(serverAssistant.status)) {
                        const aligned = data.length === local.length && data.every((msg, i) =>
                            msg.role === local[i].role && (!local[i].id || local[i].id === msg.id))
                        if (aligned) {
                            data.forEach((msg, i) => { local[i].id = msg.id })
                        } else {
                            session.messages.value = buildMessages(data)
                        }
                        session.keepOnLeave = false
                        session.synchronize = null
                        return true
                    }
                }
                const prefixMatches = data.length === userIndex && local.slice(0, userIndex).every(
                    (msg, i) => msg.id && msg.id === data[i]?.id)
                if (!userMsg.id && !receivedOutput && prefixMatches) {
                    // 请求未落库：保留草稿轮次供手动重试，后续新问题不再附加一个假历史轮次。
                    session.unpersistedRound = {userMsg, assistantMsg}
                    session.synchronize = null
                    return true
                }
                return false
            } catch (e) {
                console.error('同步上一轮结果失败', e)
                return false
            } finally {
                if (isCurrent()) session.synchronizing.value = false
                syncRequest = null
            }
        })()
        return syncRequest
    }

    session.stop = () => {
        if (!isLive()) return
        const controller = session.abortController.value
        session.keepOnLeave = true
        session.synchronize = synchronize
        // abort 不触发 fetchEventSource 的 onerror/onclose，必须主动冲刷并收尾。
        finish()
        controller?.abort()
        session.syncTimer = setTimeout(() => {
            session.syncTimer = null
            if (isCurrent()) synchronize()
        }, 500)
    }

    session.dispose = () => {
        finished = true
        clearTimeout(flushTimer)
        clearTimeout(session.syncTimer)
        clearInterval(waitTimer)
        session.onUpdate = null
        session.abortController.value?.abort()
    }

    return {
        onOpen: () => {
            if (!isLive()) return
            lastEventAt = Date.now()
            assistantMsg.streamPhase = 'waiting'
        },
        onMessage: (data) => {
            if (!isLive()) return
            lastEventAt = Date.now()
            session.streamStarted.value = true
            if (data.type === 'content') {
                receivedOutput = receivedOutput || !!data.content
                assistantMsg.streamPhase = 'generating'
                assistantMsg.loading = false
                contentBuffer += data.content
                scheduleFlush()
            } else if (data.type === 'thinking') {
                receivedOutput = receivedOutput || !!data.content
                assistantMsg.streamPhase = 'thinking'
                assistantMsg.loading = false
                thinkingBuffer += data.content
                scheduleFlush()
            } else if (data.type === 'process') {
                assistantMsg.loading = false
                if (!assistantMsg.ragProcess) assistantMsg.ragProcess = []
                const processInfo = data.payload
                const phases = {begin: 'analyzing', context_construction: 'context', generation: 'generating'}
                assistantMsg.streamPhase = phases[processInfo.step] || 'retrieving'
                const existingIndex = assistantMsg.ragProcess.findIndex(
                    p => p.step === processInfo.step && p.status === processInfo.status
                )
                if (existingIndex !== -1) {
                    assistantMsg.ragProcess[existingIndex] = processInfo
                } else {
                    assistantMsg.ragProcess.push(processInfo)
                }
                scheduleFlush()
            } else if (data.type === 'done') {
                if (userMsg && data.userMessageId) userMsg.id = data.userMessageId
                if (data.assistantMessageId) assistantMsg.id = data.assistantMessageId
                finish()
            } else if (data.type === 'usage' && data.payload) {
                if (data.payload.latency_ms != null) assistantMsg.latencyMs = data.payload.latency_ms
                if (data.payload.completion_tokens != null) assistantMsg.completionTokens = data.payload.completion_tokens
                if (data.payload.total_tokens != null) assistantMsg.totalTokens = data.payload.total_tokens
            }
        },
        onError: (err) => {
            if (!isLive()) return
            session.keepOnLeave = true
            session.synchronize = synchronize
            finish('error', err?.message || '网络异常，请检查连接后重试')
        },
        onClose: () => finish()
    }
}
