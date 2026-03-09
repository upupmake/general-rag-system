import commonApi from './commonApi'
import {API_BASE_URL} from "@/consts.js";
import {useUserStore} from "@/stores/user.js";
import {fetchEventSource} from "@microsoft/fetch-event-source"
import {message} from "ant-design-vue"

export function fetchAvailableModels() {
    return commonApi.get('/models/available')
}

export function startChat({modelId, question, kbId, options}) {
    return commonApi.post('/chat/start', {
        modelId,
        question,
        kbId,
        options
    })
}

export function fetchSessionMessages(sessionId) {
    return commonApi.get(`/chat/sessions/${sessionId}/messages`)
}

export function fetchSessions({lastActiveAt, lastId, pageSize = 20}) {
    return commonApi.post('/sessions/list', {
        lastActiveAt, lastId, pageSize
    })
}

export function searchSessions({keyword, limit, offset}) {
    return commonApi.post('/sessions/search', {
        keyword, limit, offset
    })
}

export function deleteSession(sessionId) {
    return commonApi.delete(`/sessions/${sessionId}`)
}

export function fetchSessionTitle(sessionId) {
    return commonApi.get(`/sessions/${sessionId}/title`)
}

/**
 * 通用流式请求处理函数
 */
function streamRequest(url, body, onOpen, onMessage, onError, onClose, logTag = "Stream") {
    fetchEventSource(url, {
        method: "POST",
        headers: {
            "Authorization": `Bearer ${useUserStore().token}`,
            "Content-Type": "application/json"
        },
        body: JSON.stringify(body),
        onopen(response) {
            if (response.ok) {
                console.log(`${logTag} SSE connected`)
                if (onOpen) onOpen(response)
                return
            }
            message.error("今日Token使用已达上限 或 网络发生错误").then()
            throw new Error(`${logTag} SSE Connection failed`)
        },
        onmessage(ev) {
            try {
                let json = JSON.parse(ev.data)
                if (json.type === 'content' || json.type === 'thinking') {
                    if (onMessage) onMessage({type: json.type, content: json.content})
                } else if (json.type === 'process') {
                    if (onMessage) onMessage({type: 'process', payload: json.payload})
                } else if (json.type === 'done') {
                    if (onMessage) onMessage({
                        type: 'done',
                        userMessageId: json.userMessageId,
                        assistantMessageId: json.assistantMessageId
                    })
                } else if (json.type === "usage") {
                    if (onMessage) onMessage({
                        type: 'usage',
                        payload: json.payload
                    })
                }
            } catch (e) {
                console.error(`${logTag} JSON Parse Error`, e)
            }
        },
        onclose() {
            console.log(`${logTag} SSE closed`)
            if (onClose) onClose()
        },
        onerror(err) {
            console.error(`${logTag} error`, err)
            if (onError) onError(err)
            throw err
        },
        openWhenHidden: true
    }).then()
}

export function startChatStream(sessionId, modelId, question, kbId, options, onOpen, onMessage, onError, onClose) {
    const body = {
        sessionId: sessionId,
        modelId: modelId,
        question: question,
        kbId: kbId,
        options: options
    }
    streamRequest(`${API_BASE_URL}/chat/stream`, body, onOpen, onMessage, onError, onClose, "Chat")
}

/**
 * 编辑最后一轮用户问题并重新生成回复
 */
export function editMessageStream(messageId, sessionId, modelId, kbId, newContent, options, onOpen, onMessage, onError, onClose) {
    const body = {
        sessionId: sessionId,
        modelId: modelId,
        kbId: kbId,
        newContent: newContent,
        options: options
    }
    streamRequest(`${API_BASE_URL}/chat/messages/${messageId}/edit`, body, onOpen, onMessage, onError, onClose, "Edit")
}

/**
 * 重试最后一轮AI回复
 */
export function retryMessageStream(userMessageId, sessionId, modelId, kbId, options, onOpen, onMessage, onError, onClose) {
    const body = {
        sessionId: sessionId,
        modelId: modelId,
        kbId: kbId,
        options: options
    }
    streamRequest(`${API_BASE_URL}/chat/messages/${userMessageId}/retry`, body, onOpen, onMessage, onError, onClose, "Retry")
}
