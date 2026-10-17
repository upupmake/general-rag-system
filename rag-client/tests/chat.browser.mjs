import assert from 'node:assert/strict'
import {setTimeout as sleep} from 'node:timers/promises'
import {createServer} from 'vite'

// 仅访问本地模拟 API；通过 web-access CDP Proxy 操作新建测试标签页。
const sessions = new Map(['1001', '1002'].map((id, i) => [id, [
    {id: i * 100 + 1, role: 'user', content: `初始问题${id}`, status: 'completed', modelId: 1},
    {id: i * 100 + 2, role: 'assistant', content: `初始回答${id}`, status: 'completed', modelId: 1}
]]))
const requests = []
const connections = []
let nextId = 1000
let failMessages = false
let target
const timers = new Set()
const json = (response, data) => {
    response.setHeader('Content-Type', 'application/json')
    response.end(JSON.stringify({code: 200, data}))
}

const server = await createServer({
    server: {host: '127.0.0.1', port: 0, open: false},
    plugins: [{
        name: 'local-chat-verification',
        enforce: 'pre',
        transform(source, id) {
            if (id.replaceAll('\\', '/').endsWith('/src/consts.js')) {
                return 'export const API_BASE_URL = "/mock-api"'
            }
        },
        configureServer(vite) {
            vite.middlewares.use(async (request, response, next) => {
                const path = request.url.split('?')[0]
                if (!path.startsWith('/mock-api/')) return next()
                requests.push({method: request.method, path})
                if (path === '/mock-api/models/available') {
                    return json(response, [{modelId: 1, modelName: '本地测试模型', provider: 'test', kbSupported: true, metadata: {tools: [], thinking: {default: false, editable: true}}}])
                }
                if (path === '/mock-api/kb') return json(response, {private: [], shared: [], public: [], invited: []})
                if (path === '/mock-api/users/me') return json(response, {id: 1, username: '本地测试', role: {id: 1, weight: 1}})
                if (path === '/mock-api/workspaces') return json(response, {current: {id: 1, name: '本地测试空间'}, owned: [], joined: []})
                if (path === '/mock-api/sessions/list') return json(response, {groups: [], hasMore: false})
                const messageMatch = path.match(/^\/mock-api\/chat\/sessions\/(\d+)\/messages$/)
                if (messageMatch) {
                    if (failMessages && messageMatch[1] === '1002') {
                        response.statusCode = 500
                        return json(response, null)
                    }
                    return json(response, sessions.get(messageMatch[1]) || [])
                }
                if (path === '/mock-api/chat/stream') {
                    let raw = ''
                    for await (const chunk of request) raw += chunk
                    const body = JSON.parse(raw)
                    const rows = sessions.get(String(body.sessionId))
                    const user = {id: ++nextId, role: 'user', content: body.question, status: 'generating', modelId: 1}
                    rows.push(user)
                    const answer = {id: ++nextId, role: 'assistant', content: '', status: 'completed', modelId: 1}
                    const connection = {sessionId: String(body.sessionId), aborted: false, fragments: 0}
                    connections.push(connection)
                    let ended = false
                    let interval
                    response.writeHead(200, {'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache'})
                    response.flushHeaders()
                    const send = data => response.write(`data: ${JSON.stringify(data)}\n\n`)
                    const done = () => {
                        ended = true
                        clearInterval(interval)
                        user.status = 'completed'
                        rows.push(answer)
                        send({type: 'done', userMessageId: user.id, assistantMessageId: answer.id})
                        response.end()
                    }
                    const delay = setTimeout(() => {
                        if (ended) return
                        const chunks = body.question === '等待取消' ? ['不应收到的内容']
                            : Array.from({length: 80}, (_, i) => `第${i + 1}段：\\(x_${i + 1}^2\\)。\n\n`)
                        interval = setInterval(() => {
                            const content = chunks.shift()
                            if (!content) return done()
                            connection.fragments++
                            answer.content += content
                            send({type: 'content', content})
                        }, 40)
                        timers.add(interval)
                    }, body.question === '等待取消' ? 30_000 : 50)
                    timers.add(delay)
                    response.on('close', () => {
                        clearTimeout(delay)
                        clearInterval(interval)
                        if (!ended) {
                            ended = true
                            connection.aborted = true
                            user.status = 'completed'
                            rows.push(answer)
                        }
                    })
                    return
                }
                json(response, {})
            })
        }
    }]
})

const proxy = async (path, body) => {
    const response = await fetch(`http://localhost:3456/${path}`, body === undefined ? {} : {method: 'POST', body})
    const result = await response.json()
    if (!response.ok || result.error) throw new Error(JSON.stringify(result))
    return result
}
const evaluate = async code => (await proxy(`eval?target=${target}`, code)).value
const until = async (code, label) => {
    const deadline = Date.now() + 15_000
    while (Date.now() < deadline) {
        try {
            if (await evaluate(code)) return
        } catch (e) {
            // 页面导航期间执行上下文会被替换，重试即可。
        }
        await sleep(150)
    }
    throw new Error(`${label}: ${await evaluate('location.href + "\\n" + document.title + "\\n" + document.body.innerText.slice(-2000)')}`)
}
const route = path => evaluate(`(async () => {
    const router = (await import('/src/router/index.js')).default
    await router.push(${JSON.stringify(path)})
    return true
})()`)
const stopButton = `[...document.querySelectorAll('button')].find(button => button.textContent.trim() === '停止' || button.getAttribute('aria-label') === '停止生成')`
const send = async text => {
    await evaluate('document.querySelector(".input-collapsed-bar")?.click()')
    await until('!!document.querySelector(".chat-sender textarea")', '输入框未显示')
    await evaluate(`(() => {
        const textarea = document.querySelector('.chat-sender textarea')
        Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(textarea, ${JSON.stringify(text)})
        textarea.dispatchEvent(new Event('input', {bubbles: true}))
        return true
    })()`)
    const clicked = await evaluate(`(() => {
        const button = document.querySelector('.sender-footer button.ant-btn-primary')
        if (!button) return 'missing'
        button.click()
        return 'clicked'
    })()`)
    assert.equal(clicked, 'clicked')
}

try {
    await server.listen()
    const {port} = server.httpServer.address()
    const base = `http://127.0.0.1:${port}`
    target = (await proxy('new', `${base}/login`)).targetId
    await sleep(500)
    await until('!!document.querySelector("#app")?.__vue_app__', '应用未加载')
    await evaluate(`(async () => {
        const store = (await import('/src/stores/user.js')).useUserStore()
        store.token = 'local-browser-test'
        store.user = {id: 1, username: '本地测试'}
        return true
    })()`)
    await route('/chat/1001')
    await until('!!document.querySelector(".input-collapsed-bar") && document.body.innerText.includes("初始回答1001")', '聊天未就绪')

    await send('等待取消')
    await until(`!!(${stopButton}) && !!document.querySelector(".assistant-activity")`, '等待阶段没有停止入口或阶段提示')
    await until('/[1-9]\\d* 秒/.test(document.querySelector(".activity-elapsed")?.textContent || "")', '等待时长未更新')
    await evaluate(`(${stopButton}).click()`)
    await until(`!(${stopButton})`, '停止按钮未收起')
    await sleep(650)
    assert.equal(connections[0].aborted, true)
    assert.equal(connections[0].fragments, 0)
    console.log('PASS: 首事件前可见等待时长，点击停止真实关闭本地 SSE 连接。')

    await send('公式长回答')
    await until('document.body.innerText.includes("第1段")', '正文未开始')
    await route('/kb')
    await until('document.body.innerText.includes("知识库管理")', '离开聊天页失败')
    await sleep(300)
    assert.equal(connections[1].aborted, false)
    await route('/chat/1001')
    await until(`document.body.innerText.includes("第1段") && !!(${stopButton})`, '切回未恢复生成中的回答')
    await evaluate(`(() => {
        window.__answerUpdates = 0
        window.__observer = new MutationObserver(() => window.__answerUpdates++)
        const answers = document.querySelectorAll('.message-content')
        window.__observer.observe(answers[answers.length - 1], {childList: true, subtree: true, characterData: true})
        return true
    })()`)
    await until(`document.body.innerText.includes("第80段") && !(${stopButton})`, '完整回答未刷新或未结束')
    assert.equal(connections.length, 2)
    assert.equal(connections[1].fragments, 80)
    assert.equal(await evaluate('document.querySelectorAll(".message-content svg").length > 0'), true)
    const updates = await evaluate('window.__observer.disconnect(); window.__answerUpdates')
    console.log(`PASS: 离开后原连接继续，切回未重发；80个公式分片完整渲染，观察到${updates}次正文容器变更。`)

    failMessages = true
    await route('/chat/1002')
    await until('document.body.innerText.includes("会话加载失败") && !!document.querySelector(".session-load-error button")', '加载失败没有重试入口')
    failMessages = false
    await evaluate('document.querySelector(".session-load-error button").click()')
    await until('document.body.innerText.includes("初始回答1002") && !document.querySelector(".session-load-error")', '加载重试未恢复')
    console.log('PASS: 接口500后加载状态结束，可手动重试恢复。')
    assert.equal(requests.filter(item => item.path === '/mock-api/models/available').length, 1)
    // 一次来自聊天资源缓存，一次来自知识库管理页自身的列表刷新。
    assert.equal(requests.filter(item => item.path === '/mock-api/kb').length, 2)
    console.log('PASS: 聊天多次切换只拉取一次模型列表；知识库页独立刷新自身列表。无真实后端或模型调用。')
} finally {
    if (target) await proxy(`close?target=${target}`)
    timers.forEach(timer => clearTimeout(timer))
    await server.close()
}
