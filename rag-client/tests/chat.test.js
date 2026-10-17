import assert from 'node:assert/strict'
import {beforeEach, afterEach, test} from 'node:test'
import {registerHooks} from 'node:module'
import {execFileSync} from 'node:child_process'
import {relative} from 'node:path'
import {fileURLToPath} from 'node:url'
import {computed, effectScope, nextTick, ref, watch} from 'vue'

const sourceRoot = new URL('../src/', import.meta.url)
const repoRoot = new URL('../../', import.meta.url)
const fixtures = globalThis.chatTestFixtures = {store: {token: ''}}
const stub = (source) => ({url: `data:text/javascript,${encodeURIComponent(source)}`, shortCircuit: true})

registerHooks({
    resolve(specifier, context, nextResolve) {
        if (specifier.endsWith('.css')) return stub('')
        if (specifier === '@/api/chatApi') {
            return stub(['fetchAvailableModels', 'fetchSessionMessages', 'startChatStream', 'editMessageStream', 'retryMessageStream']
                .map(name => `export const ${name} = (...args) => globalThis.chatTestFixtures.${name}(...args)`).join('\n'))
        }
        if (specifier === '@/api/kbApi') {
            return stub('export const fetchAvailableKbs = () => globalThis.chatTestFixtures.fetchAvailableKbs()')
        }
        if (specifier === '@/stores/user.js') {
            return stub('export const useUserStore = () => globalThis.chatTestFixtures.store')
        }
        if (specifier === 'ant-design-vue') {
            return stub('export const message = {success() {}, warning() {}, error() {}}')
        }
        if (specifier === './commonApi') {
            return stub('export default Object.fromEntries(["get", "post", "put", "delete"].map(method => [method, (...args) => globalThis.chatTestFixtures.commonApi[method](...args)]))')
        }
        if (specifier.startsWith('@/')) {
            const path = specifier.slice(2)
            return nextResolve(new URL(path.endsWith('.js') ? path : `${path}.js`, sourceRoot).href, context)
        }
        return nextResolve(specifier, context)
    },
    load(url, context, nextLoad) {
        const loaded = nextLoad(url, context)
        if (!url.startsWith(sourceRoot.href)) return loaded
        let source = String(loaded.source)
        // 可选地用 HEAD 中的旧实现复现回归，不改写工作区文件。
        if (process.env.CHAT_TEST_BASELINE === '1' && !url.endsWith('/chatSessions.js')) {
            const path = relative(fileURLToPath(repoRoot), fileURLToPath(url)).replaceAll('\\', '/')
            source = execFileSync('git', ['show', `HEAD:${path}`], {cwd: fileURLToPath(repoRoot), encoding: 'utf8'})
        }
        if (url.endsWith('/vars.js')) {
            // import.meta.glob 由 Vite 提供；测试不需要加载品牌图片。
            source = source.replace(/^const providerLogoModules = .*$/m, 'const providerLogoModules = {}')
        }
        return {...loaded, source}
    }
})

const {useChat} = await import('../src/views/chat/composables/useChat.js')
const {useTools} = await import('../src/views/chat/composables/useTools.js')
const {useMessageEdit} = await import('../src/views/chat/composables/useMessageEdit.js')
const {useScroll} = await import('../src/views/chat/composables/useScroll.js')
const {md, renderCached} = await import('../src/views/chat/markdown.js')
const {getChatSession} = await import('../src/views/chat/composables/chatSessions.js')
const {events} = await import('../src/events.js')
const kbApi = await import('../src/api/kbApi.js')
const chatApi = await import('../src/api/chatApi.js')
const {loadChatResources, models, kbs, selectedModel, selectedKb, contextMultiplier} = await import('../src/vars.js')
let scopes = []
let serial = 0

const deferred = () => {
    let resolve, reject
    const promise = new Promise((res, rej) => { resolve = res; reject = rej })
    return {promise, resolve, reject}
}

const settle = async () => {
    for (let i = 0; i < 12; i++) await Promise.resolve()
    await nextTick()
}

const history = (text, modelId = 1) => [
    {id: 101, role: 'user', content: text, status: 'completed', modelId},
    {id: 102, role: 'assistant', content: `${text}回答`, status: 'completed', modelId}
]

function createView(id = 'A') {
    const scope = effectScope()
    scopes.push(scope)
    return scope.run(() => {
        const sessionId = ref(id)
        const tools = useTools()
        const userScrolledUp = ref(false)
        const scrolls = []
        const scrollToBottom = (...args) => scrolls.push(args)
        const isKbSupported = computed(() => !!tools.currentModel.value?.kbSupported)
        const chat = useChat(sessionId, tools.selectedTools, tools.thinkingEnabled, tools.currentModel,
            isKbSupported, tools.isRestoring, tools.isUserUncheckedWebSearch, tools.availableTools,
            userScrolledUp, scrollToBottom)
        const edit = useMessageEdit(chat.messages, sessionId, selectedModel, selectedKb, isKbSupported,
            tools.selectedTools, tools.thinkingEnabled, tools.currentModel,
            userScrolledUp, scrollToBottom, chat.handleStreamCallbacks, chat.abortController, chat.prepareGeneration)
        return {
            ...chat, tools, edit, scope, sessionId, scrolls, userScrolledUp,
            async load(id = sessionId.value) {
                sessionId.value = id
                await chat.loadSession(id)
            }
        }
    })
}

beforeEach(() => {
    fixtures.store.token = `test-token-${++serial}`
    getChatSession('reset')
    models.value = []
    kbs.value = {private: [], shared: [], public: [], invited: []}
    selectedModel.value = null
    selectedKb.value = null
    contextMultiplier.value = null
    scopes = []
    fixtures.calls = []
    fixtures.streams = []
    fixtures.modelData = [
        {modelId: 1, provider: 'test', kbSupported: true, metadata: {tools: ['webSearch'], thinking: {editable: true, default: false}}},
        {modelId: 2, provider: 'test', kbSupported: true, metadata: {tools: [], thinking: {editable: true, default: false}}}
    ]
    fixtures.kbData = {private: [{id: 51}, {id: 52}]}
    fixtures.modelRequest = async () => fixtures.modelData
    fixtures.kbRequest = async () => fixtures.kbData
    fixtures.messageRequest = async () => []
    fixtures.commonApi = {post: async () => ({}), put: async () => ({}), delete: async () => ({})}
    fixtures.fetchAvailableModels = () => {
        fixtures.calls.push('models')
        return fixtures.modelRequest()
    }
    fixtures.fetchAvailableKbs = () => {
        fixtures.calls.push('kbs')
        return fixtures.kbRequest()
    }
    fixtures.fetchSessionMessages = id => {
        fixtures.calls.push(`messages:${id}`)
        return fixtures.messageRequest(id)
    }
    for (const type of ['startChatStream', 'editMessageStream', 'retryMessageStream']) {
        fixtures[type] = (...args) => {
            const [onOpen, onMessage, onError, onClose] = args.slice(-4)
            const controller = new AbortController()
            fixtures.streams.push({type, args, onOpen, onMessage, onError, onClose, controller})
            return controller
        }
    }
})

afterEach(() => {
    scopes.forEach(scope => scope.stop())
    fixtures.store.token = `disposed-${serial}`
    getChatSession('reset')
})

test('cancels before the first event and retains an unpersisted question', async t => {
    t.mock.timers.enable({apis: ['setTimeout', 'setInterval', 'Date']})
    const view = createView()
    await view.load()
    view.onSend('还未连接的问题')
    const stream = fixtures.streams[0]
    assert.equal(view.isGenerating.value, true)
    assert.equal(view.streamStarted.value, false)
    assert.equal(view.lastAssistantMessage.value.loading, false)
    view.stopGeneration()
    view.stopGeneration()
    assert.equal(stream.controller.signal.aborted, true)
    assert.equal(view.isGenerating.value, false)
    assert.equal(view.lastUserMessage.value.status, 'completed')
    stream.onMessage({type: 'content', content: '迟到分片'})
    stream.onClose()
    t.mock.timers.tick(1500)
    await settle()
    assert.equal(view.messages.value.length, 2)
    assert.equal(view.lastUserMessage.value.content, '还未连接的问题')
    assert.equal(view.lastAssistantMessage.value.content, '')
    assert.equal(view.lastAssistantMessage.value.elapsedSeconds, 0)
    // Bubble.List 会复制消息对象；尚无服务端 ID 时仍需展示编辑/重试入口。
    assert.equal(view.canEditOrRetry.value, true)
    assert.equal(view.isLastUserMessage({...view.lastUserMessage.value}), true)
    assert.equal(view.isLastAssistantMessage({...view.lastAssistantMessage.value}), true)
})

test('batches a burst of formula fragments into one update every 80ms', async t => {
    t.mock.timers.enable({apis: ['setTimeout', 'setInterval', 'Date']})
    const view = createView()
    await view.load()
    view.onSend('公式')
    const stream = fixtures.streams[0]
    let updates = 0
    const unwatch = watch(() => view.lastAssistantMessage.value.content, () => updates++, {flush: 'sync'})
    view.scrolls.length = 0
    const text = String.raw`\[\sum_{i=1}^{1000} x_i^2\]` + '推导'.repeat(500)
    for (const content of text) stream.onMessage({type: 'content', content})
    assert.equal(view.lastAssistantMessage.value.content, '')
    t.mock.timers.tick(79)
    assert.equal(updates, 0)
    t.mock.timers.tick(1)
    assert.equal(view.lastAssistantMessage.value.content, text)
    assert.equal(updates, 1)
    assert.equal(view.scrolls.length, 1)
    stream.onMessage({type: 'content', content: '尾部'})
    stream.onMessage({type: 'done', userMessageId: 201, assistantMessageId: 202})
    assert.equal(view.lastAssistantMessage.value.content, text + '尾部')
    assert.equal(updates, 2)
    assert.equal(view.lastUserMessage.value.id, 201)
    assert.equal(view.lastAssistantMessage.value.id, 202)
    assert.equal(view.isGenerating.value, false)
    unwatch()
})

test('reuses Markdown output when Bubble.List copies unchanged historical messages', async t => {
    fixtures.messageRequest = async () => history('历史公式')
    const view = createView()
    await view.load()
    const render = t.mock.method(md, 'render', source => `<p>${source}</p>`)
    const msg = view.lastAssistantMessage.value
    const first = renderCached({...msg}, 'content', msg.content)
    const second = renderCached({...msg}, 'content', msg.content)
    assert.equal(first, second)
    assert.equal(render.mock.callCount(), 1)
    renderCached({...msg}, 'content', msg.content + '新增')
    assert.equal(render.mock.callCount(), 2)
})

for (const terminal of ['done', 'close', 'error', 'stop']) {
    test(`flushes buffered thinking and content immediately on ${terminal}`, async t => {
        t.mock.timers.enable({apis: ['setTimeout', 'setInterval', 'Date']})
        const view = createView()
        await view.load()
        view.onSend('测试')
        const stream = fixtures.streams[0]
        stream.onMessage({type: 'thinking', content: '完整思考'})
        stream.onMessage({type: 'content', content: '完整正文'})
        if (terminal === 'done') stream.onMessage({type: 'done'})
        if (terminal === 'close') stream.onClose()
        if (terminal === 'error') stream.onError(new Error('服务异常'))
        if (terminal === 'stop') view.stopGeneration()
        assert.equal(view.lastAssistantMessage.value.thinking, '完整思考')
        assert.match(view.lastAssistantMessage.value.content, /^完整正文/)
        assert.deepEqual(view.lastAssistantMessage.value.thinkingCollapseKeys, [])
        assert.equal(view.lastAssistantMessage.value.streamPhase, null)
        assert.equal(view.isGenerating.value, false)
        t.mock.timers.tick(2000)
        await settle()
        assert.equal(view.lastAssistantMessage.value.elapsedSeconds, 0)
    })
}

test('uses real progress events for phases and tracks long waits without resending', async t => {
    t.mock.timers.enable({apis: ['setTimeout', 'setInterval', 'Date'], now: 1000})
    const view = createView()
    await view.load()
    view.onSend('检索问题')
    const stream = fixtures.streams[0]
    assert.equal(view.lastAssistantMessage.value.streamPhase, 'connecting')
    t.mock.timers.tick(21_000)
    assert.equal(view.lastAssistantMessage.value.elapsedSeconds, 21)
    assert.equal(view.lastAssistantMessage.value.idleSeconds, 21)
    stream.onOpen()
    assert.equal(view.lastAssistantMessage.value.streamPhase, 'waiting')
    for (const [step, phase] of [['begin', 'analyzing'], ['round_1', 'retrieving'], ['context_construction', 'context'], ['generation', 'generating']]) {
        stream.onMessage({type: 'process', payload: {step, status: 'running'}})
        assert.equal(view.lastAssistantMessage.value.streamPhase, phase)
    }
    t.mock.timers.tick(1000)
    assert.equal(view.lastAssistantMessage.value.idleSeconds, 1)
    stream.onMessage({type: 'usage', payload: {total_tokens: 123, completion_tokens: 100, latency_ms: 250}})
    assert.equal(view.lastAssistantMessage.value.totalTokens, 123)
    assert.equal(fixtures.streams.length, 1)
    stream.onMessage({type: 'done'})
})

test('keeps generation alive in the background and reattaches without a second request', async t => {
    t.mock.timers.enable({apis: ['setTimeout', 'setInterval', 'Date']})
    const view = createView()
    await view.load()
    selectedModel.value = 1
    await nextTick()
    selectedKb.value = 51
    view.tools.selectedTools.value = ['webSearch']
    view.tools.thinkingEnabled.value = true
    view.onSend('A的问题')
    const streamA = fixtures.streams[0]
    streamA.onMessage({type: 'content', content: '前半段'})
    await view.load('B')
    assert.equal(streamA.controller.signal.aborted, false)
    assert.equal(view.isGenerating.value, false)
    assert.equal(view.messages.value.length, 0)
    selectedModel.value = 2
    await nextTick()
    view.onSend('B的问题')
    const streamB = fixtures.streams[1]
    view.scrolls.length = 0
    streamA.onMessage({type: 'content', content: '后台内容'})
    t.mock.timers.tick(160)
    assert.equal(view.lastUserMessage.value.content, 'B的问题')
    assert.equal(view.lastAssistantMessage.value.content, '')
    assert.equal(view.scrolls.length, 0)
    await view.load('A')
    assert.equal(view.lastAssistantMessage.value.content, '前半段后台内容')
    assert.equal(view.isGenerating.value, true)
    assert.equal(selectedModel.value, 1)
    assert.equal(selectedKb.value, 51)
    assert.deepEqual(view.tools.selectedTools.value, ['webSearch'])
    assert.equal(view.tools.thinkingEnabled.value, true)
    assert.equal(fixtures.streams.length, 2)
    assert.equal(fixtures.calls.filter(call => call === 'messages:A').length, 1)
    view.stopGeneration()
    assert.equal(streamA.controller.signal.aborted, true)
    assert.equal(streamB.controller.signal.aborted, false)
    await view.load('B')
    assert.equal(view.isGenerating.value, true)
    streamA.onClose()
    assert.equal(view.isGenerating.value, true)
    streamB.onMessage({type: 'done'})
})

for (const completedWhileAway of [false, true]) {
    test(`survives page unmount and restores a ${completedWhileAway ? 'completed' : 'live'} background answer`, async t => {
        t.mock.timers.enable({apis: ['setTimeout', 'setInterval', 'Date']})
        const first = createView()
        await first.load()
        first.onSend('离开页面')
        const stream = fixtures.streams[0]
        first.scope.stop()
        first.scrolls.length = 0
        stream.onMessage({type: 'content', content: '后台接收'})
        if (completedWhileAway) stream.onMessage({type: 'done', userMessageId: 301, assistantMessageId: 302})
        const second = createView()
        await second.load()
        assert.equal(stream.controller.signal.aborted, false)
        assert.equal(second.lastAssistantMessage.value.content, '后台接收')
        assert.equal(second.isGenerating.value, !completedWhileAway)
        assert.equal(first.scrolls.length, 0)
        assert.equal(fixtures.streams.length, 1)
        assert.equal(fixtures.calls.filter(call => call === 'messages:A').length, 1)
        stream.onMessage({type: 'done'})
    })
}

test('loads resources and messages in parallel and ignores an obsolete pending session', async () => {
    const model = deferred(), kb = deferred(), a = deferred(), b = deferred()
    fixtures.modelRequest = () => model.promise
    fixtures.kbRequest = () => kb.promise
    fixtures.messageRequest = id => id === 'A' ? a.promise : b.promise
    const view = createView()
    const loadA = view.load('A')
    const loadB = view.load('B')
    assert.deepEqual(fixtures.calls, ['models', 'kbs', 'messages:A', 'messages:B'])
    model.resolve(fixtures.modelData)
    kb.resolve(fixtures.kbData)
    b.resolve(history('B', 2))
    await loadB
    assert.equal(view.lastUserMessage.value.content, 'B')
    a.resolve([{id: 99, role: 'user', content: '过期的A', status: 'pending', modelId: 1}])
    await loadA
    assert.equal(view.lastUserMessage.value.content, 'B')
    assert.equal(selectedModel.value, 2)
    assert.equal(view.loading.value, false)
    assert.equal(fixtures.streams.length, 0)
})

for (const oldFails of [false, true]) {
    test(`an old ${oldFails ? 'failure' : 'success'} cannot end a newer session load`, async () => {
        const a = deferred(), b = deferred()
        fixtures.messageRequest = id => id === 'A' ? a.promise : b.promise
        const view = createView()
        const loadA = view.load('A'), loadB = view.load('B')
        if (oldFails) a.reject(new Error('旧会话失败'))
        else a.resolve(history('旧会话'))
        await loadA
        assert.equal(view.loading.value, true)
        assert.equal(view.tools.isRestoring.value, true)
        assert.equal(view.loadError.value, '')
        b.resolve(history('B'))
        await loadB
        assert.equal(view.loading.value, false)
        assert.equal(view.tools.isRestoring.value, false)
    })
}

for (const failingResource of ['models', 'kbs', 'messages']) {
    test(`finishes loading on ${failingResource} failure and supports an explicit retry`, async () => {
        const property = {models: 'modelRequest', kbs: 'kbRequest', messages: 'messageRequest'}[failingResource]
        const original = fixtures[property]
        fixtures[property] = async () => { throw new Error(`${failingResource}失败`) }
        const view = createView()
        await view.load()
        assert.equal(view.loading.value, false)
        assert.equal(view.tools.isRestoring.value, false)
        assert.match(view.loadError.value, /失败/)
        view.question.value = '尚未发送的草稿'
        view.onSend(view.question.value)
        assert.equal(fixtures.streams.length, 0)
        assert.equal(view.question.value, '尚未发送的草稿')
        fixtures[property] = original
        await view.load()
        assert.equal(view.loadError.value, '')
        view.onSend(view.question.value)
        assert.equal(fixtures.streams.length, 1)
        fixtures.streams[0].onMessage({type: 'done'})
    })
}

test('ignores message loads that finish after the view has unmounted', async () => {
    const pending = deferred()
    fixtures.messageRequest = () => pending.promise
    const view = createView()
    const loading = view.load()
    view.scope.stop()
    pending.resolve([{role: 'user', content: '不应续跑', status: 'pending', modelId: 1}])
    await loading
    assert.equal(fixtures.streams.length, 0)
    assert.equal(view.scrolls.length, 0)
})

test('stopped-message synchronization cannot overwrite another session and gates the next generation', async t => {
    t.mock.timers.enable({apis: ['setTimeout', 'setInterval', 'Date']})
    const view = createView()
    await view.load()
    view.onSend('第一轮')
    fixtures.streams[0].onMessage({type: 'content', content: '部分回答'})
    view.stopGeneration()
    const sync = deferred()
    fixtures.messageRequest = id => id === 'A' ? sync.promise : Promise.resolve(history('B'))
    t.mock.timers.tick(500)
    await view.load('B')
    sync.resolve(history('第一轮'))
    await settle()
    assert.equal(view.lastUserMessage.value.content, 'B')
    await view.load('A')
    await view.onSend('第二轮')
    assert.equal(view.messages.value.length, 4)
    assert.equal(view.lastUserMessage.value.content, '第二轮')
    assert.equal(view.lastAssistantMessage.value.id, undefined)
    assert.equal(view.isGenerating.value, true)
    fixtures.streams[1].onMessage({type: 'done'})
})

test('retains a stopped pending question across switches without automatically restarting it', async t => {
    t.mock.timers.enable({apis: ['setTimeout', 'setInterval', 'Date']})
    const view = createView()
    await view.load()
    view.onSend('已取消')
    view.stopGeneration()
    fixtures.messageRequest = async id => id === 'A'
        ? [{id: 401, role: 'user', content: '已取消', status: 'pending', modelId: 1}] : []
    t.mock.timers.tick(500)
    await settle()
    await view.load('B')
    await view.load('A')
    assert.equal(view.lastUserMessage.value.id, 401)
    assert.equal(view.lastUserMessage.value.status, 'completed')
    assert.equal(view.messages.value.length, 2)
    assert.equal(view.isGenerating.value, false)
    assert.equal(fixtures.streams.length, 1)
})

for (const action of ['retry', 'edit']) {
    test(`${action} after early cancellation uses normal send when no message ID exists`, async t => {
        t.mock.timers.enable({apis: ['setTimeout', 'setInterval', 'Date']})
        const view = createView()
        await view.load()
        view.onSend('原问题')
        view.stopGeneration()
        if (action === 'retry') await view.edit.onRetryFromAssistant()
        else {
            view.edit.startEditByItem(view.lastUserMessage.value)
            view.edit.editingContent.value = '修改后的问题'
            await view.edit.confirmEdit()
        }
        assert.equal(fixtures.streams.length, 2)
        assert.equal(fixtures.streams[1].type, 'startChatStream')
        assert.equal(fixtures.streams[1].args[2], action === 'retry' ? '原问题' : '修改后的问题')
        assert.deepEqual(view.messages.value.map(msg => msg.role), ['user', 'assistant'])
        assert.equal(view.isGenerating.value, true)
        fixtures.streams[0].onClose()
        assert.equal(view.isGenerating.value, true)
        fixtures.streams[1].onMessage({type: 'done'})
    })
}

test('resumes a server pending round only once while navigating between sessions', async () => {
    fixtures.messageRequest = async id => id === 'A'
        ? [{id: 501, role: 'user', content: '新会话问题', status: 'pending', modelId: 1, kbId: 51, options: '{"thinking":true}'}] : []
    const view = createView()
    await view.load()
    assert.equal(fixtures.streams.length, 1)
    assert.equal(fixtures.streams[0].args[2], null)
    await view.load('B')
    await view.load('A')
    assert.equal(fixtures.streams.length, 1)
    assert.equal(view.tools.thinkingEnabled.value, true)
    fixtures.streams[0].onMessage({type: 'done'})
})

test('reuses metadata requests, expires the cache, and isolates login tokens', async t => {
    t.mock.timers.enable({apis: ['setTimeout', 'setInterval', 'Date'], now: 1000})
    await Promise.all([loadChatResources(), loadChatResources()])
    await loadChatResources()
    assert.deepEqual(fixtures.calls, ['models', 'kbs'])
    t.mock.timers.tick(60_001)
    await loadChatResources()
    assert.deepEqual(fixtures.calls, ['models', 'kbs', 'models', 'kbs'])
    fixtures.store.token = 'new-login'
    await loadChatResources()
    assert.equal(fixtures.calls.filter(call => call === 'models').length, 3)
})

test('an obsolete metadata response cannot overwrite the new login scope', async () => {
    const oldModels = deferred(), oldKbs = deferred()
    fixtures.modelRequest = () => oldModels.promise
    fixtures.kbRequest = () => oldKbs.promise
    const old = loadChatResources()
    fixtures.store.token = 'new-login'
    fixtures.modelRequest = async () => fixtures.modelData
    fixtures.kbRequest = async () => fixtures.kbData
    await loadChatResources()
    oldModels.resolve([{modelId: 999}])
    oldKbs.resolve({private: [{id: 999}]})
    await old
    assert.equal(models.value[0].modelId, 1)
    assert.equal(kbs.value.private[0].id, 51)
})

test('changing login scope disposes old background streams and their timers', async t => {
    t.mock.timers.enable({apis: ['setTimeout', 'setInterval', 'Date']})
    const view = createView()
    await view.load()
    view.onSend('旧登录的问题')
    const oldStream = fixtures.streams[0]
    fixtures.store.token = 'new-login'
    const next = createView()
    await next.load()
    assert.equal(oldStream.controller.signal.aborted, true)
    oldStream.onMessage({type: 'content', content: '不能串到新登录'})
    t.mock.timers.tick(3000)
    assert.equal(next.messages.value.length, 0)
})

test('coalesces DOM scrolling and respects a user scroll before the next tick', async t => {
    t.mock.timers.enable({apis: ['setTimeout']})
    const scope = effectScope()
    scopes.push(scope)
    const scroll = scope.run(useScroll)
    const calls = []
    scroll.messagesContainer.value = {scrollHeight: 1000, scrollTo: options => calls.push(options)}
    scroll.scrollToBottom('auto')
    scroll.scrollToBottom('auto')
    await nextTick()
    assert.equal(calls.length, 1)
    scroll.scrollToBottom('auto')
    scroll.handleWheel()
    await nextTick()
    assert.equal(calls.length, 1)
    scroll.jumpToBottom()
    await nextTick()
    assert.equal(calls.length, 2)
    assert.equal(calls[1].behavior, 'smooth')
    scroll.startTouchScrolling()
    scroll.scrollToBottom('auto', true)
    await nextTick()
    assert.equal(calls.length, 2)
})

test('immediate retry after stop waits for the saved round instead of creating a new round', async t => {
    t.mock.timers.enable({apis: ['setTimeout', 'setInterval', 'Date']})
    const view = createView()
    await view.load()
    view.onSend('原问题')
    fixtures.streams[0].onMessage({type: 'content', content: '原问题回答'})
    view.stopGeneration()
    fixtures.messageRequest = async () => history('原问题')
    await view.edit.onRetryFromAssistant()
    assert.equal(fixtures.streams.length, 2)
    assert.equal(fixtures.streams[1].type, 'retryMessageStream')
    assert.equal(fixtures.streams[1].args[0], 101)
})

test('a following send cannot duplicate the stopped server user row', async t => {
    t.mock.timers.enable({apis: ['setTimeout', 'setInterval', 'Date']})
    const view = createView()
    await view.load()
    view.onSend('已停止的问题')
    view.stopGeneration()
    fixtures.messageRequest = async () => [{id: 601, role: 'user', content: '已停止的问题', status: 'pending', modelId: 1}]
    t.mock.timers.tick(500)
    await settle()
    await view.onSend('新的问题')
    const next = fixtures.streams[1]
    if (next) next.onMessage({type: 'done', userMessageId: 601, assistantMessageId: 602})
    const ids = view.messages.value.filter(msg => msg.id).map(msg => msg.id)
    assert.equal(new Set(ids).size, ids.length)
})

test('knowledge base mutations invalidate cached chat selections immediately', async () => {
    await loadChatResources()
    fixtures.kbData = {private: [{id: 53}]}
    events.emit('knowledgeBasesChanged')
    await loadChatResources()
    assert.equal(kbs.value.private[0].id, 53)
})

for (const method of ['createKb', 'updateKb', 'deleteKb']) {
    test(`${method} invalidates the metadata cache only after a successful API mutation`, async () => {
        await loadChatResources()
        fixtures.kbData = {private: [{id: 54}]}
        await kbApi[method](51, {})
        await loadChatResources()
        assert.equal(kbs.value.private[0].id, 54)
    })
}

test('a failed stopped-round sync preserves text and drafts and can be retried', async t => {
    t.mock.timers.enable({apis: ['setTimeout', 'setInterval', 'Date']})
    const view = createView()
    await view.load()
    view.onSend('原问题')
    fixtures.streams[0].onMessage({type: 'content', content: '原问题回答'})
    view.stopGeneration()
    fixtures.messageRequest = async () => { throw new Error('同步失败') }
    view.question.value = '下一条草稿'
    assert.equal(await view.onSend(view.question.value), false)
    assert.equal(view.isSynchronizing.value, false)
    assert.equal(view.question.value, '下一条草稿')
    assert.equal(view.lastAssistantMessage.value.content, '原问题回答')
    assert.equal(fixtures.streams.length, 1)
    fixtures.messageRequest = async () => history('原问题')
    assert.equal(await view.onSend(view.question.value), true)
    assert.equal(fixtures.streams.length, 2)
    fixtures.streams[1].onMessage({type: 'done'})
})

test('switching away during stopped-round sync cannot send a draft in the next session', async t => {
    t.mock.timers.enable({apis: ['setTimeout', 'setInterval', 'Date']})
    const view = createView()
    await view.load()
    view.onSend('原问题')
    view.stopGeneration()
    const response = deferred()
    fixtures.messageRequest = id => id === 'A' ? response.promise : Promise.resolve(history('B'))
    const send = view.onSend('不应发送到B')
    await view.load('B')
    response.resolve(history('原问题'))
    assert.equal(await send, false)
    assert.equal(fixtures.streams.length, 1)
    assert.equal(view.lastUserMessage.value.content, 'B')
})

test('server-side generating state blocks another request without changing the stopped answer', async t => {
    t.mock.timers.enable({apis: ['setTimeout', 'setInterval', 'Date']})
    const view = createView()
    await view.load()
    view.onSend('原问题')
    fixtures.streams[0].onMessage({type: 'content', content: '部分答案'})
    view.stopGeneration()
    fixtures.messageRequest = async () => [{id: 701, role: 'user', content: '原问题', status: 'generating', modelId: 1}]
    await view.edit.onRetryFromAssistant()
    assert.equal(fixtures.streams.length, 1)
    assert.equal(view.lastAssistantMessage.value.content, '部分答案')
    assert.deepEqual(view.messages.value.map(msg => msg.role), ['user', 'assistant'])
})

test('deleting a session disposes its background connection and cached state', async () => {
    const view = createView()
    await view.load()
    view.onSend('要删除的会话')
    const stream = fixtures.streams[0]
    await view.load('B')
    await chatApi.deleteSession('A')
    assert.equal(stream.controller.signal.aborted, true)
    assert.equal(getChatSession('A').messages.value.length, 0)
})
