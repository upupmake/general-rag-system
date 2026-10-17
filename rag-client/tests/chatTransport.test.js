import assert from 'node:assert/strict'
import {after, beforeEach, afterEach, test} from 'node:test'
import {createServer} from 'node:http'
import {once} from 'node:events'
import {registerHooks} from 'node:module'
import {setTimeout as sleep} from 'node:timers/promises'

let handler
let requests = 0
const server = createServer((request, response) => {
    requests++
    handler(request, response)
})
server.keepAliveTimeout = 1
server.listen(0, '127.0.0.1')
await once(server, 'listening')
const base = `http://127.0.0.1:${server.address().port}`
const stub = source => ({url: `data:text/javascript,${encodeURIComponent(source)}`, shortCircuit: true})
registerHooks({
    resolve(specifier, context, nextResolve) {
        if (specifier === './commonApi') return stub('export default {}')
        if (specifier === '@/consts.js') return stub(`export const API_BASE_URL = ${JSON.stringify(base)}`)
        if (specifier === '@/stores/user.js') return stub('export const useUserStore = () => ({token: "local-test"})')
        if (specifier === 'ant-design-vue') return stub('export const message = {error: () => Promise.resolve()}')
        if (specifier === '@/events.js') return nextResolve(new URL('../src/events.js', import.meta.url).href, context)
        return nextResolve(specifier, context)
    }
})
const {startChatStream} = await import('../src/api/chatApi.js')
const oldWindow = globalThis.window
const oldDocument = globalThis.document
const deferred = () => {
    let resolve
    const promise = new Promise(res => { resolve = res })
    return {promise, resolve}
}

beforeEach(() => {
    requests = 0
    globalThis.window = {fetch: globalThis.fetch, setTimeout: globalThis.setTimeout, clearTimeout: globalThis.clearTimeout}
    globalThis.document = {addEventListener() {}, removeEventListener() {}}
})
afterEach(() => {
    globalThis.window = oldWindow
    globalThis.document = oldDocument
})
after(() => new Promise(resolve => server.close(resolve)))

test('native HTTP abort before first SSE event really closes the transport without retry', {timeout: 5000}, async () => {
    const accepted = deferred(), closed = deferred()
    handler = (request, response) => {
        response.writeHead(200, {'Content-Type': 'text/event-stream'})
        response.flushHeaders()
        response.once('close', closed.resolve)
        accepted.resolve()
    }
    let events = 0, errors = 0, closes = 0
    const controller = startChatStream(1, 1, 'question', null, {}, null,
        () => events++, () => errors++, () => closes++)
    await accepted.promise
    controller.abort()
    await closed.promise
    await sleep(30)
    assert.equal(requests, 1)
    assert.equal(events, 0)
    assert.equal(errors, 0)
    assert.equal(closes, 0)
})

test('native HTTP SSE parser preserves split fragments and final IDs', {timeout: 5000}, async () => {
    const closed = deferred()
    handler = (request, response) => {
        response.writeHead(200, {'Content-Type': 'text/event-stream'})
        const payload = 'data: {"type":"content","content":"公式\\n内容"}\n\n'
            + 'data: {"type":"done","userMessageId":21,"assistantMessageId":22}\n\n'
        response.write(payload.slice(0, 15))
        response.write(payload.slice(15, 37))
        response.end(payload.slice(37))
    }
    const events = []
    const controller = startChatStream(1, 1, 'question', null, {}, null,
        event => events.push(event), error => closed.resolve(error), () => closed.resolve())
    try {
        assert.equal(await closed.promise, undefined)
        assert.deepEqual(events, [
            {type: 'content', content: '公式\n内容'},
            {type: 'done', userMessageId: 21, assistantMessageId: 22}
        ])
        assert.equal(requests, 1)
    } finally {
        controller.abort()
    }
})

test('native HTTP failed handshake retains the server error and never repeats generation', {timeout: 5000}, async () => {
    const failed = deferred()
    handler = (request, response) => {
        response.writeHead(403, {'Content-Type': 'application/json'})
        response.end(JSON.stringify({message: '今日额度已用完'}))
    }
    const controller = startChatStream(1, 1, 'question', null, {}, null, null,
        error => failed.resolve(error), null)
    try {
        const error = await failed.promise
        assert.equal(error.status, 403)
        assert.equal(error.message, '今日额度已用完')
        await sleep(100)
        assert.equal(requests, 1)
    } finally {
        controller.abort()
    }
})
