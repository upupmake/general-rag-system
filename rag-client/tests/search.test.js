import assert from 'node:assert/strict'
import {beforeEach, test} from 'node:test'
import {createPinia, setActivePinia} from 'pinia'
import {useSearchStore} from '../src/stores/search.js'

beforeEach(() => {
    setActivePinia(createPinia())
})

test('appendResults merges snippets when a session spans pages', () => {
    const store = useSearchStore()

    store.appendResults([{
        sessionId: 1,
        sessionTitle: '会话一',
        contentList: [{content: '第一页', createdAt: '2026-10-17 10:00:00', matchType: 'content'}]
    }])
    store.appendResults([{
        sessionId: 1,
        sessionTitle: '会话一',
        contentList: [
            {content: '第一页', createdAt: '2026-10-17 10:00:00', matchType: 'content'},
            {content: '第二页', createdAt: '2026-10-17 10:01:00', matchType: 'content'}
        ]
    }, {
        sessionId: 2,
        sessionTitle: '标题命中',
        contentList: [{content: '标题命中', createdAt: '2026-10-17 10:02:00', matchType: 'title'}]
    }])

    assert.equal(store.results.length, 2)
    assert.equal(store.results[0].contentList.length, 2)
    assert.equal(store.results[0].contentList[1].content, '第二页')
    assert.equal(store.results[1].contentList[0].matchType, 'title')
})
