import {computed, ref} from "vue";
import {fetchAvailableKbs} from "@/api/kbApi";
import {fetchAvailableModels} from "@/api/chatApi";
import {useUserStore} from "@/stores/user.js";
import {events} from "@/events.js";

const providerLogoModules = import.meta.glob('./assets/providers/*-logo.*', {eager: true, query: '?url', import: 'default'})
const providerLogos = Object.fromEntries(
    Object.entries(providerLogoModules).map(([path, url]) => {
        const provider = path.match(/\/([^/]+)-logo\.[^.]+$/)?.[1]
        return [provider, url]
    }).filter(([provider]) => provider)
)

const models = ref([])

const groupedModels = computed(() => {
    const groups = {}
    for (const m of models.value) {
        if (!groups[m.provider]) {
            groups[m.provider] = []
        }
        groups[m.provider].push(m)
    }
    return groups
})
const selectedModel = ref(null)


// 知识库相关
const kbs = ref({
    private: [],
    shared: [],
    public: [],
    invited: []
})
const selectedKb = ref(null)

// 知识库分组配置
const kbGroupLabels = {
    private: '👤 我创建的',
    shared: '🏢 工作空间共享',
    public: '🌍 公共知识库',
    invited: '📧 受邀访问的'
}

let chatResources = null

events.on('knowledgeBasesChanged', () => {
    chatResources = null
})

// 工作空间切换会整页刷新；标签页内按登录 Token 隔离，缓存一分钟并合并在途请求。
const loadChatResources = () => {
    const scope = useUserStore().token
    if (chatResources?.scope === scope) {
        if (chatResources.promise) return chatResources.promise
        if (Date.now() < chatResources.expiresAt) return Promise.resolve()
    }
    const entry = {scope, expiresAt: 0, promise: null}
    chatResources = entry
    entry.promise = Promise.all([fetchAvailableModels(), fetchAvailableKbs()]).then(([availableModels, data]) => {
        if (useUserStore().token !== scope) return
        if (chatResources !== entry) return loadChatResources()
        models.value = availableModels
        kbs.value = {
            private: data.private || [],
            shared: data.shared || [],
            public: data.public || [],
            invited: data.invited || []
        }
        entry.expiresAt = Date.now() + 60_000
    }).catch(error => {
        if (chatResources === entry) chatResources = null
        throw error
    }).finally(() => {
        entry.promise = null
    })
    return entry.promise
}

// 检查 kbId 是否存在于知识库列表中
const findKbById = (kbId) => {
    if (!kbId) return null
    const allKbs = [...kbs.value.private, ...kbs.value.shared, ...kbs.value.public, ...kbs.value.invited]
    return allKbs.find(kb => kb.id === kbId)
}

export {models, groupedModels, selectedModel, kbs, selectedKb, kbGroupLabels, providerLogos, loadChatResources, findKbById}
