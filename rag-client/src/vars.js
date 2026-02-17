import {computed, ref} from "vue";
import {fetchAvailableKbs} from "@/api/kbApi";

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

// RAG模式选择：'agentic'（Agentic RAG）或 'fast'（Fast RAG）
const ragMode = ref('agentic')  // 默认使用 Agentic RAG

// 知识库分组配置
const kbGroupLabels = {
    private: '👤 我创建的',
    shared: '🏢 工作空间共享',
    public: '🌍 公共知识库',
    invited: '📧 受邀访问的'
}

// 加载知识库列表
const loadKbs = async () => {
    const data = await fetchAvailableKbs()
    kbs.value = {
        private: data.private || [],
        shared: data.shared || [],
        public: data.public || [],
        invited: data.invited || []
    }
}

// 检查 kbId 是否存在于知识库列表中
const findKbById = (kbId) => {
    if (!kbId) return null
    const allKbs = [...kbs.value.private, ...kbs.value.shared, ...kbs.value.public, ...kbs.value.invited]
    return allKbs.find(kb => kb.id === kbId)
}

export {models, groupedModels, selectedModel, kbs, selectedKb, ragMode, kbGroupLabels, loadKbs, findKbById}