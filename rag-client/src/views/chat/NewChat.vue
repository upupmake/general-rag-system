<script setup>
import {ref, onMounted, computed, watch} from 'vue'
import {useRouter} from 'vue-router'
import {message} from 'ant-design-vue'
import {Sender} from 'ant-design-x-vue'
import {
  CommentOutlined,
  RobotOutlined,
  DatabaseOutlined,
  ToolOutlined,
  GlobalOutlined,
  AppstoreOutlined,
  CodeOutlined,
  FileSearchOutlined,
  BulbOutlined
} from '@ant-design/icons-vue'

import {fetchAvailableModels, fetchSessionTitle, startChat} from '@/api/chatApi'
import {models, groupedModels, providerLogos, selectedModel, selectedKb, loadKbs} from "@/vars.js";
import {events} from "@/events.js";
import KbSelector from "@/components/KbSelector.vue";
import KbDocumentBrowser from '@/components/KbDocumentBrowser.vue'
import {useThemeStore} from '@/stores/theme';
import {usePasteUpload} from './composables/usePasteUpload.js'

const router = useRouter()
const themeStore = useThemeStore();
const {handlePaste} = usePasteUpload()
const documentBrowserVisible = ref(false)

const loading = ref(false)
const question = ref('')
const selectedTools = ref([])
const thinkingEnabled = ref(false)
const promptTemplates = [
  '总结下面这段内容，并提炼出关键结论',
  '解释这段代码的作用、风险和改进方向',
  '请润色下面的内容，使表达更清晰专业',
  '根据下面的信息生成一份结构化会议纪要'
]
const newChatDraftKey = 'chat-draft:new'

const restoreDraft = () => {
  question.value = localStorage.getItem(newChatDraftKey) || ''
}

watch(question, value => {
  if (value) {
    localStorage.setItem(newChatDraftKey, value)
  } else {
    localStorage.removeItem(newChatDraftKey)
  }
})

// 工具配置映射
const toolConfigs = {
  'webSearch': {icon: GlobalOutlined, label: '联网搜索', desc: '开启联网搜索能力，获取实时信息'}
}

onMounted(async () => {
  restoreDraft()
  // 默认选择 provider 为 minimax 的第一个模型
  models.value = await fetchAvailableModels().then()

  if (!selectedModel.value) {
    // 优先选择 provider 为 minimax 的第一个模型
    const minimaxModel = models.value.find(m => m.provider === 'minimax')

    if (minimaxModel) {
      selectedModel.value = minimaxModel.modelId
    } else {
      // 否则选择列表的最后一个模型
      selectedModel.value = models.value[models.value.length - 1]?.modelId || null
    }
  }

  // 加载知识库列表
  await loadKbs()

  // 初始化时同步一次配置
  updateModelConfig()
})

const isKbSupported = computed(() => {
  if (!selectedModel.value) return false
  const modelObj = models.value.find(m => m.modelId === selectedModel.value)
  if (!modelObj) return false
  return modelObj.kbSupported || false
})

const currentModel = computed(() => {
  if (!selectedModel.value) return null
  return models.value.find(m => m.modelId === selectedModel.value) || null
})

const availableTools = computed(() => {
  if (!currentModel.value || !currentModel.value.metadata) return []
  // 后端已统一返回对象，直接读取 tools
  return currentModel.value.metadata.tools || []
})

// 所有已知工具列表，用于渲染UI（即使模型不支持也显示，但禁用）
const allKnownTools = ['webSearch']

const updateModelConfig = () => {
  if (!isKbSupported.value) {
    selectedKb.value = null
  }

  // 处理思考模型配置
  if (currentModel.value?.metadata?.thinking) {
    const {default: isDefault} = currentModel.value.metadata.thinking
    thinkingEnabled.value = isDefault
  } else {
    thinkingEnabled.value = false
  }

  // 切换模型时，检查已选工具是否仍被支持
  // 如果不支持，则移除；
  // 如果支持，保留用户的选择（或者根据需求也可以全重置，这里选择保留支持的）
  const newSupported = availableTools.value
  selectedTools.value = selectedTools.value.filter(t => newSupported.includes(t))
}

watch(selectedModel, () => {
  updateModelConfig()
})


const toggleTool = (toolKey) => {
  if (!availableTools.value.includes(toolKey)) return // Disabled

  const index = selectedTools.value.indexOf(toolKey)
  if (index === -1) {
    selectedTools.value.push(toolKey)
  } else {
    selectedTools.value.splice(index, 1)
  }
}

const toggleThinking = () => {
  if (currentModel.value?.metadata?.thinking?.editable === false) return
  thinkingEnabled.value = !thinkingEnabled.value
}

const onSend = async (text) => {
  if (!selectedModel.value) {
    message.warning('请选择模型')
    return
  }

  loading.value = true
  try {
    const options = {}
    if (selectedTools.value.includes('webSearch')) {
      options.webSearch = true
    }

    // 思考模型参数
    let useThinking = thinkingEnabled.value
    // 如果不可编辑，强制使用默认值
    if (currentModel.value?.metadata?.thinking?.editable === false) {
      useThinking = currentModel.value.metadata.thinking.default
    }

    if (currentModel.value?.metadata?.thinking && useThinking) {
      options.thinking = true
    }

    const res = await startChat({
      modelId: selectedModel.value,
      question: text,
      kbId: isKbSupported.value ? (selectedKb.value || undefined) : undefined,
      options: options
    })
    const {sessionId} = res
    if (sessionId) {
      events.emit('sessionListRefresh')
      router.replace(`/chat/${res.sessionId}`).then(() => {
        question.value = ''
        message.success('新聊天已创建')
      })
      fetchSessionTitle(sessionId).then(titleRes => {
        events.emit('sessionTitleUpdated', {
          sessionId,
          title: titleRes.title || '新的对话'
        })
      })
    }
  } catch (error) {
    message.error('创建聊天失败，请检查网络连接后重试')
  } finally {
    loading.value = false
  }
}
</script>


<template>
  <div class="new-chat-container" :class="{ 'is-dark': themeStore.isDark }">
    <div class="content-wrapper">
      <!-- 欢迎区域 -->
      <div class="welcome-section">
        <div class="welcome-icon">
          <CommentOutlined/>
        </div>
        <h1 class="welcome-title">开始新的对话</h1>
        <p class="welcome-subtitle">选择模型和知识库，开启智能对话体验</p>
      </div>

      <!-- 配置卡片 -->
      <a-card class="config-card" :bordered="false">
        <div class="config-section">
          <div class="config-item">
            <div class="config-label">
              <RobotOutlined class="config-icon"/>
              <span>模型</span>
            </div>
            <div class="model-select-value-wrapper" style="width: 280px">
              <a-select
                  v-model:value="selectedModel"
                  class="config-select model-select-with-logo"
                  placeholder="请选择对话模型"
                  size="large"
                  style="width: 100%">
                <a-select-opt-group
                    v-for="(list, provider) in groupedModels"
                    :key="provider">
                  <template #label>
                    <span class="provider-option-label">
                      <img v-if="providerLogos[provider]" :src="providerLogos[provider]" :alt="`${provider} logo`" class="provider-logo"/>
                      <span>{{ provider.toUpperCase() }}</span>
                    </span>
                  </template>
                  <a-select-option
                      v-for="m in list"
                      :key="m.modelId"
                      :value="m.modelId"
                  >
                    {{ m.modelName }}
                  </a-select-option>
                </a-select-opt-group>
              </a-select>
              <div v-if="currentModel" class="selected-model-value">
                <img v-if="providerLogos[currentModel.provider]" :src="providerLogos[currentModel.provider]" :alt="`${currentModel.provider} logo`" class="provider-logo"/>
                <span>{{ currentModel.modelName }}</span>
              </div>
            </div>
          </div>

          <div class="config-item">
            <div class="config-label">
              <ToolOutlined class="config-icon"/>
              <span>功能</span>
            </div>

            <div class="tools-container">
              <!-- 思考模型开关 -->
              <a-tooltip
                  v-if="currentModel?.metadata?.thinking"
                  :title="currentModel.metadata.thinking.editable === false ? '当前模型强制开启或关闭思考，不可修改' : '开启深度思考模式，模型将进行详细推理'"
              >
                <div
                    class="tool-btn"
                    :class="{
                    active: thinkingEnabled,
                    disabled: currentModel.metadata.thinking.editable === false
                  }"
                    @click="toggleThinking"
                >
                  <BulbOutlined/>
                  <span class="tool-label">深度思考</span>
                </div>
              </a-tooltip>

              <template v-for="toolKey in allKnownTools" :key="toolKey">
                <a-tooltip
                    v-if="availableTools.includes(toolKey)"
                    :title="toolConfigs[toolKey]?.desc || toolKey"
                >
                  <div
                      class="tool-btn"
                      :class="{
                      active: selectedTools.includes(toolKey)
                    }"
                      @click="toggleTool(toolKey)"
                  >
                    <component :is="toolConfigs[toolKey]?.icon || AppstoreOutlined"/>
                    <span class="tool-label">{{ toolConfigs[toolKey]?.label || toolKey }}</span>
                  </div>
                </a-tooltip>
              </template>
            </div>
          </div>

          <div class="config-item">
            <div class="config-label">
              <DatabaseOutlined class="config-icon"/>
              <span>知识库</span>
              <span
                  style="font-size: 12px; color: #999; margin-left: 8px; font-weight: normal; white-space: normal;">请在您需要检索知识库中信息时选用</span>
            </div>
            <div class="kb-select-row" :class="{ 'has-file-button': selectedKb }">
              <KbSelector size="large" class="config-select kb-config-select" :disabled="!isKbSupported"/>
              <a-button v-if="selectedKb" type="text" class="kb-files-button" @click="documentBrowserVisible = true">
                <FileSearchOutlined/>
                <span>查看文件</span>
              </a-button>
            </div>
            <div v-if="!isKbSupported && selectedModel" style="color: #faad14; font-size: 12px; margin-top: 4px;">
              当前模型不支持知识库功能
            </div>
          </div>
        </div>
      </a-card>

      <!-- 输入区域 -->
      <div class="input-section" @paste="handlePaste">
        <div class="prompt-template-row" aria-label="问题模板">
          <button
              v-for="template in promptTemplates"
              :key="template"
              type="button"
              class="prompt-template"
              @click="question = template"
          >{{ template }}</button>
        </div>
        <Sender
            v-model:value="question"
            :disabled="loading"
            :auto-size="false"
            placeholder="请输入你的第一条问题，开启智能对话之旅…"
            @submit="onSend"
            class="chat-sender"
        />
      </div>

      <!-- 提示区域 -->
      <div class="tips-section">
        <div class="tip-item">💡 支持多轮对话和上下文理解</div>
        <div class="tip-item">🔍 可结合知识库进行精准问答</div>
        <div class="tip-item">⚡ 实时流式输出，响应迅速</div>
      </div>
    </div>

    <KbDocumentBrowser
        v-model:visible="documentBrowserVisible"
        :kb-id="selectedKb"/>
  </div>
</template>

<style scoped>
.new-chat-container {
  min-height: 100vh;
  background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
  padding: 40px 24px;
  display: flex;
  justify-content: center;
  align-items: center;
}

.content-wrapper {
  width: 100%;
  max-width: 900px;
  animation: fadeInUp 0.6s ease-out;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.welcome-section {
  text-align: center;
  margin-bottom: 40px;
  color: #333;
}

.welcome-icon {
  font-size: 64px;
  margin-bottom: 20px;
  animation: bounce 2s infinite;
}

@keyframes bounce {
  0%, 100% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-10px);
  }
}

.welcome-title {
  font-size: 36px;
  font-weight: 600;
  margin: 0 0 12px 0;
  color: #333;
}

.welcome-subtitle {
  font-size: 16px;
  opacity: 0.9;
  margin: 0;
}

.config-card {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  margin-bottom: 24px;
  transition: all 0.3s ease;
}

.config-card:hover {
  box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15);
  transform: translateY(-2px);
}

.config-section {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.config-item {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.config-label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 15px;
  font-weight: 500;
  color: #333;
  white-space: nowrap;
}

.config-icon {
  font-size: 18px;
  color: #1890ff;
}

.config-select {
  /* width: 100%; removed to allow manual width control */
}

.kb-config-select {
  align-self: flex-start;
  justify-content: flex-start;
}

.kb-select-row {
  display: flex;
  align-items: center;
  gap: 8px;
  width: fit-content;
  max-width: 100%;
  padding: 4px;
  background: #f5f7fa;
  border: 1px solid #e8eaed;
  border-radius: 10px;
}

.kb-files-button {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  height: 32px;
  padding: 0 10px;
  color: #1677ff;
  background: #fff;
  border-radius: 7px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
}

.provider-option-label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.provider-logo {
  width: 16px;
  height: 16px;
  object-fit: contain;
  flex-shrink: 0;
}

.model-select-value-wrapper {
  position: relative;
  display: inline-block;
}

.model-select-with-logo :deep(.ant-select-selection-item) {
  opacity: 0;
}

.selected-model-value {
  position: absolute;
  left: 12px;
  right: 32px;
  top: 50%;
  transform: translateY(-50%);
  display: flex;
  align-items: center;
  gap: 6px;
  color: #333;
  pointer-events: none;
  overflow: hidden;
  white-space: nowrap;
}

.selected-model-value span {
  overflow: hidden;
  text-overflow: ellipsis;
}

.input-section {
  position: relative;
  margin-bottom: 24px;
}

.prompt-template-row {
  display: flex;
  gap: 8px;
  margin-bottom: 10px;
  overflow-x: auto;
  scrollbar-width: none;
}

.prompt-template-row::-webkit-scrollbar {
  display: none;
}

.prompt-template {
  flex: 0 0 auto;
  padding: 6px 10px;
  color: #52606d;
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(22, 119, 255, 0.16);
  border-radius: 999px;
  cursor: pointer;
  font-size: 12px;
  transition: color 0.2s ease, border-color 0.2s ease, background 0.2s ease;
}

.prompt-template:hover,
.prompt-template:focus-visible {
  color: #1677ff;
  background: rgba(255, 255, 255, 0.95);
  border-color: #1677ff;
  outline: none;
}

.chat-sender {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
}

.chat-sender :deep(textarea) {
  height: 112px !important;
  overflow-y: auto !important;
  resize: none;
}

.chat-sender:hover {
  box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15);
}

.tips-section {
  display: flex;
  justify-content: center;
  gap: 20px;
  flex-wrap: wrap;
}

.tip-item {
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(10px);
  padding: 8px 16px;
  border-radius: 20px;
  color: #555;
  font-size: 14px;
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.tip-item:hover {
  background: rgba(255, 255, 255, 0.95);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.new-chat-container.is-dark .tip-item {
  background: rgba(255, 255, 255, 0.1);
  color: white;
  box-shadow: none;
}

.new-chat-container.is-dark .tip-item:hover {
  background: rgba(255, 255, 255, 0.15);
}

.new-chat-container.is-dark .prompt-template {
  color: #b7c3d0;
  background: rgba(255, 255, 255, 0.08);
  border-color: rgba(64, 169, 255, 0.35);
}

.new-chat-container.is-dark .prompt-template:hover,
.new-chat-container.is-dark .prompt-template:focus-visible {
  color: #69c0ff;
  background: rgba(255, 255, 255, 0.14);
  border-color: #40a9ff;
}


@media (max-width: 768px) {
  .welcome-title {
    font-size: 28px;
  }

  .welcome-icon {
    font-size: 48px;
  }

  .tips-section {
    flex-direction: column;
    align-items: center;
  }

  .config-select,
  .model-select-value-wrapper {
    width: 100% !important;
  }

  .kb-select-row {
    display: grid;
    grid-template-columns: minmax(0, 280px);
    align-items: center;
    gap: 6px;
    width: fit-content;
    max-width: 100%;
    padding: 3px;
  }

  .kb-select-row.has-file-button {
    grid-template-columns: minmax(0, 1fr) auto;
    width: 100%;
  }

  .kb-config-select {
    width: auto !important;
    min-width: 0;
  }

  .kb-select-row.has-file-button .kb-config-select {
    width: 100% !important;
  }

  .kb-files-button {
    height: 34px;
    padding: 0 9px;
    white-space: nowrap;
  }

  .tools-container {
    justify-content: flex-start;
  }

  .tool-btn {
    padding: 8px 12px;
  }

  .chat-sender :deep(textarea) {
    height: 88px !important;
  }
}
</style>

<style>
/* 工具按钮样式 */
.tools-container {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.tool-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 24px;
  border: 1px solid #e0e0e0;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  color: #666;
  background: #fff;
  user-select: none;
  font-size: 15px;
}

.tool-btn:hover {
  border-color: #1890ff;
  color: #1890ff;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(24, 144, 255, 0.1);
}

.tool-btn.active {
  background: #e6f7ff;
  border-color: #1890ff;
  color: #1890ff;
  box-shadow: 0 2px 8px rgba(24, 144, 255, 0.2);
  font-weight: 500;
}

.tool-btn.disabled {
  opacity: 0.5;
  cursor: not-allowed;
  background: #f5f5f5;
  border-color: #d9d9d9;
  color: #00000040;
  box-shadow: none;
}

.tool-btn.active.disabled {
  background: #e6f7ff;
  border-color: #1890ff;
  color: #1890ff;
  opacity: 0.6;
}

.tool-btn.disabled:hover {
  transform: none;
  box-shadow: none;
  border-color: #d9d9d9;
  color: #00000040;
}

/* Dark mode support for tools */
.new-chat-container.is-dark .tool-btn {
  background: rgba(255, 255, 255, 0.05);
  border-color: #434343;
  color: #a6a6a6;
}

.new-chat-container.is-dark .tool-btn:hover {
  border-color: #177ddc;
  color: #177ddc;
  background: rgba(255, 255, 255, 0.08);
}

.new-chat-container.is-dark .tool-btn.active {
  background: rgba(23, 125, 220, 0.2);
  border-color: #177ddc;
  color: #177ddc;
}

.new-chat-container.is-dark .tool-btn.active.disabled {
  background: rgba(23, 125, 220, 0.2);
  border-color: #177ddc;
  color: #177ddc;
  opacity: 0.6;
}

.new-chat-container.is-dark .tool-btn.disabled {
  background: rgba(255, 255, 255, 0.02);
  border-color: #303030;
  color: #434343;
}

.new-chat-container.is-dark .tool-btn.disabled:hover {
  border-color: #303030;
  color: #434343;
  background: rgba(255, 255, 255, 0.02);
}

/* 暗色模式样式 - 非 scoped 以确保优先级和覆盖 */
.new-chat-container.is-dark {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  color: #e0e0e0;
}

.new-chat-container.is-dark .welcome-section {
  color: white;
}

.new-chat-container.is-dark .welcome-title {
  color: white;
}

.new-chat-container.is-dark .config-card {
  background: rgba(30, 30, 30, 0.95);
}

.new-chat-container.is-dark .config-label {
  color: #e0e0e0;
}

.new-chat-container.is-dark .config-icon {
  color: #a0aeff;
}

.new-chat-container.is-dark .chat-sender {
  background: rgba(30, 30, 30, 0.95);
}

.new-chat-container.is-dark .kb-select-row {
  background: rgba(255, 255, 255, 0.05);
  border-color: #434343;
}

.new-chat-container.is-dark .kb-files-button {
  color: #40a9ff;
  background: rgba(255, 255, 255, 0.08);
  box-shadow: none;
}

.new-chat-container.is-dark .tip-item {
  background: rgba(255, 255, 255, 0.1);
  color: white;
  box-shadow: none;
}

.new-chat-container.is-dark .tip-item:hover {
  background: rgba(255, 255, 255, 0.15);
}
</style>
