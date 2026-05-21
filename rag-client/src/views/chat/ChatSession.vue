<script setup>
import {h, onMounted, onUnmounted, ref, watch, computed} from 'vue'
import {
  UserOutlined,
  CopyOutlined,
  EditOutlined,
  ReloadOutlined,
  CheckOutlined,
  CloseOutlined,
  AppstoreOutlined,
  BulbOutlined,
  CaretRightOutlined,
  HistoryOutlined,
  UpOutlined,
} from '@ant-design/icons-vue'
import {Bubble, Sender, ThoughtChain} from 'ant-design-x-vue'
import {useRoute} from 'vue-router'
import {Typography, theme, Spin} from 'ant-design-vue'
import {groupedModels, models, selectedKb, selectedModel, contextMultiplier} from '@/vars.js'
import KbSelector from '@/components/KbSelector.vue'
import {useThemeStore} from '@/stores/theme'

// Composables
import {useScroll} from './composables/useScroll.js'
import {useTools} from './composables/useTools.js'
import {useChat} from './composables/useChat.js'
import {useMessageEdit} from './composables/useMessageEdit.js'

// Utils
import {md} from './markdown.js'
import {allKnownTools, toolConfigs, assistantAvatar, userAvatar} from './constants.js'

const route = useRoute()
const sessionId = ref(route.params.sessionId)
const {token} = theme.useToken()
const themeStore = useThemeStore()

// Scroll management
const {
  messagesContainer,
  userScrolledUp,
  scrollToBottom,
  handleScroll
} = useScroll()

// Tools management
const {
  selectedTools,
  thinkingEnabled,
  isRestoring,
  isUserUncheckedWebSearch,
  currentModel,
  availableTools,
  toggleTool,
  toggleThinking
} = useTools()

// Computed for KB support
const isKbSupported = computed(() => {
  if (!selectedModel.value) return false
  const modelObj = models.value.find(m => m.modelId === selectedModel.value)
  if (!modelObj) return false
  return modelObj.kbSupported || false
})

// Chat management
const {
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
} = useChat(
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
)

// Message edit management
const {
  editingIndex,
  editingContent,
  startEditByItem,
  cancelEdit,
  confirmEdit,
  onRetry,
  onRetryFromAssistant
} = useMessageEdit(
    messages,
    sessionId,
    selectedModel,
    selectedKb,
    isKbSupported,
    selectedTools,
    thinkingEnabled,
    currentModel,
    isGenerating,
    userScrolledUp,
    handleStreamCallbacks
)

// Mobile detection
const isMobile = ref(false)
const inputExpanded = ref(false)
const checkIsMobile = () => {
  isMobile.value = window.innerWidth <= 768
}

const expandInput = () => {
  inputExpanded.value = true
}

const collapseInput = () => {
  inputExpanded.value = false
}

const handleSend = (text) => {
  const value = typeof text === 'string' ? text : question.value
  if (!value) return
  onSend(value)
  inputExpanded.value = false
}

// Thinking collapse handler
const handleThinkingChange = (msg, keys) => {
  if (!msg) return
  const nextKeys = Array.isArray(keys) ? keys : (keys ? [keys] : [])
  const index = messages.value.findIndex(item => item === msg || (item.id && msg.id && item.id === msg.id))
  if (index !== -1) {
    messages.value[index].thinkingCollapseKeys = [...nextKeys]
  }
}

// Lifecycle and watchers
onMounted(() => {
  checkIsMobile()
  window.addEventListener('resize', checkIsMobile)
  loadSession(sessionId.value)
})

onUnmounted(() => {
  window.removeEventListener('resize', checkIsMobile)
})

watch(selectedModel, () => {
  if (!isKbSupported.value) {
    selectedKb.value = null
  }
})

watch(
    () => route.params.sessionId,
    (newId, oldId) => {
      if (newId && newId !== oldId) {
        userScrolledUp.value = false
        loadSession(newId)
        sessionId.value = newId
      }
    }
)

const roles = computed(() => ({
  user: {
    placement: 'end',
    avatar: isMobile.value ? undefined : {icon: h(UserOutlined), style: userAvatar},
  },
  assistant: {
    placement: 'start',
    avatar: isMobile.value ? undefined : {icon: h(UserOutlined), style: assistantAvatar},
  },
}));

// 上下文长度控制
const contextCustomInput = ref(null)
const setContextMultiplier = (val) => {
  contextMultiplier.value = val
  contextCustomInput.value = null
}
const confirmContextCustom = () => {
  const v = Number(contextCustomInput.value)
  if (v >= 1) {
    contextMultiplier.value = v
    contextCustomInput.value = null
  }
}
</script>

<template>
  <div class="chat-session-container" :class="{ 'is-dark': themeStore.isDark }">
    <!-- 顶部配置栏 removed -->
    <!-- <div class="chat-header">...</div> -->

    <!-- 消息列表区域 -->
    <div class="messages-container">
      <div
          ref="messagesContainer"
          class="messages-wrapper"
          @scroll="handleScroll">
        <Spin :spinning="loading">
          <Bubble.List
              auto-scroll
              :roles="roles"
              :items="messages"
              class="bubble-list">
            <template #header="{item: msg}">
              <!-- 显示 usage header -->
              <div v-if="msg.latencyMs || msg.totalTokens"
                   style="font-size: 12px; color: #999; margin-bottom: 8px;">
                <span v-if="msg.totalTokens">Tokens: {{ msg.totalTokens }}</span>
                <a-divider type="vertical" v-if="msg.totalTokens && msg.latencyMs"/>
                <span v-if="msg.latencyMs">Latency: {{ msg.latencyMs / 1000 }}s</span>
              </div>
            </template>
            <template #message="{ item: msg, index }">
              <div v-if="msg.role === 'assistant'" class="assistant-message">
                <!-- 显示检索过程 -->
                <ThoughtChain
                    v-if="msg.ragProcess && msg.ragProcess.length > 0"
                    :items="msg.ragProcess.map((p, idx) => {
                      const item = {
                        key: `${p.step}-${p.status}-${idx}`,
                        title: p.title || '处理中',
                        description: p.description || '',
                        status: p.status || 'pending'
                      }
                      if (p.content) {
                        item.content = h(Typography, {
                          innerHTML: md.render(p.content),
                        })
                      }
                      return item
                    })"
                    collapsible
                    class="thought-chain"
                />
                <!-- 显示思考过程 -->
                <div v-if="msg.thinking" class="thinking-section">
                  <a-collapse
                      ghost
                      size="small"
                      :bordered="false"
                      :activeKey="msg.thinkingCollapseKeys"
                      @update:activeKey="(keys) => handleThinkingChange(msg, keys)"
                      expand-icon-position="start">
                    <template #expandIcon="{ isActive }">
                      <div class="expand-icon-wrapper" :class="{ 'is-active': isActive }">
                        <CaretRightOutlined/>
                      </div>
                    </template>

                    <a-collapse-panel key="thinking-panel">
                      <template #header>
                        <div class="thinking-header-content">
                          <div class="thinking-title">
                            <BulbOutlined class="thinking-icon"/>
                            <span>深度思考过程</span>
                            <span
                                v-if="(!msg.thinkingCollapseKeys || msg.thinkingCollapseKeys.length === 0) && msg.thinking"
                                class="thinking-preview">
                               - {{ msg.loading && !msg.content ? '思考中...' : '已折叠' }}
                            </span>
                          </div>

                          <div v-if="msg.loading && !msg.content" class="thinking-status">
                            <span class="thinking-dots">思考中...</span>
                          </div>

                        </div>
                      </template>

                      <div class="thinking-content-wrapper">
                        <Typography>
                          <div
                              class="markdown-body thinking-markdown" v-html="md.render(msg.thinking || '')">
                          </div>
                        </Typography>
                      </div>

                    </a-collapse-panel>
                  </a-collapse>
                </div>

                <!-- 显示回答内容 -->
                <Typography>
                  <div class="markdown-body message-content" v-html="md.render(msg.content || '')"/>
                </Typography>
              </div>

              <div v-else class="user-message">
                <!-- 编辑模式 -->
                <div v-if="isLastUserMessage(msg) && editingIndex !== -1" class="edit-mode">
                  <a-textarea
                      v-model:value="editingContent"
                      :auto-size="{ minRows: 2, maxRows: 6 }"
                      class="edit-textarea"
                  />
                  <div class="edit-actions">
                    <a-button type="primary" size="small" :icon="h(CheckOutlined)" @click="confirmEdit">确认</a-button>
                    <a-button size="small" :icon="h(CloseOutlined)" @click="cancelEdit">取消</a-button>
                  </div>
                </div>
                <!-- 正常显示模式 -->
                <Typography v-else>
                  <div class="user-message-content">{{ msg.content || '' }}</div>
                </Typography>
              </div>
            </template>
            <template #footer="{ item, index }">
              <a-space :size="token.paddingXXS">
                <a-button
                    type="text"
                    size="small"
                    :icon="h(CopyOutlined)"
                    title="复制内容"
                    @click="onCopy(item.content)"/>
                <!-- 用户消息：编辑按钮 -->
                <a-button
                    v-if="item.role === 'user' && isLastUserMessage(item) && canEditOrRetry && editingIndex === -1"
                    type="text"
                    size="small"
                    :icon="h(EditOutlined)"
                    @click="startEditByItem(item)"
                    title="编辑问题"/>
                <!-- 助手消息：重试按钮 -->
                <a-button
                    v-if="item.role === 'assistant' && isLastAssistantMessage(item) && canEditOrRetry"
                    type="text"
                    size="small"
                    :icon="h(ReloadOutlined)"
                    @click="onRetryFromAssistant()"
                    title="重试回答"/>
              </a-space>
            </template>
          </Bubble.List>
        </Spin>
      </div>
    </div>

    <!-- 输入区域 -->
    <div class="input-container">
      <div class="input-wrapper">
        <button
            v-if="!inputExpanded"
            type="button"
            class="input-collapsed-bar"
            @click="expandInput"
        >
          <span class="input-collapsed-text">{{ question ? question : '输入消息，点击展开' }}</span>
        </button>

        <Sender
            v-else
            v-model:value="question"
            :loading="loading || isGenerating || isLastUserMsgGenerating"
            :actions="false"
            :auto-size="{ minRows: 2, maxRows: 6 }"
            @submit="handleSend"
            class="chat-sender"
            placeholder="输入消息，Shift + Enter 换行，Enter 发送"
        >
          <!-- 工具栏 (Header插槽) -->
          <template #header>
            <div class="sender-header-tools"
                 v-if="allKnownTools.some(k => availableTools.includes(k)) || currentModel?.metadata?.thinking || isKbSupported">
              <div class="header-tools-wrapper">
                <!-- 思考模型开关 -->
                <div
                    v-if="currentModel?.metadata?.thinking"
                    class="header-tool-item"
                    :class="{
                    active: thinkingEnabled,
                    disabled: currentModel.metadata.thinking.editable === false
                  }"
                    @click="toggleThinking"
                    :title="currentModel.metadata.thinking.editable === false ? '当前模型强制开启或关闭思考，不可修改' : '开启深度思考模式'"
                >
                  <BulbOutlined/>
                  <span>深度思考</span>
                  <CheckOutlined v-if="thinkingEnabled" style="font-size: 10px; margin-left: 2px;"/>
                </div>

                <template v-for="toolKey in allKnownTools" :key="toolKey">
                  <div
                      v-if="availableTools.includes(toolKey)"
                      class="header-tool-item"
                      :class="{
                        active: selectedTools.includes(toolKey)
                      }"
                      @click="toggleTool(toolKey)"
                      :title="toolConfigs[toolKey]?.desc || toolKey"
                  >
                    <component :is="toolConfigs[toolKey]?.icon || AppstoreOutlined"/>
                    <span>{{ toolConfigs[toolKey]?.label || toolKey }}</span>
                    <CheckOutlined v-if="selectedTools.includes(toolKey)" style="font-size: 10px; margin-left: 2px;"/>
                  </div>
                </template>

                <!-- 知识库选择 -->
                <a-tooltip v-if="isKbSupported"
                           title="请在您需要检索知识库中信息时选用"
                           placement="topLeft">
                  <KbSelector :class="['kb-tool-item', { 'kb-tool-item--active': selectedKb }]" :bordered="false" size="small" width="150px"/>
                </a-tooltip>
              </div>
            </div>
          </template>

          <template #footer="{ info: { components: { SendButton, LoadingButton } } }">
            <div class="sender-footer">
              <div class="sender-config">
                <a-select
                    v-model:value="selectedModel"
                    class="model-select-footer"
                    placeholder="选择模型"
                    :bordered="false"
                    :dropdownMatchSelectWidth="false"
                >
                  <a-select-opt-group
                      v-for="(list, provider) in groupedModels"
                      :key="provider"
                      :label="provider.toUpperCase()">
                    <a-select-option
                        v-for="m in list"
                        :key="m.modelId"
                        :value="m.modelId"
                    >
                      {{ m.modelName }}
                    </a-select-option>
                  </a-select-opt-group>
                </a-select>
              </div>
              <div class="sender-actions">
                <!-- 上下文长度控制 -->
                <a-popover
                    trigger="click"
                    placement="topRight"
                    :arrow="false"
                    overlay-class-name="context-multiplier-popover"
                >
                  <div
                      class="header-tool-item"
                      :class="{ active: contextMultiplier !== null }"
                  >
                    <HistoryOutlined/>
                    <span>上下文{{ contextMultiplier !== null ? `：${contextMultiplier}x` : '' }}</span>
                  </div>
                  <template #content>
                    <div class="context-multiplier-panel">
                      <div class="ctx-hint">
                        仅在翻译、润色等无需长上下文的任务时限制，以节省token使用量。其他情况建议使用模型默认最大上下文长度。
                      </div>
                      <div class="ctx-divider"/>
                      <div
                          class="ctx-option"
                          :class="{ 'ctx-option-active': contextMultiplier === null }"
                          @click="setContextMultiplier(null)"
                      >默认（跟随模型）
                      </div>
                      <div class="ctx-divider"/>
                      <div class="ctx-list">
                        <div
                            v-for="n in 5"
                            :key="n"
                            class="ctx-option"
                            :class="{ 'ctx-option-active': contextMultiplier === n }"
                            @click="setContextMultiplier(n)"
                        >
                          <span>{{ n }}x</span>
                          <span class="ctx-token-hint">≈ {{ (n * 10240 / 1024).toFixed(0) }}k tokens</span>
                        </div>
                      </div>
                      <div class="ctx-divider"/>
                      <div class="ctx-custom-row">
                        <a-input-number
                            v-model:value="contextCustomInput"
                            :min="1"
                            :precision="0"
                            placeholder="自定义倍率"
                            size="small"
                            style="flex:1"
                            @pressEnter="confirmContextCustom"
                        />
                        <a-button size="small" type="primary" @click="confirmContextCustom">确认</a-button>
                      </div>
                    </div>
                  </template>
                </a-popover>

                <a-button
                    type="text"
                    size="small"
                    :icon="h(UpOutlined)"
                    title="收起输入区"
                    @click="collapseInput"/>

                <component :is="(loading || isGenerating || isLastUserMsgGenerating) ? LoadingButton : SendButton"
                           type="primary" :disabled="loading || isGenerating || isLastUserMsgGenerating || !question"
                           @click="!loading && !isGenerating && !isLastUserMsgGenerating && handleSend(question)"/>
              </div>
            </div>
          </template>
        </Sender>
      </div>
    </div>
  </div>
</template>

<style scoped src="./ChatSession.scoped.css"></style>
<style src="./ChatSession.global.css"></style>
