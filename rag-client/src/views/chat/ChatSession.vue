<script setup>
import {h, onMounted, onUnmounted, ref, watch, computed, nextTick} from 'vue'
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
  UpOutlined,
  VerticalAlignBottomOutlined,
  PauseOutlined,
  FileSearchOutlined,
} from '@ant-design/icons-vue'
import {Bubble, Sender, ThoughtChain} from 'ant-design-x-vue'
import {useRoute} from 'vue-router'
import {Typography, theme, Spin} from 'ant-design-vue'
import {groupedModels, models, providerLogos, selectedKb, selectedModel} from '@/vars.js'
import KbSelector from '@/components/KbSelector.vue'
import KbDocumentBrowser from '@/components/KbDocumentBrowser.vue'
import {useThemeStore} from '@/stores/theme'

// Composables
import {useScroll} from './composables/useScroll.js'
import {useTools} from './composables/useTools.js'
import {useChat} from './composables/useChat.js'
import {useMessageEdit} from './composables/useMessageEdit.js'
import {usePasteUpload} from './composables/usePasteUpload.js'

// Utils
import {renderCached} from './markdown.js'
import {allKnownTools, toolConfigs, assistantAvatar, userAvatar} from './constants.js'

const route = useRoute()
const sessionId = ref(route.params.sessionId)
const {token} = theme.useToken()
const themeStore = useThemeStore()

// Paste upload
const {handlePaste} = usePasteUpload()
const documentBrowserVisible = ref(false)

// Scroll management
const {
  messagesContainer,
  userScrolledUp,
  scrollToBottom,
  handleScroll,
  handleWheel,
  startTouchScrolling,
  endTouchScrolling,
  jumpToBottom
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

const modelSelectWidth = computed(() => {
  const modelName = currentModel.value?.modelName || ''
  const textWidth = [...modelName].reduce((width, char) => width + (/[^\x00-\xff]/.test(char) ? 14 : 8), 0)
  return `${Math.max(140, textWidth + 72)}px`
})

// Chat management
const {
  messages,
  loading,
  loadError,
  isGenerating,
  isSynchronizing,
  question,
  handleStreamCallbacks,
  prepareGeneration,
  loadSession,
  onSend,
  stopGeneration,
  abortController,
  onCopy,
  lastUserMessage,
  isLastUserMsgGenerating,
  lastAssistantMessage,
  canEditOrRetry,
  lastUserHasAssistant,
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
    userScrolledUp,
    scrollToBottom,
    handleStreamCallbacks,
    abortController,
    prepareGeneration
)

// Mobile detection
const isMobile = ref(false)
const inputExpanded = ref(false)
const inputWrapper = ref(null)
const shortcutHelpVisible = ref(false)
const draftStorageKey = computed(() => `chat-draft:${sessionId.value}`)

const restoreDraft = () => {
  question.value = localStorage.getItem(draftStorageKey.value) || ''
}

watch(question, (value) => {
  if (value) {
    localStorage.setItem(draftStorageKey.value, value)
  } else {
    localStorage.removeItem(draftStorageKey.value)
  }
})
const checkIsMobile = () => {
  isMobile.value = window.innerWidth <= 768
}

const focusSenderInput = () => {
  nextTick(() => {
    inputWrapper.value?.querySelector('textarea, input, [contenteditable="true"]')?.focus()
  })
}

const expandInput = () => {
  inputExpanded.value = true
  focusSenderInput()
}

const collapseInput = () => {
  inputExpanded.value = false
}

const handleKeyboardShortcut = (event) => {
  const modifier = event.ctrlKey || event.metaKey
  if (modifier && event.key.toLowerCase() === 'k') {
    event.preventDefault()
    inputExpanded.value = true
    focusSenderInput()
  } else if (event.key === 'Escape' && isGenerating.value) {
    event.preventDefault()
    stopGeneration()
  } else if (modifier && event.shiftKey && event.key.toLowerCase() === 'c') {
    event.preventDefault()
    onCopy(lastAssistantMessage.value?.content)
  } else if (modifier && event.key === '/') {
    event.preventDefault()
    if (inputExpanded.value) {
      collapseInput()
    } else {
      expandInput()
    }
  }
}

const handleSend = async (text) => {
  const value = typeof text === 'string' ? text : question.value
  if (!value || loading.value || loadError.value || isGenerating.value || isSynchronizing.value || isLastUserMsgGenerating.value) return
  if (!await onSend(value)) return
  if (!question.value) {
    localStorage.removeItem(draftStorageKey.value)
    inputExpanded.value = false
  }
}

// 小游戏提示：每次请求弹出，整个请求（流式输出）结束后隐藏
const gameTipVisible = ref(false)
let gameTipTimer = null
const gameTipText = computed(() =>
  isMobile.value ? '等待回复时，可点开菜单底部头像玩小游戏' : '等待回复时，可点击左下角头像玩小游戏'
)
watch(isGenerating, (generating) => {
  clearTimeout(gameTipTimer)
  if (generating) {
    // 延迟弹出，回复较快时不闪烁
    gameTipTimer = setTimeout(() => {
      gameTipVisible.value = true
    }, 800)
  } else {
    gameTipVisible.value = false
  }
}, {immediate: true})

// Thinking collapse handler
const handleThinkingChange = (msg, keys) => {
  if (!msg) return
  const nextKeys = Array.isArray(keys) ? keys : (keys ? [keys] : [])
  const index = messages.value.findIndex(item => item === msg || (item.id && msg.id && item.id === msg.id)
      || (item.key != null && item.key === msg.key))
  if (index !== -1) {
    messages.value[index].thinkingCollapseKeys = [...nextKeys]
  }
}

// Lifecycle and watchers
onMounted(() => {
  checkIsMobile()
  window.addEventListener('resize', checkIsMobile)
  window.addEventListener('keydown', handleKeyboardShortcut)
  loadSession(sessionId.value)
  restoreDraft()
})

onUnmounted(() => {
  window.removeEventListener('resize', checkIsMobile)
  window.removeEventListener('keydown', handleKeyboardShortcut)
  clearTimeout(gameTipTimer)
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
        cancelEdit()
        documentBrowserVisible.value = false
        userScrolledUp.value = false
        sessionId.value = newId
        loadSession(newId)
        nextTick(restoreDraft)
      }
    }
)

const streamPhaseText = {
  connecting: {
    title: '正在等待服务响应',
    description: '请求已发送，可随时停止'
  },
  waiting: {
    title: '正在等待响应',
    description: '连接已建立，等待服务端处理进展'
  },
  analyzing: {
    title: '正在分析问题',
    description: '准备知识库检索，确定检索方向'
  },
  retrieving: {
    title: '正在检索知识库',
    description: '查找与问题相关的资料'
  },
  context: {
    title: '正在整理资料',
    description: '根据检索结果准备回答上下文'
  },
  thinking: {
    title: '正在思考',
    description: '整理信息与推理'
  },
  generating: {
    title: '模型正在生成',
    description: '内容将持续更新'
  }
}

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
          @wheel="handleWheel"
          @touchstart="startTouchScrolling"
          @touchend="endTouchScrolling"
          @touchcancel="endTouchScrolling"
          @scroll="handleScroll">
        <Spin :spinning="loading">
          <a-alert
              v-if="loadError"
              type="error"
              show-icon
              message="会话加载失败"
              :description="loadError"
              class="session-load-error">
            <template #action>
              <a-button size="small" @click="loadSession(sessionId)">重新加载</a-button>
            </template>
          </a-alert>
          <Bubble.List
              :auto-scroll="false"
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
                <div
                    v-if="msg.streamPhase"
                    class="assistant-activity"
                    :class="`stream-phase-${msg.streamPhase}`"
                    role="status"
                    aria-live="polite"
                >
                  <span class="activity-indicator" aria-hidden="true">
                    <span class="activity-indicator-dot"/>
                  </span>
                  <span class="activity-copy">
                    <strong>{{ streamPhaseText[msg.streamPhase]?.title }}</strong>
                    <span>{{ streamPhaseText[msg.streamPhase]?.description }}</span>
                    <span v-if="msg.idleSeconds >= 20" class="activity-wait-hint">
                      暂未收到新的进展，服务可能仍在排队或处理。可继续等待，或点击停止。
                    </span>
                  </span>
                  <span class="activity-elapsed" aria-live="off">已用时 {{ msg.elapsedSeconds || 0 }} 秒</span>
                </div>

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
                          innerHTML: renderCached(p, 'content', p.content),
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
                              class="markdown-body thinking-markdown" v-html="renderCached(msg, 'thinking', msg.thinking)">
                          </div>
                        </Typography>
                      </div>

                    </a-collapse-panel>
                  </a-collapse>
                </div>

                <!-- 显示回答内容 -->
                <Typography>
                  <div class="markdown-body message-content" v-html="renderCached(msg, 'content', msg.content)"/>
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
                <!-- 用户消息：上一轮中断（其后没有助手消息）时的重试按钮 -->
                <a-button
                    v-if="item.role === 'user' && isLastUserMessage(item) && !lastUserHasAssistant && canEditOrRetry"
                    type="text"
                    size="small"
                    :icon="h(ReloadOutlined)"
                    @click="onRetryFromAssistant()"
                    title="重试回答"/>
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
      <!-- 回到底部悬浮按钮 -->
      <transition name="scroll-to-bottom-fade">
        <button
            v-if="userScrolledUp"
            type="button"
            class="scroll-to-bottom-btn"
            title="回到底部"
            @click="jumpToBottom"
        >
          <VerticalAlignBottomOutlined/>
        </button>
      </transition>
      <!-- 等待回复时的小游戏提示（吸附在消息区底部） -->
      <transition name="game-tip-fade">
        <div v-if="gameTipVisible" class="game-tip">
          {{ gameTipText }}
        </div>
      </transition>
    </div>

    <a-modal v-model:open="shortcutHelpVisible" title="键盘快捷键" :footer="null">
      <div class="shortcut-list">
        <div><kbd>Enter</kbd><span>发送消息</span></div>
        <div><kbd>Shift + Enter</kbd><span>换行</span></div>
        <div><kbd>Ctrl/Cmd + K</kbd><span>聚焦输入框</span></div>
        <div><kbd>Esc</kbd><span>停止生成</span></div>
        <div><kbd>Ctrl/Cmd + Shift + C</kbd><span>复制最后一条回答</span></div>
        <div><kbd>Ctrl/Cmd + /</kbd><span>快速收起/展开输入区</span></div>
      </div>
    </a-modal>

    <!-- 输入区域 -->
    <div class="input-container">
      <div class="input-wrapper" ref="inputWrapper" @paste="handlePaste">
        <div v-if="!inputExpanded" class="input-collapsed-row">
          <button
              type="button"
              class="input-collapsed-bar"
              @click="expandInput"
          >
            <span class="input-collapsed-text">{{ question ? question : '输入消息，点击展开' }}</span>
          </button>
          <a-button
              v-if="isGenerating"
              type="primary"
              danger
              :icon="h(PauseOutlined)"
              title="停止生成（Esc）"
              @click="stopGeneration">停止</a-button>
        </div>

        <Sender
            v-else
            v-model:value="question"
            :loading="loading || isGenerating || isSynchronizing || isLastUserMsgGenerating"
            :actions="false"
            :auto-size="false"
            @submit="handleSend"
            class="chat-sender"
            placeholder="输入消息，Shift + Enter 换行，Enter 发送"
        >
          <!-- 工具栏 (Header插槽) -->
          <template #header>
            <div class="sender-header-tools"
                 v-if="allKnownTools.some(k => availableTools.includes(k)) || currentModel?.metadata?.thinking">
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
              </div>
            </div>
          </template>

          <template #footer="{ info: { components: { SendButton, LoadingButton } } }">
            <div class="sender-footer">
              <div class="sender-config">
                <div class="model-select-footer-wrapper" :style="{ '--model-select-width': modelSelectWidth }">
                  <a-select
                      v-model:value="selectedModel"
                      class="model-select-footer model-select-with-logo"
                      placeholder="选择模型"
                      :bordered="false"
                      :virtual="false"
                      :dropdownMatchSelectWidth="false"
                  >
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
                <div v-if="isKbSupported" class="kb-select-row" :class="{ 'has-file-button': selectedKb }">
                  <a-tooltip title="请在您需要检索知识库中信息时选用" placement="topLeft">
                    <KbSelector size="small" width="160px" class="kb-config-select"/>
                  </a-tooltip>
                  <a-tooltip v-if="selectedKb" title="查看知识库文件">
                    <a-button type="text" size="small" class="kb-files-button" aria-label="查看知识库文件" @click="documentBrowserVisible = true">
                      <FileSearchOutlined/>
                    </a-button>
                  </a-tooltip>
                </div>
              </div>
              <div class="sender-actions">
                <a-button
                    type="text"
                    size="small"
                    class="shortcut-help-button"
                    title="键盘快捷键"
                    @click="shortcutHelpVisible = true">
                  快捷键
                </a-button>

                 <a-button
                    type="text"
                    size="small"
                    class="collapse-input-button"
                    :icon="h(UpOutlined)"
                    title="收起输入区"
                    @click="collapseInput">
                  收起输入区
                </a-button>

                <a-button
                    v-if="isGenerating"
                    type="primary"
                    danger
                    :icon="h(PauseOutlined)"
                    title="停止生成"
                    aria-label="停止生成"
                    @click="stopGeneration"/>
                <component v-else
                           :is="(loading || isGenerating || isSynchronizing || isLastUserMsgGenerating) ? LoadingButton : SendButton"
                           type="primary" :disabled="loading || !!loadError || isGenerating || isSynchronizing || isLastUserMsgGenerating || !question"
                           @click="handleSend(question)"/>
              </div>
            </div>
          </template>
        </Sender>
      </div>
    </div>

    <KbDocumentBrowser
        v-model:visible="documentBrowserVisible"
        :kb-id="selectedKb"/>
  </div>
</template>

<style scoped src="./ChatSession.scoped.css"></style>
<style src="./ChatSession.global.css"></style>
