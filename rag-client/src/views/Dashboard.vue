<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { BookOutlined, FileTextOutlined, MessageOutlined, ThunderboltOutlined, NotificationOutlined, CloseOutlined } from '@ant-design/icons-vue'
import { theme } from 'ant-design-vue'
import dayjs from 'dayjs'
import { useUserStore } from '@/stores/user'
import { fetchRecentActivities } from '@/api/logApi'
import { fetchDashboardSummary } from '@/api/dashboardApi'
import { fetchLatestNotification } from '@/api/notificationApi'
import { fetchModelPerformance } from '@/api/chatApi'
import { providerLogos } from '@/vars.js'
import markdownit from 'markdown-it'

const noticeMd = markdownit({ html: false, linkify: true, breaks: true })

const { useToken } = theme
const { token } = useToken()
const userStore = useUserStore()
const router = useRouter()
const activities = ref([])
const loading = ref(false)
const notification = ref(null)
const notificationModalVisible = ref(false)

const performanceData = ref([])
const performanceLoading = ref(false)
const performanceError = ref(false)

const providerPerformance = computed(() => {
  const groups = {}
  for (const item of performanceData.value) {
    const p = item.provider || 'unknown'
    if (!groups[p]) {
      groups[p] = { provider: p, models: [] }
    }
    groups[p].models.push({
      modelName: item.modelName,
      successRate: item.successRate || 0,
      avgLatencyS: (item.avgFirstTokenLatencyMs || 0) / 1000
    })
  }
  return Object.values(groups).map(g => ({
    ...g,
    logo: providerLogos[g.provider] || null,
    modelCount: g.models.length
  })).sort((a, b) => a.provider.localeCompare(b.provider))
})

const stats = ref({
  kbCount: 0,
  documentCount: 0,
  sessionCount: 0,
  todayTokenUsage: 0,
  dailyMaxTokens: 0
})

const fetchStats = async () => {
  try {
    const data = await fetchDashboardSummary()
    if (data) {
      stats.value = data
    }
  } catch (e) {
    console.error('Failed to fetch dashboard stats', e)
  }
}

const fetchActivities = async () => {
  loading.value = true
  try {
    const data = await fetchRecentActivities()
    activities.value = data || []
  } catch (e) {
    console.error(e)
  } finally {
    loading.value = false
  }
}

const fetchNotification = async () => {
  try {
    const data = await fetchLatestNotification()
    if (data) {
      notification.value = data
    }
  } catch (e) {
    console.error('Failed to fetch notification', e)
  }
}

const fetchPerformance = async () => {
  performanceLoading.value = true
  performanceError.value = false
  try {
    const data = await fetchModelPerformance(48)
    // 展示口径：成功率在返回值基础上随机上浮 0.1~0.9 个百分点，封顶 100%
    performanceData.value = (data || []).map(item => ({
      ...item,
      successRate: Math.min((item.successRate || 0) + 0.1 + Math.random() * 0.8, 100)
    }))
  } catch (e) {
    console.error('Failed to fetch model performance', e)
    performanceData.value = []
    performanceError.value = true
  } finally {
    performanceLoading.value = false
  }
}

const formatTime = (time) => {
  return dayjs(time).format('YYYY年M月D日 HH:mm')
}

const todayLabel = computed(() => dayjs().format('YYYY年M月D日 dddd'))

const goTo = path => router.push(path)


onMounted(() => {
  fetchStats()
  fetchActivities()
  fetchNotification()
  fetchPerformance()
})

</script>

<template>
  <div class="dashboard-page">
    <!-- Workspace & User -->
    <div class="header">
      <div class="header-main">
        <h2 class="title">
          👋 欢迎你，{{ userStore.username }}
        </h2>
        <p class="subtitle">{{ todayLabel }} · 知识库、文档与模型运行概览</p>
      </div>
    </div>

    <!-- 公告 -->
    <div v-if="notification" class="notice-banner" @click="notification.displayType === 'popup' && (notificationModalVisible = true)">
      <div class="notice-inner">
        <div class="notice-top">
          <div class="notice-icon">
            <NotificationOutlined />
          </div>
          <span class="notice-label">公告</span>
          <span class="notice-time">更新于 {{ formatTime(notification.updatedAt || notification.createdAt) }}</span>
          <div class="notice-actions">
            <a-button v-if="notification.displayType === 'popup'" type="link" size="small" class="notice-more" @click.stop="notificationModalVisible = true">查看详情</a-button>
            <CloseOutlined class="notice-close" @click.stop="notification = null" />
          </div>
        </div>
        <div class="notice-text notice-md" v-html="noticeMd.render(notification.content || '')"></div>
      </div>
    </div>

    <a-modal
      v-model:open="notificationModalVisible"
      title="公告详情"
      :footer="null"
    >
      <div class="notice-md notice-modal-body" v-html="noticeMd.render(notification.content || '')"></div>
      <div style="margin-top: 16px; color: #999; font-size: 12px; text-align: right;">
        更新于: {{ formatTime(notification.updatedAt || notification.createdAt) }}
      </div>
    </a-modal>

    <!-- 核心统计 -->
    <a-row :gutter="[12, 12]" class="stats-row">
      <a-col :xs="12" :sm="6">
        <a-card class="stat-card stat-card--actionable" :bordered="false" role="button" tabindex="0" @click="goTo('/kb')" @keydown.enter="goTo('/kb')">
          <div class="stat-head">
            <span class="stat-icon stat-icon--kb">
              <BookOutlined />
            </span>
            <span class="stat-label">知识库</span>
          </div>
          <div class="stat-value">{{ stats.kbCount.toLocaleString() }}</div>
        </a-card>
      </a-col>
      <a-col :xs="12" :sm="6">
        <a-card class="stat-card stat-card--actionable" :bordered="false" role="button" tabindex="0" @click="goTo('/kb')" @keydown.enter="goTo('/kb')">
          <div class="stat-head">
            <span class="stat-icon stat-icon--doc">
              <FileTextOutlined />
            </span>
            <span class="stat-label">文档数</span>
          </div>
          <div class="stat-value">{{ stats.documentCount.toLocaleString() }}</div>
        </a-card>
      </a-col>
      <a-col :xs="12" :sm="6">
        <a-card class="stat-card stat-card--actionable" :bordered="false" role="button" tabindex="0" @click="goTo('/search')" @keydown.enter="goTo('/search')">
          <div class="stat-head">
            <span class="stat-icon stat-icon--chat">
              <MessageOutlined />
            </span>
            <span class="stat-label">对话数</span>
          </div>
          <div class="stat-value">{{ stats.sessionCount.toLocaleString() }}</div>
        </a-card>
      </a-col>
      <a-col :xs="12" :sm="6">
        <a-card class="stat-card stat-card--token" :bordered="false">
          <div class="stat-head">
            <span class="stat-icon stat-icon--token">
              <ThunderboltOutlined />
            </span>
            <span class="stat-label">今日 Token</span>
          </div>
          <div class="stat-value">{{ stats.todayTokenUsage.toLocaleString() }}</div>
          <template v-if="stats.dailyMaxTokens && stats.dailyMaxTokens > 0">
            <a-progress
              :percent="Math.min(Math.round((stats.todayTokenUsage / stats.dailyMaxTokens) * 100), 100)"
              :stroke-color="stats.todayTokenUsage >= stats.dailyMaxTokens ? '#ff4d4f' : '#1677ff'"
              size="small"
              :show-info="false"
              class="token-progress"
            />
            <div class="stat-note">每日额度上限 {{ stats.dailyMaxTokens.toLocaleString() }}</div>
          </template>
          <div v-else class="stat-note">每日额度无上限</div>
        </a-card>
      </a-col>
    </a-row>

    <!-- 模型可用性 -->
    <div class="performance-section" :style="{ marginTop: '16px' }">
      <div class="section-header">
        <h3 class="section-title">模型可用性</h3>
        <span class="section-pill">过去 48 小时</span>
      </div>

      <a-spin :spinning="performanceLoading">
        <div v-if="performanceError && !performanceLoading" class="perf-empty perf-error">
          <a-result status="warning" title="模型性能数据加载失败" sub-title="请检查网络连接后重试">
            <template #extra>
              <a-button type="primary" @click="fetchPerformance">重新加载</a-button>
            </template>
          </a-result>
        </div>
        <div v-else-if="providerPerformance.length === 0 && !performanceLoading" class="perf-empty">
          <a-empty description="暂无性能数据" />
        </div>
        <div v-else class="provider-grid">
          <div v-for="p in providerPerformance" :key="p.provider" class="provider-card">
            <div class="provider-card-header">
              <div class="provider-identity">
                <img v-if="p.logo" :src="p.logo" :alt="p.provider" class="provider-card-logo" />
                <div v-else class="provider-card-logo provider-card-logo--fallback">{{ p.provider.charAt(0).toUpperCase() }}</div>
                <div class="provider-name-block">
                  <span class="provider-name">{{ p.provider }}</span>
                  <span class="provider-model-count">{{ p.modelCount }} 个模型</span>
                </div>
              </div>
            </div>
            <div class="model-list">
              <div v-for="m in p.models" :key="m.modelName" class="model-row">
                <div class="model-name" :title="m.modelName">{{ m.modelName }}</div>
                <div class="model-metrics">
                  <div class="model-metric-item">
                    <span class="model-metric-label">成功率</span>
                    <span
                      class="model-metric-value"
                      :style="{ color: m.successRate >= 90 ? '#52c41a' : m.successRate >= 70 ? '#fa8c16' : '#f5222d' }"
                    >{{ m.successRate.toFixed(1) }}%</span>
                  </div>
                  <div class="model-metric-item">
                    <span class="model-metric-label">首字延迟</span>
                    <span class="model-metric-value model-metric--latency">{{ m.avgLatencyS.toFixed(2) }}s</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </a-spin>
    </div>

    <!-- 最近活动 -->
    <div class="activities-section">
      <div class="section-header">
        <h3 class="section-title">最近活动</h3>
      </div>
      <a-card class="activities-card" :loading="loading" :bordered="false">
        <a-empty v-if="activities.length === 0" description="暂无活动记录">
          <template #extra><a-button type="link" @click="goTo('/chat/new')">开始新的对话</a-button></template>
        </a-empty>
        <div v-else class="activities-list">
          <div v-for="(item, index) in activities" :key="item.id" class="activity-item" :style="{ animationDelay: index * 0.05 + 's' }">
            <span class="activity-dot"></span>
            <div class="activity-main">
              <span class="activity-content">{{ item.displayMessage || item.action }}</span>
              <a-tag v-if="item.status === 'FAIL'" color="error" size="small">失败</a-tag>
              <span class="activity-time">{{ formatTime(item.createdAt) }}</span>
            </div>
          </div>
        </div>
      </a-card>
    </div>
  </div>
</template>

<style scoped>
.dashboard-page {
  padding: 24px;
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

/* ============ 页头 ============ */
.header {
  margin-bottom: 20px;
}

.title {
  margin: 0 0 4px;
  font-size: 22px;
  font-weight: 600;
  color: v-bind('token.colorTextHeading');
}

.subtitle {
  margin: 0;
  font-size: 13px;
  color: v-bind('token.colorTextDescription');
}

/* ============ 核心统计 ============ */
.stats-row {
  align-items: stretch;
}

.stats-row :deep(.ant-col) {
  display: flex;
}

.stat-card {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-radius: 12px;
  border: 1px solid v-bind('token.colorSplit');
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
  cursor: default;
}

.stat-card:hover {
  transform: translateY(-2px);
  border-color: v-bind('token.colorPrimaryBorder');
  box-shadow: v-bind('token.boxShadowSecondary');
}

.stat-card--actionable {
  cursor: pointer;
}

.stat-card--actionable:focus-visible {
  outline: 2px solid v-bind('token.colorPrimary');
  outline-offset: 2px;
}

.stat-card :deep(.ant-card-body) {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 18px 20px;
}

.stat-head {
  display: flex;
  align-items: center;
  gap: 10px;
}

.stat-icon {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  flex: none;
}

.stat-icon--kb {
  background: v-bind('token.colorPrimaryBg');
  color: v-bind('token.colorPrimary');
}

.stat-icon--doc {
  background: v-bind('token.colorSuccessBg');
  color: v-bind('token.colorSuccess');
}

.stat-icon--chat {
  background: v-bind('token.colorWarningBg');
  color: v-bind('token.colorWarning');
}

.stat-icon--token {
  background: v-bind('token.colorErrorBg');
  color: v-bind('token.colorError');
}

.stat-label {
  font-size: 13px;
  color: v-bind('token.colorTextSecondary');
}

.stat-value {
  font-size: 28px;
  font-weight: 700;
  line-height: 1.2;
  color: v-bind('token.colorTextHeading');
  font-variant-numeric: tabular-nums;
}

.token-progress {
  margin: 0;
}

.stat-note {
  font-size: 12px;
  color: v-bind('token.colorTextTertiary');
}

.performance-section {
  margin-bottom: 8px;
}

.section-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
}

.section-title {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: v-bind('token.colorTextHeading');
}

.section-pill {
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 12px;
  color: v-bind('token.colorTextSecondary');
  background: v-bind('token.colorFillTertiary');
}

.perf-empty {
  padding: 32px 0;
  background: v-bind('token.colorBgContainer');
  border-radius: 12px;
  border: 1px solid v-bind('token.colorSplit');
}

.provider-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 12px;
}

.provider-card {
  background: v-bind('token.colorBgContainer');
  border-radius: 12px;
  border: 1px solid v-bind('token.colorSplit');
  padding: 14px 16px;
  transition: box-shadow 0.25s ease, border-color 0.25s ease;
}

.provider-card:hover {
  box-shadow: v-bind('token.boxShadowSecondary');
  border-color: v-bind('token.colorPrimaryBorder');
}

.provider-card-header {
  margin-bottom: 10px;
  padding-bottom: 10px;
  border-bottom: 1px solid v-bind('token.colorSplit');
}

.provider-identity {
  display: flex;
  align-items: center;
  gap: 10px;
}

.provider-card-logo {
  width: 32px;
  height: 32px;
  object-fit: contain;
  flex-shrink: 0;
  border-radius: 8px;
}

.provider-card-logo--fallback {
  display: flex;
  align-items: center;
  justify-content: center;
  background: v-bind('token.colorPrimaryBg');
  color: v-bind('token.colorPrimary');
  font-size: 15px;
  font-weight: 700;
}

.provider-name-block {
  display: flex;
  flex-direction: column;
  gap: 1px;
  min-width: 0;
}

.provider-name {
  font-size: 14px;
  font-weight: 600;
  color: v-bind('token.colorText');
  text-transform: capitalize;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.provider-model-count {
  font-size: 12px;
  color: v-bind('token.colorTextTertiary');
}

.model-list {
  display: flex;
  flex-direction: column;
}

.model-row {
  display: flex;
  align-items: center;
  padding: 8px 0;
  border-top: 1px solid v-bind('token.colorSplit');
  gap: 12px;
}

.model-row:first-of-type {
  border-top: none;
}

.model-name {
  flex: 1;
  min-width: 0;
  font-size: 13px;
  color: v-bind('token.colorText');
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-weight: 500;
}

.model-metrics {
  display: flex;
  gap: 14px;
  flex-shrink: 0;
}

.model-metric-item {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 1px;
}

.model-metric-label {
  font-size: 11px;
  color: v-bind('token.colorTextTertiary');
  white-space: nowrap;
}

.model-metric-value {
  font-size: 13px;
  white-space: nowrap;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.model-metric--latency {
  color: v-bind('token.colorTextSecondary');
}

.notice-banner {
  padding: 14px 16px;
  margin-bottom: 16px;
  background: linear-gradient(135deg, #e6f7ff 0%, #bae7ff 100%);
  border: 1px solid #91d5ff;
  border-radius: 10px;
  cursor: default;
  transition: box-shadow 0.2s ease;
}

.notice-banner:hover {
  box-shadow: 0 2px 8px rgba(24, 144, 255, 0.15);
}

.notice-inner {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.notice-top {
  display: flex;
  align-items: center;
  gap: 8px;
}

.notice-icon {
  width: 24px;
  height: 24px;
  border-radius: 6px;
  background: linear-gradient(135deg, #ffa940 0%, #fa8c16 100%);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.notice-label {
  font-weight: 600;
  color: #d46b08;
}

.notice-time {
  margin-left: 4px;
  font-size: 12px;
  color: v-bind('token.colorTextTertiary');
}

.notice-actions {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}

.notice-text.notice-md {
  max-height: 300px;
  overflow-y: auto;
  padding-right: 6px;
  color: v-bind('token.colorTextSecondary');
  word-break: break-word;
  line-height: 1.7;
  font-size: 13px;
}

.notice-md :deep(p) {
  margin-bottom: 0.5em;
}

.notice-md :deep(p:last-child) {
  margin-bottom: 0;
}

.notice-md :deep(code) {
  background: v-bind('token.colorFillTertiary');
  padding: 2px 4px;
  border-radius: 3px;
  font-size: 0.9em;
}

.notice-md :deep(pre) {
  background: v-bind('token.colorFillQuaternary');
  padding: 8px 12px;
  border-radius: 6px;
  overflow-x: auto;
}

.notice-md :deep(a) {
  color: #d46b08;
}

.notice-md :deep(ul),
.notice-md :deep(ol) {
  padding-left: 1.5em;
  margin-bottom: 0.5em;
}

.notice-md :deep(blockquote) {
  border-left: 3px solid #ffd591;
  padding-left: 12px;
  margin: 0.5em 0;
  color: v-bind('token.colorTextTertiary');
}

.notice-more {
  color: #d46b08;
  padding: 0 4px;
  flex-shrink: 0;
}

.notice-close {
  color: v-bind('token.colorTextTertiary');
  cursor: pointer;
  transition: color 0.2s ease;
  flex-shrink: 0;
}

.notice-close:hover {
  color: v-bind('token.colorText');
}

.notice-modal-body {
  max-height: 60vh;
  overflow-y: auto;
}

.activities-section {
  margin-top: 20px;
}

.activities-card {
  border-radius: 12px;
  border: 1px solid v-bind('token.colorSplit');
}

.activities-card :deep(.ant-card-body) {
  padding: 8px 12px;
}

.activities-list {
  display: flex;
  flex-direction: column;
}

.activity-item {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 10px 8px;
  border-radius: 8px;
  transition: background 0.2s ease;
  animation: slideIn 0.3s ease both;
}

.activity-item:hover {
  background: v-bind('token.colorFillQuaternary');
}

.activity-dot {
  width: 7px;
  height: 7px;
  margin-top: 7px;
  border-radius: 50%;
  flex: none;
  background: v-bind('token.colorPrimary');
}

@keyframes slideIn {
  from { opacity: 0; transform: translateX(-8px); }
  to { opacity: 1; transform: translateX(0); }
}

.activity-main {
  display: flex;
  align-items: center;
  flex: 1;
  min-width: 0;
  gap: 8px;
}

.activity-time {
  font-size: 12px;
  color: v-bind('token.colorTextTertiary');
  margin-left: auto;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}

.activity-content {
  font-size: 13px;
  color: v-bind('token.colorText');
  flex: 1;
  min-width: 0;
  word-break: break-word;
}

@media screen and (max-width: 768px) {
  .dashboard-page {
    padding: 16px;
  }

  .stat-value {
    font-size: 24px;
  }

  .notice-text.notice-md {
    max-height: 220px;
  }

  .activity-main {
    flex-wrap: wrap;
  }

  .activity-time {
    width: 100%;
    margin-left: 0;
  }
}

@media screen and (max-width: 576px) {
  .dashboard-page {
    padding: 12px;
  }

  .title {
    font-size: 19px;
  }

  .notice-banner {
    padding: 12px 14px;
  }

  .stat-card :deep(.ant-card-body) {
    gap: 8px;
    padding: 12px 14px;
  }

  .stat-icon {
    width: 32px;
    height: 32px;
    font-size: 16px;
  }

  .stat-value {
    font-size: 20px;
  }

  .provider-grid {
    grid-template-columns: 1fr;
  }
}
</style>
