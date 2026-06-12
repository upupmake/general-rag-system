<script setup>
import { ref, computed, onMounted } from 'vue'
import { BookOutlined, FileTextOutlined, MessageOutlined, ThunderboltOutlined, NotificationOutlined, CloseOutlined } from '@ant-design/icons-vue'
import { useUserStore } from '@/stores/user'
import { fetchRecentActivities } from '@/api/logApi'
import { fetchDashboardSummary } from '@/api/dashboardApi'
import { fetchLatestNotification } from '@/api/notificationApi'
import { fetchModelPerformance } from '@/api/chatApi'
import { providerLogos } from '@/vars.js'
import md from '@/utils/markdown'
import markdownit from 'markdown-it'

const noticeMd = markdownit({ html: true, linkify: true, breaks: true })

const userStore = useUserStore()
const activities = ref([])
const loading = ref(false)
const notification = ref(null)
const notificationModalVisible = ref(false)

const performanceData = ref([])
const performanceLoading = ref(false)

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
  try {
    const data = await fetchModelPerformance(24)
    performanceData.value = data || []
  } catch (e) {
    console.error('Failed to fetch model performance', e)
  } finally {
    performanceLoading.value = false
  }
}

const formatTime = (time) => {
  return new Date(time).toLocaleString()
}


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
      <div>
        <h2 class="title">
          👋 欢迎你，{{ userStore.username }}
        </h2>
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
          <a-button v-if="notification.displayType === 'popup'" type="link" size="small" class="notice-more" @click.stop="notificationModalVisible = true">详情</a-button>
          <CloseOutlined class="notice-close" @click.stop="notification = null" />
        </div>
        <div class="notice-text notice-md" v-html="noticeMd.render(notification.content || '')"></div>
      </div>
    </div>

    <a-modal
      v-model:open="notificationModalVisible"
      title="公告详情"
      :footer="null"
    >
      <div class="notice-md" v-html="noticeMd.render(notification.content || '')"></div>
      <div style="margin-top: 16px; color: #999; font-size: 12px; text-align: right;">
        发布于: {{ formatTime(notification.createdAt) }}
      </div>
    </a-modal>

    <!-- 核心统计 -->
    <a-row :gutter="[12, 12]" class="stats-row">
      <a-col :xs="12" :sm="6">
        <a-card class="stat-mini-card" :bordered="true">
          <div class="stat-icon stat-icon--kb">
            <BookOutlined />
          </div>
          <a-statistic title="知识库" :value="stats.kbCount" />
        </a-card>
      </a-col>
      <a-col :xs="12" :sm="6">
        <a-card class="stat-mini-card" :bordered="true">
          <div class="stat-icon stat-icon--doc">
            <FileTextOutlined />
          </div>
          <a-statistic title="文档数" :value="stats.documentCount" />
        </a-card>
      </a-col>
      <a-col :xs="12" :sm="6">
        <a-card class="stat-mini-card" :bordered="true">
          <div class="stat-icon stat-icon--chat">
            <MessageOutlined />
          </div>
          <a-statistic title="对话数" :value="stats.sessionCount" />
        </a-card>
      </a-col>
      <a-col :xs="12" :sm="6">
        <a-card class="stat-mini-card stat-mini-card--token" :bordered="true">
          <div class="stat-icon stat-icon--token">
            <ThunderboltOutlined />
          </div>
          <a-statistic title="今日 Token" :value="stats.todayTokenUsage" group-separator="," />
          <template v-if="stats.dailyMaxTokens && stats.dailyMaxTokens > 0">
            <a-progress
              :percent="Math.min(Math.round((stats.todayTokenUsage / stats.dailyMaxTokens) * 100), 100)"
              :stroke-color="stats.todayTokenUsage >= stats.dailyMaxTokens ? '#ff4d4f' : '#1890ff'"
              size="small"
              :show-info="false"
              style="margin-top: 4px;"
            />
            <div class="token-limit">上限 {{ stats.dailyMaxTokens.toLocaleString() }}</div>
          </template>
          <div v-else class="token-limit">无上限</div>
        </a-card>
      </a-col>
    </a-row>

    <!-- 模型可用性 -->
    <div class="performance-section" :style="{ marginTop: '16px' }">
      <div class="section-header">
        <h3 class="section-title">模型可用性</h3>
        <span class="section-subtitle">过去 24 小时</span>
      </div>

      <a-spin :spinning="performanceLoading">
        <div v-if="providerPerformance.length === 0 && !performanceLoading" class="perf-empty">
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
      <a-card class="activities-card" :loading="loading" :bordered="true">
        <a-empty v-if="activities.length === 0" description="暂无活动记录" />
        <div v-else class="activities-list">
          <div v-for="(item, index) in activities" :key="item.id" class="activity-item" :style="{ animationDelay: index * 0.05 + 's' }">
            <div class="activity-time">{{ formatTime(item.createdAt) }}</div>
            <div class="activity-content">
              <span>{{ item.displayMessage || item.action }}</span>
              <a-tag v-if="item.status === 'FAIL'" color="error" size="small" style="margin-left: 8px;">失败</a-tag>
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

.header {
  margin-bottom: 24px;
}

.title {
  margin-bottom: 4px;
}

.subtitle {
  color: #666;
  margin: 0;
}

.stats-row {
  margin-bottom: 0;
  align-items: stretch;
}

.stats-row :deep(.ant-col) {
  display: flex;
}

.stat-mini-card {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.stat-mini-card {
  position: relative;
  overflow: hidden;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  cursor: default;
}

.stat-mini-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.stat-mini-card :deep(.ant-card-body) {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 14px 16px;
}

.stat-icon {
  width: 36px;
  height: 36px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  margin-bottom: 8px;
}

.stat-icon--kb {
  background: linear-gradient(135deg, #e6f7ff 0%, #bae7ff 100%);
  color: #1890ff;
}

.stat-icon--doc {
  background: linear-gradient(135deg, #f6ffed 0%, #d9f7be 100%);
  color: #52c41a;
}

.stat-icon--chat {
  background: linear-gradient(135deg, #fff7e6 0%, #ffe7ba 100%);
  color: #fa8c16;
}

.stat-icon--token {
  background: linear-gradient(135deg, #fff0f6 0%, #ffd6e7 100%);
  color: #eb2f96;
}

.stat-mini-card :deep(.ant-statistic-title) {
  color: #999;
  margin-bottom: 2px;
}

.stat-mini-card :deep(.ant-statistic-content-value) {
  font-size: 24px;
  font-weight: 700;
}

.token-limit {
  color: #bbb;
  margin-top: 2px;
}

.performance-section {
  margin-bottom: 8px;
}

.section-header {
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin-bottom: 14px;
}

.section-title {
  margin: 0;
  font-weight: 600;
  color: rgba(0, 0, 0, 0.85);
}

.section-subtitle {
  color: #999;
}

.perf-empty {
  padding: 40px 0;
  background: #fff;
  border-radius: 8px;
  border: 1px solid #f0f0f0;
}

.provider-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 10px;
}

.provider-card {
  background: #fff;
  border-radius: 10px;
  border: 1px solid #f0f0f0;
  padding: 14px 16px;
  transition: box-shadow 0.25s ease, border-color 0.25s ease;
}

.provider-card:hover {
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
  border-color: #e0e0e0;
}

.provider-card-header {
  margin-bottom: 12px;
  padding-bottom: 12px;
  border-bottom: 1px solid #f5f5f5;
}

.provider-identity {
  display: flex;
  align-items: center;
  gap: 12px;
}

.provider-card-logo {
  width: 36px;
  height: 36px;
  object-fit: contain;
  flex-shrink: 0;
  border-radius: 6px;
}

.provider-card-logo--fallback {
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #fff;
  font-size: 16px;
  font-weight: 700;
}

.provider-name-block {
  display: flex;
  flex-direction: column;
  gap: 1px;
  min-width: 0;
}

.provider-name {
  font-weight: 600;
  color: rgba(0, 0, 0, 0.85);
  text-transform: capitalize;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.provider-model-count {
  color: #999;
}

.model-list {
  display: flex;
  flex-direction: column;
}

.model-row {
  display: flex;
  align-items: center;
  padding: 8px 0;
  border-top: 1px solid #f5f5f5;
  gap: 12px;
}

.model-row:first-of-type {
  border-top: none;
}

.model-name {
  flex: 1;
  min-width: 0;
  color: rgba(0, 0, 0, 0.75);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-weight: 500;
}

.model-metrics {
  display: flex;
  gap: 16px;
  flex-shrink: 0;
}

.model-metric-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
}

.model-metric-label {
  font-size: 11px;
  color: #bbb;
  white-space: nowrap;
}

.model-metric-value {
  white-space: nowrap;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.model-metric--latency {
  color: rgba(0, 0, 0, 0.65);
}

.notice-banner {
  padding: 14px 16px;
  margin-bottom: 16px;
  background: linear-gradient(135deg, #e6f7ff 0%, #bae7ff 100%);
  border: 1px solid #91d5ff;
  border-radius: 10px;
  cursor: default;
  transition: all 0.2s ease;
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
  flex: 1;
}

.notice-text.notice-md {
  color: rgba(0, 0, 0, 0.75);
  word-break: break-word;
  line-height: 1.6;
  font-size: 14px;
}

.notice-md :deep(p) {
  margin-bottom: 0.5em;
}

.notice-md :deep(p:last-child) {
  margin-bottom: 0;
}

.notice-md :deep(code) {
  background: rgba(0, 0, 0, 0.06);
  padding: 2px 4px;
  border-radius: 3px;
  font-size: 0.9em;
}

.notice-md :deep(pre) {
  background: rgba(0, 0, 0, 0.04);
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
  color: rgba(0, 0, 0, 0.55);
}

.notice-more {
  color: #d46b08;
  padding: 0 4px;
  flex-shrink: 0;
}

.notice-close {
  color: rgba(0, 0, 0, 0.35);
  cursor: pointer;
  transition: color 0.2s ease;
  flex-shrink: 0;
}

.notice-close:hover {
  color: rgba(0, 0, 0, 0.65);
}

.activities-section {
  margin-top: 16px;
}

.activities-card :deep(.ant-card-body) {
  padding: 16px 20px;
}

.activities-list {
  display: flex;
  flex-direction: column;
  gap: 0;
}

.activity-item {
  display: flex;
  align-items: flex-start;
  gap: 16px;
  padding: 10px 12px;
  border-radius: 6px;
  transition: background 0.2s ease;
  animation: slideIn 0.3s ease both;
}

.activity-item:hover {
  background: #fafafa;
}

@keyframes slideIn {
  from { opacity: 0; transform: translateX(-8px); }
  to { opacity: 1; transform: translateX(0); }
}

.activity-time {
  font-size: 12px;
  color: #999;
  flex-shrink: 0;
  min-width: 130px;
  font-variant-numeric: tabular-nums;
}

.activity-content {
  font-size: 13px;
  color: rgba(0, 0, 0, 0.75);
  flex: 1;
  min-width: 0;
}

@media screen and (max-width: 576px) {
  .dashboard-page {
    padding: 12px;
  }

  .header {
    margin-bottom: 16px;
  }

  .stat-mini-card :deep(.ant-card-body) {
    padding: 10px 12px;
  }

  .stat-mini-card :deep(.ant-statistic-content-value) {
    font-size: 18px;
  }
}
</style>
