<script setup>
import {computed, onMounted, onUnmounted, ref} from 'vue'
import {message} from 'ant-design-vue'
import {
  LockOutlined,
  LogoutOutlined,
  ReloadOutlined,
  SearchOutlined,
  UserAddOutlined
} from '@ant-design/icons-vue'
import {
  ADMIN_TOKEN_EXPIRES_KEY,
  ADMIN_TOKEN_KEY,
  grantMembership,
  listMembers,
  logoutAdmin,
  updateMemberBz,
  verifyAdminPassword
} from '@/api/adminMembershipApi.js'

const hasValidSession = () => {
  const token = localStorage.getItem(ADMIN_TOKEN_KEY)
  const expiresAt = Number(localStorage.getItem(ADMIN_TOKEN_EXPIRES_KEY) || 0)
  return Boolean(token) && expiresAt > Date.now()
}

const authed = ref(hasValidSession())
const password = ref('')
const verifying = ref(false)
const loading = ref(false)
const members = ref([])
const account = ref('')
const days = ref(30)
const granting = ref(false)
const lastResult = ref(null)
const filterKeyword = ref('')
const bzDrafts = ref({})
const bzSaving = ref({})
const isMobile = ref(window.innerWidth <= 768)

const updateIsMobile = () => {
  isMobile.value = window.innerWidth <= 768
}

const compareExpireDate = (left, right) => {
  if (!left && !right) return 0
  if (!left) return 1
  if (!right) return -1
  return left.localeCompare(right)
}

const BZ_PATTERN = /^([0-9]{4}-[0-9]{2}-[0-9]{2})[\s\u3000]+开通([0-9]+)天$/

const isValidBzFormat = (value) => {
  const text = (value || '').trim()
  const match = BZ_PATTERN.exec(text)
  if (!match) {
    return false
  }

  const days = Number(match[2])
  if (!Number.isInteger(days) || days < 1 || days > 36500) {
    return false
  }

  const [year, month, day] = match[1].split('-').map(Number)
  const date = new Date(0)
  date.setHours(0, 0, 0, 0)
  date.setFullYear(year, month - 1, day)
  return date.getFullYear() === year
      && date.getMonth() === month - 1
      && date.getDate() === day
}

const columns = [
  {title: 'ID', dataIndex: 'id', key: 'id', width: 80},
  {title: '用户名', dataIndex: 'username', key: 'username', width: 140},
  {title: '邮箱', dataIndex: 'email', key: 'email', width: 220},
  {title: '角色', dataIndex: 'roleId', key: 'roleId', width: 90},
  {title: '备注 bz', dataIndex: 'bz', key: 'bz', width: 320},
  {
    title: '到期日',
    dataIndex: 'expireDate',
    key: 'expireDate',
    width: 140,
    sorter: (a, b) => compareExpireDate(a.expireDate, b.expireDate),
    sortDirections: ['ascend', 'descend']
  },
  {title: '会员状态', key: 'memberStatus', width: 110},
  {title: '账号状态', dataIndex: 'status', key: 'status', width: 110}
]

const filteredMembers = computed(() => {
  const keyword = filterKeyword.value.trim().toLowerCase()
  if (!keyword) {
    return members.value.slice()
  }
  return members.value.filter(item =>
      (item.username || '').toLowerCase().includes(keyword)
      || (item.email || '').toLowerCase().includes(keyword)
  )
})

const expiredCount = computed(() => filteredMembers.value.filter(item => item.expired).length)
const invalidCount = computed(() => filteredMembers.value.filter(item => !item.bzValid).length)

const resetLocalAdminSession = () => {
  localStorage.removeItem(ADMIN_TOKEN_KEY)
  localStorage.removeItem(ADMIN_TOKEN_EXPIRES_KEY)
  authed.value = false
  members.value = []
  lastResult.value = null
  filterKeyword.value = ''
  bzDrafts.value = {}
  bzSaving.value = {}
}

const handleLogout = () => {
  // 本地退出优先，服务端 token 删除失败也不影响页面退出
  const token = localStorage.getItem(ADMIN_TOKEN_KEY)
  logoutAdmin(token).catch(() => {
  })
  resetLocalAdminSession()
}

const handleVerify = async () => {
  if (!password.value) {
    message.warning('请输入管理员密码')
    return
  }

  verifying.value = true
  try {
    const data = await verifyAdminPassword(password.value)
    localStorage.setItem(ADMIN_TOKEN_KEY, data.token)
    localStorage.setItem(ADMIN_TOKEN_EXPIRES_KEY, String(data.expiresAt))
    password.value = ''
    authed.value = true
    message.success('验证成功')
    await loadMembers()
  } catch (error) {
    message.error(error.message || '验证失败')
  } finally {
    verifying.value = false
  }
}

const syncBzDrafts = () => {
  const drafts = {}
  for (const member of members.value) {
    drafts[member.id] = member.bz || ''
  }
  bzDrafts.value = drafts
  bzSaving.value = {}
}

const loadMembers = async () => {
  if (!authed.value) {
    return
  }

  loading.value = true
  try {
    members.value = await listMembers()
    syncBzDrafts()
  } catch (error) {
    if (error.status === 401) {
      resetLocalAdminSession()
      message.warning('管理员登录已失效，请重新输入密码')
    } else {
      message.error(error.message || '加载会员列表失败')
    }
  } finally {
    loading.value = false
  }
}

const handleGrant = async () => {
  const target = account.value.trim()
  if (!target) {
    message.warning('请输入用户名或邮箱')
    return
  }
  if (!Number.isInteger(days.value) || days.value <= 0) {
    message.warning('请输入正确的开通天数')
    return
  }

  lastResult.value = null
  granting.value = true
  try {
    const result = await grantMembership(target, days.value)
    lastResult.value = result
    message.success(result.message || '操作成功')
    await loadMembers()
  } catch (error) {
    if (error.status === 401) {
      resetLocalAdminSession()
      message.warning('管理员登录已失效，请重新输入密码')
    } else {
      message.error(error.message || '操作失败')
    }
  } finally {
    granting.value = false
  }
}

const saveBz = async (record) => {
  const draft = (bzDrafts.value[record.id] || '').trim()
  const original = record.bz || ''
  if (draft === original || bzSaving.value[record.id]) {
    return
  }
  if (!isValidBzFormat(draft)) {
    message.warning('备注格式必须为：yyyy-MM-dd 开通N天')
    return
  }

  bzSaving.value = {...bzSaving.value, [record.id]: true}
  try {
    const updated = await updateMemberBz(record.id, draft, record.bz)
    const index = members.value.findIndex(item => item.id === record.id)
    if (index !== -1) {
      members.value.splice(index, 1, updated)
    }
    bzDrafts.value = {...bzDrafts.value, [record.id]: updated.bz || ''}
    message.success('备注已保存')
  } catch (error) {
    if (error.status === 401) {
      resetLocalAdminSession()
      message.warning('管理员登录已失效，请重新输入密码')
    } else {
      message.error(error.message || '备注保存失败')
    }
  } finally {
    bzSaving.value = {...bzSaving.value, [record.id]: false}
  }
}

const memberStatus = record => {
  if (!record.bzValid) {
    return {text: '备注异常', color: 'orange'}
  }
  return record.expired
      ? {text: '已过期', color: 'red'}
      : {text: '有效', color: 'green'}
}

onMounted(() => {
  window.addEventListener('resize', updateIsMobile)
  if (authed.value) {
    loadMembers()
  }
})

onUnmounted(() => {
  window.removeEventListener('resize', updateIsMobile)
})
</script>

<template>
  <div class="member-grant-page">
    <div v-if="!authed" class="gate-wrap">
      <a-card class="gate-card" :bordered="false">
        <div class="gate-icon">
          <lock-outlined/>
        </div>
        <h1>会员管理</h1>
        <p class="gate-tip">此页面仅限管理员访问</p>
        <a-input-password
            v-model:value="password"
            class="gate-input"
            size="large"
            placeholder="请输入管理员密码"
            autocomplete="off"
            @press-enter="handleVerify"
        />
        <a-button type="primary" size="large" block :loading="verifying" @click="handleVerify">
          进入
        </a-button>
      </a-card>
    </div>

    <div v-else class="console">
      <header class="console-header">
        <div>
          <h1>会员开通 / 续费</h1>
          <p>查询当前全部会员，或按用户名 / 邮箱精准开通、续费。</p>
        </div>
        <div class="header-actions">
          <a-button :loading="loading" @click="loadMembers">
            <template #icon>
              <reload-outlined/>
            </template>
            刷新
          </a-button>
          <a-button danger @click="handleLogout">
            <template #icon>
              <logout-outlined/>
            </template>
            退出管理
          </a-button>
        </div>
      </header>

      <a-card class="grant-card" :bordered="false" title="开通 / 续费">
        <a-form :layout="isMobile ? 'vertical' : 'inline'">
          <a-form-item label="用户名或邮箱">
            <a-input
                v-model:value="account"
                class="account-input"
                placeholder="必须精准匹配其中之一"
                allow-clear
            />
          </a-form-item>
          <a-form-item label="开通天数">
            <a-input-number
                v-model:value="days"
                class="days-input"
                :min="1"
                :max="36500"
                :precision="0"
            />
          </a-form-item>
          <a-form-item>
            <a-button class="grant-submit" type="primary" :loading="granting" @click="handleGrant">
              <template #icon>
                <user-add-outlined/>
              </template>
              确认开通 / 续费
            </a-button>
          </a-form-item>
        </a-form>

        <div class="grant-hint">
          普通用户直接开通；已过期会员按当天重新开通；未过期会员在原有“开通天数”上累加。
        </div>

        <a-alert
            v-if="lastResult"
            class="grant-result"
            type="success"
            show-icon
            :message="lastResult.message"
            :description="`${lastResult.beforeBz || '无'} → ${lastResult.user?.bz || '无'}`"
        />
      </a-card>

      <a-card class="members-card" :bordered="false">
        <template #title>
          <div class="members-title">
            <span>会员列表</span>
            <a-tag color="blue">共 {{ members.length }} 人</a-tag>
            <a-tag v-if="filterKeyword.trim()" color="geekblue">筛选 {{ filteredMembers.length }} 人</a-tag>
            <a-tag v-if="expiredCount" color="red">已过期 {{ expiredCount }}</a-tag>
            <a-tag v-if="invalidCount" color="orange">备注异常 {{ invalidCount }}</a-tag>
          </div>
        </template>

        <div class="members-toolbar">
          <a-input
              v-model:value="filterKeyword"
              class="member-search"
              placeholder="按用户名或邮箱筛选"
              allow-clear
          >
            <template #prefix>
              <search-outlined/>
            </template>
          </a-input>
          <span class="members-filter-hint">
            备注格式：yyyy-MM-dd 开通N天；点击“到期日”列可排序
          </span>
        </div>

        <a-table
            :columns="columns"
            :data-source="filteredMembers"
            :loading="loading"
            :pagination="false"
            row-key="id"
            :size="isMobile ? 'small' : 'middle'"
            :scroll="{ x: 1210 }"
        >
          <template #bodyCell="{ column, record }">
            <template v-if="column.key === 'roleId'">
              <a-tag color="gold">{{ record.roleId === 1 ? '会员' : record.roleId }}</a-tag>
            </template>
            <template v-else-if="column.key === 'bz'">
              <div class="bz-editor">
                <a-input
                    v-model:value="bzDrafts[record.id]"
                    class="bz-input"
                    size="small"
                    :maxlength="250"
                    :status="!isValidBzFormat(bzDrafts[record.id]) ? 'error' : undefined"
                    :disabled="bzSaving[record.id]"
                    placeholder="例如 2026-10-18 开通30天"
                    @press-enter="saveBz(record)"
                />
                <a-button
                    type="link"
                    size="small"
                    :loading="bzSaving[record.id]"
                    :disabled="bzSaving[record.id] || !isValidBzFormat(bzDrafts[record.id]) || (bzDrafts[record.id] || '') === (record.bz || '')"
                    @click="saveBz(record)"
                >
                  保存
                </a-button>
              </div>
            </template>
            <template v-else-if="column.key === 'expireDate'">
              {{ record.expireDate || '-' }}
            </template>
            <template v-else-if="column.key === 'memberStatus'">
              <a-tag :color="memberStatus(record).color">{{ memberStatus(record).text }}</a-tag>
            </template>
            <template v-else-if="column.key === 'status'">
              <a-tag :color="record.status === 'active' ? 'green' : 'red'">
                {{ record.status || '-' }}
              </a-tag>
            </template>
          </template>
        </a-table>
      </a-card>
    </div>
  </div>
</template>

<style scoped>
.member-grant-page {
  min-height: 100vh;
  padding: 24px;
  background:
      radial-gradient(circle at 12% 8%, rgba(22, 119, 255, 0.12), transparent 32%),
      radial-gradient(circle at 88% 0%, rgba(114, 46, 209, 0.10), transparent 30%),
      #f5f7fb;
  box-sizing: border-box;
}

.gate-wrap {
  min-height: calc(100vh - 48px);
  display: flex;
  align-items: center;
  justify-content: center;
}

.gate-card {
  width: 420px;
  max-width: 100%;
  padding: 20px 8px;
  text-align: center;
  border-radius: 20px;
  box-shadow: 0 24px 70px rgba(15, 23, 42, 0.12);
}

.gate-icon {
  width: 64px;
  height: 64px;
  margin: 0 auto 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 20px;
  background: #e8f0ff;
  color: #1677ff;
  font-size: 28px;
}

.gate-card h1 {
  margin: 0 0 8px;
  font-size: 24px;
}

.gate-tip {
  margin: 0 0 24px;
  color: #64748b;
}

.gate-input {
  margin-bottom: 16px;
}

.console {
  max-width: 1600px;
  margin: 0 auto;
}

.console-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 20px;
}

.console-header h1 {
  margin: 0 0 6px;
  font-size: 26px;
}

.console-header p {
  margin: 0;
  color: #64748b;
}

.header-actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.grant-card,
.members-card {
  margin-bottom: 20px;
  border-radius: 16px;
  box-shadow: 0 10px 34px rgba(15, 23, 42, 0.06);
}

.grant-hint {
  margin-top: 6px;
  color: #64748b;
  font-size: 13px;
}

.grant-result {
  margin-top: 16px;
}

.members-title {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.members-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.member-search {
  width: 320px;
  max-width: 100%;
}

.account-input {
  width: 280px;
}

.days-input {
  width: 140px;
}

.grant-submit {
  min-width: 150px;
}

.members-filter-hint {
  color: #64748b;
  font-size: 13px;
}

.bz-editor {
  display: flex;
  align-items: center;
  gap: 4px;
}

.bz-input {
  min-width: 180px;
}

@media (max-width: 768px) {
  .member-grant-page {
    padding: 12px;
  }

  .console-header {
    flex-direction: column;
  }

  .console-header h1 {
    font-size: 22px;
  }

  .header-actions {
    width: 100%;
  }

  .header-actions :deep(.ant-btn) {
    flex: 1;
  }

  .gate-card {
    padding: 16px 4px;
  }

  .gate-card h1 {
    font-size: 22px;
  }

  .grant-card :deep(.ant-card-body),
  .members-card :deep(.ant-card-body) {
    padding: 12px;
  }

  .members-card :deep(.ant-card-head) {
    padding: 0 12px;
  }

  .members-card :deep(.ant-table) {
    font-size: 12px;
  }

  .members-card :deep(.ant-table-cell) {
    padding: 8px;
  }

  .account-input,
  .days-input,
  .member-search {
    width: 100%;
  }

  .grant-submit {
    width: 100%;
  }

  .members-toolbar {
    align-items: stretch;
  }

  .members-filter-hint {
    width: 100%;
  }

  .bz-input {
    min-width: 150px;
  }

  .gate-wrap {
    min-height: calc(100vh - 24px);
  }
}
</style>
