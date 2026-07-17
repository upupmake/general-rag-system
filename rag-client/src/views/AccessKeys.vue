<script setup>
import {computed, onMounted, ref} from 'vue'
import {message, Modal} from 'ant-design-vue'
import {
  CopyOutlined,
  KeyOutlined,
  PlusOutlined,
  StopOutlined
} from '@ant-design/icons-vue'
import dayjs from 'dayjs'
import {createAccessKey, listAccessKeys, revokeAccessKey} from '@/api/accessKeyApi.js'

const accessKeys = ref([])
const loading = ref(false)
const createVisible = ref(false)
const creating = ref(false)
const keyName = ref('')
const createdKey = ref(null)

const activeCount = computed(() => accessKeys.value.filter(item => item.active).length)

const columns = [
  {title: '名称', dataIndex: 'name', key: 'name'},
  {title: 'Key 前缀', dataIndex: 'keyPrefix', key: 'keyPrefix'},
  {title: '状态', key: 'status', width: 100},
  {title: '创建时间', dataIndex: 'createdAt', key: 'createdAt', width: 180},
  {title: '最后使用', dataIndex: 'lastUsedAt', key: 'lastUsedAt', width: 180},
  {title: '操作', key: 'action', width: 100, align: 'right'}
]

const loadAccessKeys = async () => {
  loading.value = true
  try {
    accessKeys.value = await listAccessKeys()
  } finally {
    loading.value = false
  }
}

const openCreate = () => {
  keyName.value = ''
  createVisible.value = true
}

const handleCreate = async () => {
  const name = keyName.value.trim()
  if (!name) {
    message.warning('请输入 Access Key 名称')
    return
  }

  creating.value = true
  try {
    createdKey.value = await createAccessKey(name)
    createVisible.value = false
    await loadAccessKeys()
  } finally {
    creating.value = false
  }
}

const copyCreatedKey = async () => {
  try {
    await navigator.clipboard.writeText(createdKey.value.accessKey)
    message.success('Access Key 已复制')
  } catch (error) {
    message.error('复制失败，请手动复制')
  }
}

const closeCreatedKey = () => {
  createdKey.value = null
}

const handleRevoke = (record) => {
  Modal.confirm({
    title: `撤销“${record.name}”？`,
    content: '撤销后使用该 Key 的外部服务将立即失去访问能力，且无法恢复。',
    okText: '确认撤销',
    okType: 'danger',
    cancelText: '取消',
    async onOk() {
      await revokeAccessKey(record.id)
      message.success('Access Key 已撤销')
      await loadAccessKeys()
    }
  })
}

const formatTime = value => value ? dayjs(value).format('YYYY-MM-DD HH:mm:ss') : '从未使用'

onMounted(loadAccessKeys)
</script>

<template>
  <main class="access-key-page">
    <header class="page-header">
      <div>
        <div class="title-row">
          <key-outlined class="title-icon" />
          <h1>Access Key</h1>
          <a-tag>{{ activeCount }} 个有效</a-tag>
        </div>
        <p>管理用于外部服务访问个人资产的永久凭证。</p>
      </div>
      <a-button type="primary" @click="openCreate">
        <template #icon><plus-outlined /></template>
        创建 Access Key
      </a-button>
    </header>

    <a-alert
      class="security-alert"
      type="warning"
      show-icon
      message="Access Key 等同于账户凭证，请勿提交到代码仓库或发送给他人。"
    />

    <section class="mcp-example" aria-labelledby="mcp-example-title">
      <div class="mcp-example-heading">
        <div>
          <h2 id="mcp-example-title">用于 MCP 知识库检索</h2>
          <p>当前 Access Key 可用于配置 MCP 检索服务。请将示例中的占位内容替换为创建时保存的完整 Key。</p>
        </div>
        <a-tag color="blue">MCP</a-tag>
      </div>
      <div class="command-box">
        <code>mcp-add kb --url https://starvpn.forwardforever.top:7777/mcp --header Authorization="Bearer grs_ak_你的完整AccessKey"</code>
      </div>
    </section>

    <section class="key-table" aria-label="Access Key 列表">
      <a-table
        :columns="columns"
        :data-source="accessKeys"
        :loading="loading"
        :pagination="false"
        :row-key="record => record.id"
        :scroll="{ x: 860 }"
      >
        <template #emptyText>
          <a-empty description="暂无 Access Key" />
        </template>
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'name'">
            <span class="key-name">{{ record.name }}</span>
          </template>
          <template v-else-if="column.key === 'keyPrefix'">
            <code>{{ record.keyPrefix }}••••••••</code>
          </template>
          <template v-else-if="column.key === 'status'">
            <a-tag :color="record.active ? 'green' : 'default'">
              {{ record.active ? '有效' : '已撤销' }}
            </a-tag>
          </template>
          <template v-else-if="column.key === 'createdAt'">
            {{ formatTime(record.createdAt) }}
          </template>
          <template v-else-if="column.key === 'lastUsedAt'">
            {{ formatTime(record.lastUsedAt) }}
          </template>
          <template v-else-if="column.key === 'action'">
            <a-tooltip v-if="record.active" title="撤销 Access Key">
              <a-button
                type="text"
                danger
                aria-label="撤销 Access Key"
                @click="handleRevoke(record)"
              >
                <template #icon><stop-outlined /></template>
              </a-button>
            </a-tooltip>
            <span v-else class="revoked-time">{{ formatTime(record.revokedAt) }}</span>
          </template>
        </template>
      </a-table>
    </section>

    <a-modal
      v-model:open="createVisible"
      title="创建 Access Key"
      ok-text="创建"
      cancel-text="取消"
      :confirm-loading="creating"
      @ok="handleCreate"
    >
      <a-form layout="vertical" @submit.prevent="handleCreate">
        <a-form-item label="名称" required>
          <a-input
            v-model:value="keyName"
            :maxlength="100"
            placeholder="例如：MCP 服务"
            autofocus
          />
        </a-form-item>
      </a-form>
    </a-modal>

    <a-modal
      :open="Boolean(createdKey)"
      title="Access Key 已创建"
      :closable="false"
      :mask-closable="false"
      :keyboard="false"
      @ok="closeCreatedKey"
    >
      <a-alert
        type="warning"
        show-icon
        message="请立即复制并妥善保存。关闭后将无法再次查看完整 Key。"
      />
      <div v-if="createdKey" class="created-key-box">
        <code>{{ createdKey.accessKey }}</code>
        <a-tooltip title="复制 Access Key">
          <a-button aria-label="复制 Access Key" @click="copyCreatedKey">
            <template #icon><copy-outlined /></template>
          </a-button>
        </a-tooltip>
      </div>
      <template #footer>
        <a-button type="primary" @click="closeCreatedKey">我已保存</a-button>
      </template>
    </a-modal>
  </main>
</template>

<style scoped>
.access-key-page {
  width: min(1180px, 100%);
  margin: 0 auto;
  padding: 28px 32px 48px;
}

.page-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 24px;
  margin-bottom: 20px;
}

.title-row {
  display: flex;
  align-items: center;
  gap: 10px;
}

.title-row h1 {
  margin: 0;
  font-size: 24px;
  line-height: 32px;
  letter-spacing: 0;
}

.title-icon {
  color: #1677ff;
  font-size: 22px;
}

.page-header p {
  margin: 6px 0 0 32px;
  color: rgba(0, 0, 0, 0.55);
}

.security-alert {
  margin-bottom: 16px;
}

.mcp-example {
  margin-bottom: 16px;
  padding: 18px 20px;
  border: 1px solid rgba(22, 119, 255, 0.2);
  border-radius: 6px;
  background: rgba(22, 119, 255, 0.035);
}

.mcp-example-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.mcp-example h2 {
  margin: 0;
  font-size: 16px;
  line-height: 24px;
}

.mcp-example p {
  margin: 4px 0 0;
  color: rgba(0, 0, 0, 0.55);
}

.command-box {
  margin-top: 14px;
  padding: 12px 14px;
  overflow-x: auto;
  border: 1px solid rgba(5, 5, 5, 0.1);
  border-radius: 6px;
  background: rgba(0, 0, 0, 0.04);
  white-space: nowrap;
}

.key-table {
  overflow: hidden;
  border: 1px solid rgba(5, 5, 5, 0.08);
  border-radius: 6px;
}

.key-name {
  font-weight: 600;
}

code {
  font-family: Consolas, 'Courier New', monospace;
  font-size: 13px;
}

.revoked-time {
  color: rgba(0, 0, 0, 0.45);
  font-size: 12px;
  white-space: nowrap;
}

.created-key-box {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 18px;
  padding: 12px;
  border: 1px solid rgba(5, 5, 5, 0.12);
  border-radius: 6px;
  background: rgba(0, 0, 0, 0.025);
}

.created-key-box code {
  min-width: 0;
  flex: 1;
  overflow-wrap: anywhere;
  user-select: all;
}

:global(body[data-theme='dark']) .page-header p,
:global(body[data-theme='dark']) .mcp-example p,
:global(body[data-theme='dark']) .revoked-time {
  color: rgba(255, 255, 255, 0.55);
}

:global(body[data-theme='dark']) .mcp-example {
  border-color: rgba(22, 119, 255, 0.35);
  background: rgba(22, 119, 255, 0.08);
}

:global(body[data-theme='dark']) .key-table,
:global(body[data-theme='dark']) .command-box,
:global(body[data-theme='dark']) .created-key-box {
  border-color: rgba(255, 255, 255, 0.12);
}

:global(body[data-theme='dark']) .command-box,
:global(body[data-theme='dark']) .created-key-box {
  background: rgba(255, 255, 255, 0.04);
}

@media (max-width: 768px) {
  .access-key-page {
    padding: 20px 16px 36px;
  }

  .page-header {
    align-items: stretch;
    flex-direction: column;
    gap: 16px;
  }

  .page-header p {
    margin-left: 0;
  }

  .page-header > .ant-btn {
    align-self: flex-start;
  }
}
</style>
