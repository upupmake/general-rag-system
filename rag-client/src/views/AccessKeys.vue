<script setup>
import {computed, onMounted, ref} from 'vue'
import {message, Modal} from 'ant-design-vue'
import {
  CopyOutlined,
  DownloadOutlined,
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

const mcpCommand = computed(() => {
  if (!createdKey.value?.accessKey) return ''
  return `/mcp-add kb --url https://starvpn.forwardforever.top:7777/mcp --header Authorization="Bearer ${createdKey.value.accessKey}"`
})

const copyText = async (text, successMessage) => {
  try {
    await navigator.clipboard.writeText(text)
    message.success(successMessage)
  } catch (error) {
    message.error('复制失败，请手动复制')
  }
}

const copyCreatedKey = () => copyText(createdKey.value.accessKey, 'Access Key 已复制')
const copyMcpCommand = () => copyText(mcpCommand.value, 'MCP 命令已复制')

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
      <a-button class="create-key-button" type="primary" @click="openCreate">
        <template #icon><plus-outlined /></template>
        <span class="create-label-desktop">创建 Access Key</span>
        <span class="create-label-mobile">新建</span>
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
          <p>将示例中的占位内容替换为创建时保存的完整 Key，然后复制整条命令，在 Agent 的对话框中运行。</p>
        </div>
        <a-tag color="blue">MCP</a-tag>
      </div>
      <div class="command-box">
        <code>/mcp-add kb --url https://starvpn.forwardforever.top:7777/mcp --header Authorization="Bearer grs_ak_你的完整AccessKey"</code>
      </div>
      <div class="agent-download">
        <router-link class="mcp-doc-link" to="/docs/mcp">查看完整接入文档</router-link>
        <span>还没有可使用 MCP 的 Agent？</span>
        <a-button
          href="https://makecode.forwardforever.top/"
          target="_blank"
          rel="noopener noreferrer"
        >
          <template #icon><download-outlined /></template>
          前往下载 Agent
        </a-button>
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

    <section class="key-cards" aria-label="Access Key 列表">
      <a-spin :spinning="loading">
        <a-empty v-if="!loading && accessKeys.length === 0" description="暂无 Access Key" />
        <article v-for="record in accessKeys" :key="record.id" class="key-card">
          <div class="key-card-header">
            <div class="key-card-title">
              <span class="key-name">{{ record.name }}</span>
              <a-tag :color="record.active ? 'green' : 'default'">
                {{ record.active ? '有效' : '已撤销' }}
              </a-tag>
            </div>
            <a-button
              v-if="record.active"
              type="text"
              danger
              aria-label="撤销 Access Key"
              @click="handleRevoke(record)"
            >
              <template #icon><stop-outlined /></template>
            </a-button>
          </div>
          <code class="key-prefix">{{ record.keyPrefix }}••••••••</code>
          <dl class="key-meta">
            <div>
              <dt>创建时间</dt>
              <dd>{{ formatTime(record.createdAt) }}</dd>
            </div>
            <div>
              <dt>{{ record.active ? '最后使用' : '撤销时间' }}</dt>
              <dd>{{ formatTime(record.active ? record.lastUsedAt : record.revokedAt) }}</dd>
            </div>
          </dl>
        </article>
      </a-spin>
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
      <div v-if="createdKey" class="created-command-box">
        <div class="created-command-label">MCP 配置命令</div>
        <code>{{ mcpCommand }}</code>
        <a-button type="primary" ghost @click="copyMcpCommand">
          <template #icon><copy-outlined /></template>
          复制 MCP 命令
        </a-button>
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

.agent-download {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 12px;
  margin-top: 12px;
  color: rgba(0, 0, 0, 0.55);
  font-size: 13px;
}

.mcp-doc-link {
  margin-right: auto;
}

.key-table {
  overflow: hidden;
  border: 1px solid rgba(5, 5, 5, 0.08);
  border-radius: 6px;
}

.key-cards,
.create-label-mobile {
  display: none;
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

.created-command-box {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 12px;
  padding: 12px;
  border: 1px solid rgba(22, 119, 255, 0.2);
  border-radius: 6px;
  background: rgba(22, 119, 255, 0.04);
}

.created-command-label {
  color: rgba(0, 0, 0, 0.55);
  font-size: 12px;
}

.created-command-box code {
  overflow-wrap: anywhere;
  user-select: all;
}

:global(body[data-theme='dark']) .page-header p,
:global(body[data-theme='dark']) .mcp-example p,
:global(body[data-theme='dark']) .agent-download,
:global(body[data-theme='dark']) .revoked-time {
  color: rgba(255, 255, 255, 0.55);
}

:global(body[data-theme='dark']) .mcp-example {
  border-color: rgba(22, 119, 255, 0.35);
  background: rgba(22, 119, 255, 0.08);
}

:global(body[data-theme='dark']) .key-table,
:global(body[data-theme='dark']) .key-card,
:global(body[data-theme='dark']) .command-box,
:global(body[data-theme='dark']) .created-key-box {
  border-color: rgba(255, 255, 255, 0.12);
}

:global(body[data-theme='dark']) .key-card,
:global(body[data-theme='dark']) .command-box,
:global(body[data-theme='dark']) .created-key-box,
:global(body[data-theme='dark']) .created-command-box {
  background: rgba(255, 255, 255, 0.04);
}

:global(body[data-theme='dark']) .created-command-label {
  color: rgba(255, 255, 255, 0.55);
}

:global(body[data-theme='dark']) .key-meta dt {
  color: rgba(255, 255, 255, 0.45);
}

@media (max-width: 768px) {
  .access-key-page {
    padding: 18px 14px 32px;
  }

  .page-header {
    align-items: flex-start;
    flex-direction: row;
    gap: 12px;
    margin-bottom: 16px;
  }

  .page-header > div {
    min-width: 0;
    flex: 1;
  }

  .title-row {
    flex-wrap: wrap;
    gap: 6px 8px;
  }

  .title-row h1 {
    font-size: 21px;
    line-height: 28px;
  }

  .title-icon {
    font-size: 20px;
  }

  .page-header p {
    margin: 5px 0 0;
    font-size: 13px;
    line-height: 20px;
  }

  .create-key-button {
    flex: none;
    margin-left: auto;
  }

  .create-label-desktop {
    display: none;
  }

  .create-label-mobile {
    display: inline;
  }

  .security-alert {
    margin-bottom: 12px;
  }

  .mcp-example {
    margin-bottom: 12px;
    padding: 14px;
  }

  .mcp-example-heading {
    gap: 10px;
  }

  .mcp-example h2 {
    font-size: 15px;
    line-height: 22px;
  }

  .mcp-example p {
    margin-top: 3px;
    font-size: 13px;
    line-height: 20px;
  }

  .command-box {
    margin-top: 10px;
    padding: 10px 12px;
    overflow-x: visible;
    white-space: normal;
  }

  .command-box code {
    overflow-wrap: anywhere;
    line-height: 20px;
    user-select: all;
  }

  .agent-download {
    align-items: stretch;
    flex-direction: column;
    gap: 7px;
    margin-top: 10px;
  }

  .mcp-doc-link {
    margin-right: 0;
  }

  .agent-download .ant-btn {
    width: 100%;
  }

  .key-table {
    display: none;
  }

  .key-cards {
    display: block;
  }

  .key-card {
    padding: 14px;
    border: 1px solid rgba(5, 5, 5, 0.08);
    border-radius: 8px;
    background: #fff;
  }

  .key-card + .key-card {
    margin-top: 10px;
  }

  .key-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
  }

  .key-card-title {
    display: flex;
    min-width: 0;
    align-items: center;
    gap: 8px;
  }

  .key-card-title .key-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .key-prefix {
    display: block;
    width: fit-content;
    max-width: 100%;
    margin-top: 9px;
    padding: 5px 8px;
    overflow: hidden;
    border-radius: 4px;
    background: rgba(0, 0, 0, 0.04);
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .key-meta {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    margin: 13px 0 0;
    padding-top: 11px;
    border-top: 1px solid rgba(5, 5, 5, 0.06);
  }

  .key-meta dt {
    margin-bottom: 3px;
    color: rgba(0, 0, 0, 0.45);
    font-size: 11px;
    line-height: 16px;
  }

  .key-meta dd {
    margin: 0;
    font-size: 12px;
    line-height: 18px;
    overflow-wrap: anywhere;
  }

  :global(body[data-theme='dark']) .key-prefix {
    background: rgba(255, 255, 255, 0.06);
  }

  :global(body[data-theme='dark']) .key-meta {
    border-top-color: rgba(255, 255, 255, 0.08);
  }
}
</style>
