<template>
  <a-card hoverable class="workspace-card" :class="{ 'workspace-card-current': isCurrent }">
    <div class="ws-head">
      <span class="ws-icon">
        <AppstoreOutlined/>
      </span>
      <div class="ws-title-wrap">
        <div class="ws-name" :title="workspace.name">{{ workspace.name }}</div>
        <div class="ws-tags">
          <a-tag v-if="isCurrent" color="green">当前</a-tag>
          <a-tag :color="isOwner ? 'blue' : 'default'">{{ isOwner ? '拥有者' : '成员' }}</a-tag>
        </div>
      </div>
      <a-dropdown>
        <a-button type="text" size="small" class="ws-more">
          <template #icon>
            <MoreOutlined/>
          </template>
        </a-button>
        <template #overlay>
          <a-menu>
            <a-menu-item v-if="!isCurrent" key="switch" @click="$emit('switch', workspace)">
              <template #icon>
                <SwapOutlined/>
              </template>
              切换到此工作空间
            </a-menu-item>
            <a-menu-item key="members" @click="$emit('view-members', workspace)">
              <template #icon>
                <TeamOutlined/>
              </template>
              查看成员
            </a-menu-item>
            <template v-if="isOwner">
              <a-menu-divider/>
              <a-menu-item key="invite" @click="$emit('invite', workspace)">
                <template #icon>
                  <UserAddOutlined/>
                </template>
                邀请成员
              </a-menu-item>
              <a-menu-item key="edit" @click="$emit('edit', workspace)">
                <template #icon>
                  <EditOutlined/>
                </template>
                编辑
              </a-menu-item>
              <a-menu-divider/>
              <a-menu-item key="delete" danger @click="$emit('delete', workspace)">
                <template #icon>
                  <DeleteOutlined/>
                </template>
                删除
              </a-menu-item>
            </template>
          </a-menu>
        </template>
      </a-dropdown>
    </div>

    <p class="ws-desc">{{ workspace.description || '暂无描述' }}</p>

    <div class="ws-footer">
      <span class="ws-meta-label">创建时间</span>
      <span class="ws-meta-value">{{ formatDate(workspace.createdAt) }}</span>
    </div>
  </a-card>
</template>

<script setup>
import dayjs from 'dayjs'
import {theme} from 'ant-design-vue'
import {
  MoreOutlined,
  SwapOutlined,
  TeamOutlined,
  UserAddOutlined,
  EditOutlined,
  DeleteOutlined,
  AppstoreOutlined
} from '@ant-design/icons-vue'

const {useToken} = theme
const {token} = useToken()

defineProps({
  workspace: {
    type: Object,
    required: true
  },
  isCurrent: {
    type: Boolean,
    default: false
  },
  isOwner: {
    type: Boolean,
    default: false
  }
})

defineEmits(['switch', 'edit', 'delete', 'invite', 'view-members'])

const formatDate = (date) => {
  return dayjs(date).format('YYYY-MM-DD')
}
</script>

<style scoped>
.workspace-card {
  height: 100%;
  border-radius: 12px;
  transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}

.workspace-card:hover {
  transform: translateY(-3px);
  border-color: v-bind('token.colorPrimaryBorder');
  box-shadow: v-bind('token.boxShadowSecondary');
}

.workspace-card-current {
  border-color: v-bind('token.colorPrimary');
}

.workspace-card :deep(.ant-card-body) {
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 20px;
}

.ws-head {
  display: flex;
  align-items: flex-start;
  gap: 10px;
}

.ws-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: 10px;
  font-size: 18px;
  flex: none;
  color: v-bind('token.colorPrimary');
  background: v-bind('token.colorPrimaryBg');
}

.ws-title-wrap {
  flex: 1;
  min-width: 0;
}

.ws-name {
  font-size: 15px;
  font-weight: 600;
  line-height: 1.4;
  color: v-bind('token.colorText');
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ws-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 6px;
}

.ws-tags :deep(.ant-tag) {
  margin-inline-end: 0;
}

.ws-more {
  flex: none;
  color: v-bind('token.colorTextTertiary');
  transition: color 0.2s ease;
}

.ws-more:hover {
  color: v-bind('token.colorText');
}

.ws-desc {
  flex: 1;
  margin: 12px 0 0;
  min-height: 42px;
  font-size: 13px;
  line-height: 1.6;
  color: v-bind('token.colorTextSecondary');
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.ws-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px solid v-bind('token.colorSplit');
  font-size: 12px;
}

.ws-meta-label {
  color: v-bind('token.colorTextTertiary');
}

.ws-meta-value {
  color: v-bind('token.colorTextSecondary');
}

@media (hover: none) {
  .workspace-card:hover {
    transform: none;
    box-shadow: none;
  }
}
</style>
