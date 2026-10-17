<script setup>
import {onMounted, ref, computed} from 'vue';
import {useRouter} from 'vue-router';
import {message, theme} from 'ant-design-vue';
import {
  PlusOutlined,
  DeleteOutlined,
  UserOutlined,
  TeamOutlined,
  GlobalOutlined,
  MailOutlined
} from '@ant-design/icons-vue';
import dayjs from 'dayjs';
import {createKb, deleteKb, fetchAvailableKbs} from "@/api/kbApi.js";
import {useUserStore} from '@/stores/user';

const {useToken} = theme;
const {token} = useToken();
const router = useRouter();
const userStore = useUserStore();

const isModalVisible = ref(false);
const formState = ref({
  name: '',
  description: '',
  visibility: 'private'
});

// 分类配置：key 对应后端 /kb 接口返回 Map 的字段
const categories = [
  {key: 'private', label: '我创建的', icon: UserOutlined, color: '#1677ff', tint: 'rgba(22, 119, 255, 0.12)', emptyText: '还没有创建任何知识库'},
  {key: 'shared', label: '工作空间共享', icon: TeamOutlined, color: '#52c41a', tint: 'rgba(82, 196, 26, 0.12)', emptyText: '当前工作空间暂无共享的知识库'},
  {key: 'public', label: '公共知识库', icon: GlobalOutlined, color: '#fa8c16', tint: 'rgba(250, 140, 22, 0.12)', emptyText: '暂无公开的知识库'},
  {key: 'invited', label: '受邀访问', icon: MailOutlined, color: '#722ed1', tint: 'rgba(114, 46, 209, 0.12)', emptyText: '没有被邀请访问的知识库'}
];

const kbMap = ref({
  private: [],
  shared: [],
  public: [],
  invited: []
});

const activeTab = ref('private');
// 初始为 true：首屏直接渲染骨架，避免先闪一帧空状态
const isLoading = ref(true);

const totalCount = computed(() => categories.reduce((sum, c) => sum + kbMap.value[c.key].length, 0));

// 封装加载数据的方法，便于复用
const loadData = async () => {
  isLoading.value = true;
  try {
    const data = await fetchAvailableKbs();
    if (data) {
      kbMap.value = {
        private: data.private || [],
        shared: data.shared || [],
        public: data.public || [],
        invited: data.invited || []
      };
      // 当前分类无数据时，自动切换到第一个有数据的分类
      if (!kbMap.value[activeTab.value].length) {
        const firstNonEmpty = categories.find(c => kbMap.value[c.key].length > 0);
        if (firstNonEmpty) activeTab.value = firstNonEmpty.key;
      }
    }
  } catch (e) {
    message.error('加载知识库列表失败');
  } finally {
    isLoading.value = false;
  }
};

onMounted(() => {
  loadData();
});

const handleCreate = async () => {
  if (!formState.value.name) return message.error('请输入名称');
  // 新建后切换到对应分类，避免停留在当前分类看不到新建结果
  const targetTab = categories.some(c => c.key === formState.value.visibility)
      ? formState.value.visibility
      : 'private';
  try {
    await createKb(formState.value);
    message.success('创建成功');
    isModalVisible.value = false;
    formState.value = {name: '', description: '', visibility: 'private'};
    activeTab.value = targetTab;
    await loadData();
  } catch (e) {
    // 错误已由拦截器处理
  }
};

const goToDetail = (id) => {
  router.push(`/kb/${id}`);
};

// 仅在卡片本身聚焦时响应回车，避免内部删除按钮的按键冒泡触发跳转
const onCardEnter = (e, id) => {
  if (e.target !== e.currentTarget) return;
  goToDetail(id);
};

const handleDelete = async (id) => {
  try {
    await deleteKb(id);
    message.success('已删除');
    loadData();
  } catch (e) {
    // 错误已由拦截器处理
  }
};

const isOwner = (item) => item.ownerUserId === userStore.userId;

const formatDate = (date) => (date ? dayjs(date).format('YYYY-MM-DD') : '');

// 描述被两行截断时，hover 展示完整内容；未截断则不弹提示
// activeDescId 记录当前弹出的卡片 id，保证同一时刻只有一张卡片的提示展示
// 标题直接绑定各卡片自身的 description，避免提示淡出过程中渲染出空白内容
const activeDescId = ref(null);

const onDescEnter = (e, item) => {
  const el = e.currentTarget;
  // 触屏设备无 hover 语义，避免点击跳转时闪烁
  if (!window.matchMedia('(hover: hover)').matches) return;
  if (el.scrollHeight > el.clientHeight + 1) {
    activeDescId.value = item.id;
  }
};

const onDescLeave = (item) => {
  if (activeDescId.value === item.id) {
    activeDescId.value = null;
  }
};

// 后端 visibility 可能是字符串或枚举对象，统一取出值
const visibilityMeta = (item) => {
  const value = typeof item.visibility === 'string' ? item.visibility : (item.visibility?.value || '');
  if (value === 'shared') return {label: '工作空间共享', color: 'green'};
  if (value === 'public') return {label: '公开', color: 'orange'};
  return {label: '私有', color: 'default'};
};
</script>

<template>
  <div class="kb-page">
    <div class="kb-page-header">
      <div class="kb-page-heading">
        <h2>知识库管理</h2>
        <p>共 {{ totalCount }} 个知识库，点击卡片进入详情管理文档</p>
      </div>
      <a-button type="primary" class="kb-create-btn" @click="isModalVisible = true">
        <template #icon><PlusOutlined/></template>
        新建知识库
      </a-button>
    </div>

    <a-tabs v-model:activeKey="activeTab" class="kb-tabs">
      <a-tab-pane v-for="cat in categories" :key="cat.key">
        <template #tab>
          <span class="kb-tab-label">
            <component :is="cat.icon"/>
            {{ cat.label }}
            <span class="kb-tab-count">{{ kbMap[cat.key].length }}</span>
          </span>
        </template>

        <!-- 加载骨架屏 -->
        <div v-if="isLoading" class="kb-grid">
          <div v-for="n in 4" :key="n" class="kb-skeleton">
            <div class="kb-skeleton-head">
              <span class="kb-skeleton-icon"></span>
              <span class="kb-skeleton-line w-50"></span>
            </div>
            <span class="kb-skeleton-line w-90"></span>
            <span class="kb-skeleton-line w-30"></span>
          </div>
        </div>

        <!-- 空状态 -->
        <div v-else-if="!kbMap[cat.key].length" class="kb-empty">
          <a-empty :description="cat.emptyText">
            <a-button v-if="cat.key === 'private'" type="primary" @click="isModalVisible = true">新建知识库</a-button>
          </a-empty>
        </div>

        <!-- 卡片网格 -->
        <div v-else class="kb-grid">
          <a-card
              v-for="item in kbMap[cat.key]"
              :key="item.id"
              hoverable
              class="kb-card"
              role="button"
              tabindex="0"
              @click="goToDetail(item.id)"
              @keydown.enter="onCardEnter($event, item.id)"
          >
            <div class="kb-card-main">
              <div class="kb-card-head">
                <span class="kb-card-icon" :style="{color: cat.color, backgroundColor: cat.tint}">
                  <component :is="cat.icon"/>
                </span>
                <span class="kb-card-name" :title="item.name">{{ item.name }}</span>
                <a-popconfirm
                    v-if="isOwner(item)"
                    title="确认删除该知识库？"
                    description="删除后无法恢复，且会删除关联的所有文档。"
                    ok-text="确认删除"
                    cancel-text="取消"
                    @confirm="handleDelete(item.id)"
                >
                  <a-button type="text" danger size="small" class="kb-card-delete" @click.stop>
                    <template #icon><DeleteOutlined/></template>
                  </a-button>
                </a-popconfirm>
              </div>
              <a-tooltip :open="activeDescId === item.id" :title="item.description || '暂无描述'" placement="top" overlay-class-name="kb-desc-tooltip">
                <p class="kb-card-desc" @mouseenter="onDescEnter($event, item)" @mouseleave="onDescLeave(item)">{{ item.description || '暂无描述' }}</p>
              </a-tooltip>
            </div>
            <div class="kb-card-footer">
              <a-tag :color="visibilityMeta(item).color">{{ visibilityMeta(item).label }}</a-tag>
              <span class="kb-card-date">{{ formatDate(item.createdAt) }}</span>
            </div>
          </a-card>
        </div>
      </a-tab-pane>
    </a-tabs>

    <a-modal v-model:open="isModalVisible" title="新建知识库" @ok="handleCreate" class="create-kb-modal">
      <a-form layout="vertical">
        <a-form-item label="名称" required>
          <a-input v-model:value="formState.name" placeholder="请输入知识库名称"/>
        </a-form-item>
        <a-form-item label="描述">
          <a-textarea v-model:value="formState.description" placeholder="请输入简要描述"/>
        </a-form-item>
        <a-form-item label="可见性">
          <a-radio-group v-model:value="formState.visibility" option-type="button">
            <a-radio value="private">私有 (仅自己可见)</a-radio>
            <a-radio value="shared">共享 (同工作空间可见)</a-radio>
            <a-radio value="public">公开 (所有人可见)</a-radio>
          </a-radio-group>
        </a-form-item>
      </a-form>
    </a-modal>
  </div>
</template>

<style scoped>
.kb-page {
  padding: 24px;
}

.kb-page-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  margin-bottom: 20px;
}

.kb-page-heading h2 {
  margin: 0 0 4px;
  font-size: 22px;
  font-weight: 600;
  color: v-bind('token.colorTextHeading');
}

.kb-page-heading p {
  margin: 0;
  font-size: 13px;
  color: v-bind('token.colorTextDescription');
}

/* ============ Tabs ============ */
.kb-tab-label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.kb-tab-count {
  padding: 0 8px;
  border-radius: 10px;
  font-size: 12px;
  line-height: 20px;
  background: v-bind('token.colorFillSecondary');
  color: v-bind('token.colorTextDescription');
}

.ant-tabs-tab-active .kb-tab-count {
  background: v-bind('token.colorPrimaryBg');
  color: v-bind('token.colorPrimary');
}

/* ============ 卡片网格 ============ */
.kb-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 16px;
  align-items: stretch;
}

.kb-card {
  height: 100%;
  cursor: pointer;
  border-radius: 12px;
  transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}

.kb-card:hover {
  transform: translateY(-3px);
  border-color: v-bind('token.colorPrimaryBorder');
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.12);
}

.kb-card :deep(.ant-card-body) {
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 20px;
}

.kb-card-main {
  flex: 1;
}

.kb-card-head {
  display: flex;
  align-items: center;
  gap: 10px;
}

.kb-card-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: 10px;
  font-size: 18px;
  flex: none;
}

.kb-card-name {
  flex: 1;
  min-width: 0;
  font-size: 15px;
  font-weight: 600;
  color: v-bind('token.colorText');
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.kb-card-delete {
  flex: none;
  opacity: 0;
  transition: opacity 0.2s ease;
}

.kb-card:hover .kb-card-delete,
.kb-card:focus-within .kb-card-delete {
  opacity: 1;
}

.kb-card:focus-visible {
  outline: 2px solid v-bind('token.colorPrimaryBorder');
  outline-offset: 2px;
}

.kb-card-desc {
  margin: 12px 0 0;
  font-size: 13px;
  line-height: 1.6;
  color: v-bind('token.colorTextSecondary');
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  min-height: 42px;
}

.kb-card-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px solid v-bind('token.colorSplit');
}

.kb-card-date {
  font-size: 12px;
  color: v-bind('token.colorTextTertiary');
}

/* 描述 Tooltip 悬浮在 body 下，需 :global 穿透 scoped */
:global(.kb-desc-tooltip .ant-tooltip-inner) {
  max-width: 340px;
}

/* ============ 骨架屏 ============ */
.kb-skeleton {
  border: 1px solid v-bind('token.colorSplit');
  border-radius: 12px;
  padding: 20px;
  background: v-bind('token.colorBgContainer');
}

.kb-skeleton-head {
  display: flex;
  align-items: center;
  gap: 12px;
}

.kb-skeleton-icon {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  background: v-bind('token.colorFillSecondary');
  flex: none;
}

.kb-skeleton-line {
  display: block;
  height: 12px;
  border-radius: 6px;
  margin-top: 12px;
  background: v-bind('token.colorFillSecondary');
  animation: kb-shimmer 1.2s ease-in-out infinite;
}

.kb-skeleton-head .kb-skeleton-line {
  margin-top: 0;
  flex: 1;
}

.w-50 { width: 50%; }
.w-90 { width: 90%; }
.w-30 { width: 30%; }

@keyframes kb-shimmer {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.45; }
}

/* ============ 空状态 ============ */
.kb-empty {
  padding: 48px 0;
}

/* ============ 响应式：PC / 移动端 ============ */
@media (hover: none) {
  /* 触屏设备无悬停态，删除按钮常驻 */
  .kb-card-delete {
    opacity: 1;
  }

  /* 触屏点击后 :hover 会残留，取消上浮反馈 */
  .kb-card:hover {
    transform: none;
    box-shadow: none;
  }
}

@media (max-width: 768px) {
  .kb-page {
    padding: 16px;
  }

  .kb-page-header {
    flex-direction: column;
    align-items: stretch;
    gap: 12px;
  }

  .kb-create-btn {
    width: 100%;
  }
}

@media (max-width: 576px) {
  .kb-grid {
    grid-template-columns: 1fr;
    gap: 12px;
  }
}

.create-kb-modal {
  min-width: 380px;
}

@media (max-width: 768px) {
  .create-kb-modal {
    min-width: auto;
    width: 90% !important;
  }
}
</style>