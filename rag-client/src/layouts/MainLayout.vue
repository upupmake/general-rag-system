<script setup>
import {ref, watch, computed, nextTick, onMounted, onUnmounted, defineAsyncComponent} from 'vue'
import {useRouter, useRoute} from 'vue-router'
import SessionList from '@/components/SessionList.vue'
// 游戏组件按需异步加载，避免求解器/素材进入首屏 bundle
const Game2048 = defineAsyncComponent(() => import('@/components/Game2048.vue'))
const Gomoku = defineAsyncComponent(() => import('@/components/Gomoku.vue'))
const PuzzleGame = defineAsyncComponent(() => import('@/components/PuzzleGame.vue'))
const SokobanGame = defineAsyncComponent(() => import('@/components/SokobanGame.vue'))
import {useThemeStore} from '@/stores/theme'
import {useUserStore} from '@/stores/user'
import {
  LogoutOutlined, 
  UserOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  GithubOutlined,
  KeyOutlined,
  AppstoreOutlined,
  DownOutlined,
  GiftOutlined,
  BorderOutlined,
  PictureOutlined,
  BoxPlotOutlined
} from '@ant-design/icons-vue';
// 引入本地静态资源 URL
import lightThemeUrl from '@/assets/github-markdown.min.css?url';
import workspaceApi from '@/api/workspaceApi.js';
import {events} from '@/events.js'

const router = useRouter()
const route = useRoute()
const selectedKeys = ref([])
const themeStore = useThemeStore();
const userStore = useUserStore();
const currentWorkspace = ref(null)
const isFooterExpanded = ref(false) // 控制底部用户菜单展开/收起
const footerMenuRef = ref(null)
const collapsed = ref(false) // 控制侧边栏收起/展开
const isMobile = ref(false)
const isGameOpen = ref(false) // 2048 小游戏弹窗
const isGomokuOpen = ref(false) // 五子棋 小游戏弹窗
const isPuzzleOpen = ref(false) // 图片拼图 小游戏弹窗
const isSokobanOpen = ref(false) // 推箱子 小游戏弹窗

// 加载当前工作空间信息
const loadCurrentWorkspace = async () => {
  try {
    const data = await workspaceApi.getWorkspaces()
    currentWorkspace.value = data.current
  } catch (error) {
    console.error('加载工作空间失败:', error)
  }
}

const checkIsMobile = () => {
  isMobile.value = window.innerWidth <= 768
}

const handleDocumentClick = (event) => {
  if (footerMenuRef.value && !footerMenuRef.value.contains(event.target)) {
    isFooterExpanded.value = false
  }
}

onMounted(() => {
  checkIsMobile()
  window.addEventListener('resize', handleWindowResize)
  document.addEventListener('click', handleDocumentClick)
  loadCurrentWorkspace()
  if (userStore.token) {
    userStore.fetchCurrentUser().catch(() => {
      router.push('/login')
    })
  }
  events.on('sessionTitleUpdated', ({sessionId: sid, title}) => {
    if (String(route.params.sessionId) === String(sid)) {
      currentSessionTitle.value = title
      updatePageTitle()
    }
  })
  restoreMenuBtnPos()
})

onUnmounted(() => {
  window.removeEventListener('resize', handleWindowResize)
  document.removeEventListener('click', handleDocumentClick)
  events.off('sessionTitleUpdated')
})

// 计算用户显示名称
const userDisplayName = computed(() => {
  return userStore.user?.username || userStore.user?.email || '未登录'
})

const isMemberUser = computed(() => {
  const user = userStore.user
  const roleId = user?.roleId ?? user?.role_id ?? user?.role?.id
  return Number(roleId) === 1
})

const userRoleLabel = computed(() => isMemberUser.value ? '会员用户' : '普通用户')

const membershipExpireDate = computed(() => {
  const bz = userStore.user?.bz || ''
  const match = bz.match(/^(\d{4}-\d{2}-\d{2})\s+开通(\d+)天$/)
  if (!match) {
    return ''
  }

  const startDate = new Date(`${match[1]}T00:00:00`)
  startDate.setDate(startDate.getDate() + Number(match[2]))
  const year = startDate.getFullYear()
  const month = String(startDate.getMonth() + 1).padStart(2, '0')
  const day = String(startDate.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
})

// 计算工作空间显示名称
const workspaceDisplayName = computed(() => {
  return currentWorkspace.value?.name || '未选择工作空间'
})

// 切换底部用户菜单展开状态
const toggleFooter = () => {
  isFooterExpanded.value = !isFooterExpanded.value
}

// 打开 2048 小游戏弹窗(同时收起用户菜单)
const openGame = () => {
  isFooterExpanded.value = false
  isGameOpen.value = true
}

// 弹窗可见性同步到游戏状态: 打开时恢复 AI 托管, 关闭时挂起(棋盘/分数/开关状态保留)
watch(isGameOpen, async (open) => {
  const {useGame2048Store} = await import('@/stores/game2048')
  useGame2048Store().setActive(open)
})

// 打开 五子棋 小游戏弹窗(同时收起用户菜单)
const openGomoku = () => {
  isFooterExpanded.value = false
  isGomokuOpen.value = true
}

// 关闭弹窗终止 AI 计算并保留棋局, 重新打开后按需续算
watch(isGomokuOpen, async (open) => {
  const {useGameGomokuStore} = await import('@/stores/gameGomoku')
  useGameGomokuStore().setActive(open)
})

// 打开 图片拼图 小游戏弹窗(同时收起用户菜单)
const openPuzzle = () => {
  isFooterExpanded.value = false
  isPuzzleOpen.value = true
}

// 关闭弹窗强制关停 AI 托管(含 Worker), 拼图状态保留
watch(isPuzzleOpen, async (open) => {
  const {useGamePuzzleStore} = await import('@/stores/gamePuzzle')
  useGamePuzzleStore().setActive(open)
})

// 打开 推箱子 小游戏弹窗(同时收起用户菜单)
const openSokoban = () => {
  isFooterExpanded.value = false
  isSokobanOpen.value = true
}

// 关闭弹窗强制关停 AI 托管(含 Worker), 棋盘状态保留
watch(isSokobanOpen, async (open) => {
  const {useGameSokobanStore} = await import('@/stores/gameSokoban')
  useGameSokobanStore().setActive(open)
})

// 切换侧边栏收起/展开
const toggleCollapsed = () => {
  collapsed.value = !collapsed.value
  isFooterExpanded.value = false
}

// ============ 移动端展开按钮拖动 ============
const MENU_BTN_POS_KEY = 'mobile-menu-btn-pos'
const MENU_BTN_MARGIN = 8
const MENU_BTN_DEFAULT_SIZE = 40

const menuBtnRef = ref(null)
const menuBtnPos = ref(null) // {left, top}，null 表示使用默认位置
const isMenuBtnDragging = ref(false)
let menuDrag = null
let menuDragRaf = 0
let menuDragPending = null
let suppressMenuBtnClick = false

const getMenuBtnEl = () => menuBtnRef.value?.$el || menuBtnRef.value

const clampMenuBtnPos = (left, top) => {
  const rect = getMenuBtnEl()?.getBoundingClientRect()
  const width = rect?.width || MENU_BTN_DEFAULT_SIZE
  const height = rect?.height || MENU_BTN_DEFAULT_SIZE
  const maxLeft = Math.max(MENU_BTN_MARGIN, window.innerWidth - width - MENU_BTN_MARGIN)
  const maxTop = Math.max(MENU_BTN_MARGIN, window.innerHeight - height - MENU_BTN_MARGIN)
  return {
    left: Math.min(Math.max(left, MENU_BTN_MARGIN), maxLeft),
    top: Math.min(Math.max(top, MENU_BTN_MARGIN), maxTop)
  }
}

const restoreMenuBtnPos = () => {
  try {
    const saved = JSON.parse(localStorage.getItem(MENU_BTN_POS_KEY) || 'null')
    if (saved && Number.isFinite(saved.left) && Number.isFinite(saved.top)) {
      menuBtnPos.value = clampMenuBtnPos(saved.left, saved.top)
    }
  } catch (e) {
    localStorage.removeItem(MENU_BTN_POS_KEY)
  }
}

const onMenuBtnPointerDown = (e) => {
  if (e.pointerType === 'mouse' && e.button !== 0) return
  const el = getMenuBtnEl()
  if (!el) return
  const rect = el.getBoundingClientRect()
  menuDrag = {
    pointerId: e.pointerId,
    startX: e.clientX,
    startY: e.clientY,
    offsetX: e.clientX - rect.left,
    offsetY: e.clientY - rect.top,
    // 缓存尺寸，拖动过程中不再读取布局，避免每帧强制 reflow
    width: rect.width,
    height: rect.height,
    left: rect.left,
    top: rect.top,
    moved: false
  }
  window.addEventListener('pointermove', onMenuBtnPointerMove)
  window.addEventListener('pointerup', onMenuBtnPointerUp)
  window.addEventListener('pointercancel', onMenuBtnPointerUp)
}

const computeMenuBtnPos = (x, y) => {
  const maxLeft = Math.max(MENU_BTN_MARGIN, window.innerWidth - menuDrag.width - MENU_BTN_MARGIN)
  const maxTop = Math.max(MENU_BTN_MARGIN, window.innerHeight - menuDrag.height - MENU_BTN_MARGIN)
  return {
    left: Math.min(Math.max(x - menuDrag.offsetX, MENU_BTN_MARGIN), maxLeft),
    top: Math.min(Math.max(y - menuDrag.offsetY, MENU_BTN_MARGIN), maxTop)
  }
}

const onMenuBtnPointerMove = (e) => {
  if (!menuDrag || e.pointerId !== menuDrag.pointerId) return
  if (!menuDrag.moved && Math.hypot(e.clientX - menuDrag.startX, e.clientY - menuDrag.startY) < 5) return
  if (!menuDrag.moved) {
    menuDrag.moved = true
    isMenuBtnDragging.value = true
  }
  e.preventDefault()
  menuDragPending = {x: e.clientX, y: e.clientY}
  if (menuDragRaf) return
  // 每帧最多更新一次，并直接写入 DOM 样式，保证跟手
  menuDragRaf = requestAnimationFrame(() => {
    menuDragRaf = 0
    if (!menuDrag || !menuDragPending) return
    const pos = computeMenuBtnPos(menuDragPending.x, menuDragPending.y)
    menuDrag.left = pos.left
    menuDrag.top = pos.top
    const el = getMenuBtnEl()
    if (el) {
      el.style.left = `${pos.left}px`
      el.style.top = `${pos.top}px`
    }
    // 同步响应式状态，避免其他状态更新重渲染时位置回退
    menuBtnPos.value = pos
  })
}

const onMenuBtnPointerUp = (e) => {
  if (!menuDrag || (e.pointerId !== undefined && e.pointerId !== menuDrag.pointerId)) return
  if (menuDragRaf) {
    cancelAnimationFrame(menuDragRaf)
    menuDragRaf = 0
  }
  if (menuDrag.moved) {
    // 用松手时的坐标补齐最后一帧，保证落点与手指一致
    const pointerX = e.type !== 'pointercancel' ? e.clientX : menuDragPending?.x
    const pointerY = e.type !== 'pointercancel' ? e.clientY : menuDragPending?.y
    if (Number.isFinite(pointerX) && Number.isFinite(pointerY)) {
      const pos = computeMenuBtnPos(pointerX, pointerY)
      menuDrag.left = pos.left
      menuDrag.top = pos.top
    }
    menuBtnPos.value = {left: menuDrag.left, top: menuDrag.top}
    // 直接写入 DOM，避免 Vue 异步更新造成松手瞬间的位置回跳
    const el = getMenuBtnEl()
    if (el) {
      el.style.left = `${menuDrag.left}px`
      el.style.top = `${menuDrag.top}px`
    }
    localStorage.setItem(MENU_BTN_POS_KEY, JSON.stringify(menuBtnPos.value))
    // 拖动结束后浏览器仍会触发 click，短暂拦截避免误展开菜单
    suppressMenuBtnClick = true
    setTimeout(() => {
      suppressMenuBtnClick = false
    }, 300)
  }
  menuDrag = null
  menuDragPending = null
  isMenuBtnDragging.value = false
  window.removeEventListener('pointermove', onMenuBtnPointerMove)
  window.removeEventListener('pointerup', onMenuBtnPointerUp)
  window.removeEventListener('pointercancel', onMenuBtnPointerUp)
}

const handleMenuBtnClick = () => {
  if (suppressMenuBtnClick) return
  toggleCollapsed()
}

const handleWindowResize = () => {
  checkIsMobile()
  if (menuBtnPos.value) {
    menuBtnPos.value = clampMenuBtnPos(menuBtnPos.value.left, menuBtnPos.value.top)
  }
}

// 按钮出现后按实际尺寸再校正一次位置（如横竖屏切换后）
watch(collapsed, (visible) => {
  if (visible && menuBtnPos.value) {
    nextTick(() => {
      menuBtnPos.value = clampMenuBtnPos(menuBtnPos.value.left, menuBtnPos.value.top)
    })
  }
})

// 退出登录
const handleLogout = () => {
  userStore.logout()
}

// 动态加载 CSS 的函数
const loadMarkdownTheme = () => {
  const existingLink = document.getElementById('markdown-theme-css');
  if (existingLink) {
    existingLink.remove();
  }

  const link = document.createElement('link');
  link.id = 'markdown-theme-css';
  link.rel = 'stylesheet';
  link.href = lightThemeUrl;
  document.head.appendChild(link);
};

loadMarkdownTheme();

// 页面标题映射
const pageTitleMap = {
  '/chat/new': '新聊天',
  '/dashboard': 'Dashboard',
  '/kb': '知识库',
  '/search': '搜索对话',
  '/access-keys': 'Access Key',
  '/workspaces': '工作空间',
}

const currentSessionTitle = ref('')

const updatePageTitle = () => {
  let pageTitle = ''
  if (route.path.startsWith('/chat/') && !route.path.startsWith('/chat/new')) {
    pageTitle = currentSessionTitle.value || '对话'
  } else if (route.path.startsWith('/kb/')) {
    pageTitle = '知识库'
  } else {
    pageTitle = pageTitleMap[route.path] || ''
  }
  document.title = pageTitle ? `${pageTitle} - RAG知识系统` : 'RAG知识系统'
}

/**
 * 同步菜单选中状态
 */
watch(
    () => route.path,
    () => {
      if (route.path.startsWith('/chat/new')) {
        selectedKeys.value = ['new-chat']
      } else if (route.path.startsWith('/dashboard')) {
        selectedKeys.value = ['dashboard']
      } else if (route.path.startsWith('/kb')) {
        selectedKeys.value = ['kb']
      } else if (route.path.startsWith('/search')) {
        selectedKeys.value = ['search']
      } else if (route.path.startsWith('/access-keys')) {
        selectedKeys.value = ['access-keys']
      } else if (route.path.startsWith('/workspaces')) {
        selectedKeys.value = ['workspaces']
      } else {
        selectedKeys.value = []
      }
      updatePageTitle()
    },
    {immediate: true}
)

// 切换会话时重置标题，实际标题由 SessionList 通过 sessionTitleUpdated 事件提供
watch(
    () => route.params.sessionId,
    () => {
      currentSessionTitle.value = ''
      updatePageTitle()
    },
    {immediate: true}
)

// 初始化加载工作空间信息（移至 onMounted）
// loadCurrentWorkspace()

const go = (path) => {
  // 在对话页面切换到其他页面时，在新标签页打开
  const isOnChatSession = route.path.startsWith('/chat/') && !route.path.startsWith('/chat/new')
  if (isOnChatSession) {
    window.open(path, '_blank')
    return
  }
  router.push(path)
}
</script>

<template>
  <a-layout style="min-height: 100vh">
    <!-- 移动端遮罩层 -->
    <div 
      v-if="!collapsed" 
      class="sidebar-overlay" 
      @click="toggleCollapsed"
    ></div>
    
    <a-layout-sider 
      v-model:collapsed="collapsed"
      :width="240" 
      :collapsed-width="isMobile ? 0 : 64"
      :theme="themeStore.isDark ? 'dark' : 'light'" 
      style="position: fixed; left: 0; top: 0; bottom: 0; z-index: 100;"
      collapsible
      :trigger="null">
      <div class="sidebar-container">
        <!-- Logo -->
        <div class="logo"
             :style="{ color: themeStore.isDark ? '#fff' : 'inherit', borderBottomColor: themeStore.isDark ? '#303030' : '#f0f0f0' }"
             @click="go('/dashboard')">
          <img v-if="!collapsed" src="@/assets/logo.png" alt="Logo" style="height: 50px"/>
          <img v-else src="@/assets/logo.png" alt="Logo" style="height: 32px"/>
          <span v-if="!collapsed">RAG 系统</span>
        </div>
        
        <!-- 功能菜单 -->
        <a-menu class="function-menu" mode="inline" :selectedKeys="selectedKeys" :theme="themeStore.isDark ? 'dark' : 'light'" :inline-collapsed="collapsed">
          <a-menu-item key="new-chat" @click="go('/chat/new')">
            <template #icon>
              <span style="font-size: 16px;">➕</span>
            </template>
            新聊天
          </a-menu-item>
          <a-menu-item key="dashboard" @click="go('/dashboard')">
            <template #icon>
              <span style="font-size: 16px;">📊</span>
            </template>
            Dashboard
          </a-menu-item>

          <a-menu-item key="kb" @click="go('/kb')">
            <template #icon>
              <span style="font-size: 16px;">📚</span>
            </template>
            知识库
          </a-menu-item>
          <a-menu-item key="workspaces" @click="go('/workspaces')">
            <template #icon>
              <span style="font-size: 16px;">🏢</span>
            </template>
            工作空间
          </a-menu-item>
          <a-menu-item key="search" @click="go('/search')">
            <template #icon>
              <span style="font-size: 16px;">🔍</span>
            </template>
            搜索对话
          </a-menu-item>
          <a-menu-item key="access-keys" @click="go('/access-keys')">
            <template #icon>
              <key-outlined />
            </template>
            Access Key
          </a-menu-item>
        </a-menu>

        <div class="mobile-function-nav">
          <a-dropdown :trigger="['click']" placement="bottomLeft" overlay-class-name="mobile-function-dropdown">
            <a-button class="mobile-function-nav-trigger">
              <template #icon>
                <appstore-outlined />
              </template>
              <span>功能菜单</span>
              <down-outlined class="mobile-function-nav-chevron" />
            </a-button>
            <template #overlay>
              <a-menu :selectedKeys="selectedKeys" :theme="themeStore.isDark ? 'dark' : 'light'">
                <a-menu-item key="new-chat" @click="go('/chat/new')">
                  <template #icon><span>➕</span></template>
                  新聊天
                </a-menu-item>
                <a-menu-item key="dashboard" @click="go('/dashboard')">
                  <template #icon><span>📊</span></template>
                  Dashboard
                </a-menu-item>
                <a-menu-item key="kb" @click="go('/kb')">
                  <template #icon><span>📚</span></template>
                  知识库
                </a-menu-item>
                <a-menu-item key="workspaces" @click="go('/workspaces')">
                  <template #icon><span>🏢</span></template>
                  工作空间
                </a-menu-item>
                <a-menu-item key="search" @click="go('/search')">
                  <template #icon><span>🔍</span></template>
                  搜索对话
                </a-menu-item>
                <a-menu-item key="access-keys" @click="go('/access-keys')">
                  <template #icon><key-outlined /></template>
                  Access Key
                </a-menu-item>
              </a-menu>
            </template>
          </a-dropdown>
        </div>

        <!-- 最近会话（独立区域） - 收起时隐藏 -->
        <div v-if="!collapsed" class="session-wrapper">
          <div class="session-title">最近聊天</div>
          <div class="session-list-container">
            <SessionList/>
          </div>
        </div>

        <!-- 底部信息区域 -->
        <div class="sidebar-footer" :style="{ 
          borderTopColor: themeStore.isDark ? '#303030' : '#f0f0f0',
          backgroundColor: themeStore.isDark ? '#141414' : '#fafafa'}">
          <div v-if="!collapsed" ref="footerMenuRef" class="footer-menu-wrapper">
            <!-- 用户菜单 -->
            <transition name="slide-fade-up">
              <div v-if="isFooterExpanded" class="action-buttons">
                <!-- 收起/展开按钮 -->
                <a-button type="text" @click="toggleCollapsed" class="collapse-btn">
                  <template #icon>
                    <menu-fold-outlined v-if="!collapsed" />
                    <menu-unfold-outlined v-else />
                  </template>
                  <span v-if="!collapsed">收起侧边栏</span>
                </a-button>
                
                <!-- GitHub 开源地址 -->
                <a-button type="text" href="https://github.com/cockmake/general-rag-system" target="_blank" title="GitHub 开源地址">
                  <template #icon>
                    <github-outlined />
                  </template>
                  GitHub 开源地址
                </a-button>

                <!-- 2048 小游戏 -->
                <a-button type="text" @click.stop="openGame" title="2048">
                  <template #icon>
                    <gift-outlined />
                  </template>
                  2048
                </a-button>

                <!-- 五子棋 小游戏 -->
                <a-button type="text" @click.stop="openGomoku" title="五子棋">
                  <template #icon>
                    <border-outlined />
                  </template>
                  五子棋
                </a-button>

                <!-- 图片拼图 小游戏 -->
                <a-button type="text" @click.stop="openPuzzle" title="图片拼图">
                  <template #icon>
                    <picture-outlined />
                  </template>
                  图片拼图
                </a-button>

                <!-- 推箱子 小游戏 -->
                <a-button type="text" @click.stop="openSokoban" title="推箱子">
                  <template #icon>
                    <box-plot-outlined />
                  </template>
                  推箱子
                </a-button>

                <!-- 退出登录 -->
                <a-button type="text" danger @click.stop="handleLogout" title="退出登录">
                  <template #icon>
                    <logout-outlined />
                  </template>
                  退出登录
                </a-button>
              </div>
            </transition>
            
            <!-- 用户信息卡片 -->
            <div class="footer-info-section" @click.stop="toggleFooter">
              <div class="user-avatar" :class="{ 'member-avatar': isMemberUser }">
                <user-outlined />
              </div>
              <div class="user-meta">
                <div class="user-info" :style="{ color: themeStore.isDark ? '#fff' : '#000' }">
                  <span class="user-name" :class="{ 'member-user-name': isMemberUser }">{{ userDisplayName }}</span>
                  <span class="user-role-tag" :class="{ 'member-role-tag': isMemberUser }">{{ userRoleLabel }}</span>
                </div>
                <div v-if="isMemberUser && membershipExpireDate" class="membership-expire" :style="{ color: themeStore.isDark ? '#999' : '#666' }">
                  到期：{{ membershipExpireDate }}
                </div>
                <div class="workspace-info" :style="{ color: themeStore.isDark ? '#999' : '#666' }">
                  🏢 {{ workspaceDisplayName }}
                </div>
              </div>
            </div>
          </div>
          
          <!-- 收起状态下的用户图标 -->
          <div v-else class="footer-collapsed-icon" @click="toggleCollapsed" :style="{ 
            color: themeStore.isDark ? '#fff' : '#000',
            cursor: 'pointer'
          }">
            <user-outlined style="font-size: 20px;" />
          </div>
        </div>
      </div>
    </a-layout-sider>

    <!-- PC端侧边栏边缘半圆切换手柄 -->
    <div
      v-if="!isMobile"
      class="sidebar-toggle-handle"
      :style="{ left: collapsed ? '56px' : '232px' }"
      @click="toggleCollapsed"
    >
      <menu-fold-outlined v-if="!collapsed" />
      <menu-unfold-outlined v-else />
    </div>

    <a-layout :style="{ marginLeft: isMobile ? '0px' : (collapsed ? '64px' : '240px'), transition: 'margin-left 0.2s' }">
      <!-- 移动端展开按钮 -->
      <a-button 
        v-if="collapsed"
        ref="menuBtnRef"
        class="mobile-menu-btn"
        :class="{ 'is-dragging': isMenuBtnDragging }"
        type="primary" 
        shape="circle"
        :style="menuBtnPos ? { left: menuBtnPos.left + 'px', top: menuBtnPos.top + 'px' } : null"
        @pointerdown="onMenuBtnPointerDown"
        @click="handleMenuBtnClick"
      >
        <template #icon>
          <menu-unfold-outlined />
        </template>
      </a-button>
      
      <a-layout-content style="padding: 0">
        <router-view/>
      </a-layout-content>
    </a-layout>

    <!-- 2048 小游戏弹窗 -->
    <a-modal
      v-model:open="isGameOpen"
      :title="null"
      :footer="null"
      :centered="true"
      :width="isMobile ? '94%' : 440"
      :body-style="{ padding: isMobile ? '12px' : '24px' }"
      :mask-closable="false"
      destroy-on-close
    >
      <Game2048 />
    </a-modal>

    <!-- 五子棋 小游戏弹窗 -->
    <a-modal
      v-model:open="isGomokuOpen"
      :title="null"
      :footer="null"
      :centered="true"
      :width="isMobile ? '96%' : 640"
      :body-style="{ padding: isMobile ? '8px' : '20px' }"
      :mask-closable="false"
      destroy-on-close
    >
      <Gomoku />
    </a-modal>

    <!-- 图片拼图 小游戏弹窗 -->
    <a-modal
      v-model:open="isPuzzleOpen"
      :title="null"
      :footer="null"
      :centered="true"
      :width="isMobile ? '94%' : 460"
      :body-style="{ padding: isMobile ? '12px' : '24px' }"
      :mask-closable="false"
      destroy-on-close
    >
      <PuzzleGame />
    </a-modal>

    <!-- 推箱子 小游戏弹窗 -->
    <a-modal
      v-model:open="isSokobanOpen"
      :title="null"
      :footer="null"
      :centered="true"
      :width="isMobile ? '94%' : 600"
      :body-style="{ padding: isMobile ? '12px' : '24px' }"
      :mask-closable="false"
      destroy-on-close
    >
      <SokobanGame />
    </a-modal>
  </a-layout>
</template>

<style scoped>
.sidebar-container {
  height: 100%;
  display: flex;
  flex-direction: column;
  position: relative;
}

.logo {
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  border-bottom: 1px solid #f0f0f0;
  flex-shrink: 0;
  gap: 8px;
  transition: all 0.2s;
  cursor: pointer;
}

.session-wrapper {
  border-top: 1px solid #f0f0f0;
  padding: 8px 4px;
  overflow: hidden;
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.session-title {
  font-size: 12px;
  color: #999;
  padding: 4px 8px;
  flex-shrink: 0;
}

.session-list-container {
  flex: 1;
  min-height: 0;
  overflow: hidden; /* Ensure SessionList's height:100% works against this */
}

.sidebar-footer {
  flex-shrink: 0;
  margin-top: auto;
  padding: 12px;
  border-top: 1px solid #f0f0f0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.footer-menu-wrapper {
  position: relative;
}

.collapse-btn {
  justify-content: flex-start;
  text-align: left;
  width: 100%;
}

.footer-info-section {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px;
  border-radius: 12px;
  cursor: pointer;
  background: rgba(0, 0, 0, 0.03);
  border: 1px solid rgba(0, 0, 0, 0.06);
  transition: background-color 0.2s, border-color 0.2s, box-shadow 0.2s;
}

.footer-info-section:hover {
  background-color: rgba(0, 0, 0, 0.05);
  border-color: rgba(0, 0, 0, 0.1);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
}

[data-theme="dark"] .footer-info-section {
  background: rgba(255, 255, 255, 0.04);
  border-color: rgba(255, 255, 255, 0.08);
}

[data-theme="dark"] .footer-info-section:hover {
  background-color: rgba(255, 255, 255, 0.08);
  border-color: rgba(255, 255, 255, 0.14);
}

.user-avatar {
  width: 34px;
  height: 34px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  color: #1677ff;
  background: rgba(22, 119, 255, 0.12);
}

.member-avatar {
  color: #722ed1;
  background: linear-gradient(135deg, rgba(114, 46, 209, 0.16), rgba(250, 173, 20, 0.2));
}

.user-meta {
  min-width: 0;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.footer-collapsed-icon {
  display: flex;
  justify-content: center;
  padding: 8px 0;
  transition: all 0.2s;
}

.footer-collapsed-icon:hover {
  background-color: rgba(0, 0, 0, 0.04);
  border-radius: 4px;
}

[data-theme="dark"] .footer-collapsed-icon:hover {
  background-color: rgba(255, 255, 255, 0.08);
}

.user-info {
  display: flex;
  align-items: center;
  font-size: 14px;
  font-weight: 500;
}

.user-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}

.member-user-name {
  color: #722ed1;
}

.user-role-tag {
  flex-shrink: 0;
  margin-left: 8px;
  padding: 1px 6px;
  border-radius: 10px;
  font-size: 12px;
  background: rgba(0, 0, 0, 0.06);
}

.member-role-tag {
  color: #722ed1;
  background: rgba(114, 46, 209, 0.12);
}

.membership-expire {
  font-size: 12px;
  line-height: 1.2;
}

.workspace-info {
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.action-buttons {
  position: absolute;
  left: 0;
  right: 0;
  bottom: calc(100% + 8px);
  z-index: 10;
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 8px;
  border-radius: 12px;
  border: 1px solid rgba(0, 0, 0, 0.06);
  background: #fff;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
}

[data-theme="dark"] .action-buttons {
  border-color: rgba(255, 255, 255, 0.12);
  background: #1f1f1f;
}

.action-buttons .ant-btn {
  justify-content: flex-start;
  text-align: left;
}

/* 向上滑动淡入淡出动画 */
.slide-fade-up-enter-active {
  transition: all 0.3s ease-out;
}

.slide-fade-up-leave-active {
  transition: all 0.2s ease-in;
}

.slide-fade-up-enter-from {
  transform: translateY(10px);
  opacity: 0;
}

.slide-fade-up-leave-to {
  transform: translateY(10px);
  opacity: 0;
}

.mobile-function-nav {
  display: none;
}

:global(.mobile-function-dropdown .ant-dropdown-menu) {
  min-width: 208px;
}

/* 移动端响应式 */
@media (max-width: 768px) {
  /* 在移动端，侧边栏在收起时完全隐藏 */
  .sidebar-container {
    overflow-x: hidden;
  }

  .function-menu {
    display: none;
  }

  .mobile-function-nav {
    display: block;
    flex-shrink: 0;
    padding: 8px 12px 6px;
  }

  .mobile-function-nav-trigger {
    width: 100%;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    border-radius: 6px;
  }

  .mobile-function-nav-chevron {
    margin-left: auto;
    color: rgba(0, 0, 0, 0.45);
    font-size: 11px;
  }

  :global(body[data-theme='dark']) .mobile-function-nav-chevron {
    color: rgba(255, 255, 255, 0.45);
  }
}

/* 侧边栏边缘半圆切换手柄 */
.sidebar-toggle-handle {
  position: fixed;
  top: 50%;
  transform: translateY(-50%);
  width: 24px;
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  z-index: 101;
  transition: left 0.2s, background-color 0.2s, color 0.2s, box-shadow 0.2s, width 0.2s, height 0.2s;
  border-radius: 0 32px 32px 0;
  background: #fff;
  border: 1px solid #e8e8e8;
  border-left: none;
  color: #bbb;
  box-shadow: 3px 2px 12px rgba(0, 0, 0, 0.15);
  font-size: 12px;
}

.sidebar-toggle-handle:hover {
  width: 30px;
  height: 72px;
  background: #1677ff;
  color: #fff;
  box-shadow: 3px 2px 18px rgba(22, 119, 255, 0.4);
  border-color: #1677ff;
}

[data-theme="dark"] .sidebar-toggle-handle {
  background: #1f1f1f;
  border-color: #303030;
  color: #666;
}

[data-theme="dark"] .sidebar-toggle-handle:hover {
  background: #1677ff;
  color: #fff;
}

/* 移动端遮罩层 */
.sidebar-overlay {
  display: none;
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.45);
  z-index: 99;
}

/* 移动端展开按钮 */
.mobile-menu-btn {
  display: none;
  position: fixed;
  top: 16px;
  left: 80px;
  z-index: 101;
  cursor: grab;
  touch-action: none;
  user-select: none;
  /* 禁用 ant Button 默认的 transition: all，否则 left/top 变化会滞后于手指 */
  transition: none;
}

.mobile-menu-btn.is-dragging {
  cursor: grabbing;
}

@media (max-width: 768px) {
  .mobile-menu-btn {
    display: block;
    left: 16px;
    top: 80px;
  }
  
  .sidebar-overlay {
    display: block;
  }
}

</style>
