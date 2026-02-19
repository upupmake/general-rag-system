# RAG Client - 前端应用

前端界面模块，基于 Vue 3 + Vite + Ant Design Vue 构建。

## 核心技术

- **Vue 3.5.24** - Composition API
- **Vite 7.2.5** - rolldown-vite 构建工具
- **Ant Design Vue 4.2.6** + **Ant Design X Vue 1.5.0** - UI 组件库
- **Pinia 3.0.4** - 状态管理（支持 localStorage 持久化）
- **Vue Router 4.6.4** - 路由管理（Hash 模式）
- **Axios 1.13.2** - HTTP 客户端（Bearer Token 拦截器）
- **@microsoft/fetch-event-source** - SSE 流式响应
- **markdown-it** - Markdown 渲染引擎
  - highlight.js - 代码高亮
  - MathJax3 - 数学公式
  - markdown-it-task-lists - 任务列表
  - markdown-it-emoji - Emoji 支持

## 项目结构

```
src/
├── api/                    # API 接口封装
│   ├── request.js          # Axios 实例（Bearer Token拦截器）
│   ├── chatApi.js          # 对话相关 API
│   ├── kbApi.js            # 知识库 API
│   ├── workspaceApi.js     # 工作空间 API
│   ├── documentApi.js      # 文档管理 API
│   └── ...
│
├── stores/                 # Pinia 状态管理
│   ├── user.js             # 用户状态（登录信息、Token）
│   ├── theme.js            # 主题状态（深色/浅色模式）
│   ├── search.js           # 搜索状态
│   └── ...
│
├── router/                 # 路由配置
│   └── index.js            # 路由定义（beforeEach 认证守卫）
│
├── views/                  # 页面组件
│   ├── Login.vue           # 登录页面
│   ├── Dashboard.vue       # 仪表盘
│   ├── chat/               # 对话相关页面
│   │   ├── ChatPage.vue    # 对话主页面
│   │   └── ...
│   ├── kb/                 # 知识库相关页面
│   ├── workspace/          # 工作空间相关页面
│   └── ...
│
├── components/             # 公共组件
│   ├── ChatMessage.vue     # 消息组件（支持Markdown渲染）
│   ├── DocumentList.vue    # 文档列表
│   └── ...
│
├── layouts/                # 布局组件
│   └── MainLayout.vue      # 主布局（侧边栏+顶栏）
│
├── utils/                  # 工具函数
│   ├── auth.js             # 认证工具（Token管理）
│   ├── markdown.js         # Markdown 渲染配置
│   └── ...
│
├── consts.js               # 常量定义（API_BASE_URL等）
├── events.js               # 事件总线（mitt）
├── vars.js                 # 全局变量
├── style.css               # 全局样式
├── App.vue                 # 根组件
└── main.js                 # 应用入口
```

## 快速开始

### 安装依赖
```bash
npm install
```

### 启动开发服务器
```bash
npm run dev
```
默认端口：`5173`，访问 http://localhost:5173

### 构建生产版本
```bash
npm run build
```
构建产物在 `dist/` 目录

### 预览生产构建
```bash
npm run preview
```

## 核心技术实现

### 1. 状态管理（Pinia）

所有 Store 默认持久化到 localStorage：

**user.js** - 用户认证
```javascript
const userStore = useUserStore()
userStore.token      // JWT Token
userStore.userInfo   // 用户信息（id, username, email）
userStore.login()    // 登录
userStore.logout()   // 登出
```

**theme.js** - 主题切换
```javascript
const themeStore = useThemeStore()
themeStore.isDark    // 是否深色模式
themeStore.toggleTheme()
```

**search.js** - 搜索状态
```javascript
const searchStore = useSearchStore()
searchStore.keyword  // 搜索关键词
searchStore.results  // 搜索结果
```

### 2. 路由守卫（Vue Router）

**认证守卫** (`router/index.js`)
```javascript
router.beforeEach((to, from, next) => {
  const userStore = useUserStore()
  
  // 公开路由：登录、注册
  if (to.path === '/login' || to.path === '/register') {
    return next()
  }
  
  // 检查 Token
  if (!userStore.token) {
    return next('/login')
  }
  
  next()
})
```

**路由模式**：Hash 模式（避免后端配置 history fallback）

### 3. API 通信

**Axios 实例配置** (`api/request.js`)
```javascript
const request = axios.create({
  baseURL: API_BASE_URL,  // https://www.forwardforever.top:5616/api
  timeout: 30000
})

// 请求拦截器：自动添加 Bearer Token
request.interceptors.request.use(config => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// 响应拦截器：401 自动跳转登录
request.interceptors.response.use(
  response => response.data,
  error => {
    if (error.response?.status === 401) {
      userStore.logout()
      router.push('/login')
    }
    return Promise.reject(error)
  }
)
```

**SSE 流式响应** (`@microsoft/fetch-event-source`)
```javascript
import { fetchEventSource } from '@microsoft/fetch-event-source'

fetchEventSource('/chat/stream', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ query: '...' }),
  onmessage(event) {
    const data = JSON.parse(event.data)
    // 处理流式消息
  },
  onerror(err) {
    // 自动重连
  }
})
```

### 4. Markdown 渲染引擎

**配置** (`utils/markdown.js`)
```javascript
const md = markdownIt({
  html: true,           // 允许 HTML
  linkify: true,        // 自动转链接
  typographer: true,    // 智能排版
  highlight: (str, lang) => {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(str, { language: lang }).value
    }
    return ''
  }
})
  .use(markdownItTaskLists)  // [ ] [x] 任务列表
  .use(markdownItEmoji)      // :smile: Emoji
  .use(mathjaxPlugin)        // $...$ 数学公式
```

**使用示例**
```vue
<template>
  <div v-html="renderedContent" class="markdown-body"></div>
</template>

<script setup>
import { computed } from 'vue'
import { renderMarkdown } from '@/utils/markdown'

const props = defineProps({
  content: String
})

const renderedContent = computed(() => renderMarkdown(props.content))
</script>
```

### 5. API 模块化封装

**示例：chatApi.js**
```javascript
import request from './request'

export default {
  // 流式对话（SSE）
  streamChat(sessionId, query, kbIds) {
    return fetchEventSource(`${API_BASE_URL}/chat/stream`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` },
      body: JSON.stringify({ sessionId, query, kbIds })
    })
  },
  
  // Agentic RAG 对话
  agenticChat(sessionId, query, kbIds) {
    return request.post('/chat/agentic', { sessionId, query, kbIds })
  },
  
  // 获取会话列表
  getSessions(workspaceId) {
    return request.get('/chat/sessions', { params: { workspaceId } })
  }
}
```

## 开发指南

**代码规范**
- Composition API：`<script setup>` 语法
- 组件命名：PascalCase（`ChatMessage.vue`）
- 事件命名：kebab-case（`@message-sent`）
- Props 类型检查：使用 `defineProps` 定义类型

**调试技巧**
- Vue DevTools：查看组件树、Pinia 状态
- Network 面板：查看 API 请求（EventStream 类型）
- 热重载失效：删除 `node_modules/.vite` 缓存

**常用命令**
```bash
rm -rf node_modules/.vite           # 清理缓存
npm run build -- --report           # 依赖分析
rm -rf node_modules && npm install  # 重新安装
```

## 配置说明

**API 地址配置** (`consts.js`)
```javascript
export const API_BASE_URL = 'https://www.forwardforever.top:5616/api'
```

**Vite 配置** (`vite.config.js`)
```javascript
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://localhost:8080', changeOrigin: true }
    }
  }
})
```

## 注意事项

⚠️ **关键约束**
1. Token 存储：localStorage，注意 XSS 风险
2. 路由模式：Hash 模式（带 `#/`），避免后端配置
3. API Base URL：硬编码在 `consts.js`，部署时修改
4. SSE 兼容性：需现代浏览器（不支持 IE）
5. Token 有效期：24 小时，过期需重新登录

## 常见问题

**Q: 开发服务器启动失败？**  
A: 检查 Node.js 版本（需 18+），删除 `node_modules` 重新安装

**Q: API 请求 401 错误？**  
A: Token 过期，拦截器会自动跳转登录页

**Q: 如何切换后端地址？**  
A: 修改 `src/consts.js` 中的 `API_BASE_URL`

---

**返回主文档**：[../README.md](../README.md)
