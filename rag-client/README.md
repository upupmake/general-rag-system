# RAG 前端客户端

`rag-client` 是通用 RAG 系统的浏览器端应用，负责用户认证、工作空间与知识库管理、知识库问答、流式会话、会话检索、仪表盘和 Access Key 管理。项目使用 Vue 3 组合式 API 与 Vite 构建，并以 Ant Design Vue、Ant Design X Vue 提供主要界面组件。

## 技术栈

版本以 `package.json` 为准：

- Vue `3.5.24`
- Vite `7.2.5`，实际依赖映射为 `rolldown-vite`
- Vue Router `4.6.4`
- Pinia `3.0.4`
- Ant Design Vue `4.2.6`
- Ant Design X Vue `1.5.0`
- Axios `1.13.2`
- `@microsoft/fetch-event-source` `2.0.1`
- Markdown It `14.1.0`、Highlight.js `11.11.1`

## 目录结构

```text
rag-client/
├── src/
│   ├── api/                 # Axios 接口与 SSE 对话接口
│   ├── assets/              # 静态资源和 Markdown 主题样式
│   ├── components/          # 会话列表、知识库选择器、文件浏览器等共享组件
│   ├── layouts/             # 认证布局和主应用布局
│   ├── router/              # 路由表与登录守卫
│   ├── stores/              # 用户、主题和搜索状态
│   ├── utils/               # 通用 Markdown 渲染等工具
│   ├── views/               # 页面组件
│   │   ├── chat/            # 新建对话、已有会话及相关组合式函数
│   │   ├── docs/            # MCP 接入文档页
│   │   ├── kb/              # 知识库列表与详情页
│   │   └── workspace/       # 工作空间管理页
│   ├── consts.js            # 后端 API 根地址
│   ├── vars.js              # 跨页面共享的模型、知识库等状态
│   ├── App.vue
│   └── main.js
├── vite.config.js
└── package.json
```

## 配置

### 后端 API 地址

`src/consts.js` 直接导出 `API_BASE_URL`。当前启用值为：

```js
const API_BASE_URL = 'https://starvpn.forwardforever.top:5616/api'
```

同一文件保留了本地调试地址 `http://127.0.0.1:8080/api` 的注释行。项目目前没有通过环境变量读取 API 地址，`vite.config.js` 也没有配置开发代理；切换后端时需要修改 `src/consts.js`。

### HTTP 请求约定

`src/api/commonApi.js` 创建 Axios 实例，并在请求拦截器中从用户 Store 读取令牌，写入 `Authorization: Bearer <token>`。普通接口预期响应结构如下：

```json
{
  "code": 200,
  "message": "ok",
  "data": {}
}
```

响应码字段不是 `200` 时进入统一业务错误处理；下载和预览使用 `blob` 响应时直接返回二进制数据。登录令牌保存在浏览器 `localStorage` 的 `rag_token` 项中，刷新页面后再通过 `/users/me` 获取用户资料。

## 路由

路由使用 `createWebHistory()`，不是 Hash 路由。生产部署必须将未知前端路径回退到 `index.html`，否则直接访问或刷新子路由会返回服务器的 404。

| 路径 | 访问范围 | 页面与用途 |
| --- | --- | --- |
| `/login` | 公开 | 登录 |
| `/register` | 公开 | 注册 |
| `/forgot-password` | 公开 | 找回密码 |
| `/docs/mcp` | 公开 | MCP 接入文档 |
| `/` | 需登录 | 主布局入口，重定向到 `/dashboard` |
| `/dashboard` | 需登录 | 仪表盘 |
| `/kb` | 需登录 | 知识库列表 |
| `/kb/:kbId` | 需登录 | 知识库详情与文档管理 |
| `/chat/new` | 需登录 | 新建对话 |
| `/chat/:sessionId` | 需登录 | 已有会话与 SSE 流式对话 |
| `/search` | 需登录 | 会话内容搜索 |
| `/access-keys` | 需登录 | Access Key 管理与 MCP 配置 |
| `/workspaces` | 需登录 | 工作空间管理 |

路由守卫以 `userStore.isLogin` 判断是否存在令牌，非公开路由在未登录时跳转 `/login`。位于已有会话 `/chat/:sessionId` 时，主导航、徽标和切换到其他会话会打开新标签页；其他页面仍使用 Vue Router 在当前标签页跳转。

## 安装与运行

在 `rag-client` 目录执行：

```bash
npm install
npm run dev
```

`npm run dev` 实际执行 `vite --host`。项目未覆盖 Vite 开发端口，因此未被占用时使用默认端口 `5173`，并监听可供局域网访问的主机地址。

生产构建与本地预览：

```bash
npm run build
npm run preview
```

构建产物输出到 `dist/`。仓库未声明 Node.js 版本约束，安装与构建环境应满足当前 Vite 版本的运行要求。

## 核心实现

### SSE 流式对话与主动停止

已有会话通过 `src/api/chatApi.js` 的 `streamRequest()` 调用 `@microsoft/fetch-event-source`：

1. 每次流式请求创建独立的 `AbortController`，并将 `signal` 交给 `fetchEventSource`。
2. 握手阶段同时检查 HTTP 状态和 `Content-Type: text/event-stream`。失败时优先读取后端 JSON 的 `message`，并在错误对象中保留 HTTP 状态。
3. SSE 消息按 `content`、`thinking`、`process`、`usage`、`done` 分发。流式阶段在助手气泡内显示连接、检索、思考或生成状态。
4. `useChat.js` 共享当前控制器、用户消息、助手消息、`streamStarted` 和停止状态。只有收到至少一个 SSE 事件后才显示停止按钮。
5. 用户停止时先调用 `abort()`，随后直接执行幂等的 `finalizeStopped()`。这是必要步骤，因为 `fetchEventSource` 被中止时通常不会触发 `onerror` 或 `onclose`。
6. `finalizeStopped()` 保留已生成内容，将当前用户消息和助手消息标为完成，清空流状态，并在 `500ms` 后重新获取消息以补齐后端生成的消息编号。
7. 非用户触发的连接或流式错误会把实际错误信息附加到助手内容，将助手消息标为错误，并清理生成状态。

流式增量到达时使用即时滚动，避免 `smooth` 动画在持续输出期间叠加抖动。

### 消息滚动跟随

`src/views/chat/composables/useScroll.js` 明确区分用户滚动和程序滚动：

- 距离底部不超过 `50px` 才视为接近底部，该阈值是固定交互约束。
- 桌面端发生滚轮操作就立即暂停自动跟随；后续滚动事件只按实际滚动条位置更新状态。
- 移动端在消息区 `touchstart` 后暂停自动滚动，`touchend` 或 `touchcancel` 时重新判断是否接近底部。
- 程序滚动期间不反向修改用户滚动状态。
- 用户离开底部后显示“回到底部”按钮；点击时解除触摸保持状态并强制平滑滚到底部。

### 会话列表、游标分页与搜索

`SessionList.vue` 调用 `POST /sessions/list`，每页请求 `20` 条记录，并用 `lastActiveAt` 与 `lastId` 传递下一页游标。响应中的 `nextCursor` 和 `hasMore` 控制“查看更多历史”。

侧边栏搜索由服务端执行：输入变化后等待 `300ms` 再请求，新关键词会清空当前游标和列表。每次请求递增请求编号，过期响应不能覆盖较新的搜索结果。会话可导出 Markdown、重命名和删除；重命名后同时更新当前列表项并通过事件总线同步页面标题。

### Sender 固定高度

新建对话和已有会话的 `Sender` 都设置 `:auto-size="false"`，长文本在固定高度文本框内纵向滚动，避免持续测量和扩张输入区：

| 场景 | 桌面端 | 移动端 |
| --- | ---: | ---: |
| 已有会话 | `72px` | `56px` |
| 新建对话 | `112px` | `88px` |

已有会话支持展开或收起输入区；`Ctrl/Cmd + /` 切换状态，展开后聚焦输入框。移动端隐藏可见的快捷键帮助按钮。

### 知识库文件浏览器

`KbDocumentBrowser.vue` 被 `NewChat.vue` 和 `ChatSession.vue` 共用：

- 抽屉桌面宽度为 `620px`，视口宽度不超过 `768px` 时使用 `100%`。
- 文件名中的反斜杠在展示时归一化为 `/`，按相对路径生成目录层级；目录优先，再按名称排序。
- 展示文件或目录大小以及 `processing`、`ready`、`failed` 状态。
- 抽屉打开且仍有文件处于 `processing` 时，每 `3` 秒重新获取列表；关闭抽屉、处理完成、切换知识库或组件卸载时停止原有轮询。
- 可预览 PDF、Markdown、常见图片和文本或代码文件；关闭预览或卸载组件时释放 PDF、图片使用的 Blob URL。
- 知识库所有者可删除其中任意文件，上传者可删除自己上传的文件。只有目录内全部文件均可删除时才显示目录删除入口；目录删除逐个调用现有文件删除接口。

### Dashboard 与 Markdown

Dashboard 在挂载时并行发起统计摘要、最近活动、最新公告和最近 `24` 小时模型性能请求。模型性能按提供商分组，优先显示 `providerLogos` 中的提供商标识，并展示成功率与首字延迟。

公告不使用全局 `.markdown-body`，而使用独立的 `notice-md` 样式，避免 GitHub Markdown 样式给公告区域添加不透明背景。公告渲染器配置为：

```js
markdownit({ html: false, linkify: true, breaks: true })
```

因此公告禁止原始 HTML、自动识别链接，并保留单换行。聊天使用单独的 Markdown 实例，同样设置 `html: false`，支持任务列表与 Highlight.js 代码高亮。`src/utils/markdown.js` 用于其他文档预览页面，目前允许原始 HTML；向该渲染器传入外部内容前必须确认内容可信或先完成净化。

当前浏览器端 Markdown 渲染器没有注册 `markdown-it-mathjax3`。虽然依赖清单仍包含该包，公式目前按普通 Markdown 文本显示，不应在未验证 Vite 兼容性的情况下直接恢复插件。

### Access Key 与 MCP

`/access-keys` 可创建、列出和撤销个人 Access Key。完整 Key 仅保存在创建成功后的临时页面状态中，关闭提示后无法再次查看；列表只展示名称、前缀、状态和时间。页面提供“复制 Access Key”和“复制 MCP 命令”。

MCP 服务地址：

```text
https://starvpn.forwardforever.top:7777/mcp
```

在 Agent 对话框执行的命令格式：

```text
/mcp-add kb --transport http --url https://starvpn.forwardforever.top:7777/mcp --header Authorization="Bearer grs_ak_你的完整AccessKey"
```

需要将占位内容替换为创建时得到的完整 Key，并复制整条命令。Agent 下载地址为 <https://makecode.forwardforever.top/>，页面以新窗口打开，并设置 `noopener noreferrer`。

## 关键约束

- History 路由部署必须配置 `index.html` 回退。
- `API_BASE_URL` 当前是源码常量，切换本地或生产后端时不要误提交无关地址变更。
- 流式停止必须保留 `AbortController` 与直接调用 `finalizeStopped()` 的组合，不能依赖中止后触发关闭或错误回调。
- 自动跟随的接近底部阈值保持 `50px`；触摸按住消息区时，即使强制滚动也不能越过触摸保护。
- 流式输出使用即时滚动，只有用户点击“回到底部”等明确操作使用平滑滚动。
- 会话搜索必须重置游标、保持 `300ms` 防抖，并防止旧请求覆盖新结果。
- `Sender` 保持关闭自动扩高和固定高度，长输入由文本框内部滚动。
- 知识库文件轮询只在抽屉可见且存在处理中项目时运行，切换知识库和卸载时必须停止旧轮询。
- Access Key 等同账户凭证，完整值不得持久化到列表状态、日志或代码仓库。
- 聊天和公告渲染不允许原始 HTML；不要把公告容器改为 `.markdown-body`。

## 相关文档

返回项目总览：[根目录 README](../README.md)
