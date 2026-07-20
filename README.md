# General RAG System

<div align="center">

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Spring Boot](https://img.shields.io/badge/Spring%20Boot-2.7.6-brightgreen)
![Vue.js](https://img.shields.io/badge/Vue.js-3.5-42b883)

**企业级 Agentic RAG 知识库问答系统**

多用户 · 多工作空间 · 智能代理检索 · 文档向量化 · MCP 开放协议

[系统架构](#系统架构) · [核心机制](#核心机制) · [快速开始](#快速开始) · [模块文档](#模块文档) · [部署边界](#部署边界)

</div>

---

## 项目简介

General RAG System 是一个**企业级 Agentic RAG 知识库问答系统**。它使用 LangChain 原生工具调用和自定义多轮控制器，让 LLM 根据问题与检索历史选择检索工具、补充上下文并判断何时停止。系统还通过 MCP 协议向外部 Agent 开放经过授权的知识库能力。

### 产品特点

**Agentic RAG 引擎**
- 基于 LangChain Function Calling 的可控多轮检索编排，LLM 自主选择 6 个工具
- 对话入口默认最多 10 轮检索，支持结果去重、连续片段读取和上下文扩展
- RAG Gateway 判断是否需要检索；不需要知识库时直接进入纯 LLM 模式
- Agentic RAG 是当前唯一知识库检索路径，传统单次 RAG 已移除

**MCP 开放协议**
- 基于 FastMCP 3 实现 Streamable HTTP 协议
- 提供 9 个标准化工具：知识库列出与创建、关键词/语义检索、文件发现与上下文读取、文件上传/删除
- Access Key 鉴权（`grs_ak_` + unpadded Base64URL），仅返回一次明文，服务端存储 SHA-256
- 每次工具调用异步发布审计消息到 RabbitMQ（Access Key 明文不进入消息体）
- 公网通过 Nginx TLS 代理 `/mcp` 路径，不设 `MCP_PUBLIC_URL`

**多租户与权限**
- 工作空间（Workspace）隔离，支持 owned / workspace_shared / invited / public 四级分类
- 知识库可见性：私有、公开、工作空间共享
- 成员角色管理，操作审计日志

**文档管理**
- 文件上传请求限制为 100MB，解析 PDF、纯文本、Markdown、JSON、常见代码与标记文件及图片
- 文档以 `kbId + MD5 checksum` 去重；文件名空格转 `_`，反斜杠转 `/`
- 按文件类型解析和分块，批量向量化后写入 Milvus

**安全机制**
- JWT 认证与 Redis 登录状态校验，前端路由守卫配合 401 拦截处理
- `/rag/retrieval/*` 内部接口无公开鉴权，必须部署在受控内网
- `rag-mcp` 不直接访问 MySQL/Milvus，权限判断和所有变更操作由 Java 后端负责
- CORS、Druid 连接池与 MCP 工具调用审计

## 系统架构

### 模块组成

```
general-rag-system/
├── rag-client/          # 前端（Vue 3.5 + Vite 7.2 + Ant Design Vue 4.2 + ant-design-x-vue 1.5）
├── rag-server/          # 业务后端（Spring Boot 2.7 + MyBatis Plus 3.5 + Java 11）
├── rag-llm/             # AI 服务（FastAPI + LangChain，端口 8848）
├── rag-mcp/             # MCP 服务（FastMCP 3，端口 8858）
└── embedding_rerank/    # 本地模型服务（Embedding 当前检索路径必需；Rerank 为独立服务）
```

### 技术选型

| 模块 | 技术栈 | 端口 | 说明 |
|------|--------|------|------|
| **前端** | Vue 3.5、Vite 7.2、Ant Design Vue 4.2、ant-design-x-vue 1.5、Pinia 3.0 | 5173 | History 路由、localStorage 状态、SSE 流式 |
| **后端** | Spring Boot 2.7、MyBatis Plus 3.5、JWT (jjwt 0.11.5)、Java 11 | 8080 | REST API，`/api` 前缀，权限与持久化 |
| **AI 服务** | FastAPI、LangChain、Pydantic | 8848 | `root_path="/rag"`，Agentic RAG 与 SSE |
| **MCP 服务** | FastMCP 3、Streamable HTTP | 8858 | `/mcp` 端点，Access Key 鉴权编排与审计 |
| **本地模型服务** | vLLM、FastAPI | 8890/8891 | Embedding 1024 维；Rerank 独立服务 |
| **向量数据库** | Milvus Java/Python 客户端 | 按部署配置 | 知识库向量与文档 chunk 元数据 |
| **对象存储** | MinIO | 按部署配置 | 原始文档对象存储 |
| **关系数据库** | MySQL | 按部署配置 | 用户、权限、会话和业务数据 |
| **缓存** | Redis | 按部署配置 | JWT `jti` 登录状态与验证码 |
| **消息队列** | RabbitMQ | 按部署配置 | 文档处理状态与 MCP 审计 |

### 系统拓扑

```text
浏览器 ──HTTPS──► Nginx ──► rag-client
                           │
                           ▼
                     rag-server:8080/api
                     │      │       │
                     ▼      ▼       ▼
                  MySQL   MinIO  RabbitMQ
                     │              │
                     └──────┬───────┘
                            ▼
                     rag-llm:8848/rag
                     │      │       │
                     ▼      ▼       ▼
                  Milvus  LLM API  Embedding:8890

第三方 Agent/客户端 ──HTTPS──► Nginx /mcp ──► rag-mcp:8858/mcp
                                                   │
                                      ┌────────────┴────────────┐
                                      ▼                         ▼
                              Java OpenAPI              rag-llm 内网检索 API
                              /api/openapi/v1           /rag/retrieval/*
```

## 核心机制

### Agentic RAG 检索循环

知识库对话首先由 RAG Gateway 判断是否需要检索。需要检索时，系统进入唯一的 Agentic RAG 路径；控制器根据用户问题、对话历史和已有工具结果，逐轮选择以下工具：

| 工具 | 作用 |
| --- | --- |
| `keyword_search` | 用明确术语、函数名、配置项或错误码精确检索正文 |
| `read_file_chunks` | 按 `documentId` 和索引范围连续读取原文，单次最多 20 个 chunk |
| `expand_context` | 围绕已命中 chunk 扩展前后文 |
| `semantic_search` | 多查询语义召回、去重、Rerank 与相关性过滤 |
| `find_files` | 使用文件名模式发现文件及其 `documentId` |
| `stop_search` | 信息足够或继续检索无收益时终止循环 |

对话入口从 `options.maxRounds` 读取轮次，默认值为 10。关键词和语义检索会排除已进入参考集合的 chunk，连续读取与上下文扩展则保留原文连续性；最终引用按 chunk 主键去重，并以 `fileName` 标注来源。

### 文档入库链路

```text
网页或 OpenAPI 上传
        │
        ▼
rag-server：权限校验、MD5 去重、MinIO 保存、写入 processing 状态
        │ RabbitMQ: rag.document.process.key
        ▼
rag-llm：下载文件、按扩展名解析与分块、调用 Embedding
        │
        ├──► Milvus：保存向量和 documentId/chunkIndex/fileName 元数据
        └──► RabbitMQ：回传成功或失败以及 chunk 信息
```

上传以知识库为去重边界，使用 `kbId + MD5 checksum` 判断重复内容。文件名先把反斜杠转换为 `/`，再把空格转换为 `_`。Java 服务负责业务记录与对象存储，Python 服务负责异步解析、向量化和状态回传。

### 权限与外部访问

网页业务接口使用 JWT，JWT 的 `jti` 必须同时存在于 Redis 登录状态中。MCP/OpenAPI 使用用户创建的 Access Key：格式是 `grs_ak_` 加 32 个安全随机字节的无填充 Base64URL 编码；明文只在创建响应中返回一次，服务端只保存 SHA-256 摘要。

知识库读取来源按 `owned > workspace_shared > invited > public` 确定唯一分类。`rag-mcp` 只承担协议、参数校验和调用编排：身份验证、知识库授权、创建、上传和删除都由 Java OpenAPI 执行；检索才会在授权后转发给内网 `rag-llm`。因此 `rag-mcp` 不需要也不应获得 MySQL 或 Milvus 访问权限。

### SSE 生命周期

`rag-llm` 通过 SSE 输出检索过程、思考内容、回答内容和用量。Java 后端代理该流并持久化消息：正常完成、异常和客户端取消共用一个 `AtomicBoolean saved`，保证终止路径最多保存一次。前端主动停止时使用 `AbortController.abort()`，并立即调用幂等的 `finalizeStopped()`，因为中止请求不保证触发流的错误或关闭回调。

前端还保留以下流式交互约束：消息区距离底部 50px 以内才自动跟随；滚轮操作或 `touchstart` 暂停跟随；离开底部后提供返回底部按钮。

### 模型与回退

- 会话标题：`glm-5.2/z-ai`。
- RAG Gateway：`glm-5.2/z-ai`。
- 检索决策控制器：`MiniMax-M3/minimax`。
- 当前 Embedding 执行路径：本地 `Qwen/Qwen3-Embedding-0.6B` 服务；配置标识为 `text-embedding-v4/qwen`。
- Rerank：`qwen3-rerank/qwen`。

`model_config.json` 每次使用时从磁盘读取，不使用进程内缓存。官方聊天 LLM 默认超时 60 秒；候选回退只允许发生在流开始之前，已经输出有效内容后不会切换候选。特定 OpenAI 兼容 Gemini 路径通过 `ainvoke` 完成非流式调用。

## 快速开始

### 运行前准备

完整系统依赖 MySQL、Redis、RabbitMQ、MinIO 和 Milvus。数据库结构脚本为根目录的 `general_rag_database.sql`。各服务的地址和凭据需要按部署环境修改；不要直接复用仓库中的开发环境值。

后端必须使用 Java 11。当前项目约定的本机 JDK 路径是 `D:\JDK-11`，JDK 21 会触发当前 Lombok 与编译器组合的 `JCTree$JCImport.qualid` 错误。

当前 `rag-llm` 的 Embedding 入口固定连接本地 `8890` 服务，因此运行知识库向量化与检索前需要启动 `embedding_rerank` 的 Embedding 服务。`8891` 的本地 Rerank 服务是独立实现；当前 `rag-llm` 的 Rerank 执行路径仍使用 `model_config.json` 中的 `qwen3-rerank/qwen` endpoint。

### 配置入口

- `rag-client/src/consts.js`：Java API 基础地址。
- `rag-server/src/main/resources/application-*.yml`：MySQL、Redis、RabbitMQ、MinIO、Milvus、邮件、JWT 和 `rag-llm` 地址。
- `rag-llm/main.py`：当前入口在导入服务前直接设置 RabbitMQ、MinIO、Milvus 连接值；部署时必须修改该入口或先改造为不覆盖外部配置。
- `rag-llm/model_config.json`：聊天模型候选和 Rerank 配置；每次使用时重新从磁盘读取。
- `rag-mcp/rag_mcp/config.py` 或同名环境变量：Java OpenAPI、内网 `rag-llm`、监听地址和审计消息配置。
- `embedding_rerank/config/*.py`：本地 Embedding/Rerank 模型和服务参数；当前实现直接读取类属性。

详细字段和约束见各模块 README。

### 启动服务

以下命令均从仓库根目录执行。先启动基础设施，再启动模型服务、AI 服务、Java 后端和前端。

```bash
# 本地 Embedding：8890
cd embedding_rerank
python embedding_start.py
```

```bash
# AI 服务：8848，root_path=/rag
cd rag-llm
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8848
```

```powershell
# Java 后端：8080，context-path=/api
cd rag-server
$env:JAVA_HOME = "D:\JDK-11"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
mvn spring-boot:run
```

```bash
# 前端：Vite 默认端口 5173
cd rag-client
npm install
npm run dev
```

可选服务：

```bash
# MCP：8858/mcp
cd rag-mcp
python -m pip install -r requirements.txt
python -m rag_mcp.server

# 独立本地 Rerank：8891
cd embedding_rerank
python rerank_start.py
```

服务配置、运行目录和生产部署要求分别见各模块文档。

### 数据库初始化

先创建数据库，再导入根目录脚本：

```bash
mysql -u root -p -e "CREATE DATABASE general_rag DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
mysql -u root -p general_rag < general_rag_database.sql
```

## 项目结构

```text
general-rag-system/
├── rag-client/              # Vue 浏览器端：页面、SSE 交互、知识库与 Access Key 管理
├── rag-server/              # Java 业务后端：认证、权限、持久化、OpenAPI 与审计
├── rag-llm/                 # Python AI 服务：文档处理、Agentic RAG 与聊天流
├── rag-mcp/                 # FastMCP 服务：外部 Agent 工具、Access Key 鉴权编排
├── embedding_rerank/        # 本地 Embedding 与 Rerank 模型服务
├── general_rag_database.sql # 数据库结构脚本
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

模块之间的边界如下：

- `rag-client` 只访问 Java API，不直接调用内部检索接口。
- `rag-server` 拥有用户身份、工作空间、知识库权限和所有变更操作。
- `rag-llm` 处理 AI 工作负载；其中 `/rag/retrieval/*` 只面向受信任内网调用。
- `rag-mcp` 使用 Access Key 请求 Java OpenAPI，获得授权路由后再调用 `rag-llm`。
- `embedding_rerank` 提供独立模型服务；当前 `rag-llm` 的 Embedding 路径连接本地 `8890`。

## 部署边界

- Java 生产构建必须使用 `D:\JDK-11`。
- `/rag/retrieval/*` 没有公共鉴权，不得暴露到公网。
- MCP 公网入口由 Nginx 终止 TLS，并将 `/mcp` 原路径代理到 `rag-mcp:8858/mcp`；当前服务不配置 `MCP_PUBLIC_URL`。
- Access Key、JWT Secret、模型 API Key、数据库和基础设施密码不得写入 README、日志或前端代码。
- MCP 审计发布到 `rag.audit.exchange`，路由键为 `rag.mcp.tool.log.v1`，由 `rag.mcp.tool.log.queue` 消费，并配置死信交换机与死信队列。
- 前端使用 History 路由，静态站点必须把未知路径回退到 `index.html`。

## 模块文档

- [rag-client/README.md](./rag-client/README.md)：依赖、路由、SSE 交互、页面实现、配置与构建。
- [rag-server/README.md](./rag-server/README.md)：接口、认证、权限、配置、数据库、RabbitMQ 与 Java 11 构建。
- [rag-llm/README.md](./rag-llm/README.md)：聊天 API、六工具 Agentic RAG、内部检索接口、模型配置和回退机制。
- [rag-mcp/README.md](./rag-mcp/README.md)：九个 MCP 工具、环境变量、Access Key、审计和 Nginx 部署。
- [embedding_rerank/README.md](./embedding_rerank/README.md)：Embedding/Rerank 统一入口及 API、配置、快速开始文档。

## 开源协议

本项目采用 [Apache License 2.0](./LICENSE)。贡献流程见 [CONTRIBUTING.md](./CONTRIBUTING.md)。
