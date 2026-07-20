# RAG Server 后端服务

`rag-server` 是通用 RAG 系统的 Java 业务服务，负责用户与工作空间、知识库与文档、会话与消息、模型权限、访问密钥和审计数据。服务通过 HTTP 与 `rag-client`、`rag-mcp` 交互，通过 WebClient、RabbitMQ、Milvus、MinIO、MySQL 和 Redis 完成模型调用、异步文档处理、向量管理、对象存储及认证状态管理。

## 技术基线

| 项目 | 当前实现 |
| --- | --- |
| 应用框架 | Spring Boot 2.7.6，Spring MVC 与 WebFlux WebClient |
| Java | Java 11，项目环境必须使用 `D:\JDK-11` |
| 数据访问 | MyBatis-Plus 3.5.15、MySQL、Druid 1.2.27 |
| 缓存与登录状态 | Redis、Lettuce |
| 对象与向量存储 | MinIO SDK 8.6.0、Milvus SDK 2.6.11 |
| 消息队列 | Spring AMQP、RabbitMQ |
| 认证 | JWT 0.11.5、OpenAPI Access Key |
| 构建 | Maven；仓库没有 Maven Wrapper |

服务默认监听 `http://localhost:8080/api`：

```yaml
server:
  port: 8080
  servlet:
    context-path: /api
```

## 主要职责

- 用户注册、邮件验证码、登录、退出和当前用户信息。
- 工作空间创建、切换、邀请、成员管理和权限校验。
- 私有、工作空间共享、受邀和公开知识库的访问控制。
- 文档上传、MinIO 存储、RabbitMQ 异步处理、分块查询、预览和跨存储删除。
- 对话创建、SSE 流式生成、消息编辑重生成、重试、会话搜索和游标分页。
- 用户级 Access Key 管理，以及供 MCP 等外部调用方使用的 OpenAPI。
- 普通业务审计、仪表盘统计和 MCP 工具调用审计落库。

## 目录结构

```text
rag-server/
├── pom.xml
├── src/main/java/com/rag/ragserver/
│   ├── controller/       HTTP 接口
│   ├── interceptor/      JWT 与 Access Key 拦截器
│   ├── service/impl/     业务实现
│   ├── mapper/           MyBatis Mapper 接口
│   ├── domain/           实体、枚举和领域 VO
│   ├── dto/              请求及查询 DTO
│   ├── configuration/    MySQL、Redis、RabbitMQ、MinIO、Milvus 等配置
│   ├── rabbit/           文档消息与 MCP 审计消费者
│   ├── aspect/           普通业务审计切面
│   └── exception/        业务异常和统一异常处理
└── src/main/resources/
    ├── application.yml
    ├── application-dev.yml
    ├── application-prod.yml
    ├── application-xy.yml
    └── com/rag/ragserver/mapper/   MyBatis XML
```

## 认证方式

### JWT

除公开用户接口和 `/openapi/v1/**` 外，业务接口由 `JwtInterceptor` 校验 JWT。请求可使用：

```http
Authorization: Bearer <JWT>
```

登录成功后，JWT 的 `jti` 同时写入 Redis。请求需要通过签名、有效期、Redis 登录状态、用户启用状态和当前工作空间成员关系校验。普通登录默认有效期为 24 小时；勾选“记住我”时为 30 天。退出登录会删除 Redis 中对应的 `jti`。

无需 JWT 的用户接口为：

- `GET /api/users/test`
- `POST /api/users/send-code`
- `POST /api/users/register`
- `POST /api/users/send-reset-code`
- `POST /api/users/reset-password`
- `POST /api/users/login`

### OpenAPI Access Key

`/api/openapi/v1/**` 不使用 JWT，统一由 `AccessKeyInterceptor` 校验：

```http
Authorization: Bearer grs_ak_<密钥主体>
```

Access Key 是用户级凭据，格式为 `grs_ak_` 加 32 个安全随机字节的无填充 Base64URL 编码。明文只在创建响应中返回一次；数据库仅保存 SHA-256 摘要、展示前缀、名称和时间字段。已撤销密钥既不能认证，也不会出现在有效密钥列表中。每次成功认证会向请求写入 `userId`、`accessKeyId` 并更新 `last_used_at`。

Access Key 的创建、列表和撤销仍是 JWT 业务接口，用户只能管理自己的密钥。

## 主要接口

以下路径均包含服务的 `/api` 上下文前缀。除“公开用户接口”和 OpenAPI 外，其余接口均需要 JWT。

### 用户与 Access Key

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `POST` | `/api/users/send-code` | 发送注册验证码 |
| `POST` | `/api/users/register` | 注册用户并初始化工作空间 |
| `POST` | `/api/users/send-reset-code` | 发送密码重置验证码 |
| `POST` | `/api/users/reset-password` | 使用验证码重置密码 |
| `POST` | `/api/users/login` | 用户名或邮箱登录 |
| `GET` | `/api/users/me` | 查询当前用户 |
| `POST` | `/api/users/logout` | 退出并失效当前 JWT |
| `GET` | `/api/access-keys` | 查询当前用户的有效 Access Key |
| `POST` | `/api/access-keys` | 创建 Access Key，明文仅本次返回 |
| `DELETE` | `/api/access-keys/{accessKeyId}` | 撤销自己的 Access Key |

### 工作空间

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/api/workspaces/list` | 查询拥有、加入和当前工作空间 |
| `POST` | `/api/workspaces` | 创建工作空间 |
| `PUT` | `/api/workspaces/{workspaceId}` | 修改本人创建的工作空间 |
| `DELETE` | `/api/workspaces/{workspaceId}` | 删除本人创建且允许删除的工作空间 |
| `POST` | `/api/workspaces/{workspaceId}/switch` | 切换当前工作空间 |
| `POST` | `/api/workspaces/invite` | 邀请用户加入工作空间 |
| `GET` | `/api/workspaces/{workspaceId}/members` | 查询工作空间成员 |
| `DELETE` | `/api/workspaces/{workspaceId}/members/{userId}` | 移除工作空间成员 |

### 知识库与文档

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/api/kb` | 查询当前用户和工作空间可见的知识库 |
| `POST` | `/api/kb` | 创建知识库 |
| `PUT` | `/api/kb/{kbId}` | 修改本人知识库 |
| `DELETE` | `/api/kb/{kbId}` | 删除本人知识库 |
| `GET` | `/api/kb/{kbId}/documents` | 查询知识库文档 |
| `POST` | `/api/kb/{kbId}/documents` | 以多部分字段 `files` 上传文档 |
| `DELETE` | `/api/kb/{kbId}/documents/{docId}` | 删除文档 |
| `GET` | `/api/kb/{kbId}/documents/{docId}/chunks` | 分页查询文档分块 |
| `GET` | `/api/kb/{kbId}/documents/{docId}/preview` | 流式预览原始文档 |
| `POST` | `/api/kb/{kbId}/invite` | 邀请用户访问知识库 |
| `GET` | `/api/kb/{kbId}/invited-users` | 查询受邀用户 |
| `DELETE` | `/api/kb/{kbId}/invited-users/{userId}` | 移除受邀用户 |

普通上传和写操作仅允许知识库拥有者。公开、工作空间共享和受邀权限只提供符合规则的读取能力，不自动获得写权限。

### 对话与会话

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `POST` | `/api/chat/start` | 校验模型和知识库权限，创建会话及首条用户消息 |
| `POST` | `/api/chat/stream` | 通过 SSE 流式生成回答 |
| `GET` | `/api/chat/sessions/{sessionId}/messages` | 查询会话消息 |
| `POST` | `/api/chat/messages/{messageId}/edit` | 编辑用户消息并以 SSE 重新生成 |
| `POST` | `/api/chat/messages/{userMessageId}/retry` | 对指定用户消息以 SSE 重试 |
| `POST` | `/api/sessions/list` | 按最后活跃时间和编号进行游标分页，可传 `keyword` 搜索标题 |
| `POST` | `/api/sessions/search` | 搜索会话消息内容 |
| `DELETE` | `/api/sessions/{sessionId}` | 删除当前用户、当前工作空间中的会话 |
| `GET` | `/api/sessions/{sessionId}/title` | 生成并保存会话标题 |
| `PUT` | `/api/sessions/{sessionId}/title` | 修改会话标题 |

### 统计与辅助查询

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/api/dashboard/summary` | 查询用户、知识库、文档、会话和 Token 用量摘要 |
| `GET` | `/api/models/available` | 查询当前用户可用模型 |
| `GET` | `/api/models/performance?hours=24` | 查询指定时间窗口的模型表现 |
| `GET` | `/api/audit-logs/recent?limit=10` | 查询当前用户最近的重要操作 |
| `GET` | `/api/notifications/latest` | 查询最新通知 |

### OpenAPI

下列接口必须携带有效 Access Key：

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/api/openapi/v1/auth/verify` | 验证 Access Key，返回对应用户和密钥编号 |
| `GET` | `/api/openapi/v1/knowledge-bases` | 分类查询用户可读知识库 |
| `POST` | `/api/openapi/v1/knowledge-bases` | 创建个人私有或公开知识库 |
| `GET` | `/api/openapi/v1/knowledge-bases/{kbId}/access` | 检查读取权限和权限来源 |
| `GET` | `/api/openapi/v1/knowledge-bases/{kbId}/private-access` | 检查是否为本人创建的私有知识库 |
| `POST` | `/api/openapi/v1/knowledge-bases/{kbId}/documents` | 向本人私有知识库上传 `files` |
| `DELETE` | `/api/openapi/v1/knowledge-bases/{kbId}/documents/{docId}` | 删除本人私有知识库中的指定文档 |

OpenAPI 创建知识库的请求字段只有：

- `name`：必填，非空，最长 100 个字符。
- `description`：可选，最长 200 个字符。
- `visibility`：必填，只接受 `private` 或 `public`，不接受 `shared`。

## 配置

`application.yml` 固定公共配置并默认启用 `dev`；`application-dev.yml`、`application-prod.yml` 和 `application-xy.yml` 提供环境连接参数。部署时应通过外部配置或环境变量覆盖敏感值，不应把真实密码、Access Key 或令牌提交到仓库。

需要配置的主要属性如下：

```yaml
mysql:
  host: <MySQL 主机>
  port: <MySQL 端口>
  database: general_rag
  username: <MySQL 用户名>
  password: <MySQL 密码>

jwt:
  secret: <至少 32 字节的随机密钥>

llm:
  host: <rag-llm 主机>
  port: <rag-llm 端口>

minio:
  endpoint: <MinIO 地址>
  bucketName: files
  accessKey: <MinIO 用户名>
  secretKey: <MinIO 密码>

milvus:
  uri: <Milvus 地址>
  token: <Milvus 令牌>

spring:
  redis:
    host: <Redis 主机>
    port: <Redis 端口>
    password: <Redis 密码>
  rabbitmq:
    host: <RabbitMQ 主机>
    port: <RabbitMQ 端口>
    username: <RabbitMQ 用户名>
    password: <RabbitMQ 密码>
  mail:
    host: <SMTP 主机>
    port: <SMTP 端口>
    username: <发件邮箱>
    password: <SMTP 授权码>
```

上传大小由公共配置统一限制为单个文件 100MB、单次多部分请求 100MB：

```yaml
spring:
  servlet:
    multipart:
      max-file-size: 100MB
      max-request-size: 100MB
```

## 初始化与运行

### 前置服务

启动前需要可访问的 MySQL、Redis、RabbitMQ、MinIO、Milvus、SMTP 服务和 `rag-llm`。数据库结构脚本位于仓库根目录 `general_rag_database.sql`。

从 `rag-server` 目录执行：

```powershell
mysql -u root -p -e "CREATE DATABASE general_rag DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
cmd /c "mysql -u root -p general_rag < ..\general_rag_database.sql"
```

### 强制使用 Java 11

默认 JDK 21 会导致当前 Lombok 与编译器组合出现 `JCTree$JCImport.qualid` 错误。构建和运行前必须显式切换到 `D:\JDK-11`：

```powershell
cd D:\waibao\general-rag-system\code\rag-server
$env:JAVA_HOME = "D:\JDK-11"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
java -version
mvn -version
```

`java -version` 和 `mvn -version` 都应显示 Java 11。项目没有 `mvnw.cmd`，必须使用本机 `mvn`。

### 构建与启动

```powershell
mvn clean test
mvn spring-boot:run
```

生产打包和运行：

```powershell
mvn clean package
java -jar target\rag-server-1.0.0.jar --spring.profiles.active=prod
```

启动后，业务基地址为 `http://localhost:8080/api`。

## 关键实现约束

### 文档上传、去重与删除

1. 上传始终以知识库为边界，服务器端再次校验知识库拥有者权限。
2. 单文件和单次请求上限均为 100MB。
3. 文件名先把反斜杠 `\` 规范为 `/`，再把空格替换为下划线 `_`；MinIO 对象名使用分组目录、知识库编号和随机 UUID，不直接使用原文件名作为对象键。
4. 去重键是 `kbId + 文件内容 MD5`，不是文件名。同一知识库中允许同名但内容不同的文件；即使改名，内容相同仍会被拒绝。不同知识库之间不互相去重。
5. 上传成功写入文档记录后，服务向 `server.interact.llm.exchange` 发送路由键 `rag.document.process.key`，由异步流程完成解析、分块和向量化。
6. 删除时先按 `documentId` 删除 Milvus 向量，再清理 `document_chunks`、逻辑删除 `documents`，最后删除 MinIO 对象。MySQL、Milvus 和 MinIO 不具备跨存储事务原子性，失败恢复和运维处理必须考虑部分完成状态。

### SSE 保存生命周期

`ChatController.executeStreamChat` 调用 `rag-llm` 的 `/rag/chat/stream`，解析内容、思考过程、检索过程和用量事件。每次流式请求只创建一个 `AtomicBoolean saved`，并由正常完成、异常和取消三个终止路径共同使用 `compareAndSet(false, true)`，保证用户消息与助手消息最多保存一次。

`doOnError` 和 `doOnCancel` 位于完整的解析、拼接及正常保存链之后，因此下游连接失败、SSE 解析失败和客户端取消都受同一保护。正常完成保存 `completed`；取消保存已有部分内容并按停止流程结束；服务或解析异常保存已有部分内容并标记错误状态。

### OpenAPI 权限与数据边界

- 每个可读知识库只归入一个分类，优先级固定为 `owned > workspace_shared > invited > public`。
- `owned` 包含用户创建的全部知识库；`workspace_shared` 覆盖该用户加入的所有工作空间，而不是只看当前工作空间。
- OpenAPI 创建时从 Access Key 认证结果取得 `ownerUserId`，不接受调用方传入所有者，且不创建 `shared` 知识库。
- OpenAPI 文档管理只允许本人创建且当前可见性为 `private` 的知识库；删除时还会校验文档确实属于路径中的 `kbId`。
- 外部 VO 不暴露系统提示词、任意内部元数据或 Milvus 细节。`ownerUserId` 只在通过权限检查后按授权结果返回。

### RabbitMQ MCP 审计与死信

MCP 工具调用最终只发布一条 `SUCCESS` 或 `FAIL` 审计事件。Java 服务使用以下持久拓扑消费并写入 `mcp_tool_logs`：

| 用途 | 名称 |
| --- | --- |
| 审计交换机 | `rag.audit.exchange` |
| 路由键 | `rag.mcp.tool.log.v1` |
| 消费队列 | `rag.mcp.tool.log.queue` |
| 死信交换机 | `rag.audit.dlx` |
| 死信路由键 | `rag.mcp.tool.log.dead` |
| 死信队列 | `rag.mcp.tool.log.dead.queue` |

消费者最多尝试 3 次，最终拒绝时不重新入原队列，由队列的 DLX 参数转入死信队列。消息必须包含调用编号、用户编号、Access Key 编号、工具名、最终状态和创建时间；`invocation_id` 配合 `INSERT IGNORE` 保证重复投递幂等。审计摘要不得保存 Authorization、完整 Access Key、文档分块正文或上传文件内容。

### DTO、VO 与 Mapper

- 控制器请求应使用专用 DTO 承载并执行 Bean Validation，不能依赖调用方传入用户编号、工作空间权限或所有者身份。
- 对外响应使用用途明确的 VO，尤其是 OpenAPI，不直接暴露包含系统提示词、存储路径或内部字段的实体。
- 新增 Mapper 聚合或跨表查询时必须定义强类型 DTO/VO，字段使用 `Long`、`Double` 等准确类型，并通过列别名匹配属性名；禁止返回 `List<Map<String, Object>>`。
- MyBatis XML 固定放在 `src/main/resources/com/rag/ragserver/mapper/`，`namespace` 必须与 Mapper 接口全限定名一致。

### 邮箱域名校验

注册验证码和注册接口使用相同的域名白名单。匹配规则是域名完全相等，或以 `.` 加白名单域名结尾；教育邮箱另按 `.edu.cn` 和 `.edu` 后缀处理。禁止使用简单子串匹配，例如白名单中的 `mail.com` 不能匹配 `tankmail.com` 或 `58mail.com`。

## 返回主文档

[返回项目总览](../README.md)
