# RAG Server - 后端服务

基于 Spring Boot + MyBatis Plus 的后端业务服务。

## 核心技术

- **Spring Boot 2.7.6** - 企业级应用框架
- **Java 11** - ⚠️ 必须使用 Java 11（不支持其他版本）
- **MyBatis Plus 3.5.15** - ORM 框架（分页拦截器）
- **MySQL 8.0+** - 关系型数据库
- **Redis 6.x/7.x** - 缓存与 Session 存储（Lettuce driver）
- **Milvus SDK 2.6.11** - 向量数据库客户端
- **MinIO SDK 8.6.0** - 对象存储客户端（S3 兼容）
- **JWT (jjwt 0.11.5)** - Token 认证（HS256, 24h 过期）
- **RabbitMQ** - 消息队列（DirectExchange, Jackson2Json）
- **Druid 1.2.27** - 数据库连接池

## 项目结构

```
com.rag.ragserver/
├── controller/                        # REST 控制器层（9 个）
│   ├── UserController.java            # 用户管理（登录、注册、信息）
│   ├── WorkspaceController.java       # 工作空间管理
│   ├── KbController.java              # 知识库管理
│   ├── ChatController.java            # 对话管理（流式/Agentic RAG）
│   ├── QuerySessionController.java    # 会话查询
│   ├── DashboardController.java       # 仪表盘统计
│   ├── ModelsController.java          # 模型管理
│   ├── AuditLogsController.java       # 审计日志查询
│   └── NotificationsController.java   # 通知管理
│
├── service/                           # 业务逻辑层
│   ├── impl/                          # 实现类
│   ├── UserService.java
│   ├── WorkspaceService.java
│   ├── KbService.java
│   ├── DocumentService.java
│   ├── ChatService.java
│   └── ...
│
├── mapper/                            # MyBatis 数据访问层（15 个）
│   ├── UserMapper.java
│   ├── WorkspaceMapper.java
│   ├── KbMapper.java
│   ├── DocumentMapper.java
│   ├── QuerySessionMapper.java
│   ├── ModelMapper.java
│   ├── AuditLogMapper.java
│   └── ...
│
├── domain/                            # 实体类
│   ├── User.java
│   ├── Workspace.java
│   ├── Knowledgebase.java
│   ├── Document.java
│   ├── QuerySession.java
│   ├── vo/                            # VO（视图对象）
│   └── ...
│
├── dto/                               # 数据传输对象
│   ├── LoginDTO.java
│   ├── CreateWorkspaceDTO.java
│   ├── CreateKbDTO.java
│   ├── ChatRequestDTO.java
│   └── ...
│
├── configuration/                     # Spring 配置类
│   ├── CorsConfig.java                # CORS 跨域配置
│   ├── JwtConfig.java                 # JWT 配置
│   ├── MyBatisPlusConfig.java         # MyBatis Plus 配置
│   ├── MilvusConfig.java              # Milvus 客户端配置
│   ├── MinioConfig.java               # MinIO 客户端配置
│   ├── RabbitConfig.java              # RabbitMQ 配置
│   └── RedisConfig.java               # Redis 配置
│
├── interceptor/                       # 拦截器
│   └── JwtInterceptor.java            # JWT Token 验证（excludes /users/*）
│
├── aspect/                            # AOP 切面
│   └── AuditLogAspect.java            # 审计日志切面（@AuditLog）
│
├── exception/                         # 异常处理
│   ├── GlobalExceptionHandler.java    # 全局异常处理器
│   ├── UnauthorizedException.java
│   └── BusinessException.java
│
├── rabbit/                            # RabbitMQ 消费者
│   ├── SessionNameConsumer.java       # 会话名称生成消费者
│   └── DocumentEmbeddingConsumer.java # 文档向量化消费者
│
├── assembler/                         # 对象转换器（Entity <-> VO/DTO）
├── common/                            # 公共类（Result、Constants 等）
└── utils/                             # 工具类（JWT、加密、文件处理等）

src/main/resources/
├── application.yml                    # 主配置（profile, port 8080, /api context）
├── application-dev.yml                # 开发环境配置
├── application-prod.yml               # 生产环境配置
└── com/rag/ragserver/mapper/          # MyBatis XML 映射文件（⚠️ 关键位置）
    ├── UserMapper.xml
    ├── WorkspaceMapper.xml
    ├── KbMapper.xml
    ├── DocumentMapper.xml
    └── ...（15 个 XML 文件）
```

## API 路由

### 用户管理 (`/api/users`)
| 方法 | 路径 | 功能 | 认证 |
|------|-----|------|------|
| POST | `/users/login` | 用户登录 | 否 |
| POST | `/users/register` | 用户注册 | 否 |
| GET | `/users/me` | 获取当前用户信息 | 是 |
| PUT | `/users/me` | 更新用户信息 | 是 |
| PUT | `/users/password` | 修改密码 | 是 |

### 工作空间管理 (`/api/workspaces`)
| 方法 | 路径 | 功能 | 认证 |
|------|-----|------|------|
| GET | `/workspaces` | 工作空间列表 | 是 |
| POST | `/workspaces` | 创建工作空间 | 是 |
| GET | `/workspaces/{id}` | 工作空间详情 | 是 |
| PUT | `/workspaces/{id}` | 更新工作空间 | 是 |
| DELETE | `/workspaces/{id}` | 删除工作空间 | 是 |
| GET | `/workspaces/{id}/members` | 成员列表 | 是 |
| POST | `/workspaces/{id}/members` | 添加成员 | 是 |
| DELETE | `/workspaces/{id}/members/{userId}` | 移除成员 | 是 |

### 知识库管理 (`/api/knowledgebases`)
| 方法 | 路径 | 功能 | 认证 |
|------|-----|------|------|
| GET | `/knowledgebases` | 知识库列表 | 是 |
| POST | `/knowledgebases` | 创建知识库 | 是 |
| GET | `/knowledgebases/{id}` | 知识库详情 | 是 |
| PUT | `/knowledgebases/{id}` | 更新知识库 | 是 |
| DELETE | `/knowledgebases/{id}` | 删除知识库 | 是 |
| GET | `/knowledgebases/{id}/documents` | 文档列表 | 是 |
| POST | `/knowledgebases/{id}/documents` | 上传文档 | 是 |
| DELETE | `/knowledgebases/{kbId}/documents/{docId}` | 删除文档 | 是 |

### 对话管理 (`/api/chat`)
| 方法 | 路径 | 功能 | 认证 |
|------|-----|------|------|
| POST | `/chat/stream` | 流式对话（SSE） | 是 |
| POST | `/chat/agentic` | Agentic RAG 对话 | 是 |
| GET | `/chat/sessions` | 会话列表 | 是 |
| GET | `/chat/sessions/{id}` | 会话详情 | 是 |
| DELETE | `/chat/sessions/{id}` | 删除会话 | 是 |
| GET | `/chat/sessions/{id}/messages` | 消息历史 | 是 |

### 仪表盘统计 (`/api/dashboard`)
| 方法 | 路径 | 功能 | 认证 |
|------|-----|------|------|
| GET | `/dashboard/stats` | 系统统计（工作空间、知识库、会话数） | 是 |
| GET | `/dashboard/recent-sessions` | 最近会话 | 是 |

### 模型管理 (`/api/models`)
| 方法 | 路径 | 功能 | 认证 |
|------|-----|------|------|
| GET | `/models` | 可用模型列表 | 是 |
| GET | `/models/permissions` | 用户模型权限 | 是 |

### 审计日志 (`/api/audit-logs`)
| 方法 | 路径 | 功能 | 认证 |
|------|-----|------|------|
| GET | `/audit-logs` | 审计日志查询（分页） | 是 |

**认证方式**：所有需要认证的接口，请求头携带 `Authorization: Bearer <token>`

## 快速开始

### 数据库初始化
```bash
mysql -u root -p
CREATE DATABASE general_rag DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
mysql -u root -p general_rag < ../1_general_rag.sql
```

### 配置文件

编辑 `src/main/resources/application-dev.yml`：

```yaml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/general_rag?useUnicode=true&characterEncoding=utf8mb4&serverTimezone=Asia/Shanghai
    username: root
    password: your_mysql_password
  redis:
    host: localhost
    port: 6379
    password: your_redis_password
  rabbitmq:
    host: localhost
    port: 5672
    username: admin
    password: your_rabbitmq_password

# JWT 配置（至少 32 字符）
jwt:
  secret: your-jwt-secret-key-at-least-32-characters
  expiration: 86400000  # 24小时
```

### 构建运行
```bash
# 安装依赖并编译
mvn clean install

# 启动开发服务器（默认 8080 端口）
mvn spring-boot:run

# 打包
mvn clean package
java -jar target/rag-server-1.0.0.jar --spring.profiles.active=prod
```

## 核心技术实现

### 1. JWT 认证

**JwtInterceptor** (`interceptor/JwtInterceptor.java`)
- 拦截所有请求（除 `/users/*`）
- 验证 Bearer Token
- Token 格式：`Authorization: Bearer <token>`
- 过期时间：24 小时（86400000ms）

**使用示例**
```bash
curl -X POST http://localhost:8080/api/users/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"password"}'

curl -X GET http://localhost:8080/api/workspaces \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### 2. MyBatis Plus 配置

⚠️ **关键约束**：MyBatis XML 映射文件必须位于 `src/main/resources/com/rag/ragserver/mapper/`

**Mapper 接口与 XML 同步规则**
```java
// Mapper 接口：src/main/java/com/rag/ragserver/mapper/UserMapper.java
package com.rag.ragserver.mapper;

@Mapper
public interface UserMapper extends BaseMapper<User> {
    List<User> selectByWorkspaceId(Long workspaceId);
}
```

```xml
<!-- XML 文件：src/main/resources/com/rag/ragserver/mapper/UserMapper.xml -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.rag.ragserver.mapper.UserMapper">
    <select id="selectByWorkspaceId" resultType="com.rag.ragserver.domain.User">
        SELECT * FROM users WHERE workspace_id = #{workspaceId}
    </select>
</mapper>
```

**分页插件**（`configuration/MyBatisPlusConfig.java`）
```java
@Bean
public MybatisPlusInterceptor mybatisPlusInterceptor() {
    MybatisPlusInterceptor interceptor = new MybatisPlusInterceptor();
    interceptor.addInnerInterceptor(new PaginationInnerInterceptor(DbType.MYSQL));
    return interceptor;
}
```

### 3. 全局异常处理

**GlobalExceptionHandler** (`exception/GlobalExceptionHandler.java`)
```java
@RestControllerAdvice
public class GlobalExceptionHandler {
    @ExceptionHandler(UnauthorizedException.class)
    public Result<?> handleUnauthorized(UnauthorizedException e) {
        return Result.error(401, "未授权");
    }
    
    @ExceptionHandler(Exception.class)
    public Result<?> handleException(Exception e) {
        log.error("系统异常", e);
        return Result.error(500, "系统异常");
    }
}
```

### 4. 审计日志（AOP）

**AuditLogAspect** (`aspect/AuditLogAspect.java`)
- 切点：`@AuditLog` 注解
- 记录操作人、操作时间、IP、请求参数

```java
@AuditLog(operation = "创建知识库")
@PostMapping("/knowledgebases")
public Result<Knowledgebase> create(@RequestBody CreateKbDTO dto) {
    // ...
}
```

### 5. RabbitMQ 消费者

**配置** (`configuration/RabbitConfig.java`)
- DirectExchange
- 3 次重试
- Jackson2JsonMessageConverter

**消费者示例** (`rabbit/SessionNameConsumer.java`)
```java
@RabbitListener(queues = "session.name.generate.producer.queue")
public void handleMessage(SessionNameMessage message) {
    // 处理会话名称生成
}
```

## 注意事项

⚠️ **关键约束**

1. **Java 版本**：必须使用 Java 11（不支持 Java 8 或 Java 17+）
2. **MyBatis XML 位置**：`src/main/resources/com/rag/ragserver/mapper/`
   - 修改 Mapper 接口时必须同步更新 XML
   - 命名空间必须与接口全限定名一致
3. **JWT Secret**：最少 32 字符，使用环境变量或强密钥
4. **Context Path**：所有接口前缀为 `/api`（配置在 `application.yml`）
5. **认证排除**：`/users/login` 和 `/users/register` 不需要 Token
6. **CORS**：已配置全局 CORS（允许所有来源）
7. **软删除**：实体使用 `is_deleted` 标记，不使用硬删除
8. **文件上传大小**：默认限制 50MB（`spring.servlet.multipart.max-file-size`）

## 常见问题

**Q: 启动失败，端口已被占用？**  
A: 检查 8080 端口是否被占用，或在配置文件中修改 `server.port`

**Q: MyBatis 映射文件找不到？**  
A: 确认 XML 文件在 `src/main/resources/com/rag/ragserver/mapper/` 目录

**Q: JWT Token 验证失败？**  
A: 检查 Secret 配置一致性，确认 Token 未过期（24h 有效期）

**Q: 数据库连接失败？**  
A: 确认 MySQL 已启动，数据库已创建，用户名密码正确

**Q: 如何生成 JWT Secret？**  
A: `openssl rand -base64 32` 或 `python -c "import secrets; print(secrets.token_urlsafe(32))"`

---

**返回主文档**：[../README.md](../README.md)
