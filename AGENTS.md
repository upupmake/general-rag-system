# AGENTS.md

本文件面向在此仓库工作的编码代理。先读本文件，再读目标模块的 `README.md`；模块细节以模块文档和当前代码为准。

## 项目概览

General RAG System 是企业级 Agentic RAG 知识库问答系统，采用多模块单仓库：

| 目录 | 技术栈 | 默认端口 | 职责 |
| --- | --- | --- | --- |
| `rag-client/` | Vue 3.5、Vite 7.2（rolldown-vite）、Ant Design Vue 4.2、Pinia 3 | 5173 | 浏览器端页面、SSE 对话、知识库与 Access Key 管理 |
| `rag-server/` | Spring Boot 2.7.6、MyBatis-Plus 3.5、Java 11 | 8080（`/api`） | 用户、工作空间、知识库权限、持久化、OpenAPI、审计 |
| `rag-llm/` | FastAPI、LangChain、Pydantic | 8848（`root_path=/rag`） | 文档解析向量化、Agentic RAG、聊天 SSE、内部检索接口 |
| `rag-mcp/` | FastMCP 3、Streamable HTTP | 8858（`/mcp`） | 对外 MCP 工具、Access Key 鉴权编排、审计发布 |
| `embedding_rerank/` | vLLM、FastAPI | 8890 / 8891 | 本地 Embedding（1024 维）与独立 Rerank 服务 |

模块边界：

- `rag-client` 只调用 Java API，不直接访问内部检索接口。
- `rag-server` 是身份、权限、知识库/文档变更的唯一控制方，也是 OpenAPI 的实现方。
- `rag-llm` 承担 AI 工作负载；`/rag/retrieval/*` 没有公共鉴权，只能在内网被 `rag-mcp` 调用。
- `rag-mcp` 不连接 MySQL/Milvus，不复制或绕过 Java 权限判断。
- `embedding_rerank` 是独立模型服务；`rag-llm` 当前 Embedding 路径固定连接本地 `8890`，检索链路不再调用 Rerank。

## 文档优先级

冲突时按以下顺序取信：

1. 目标模块的 `README.md` 与当前代码。
2. 根 `README.md` 的架构与机制说明。
3. `CONTRIBUTING.md`（部分工具命令已过时，见下文）。

根 `README.md` 与模块文档的不一致点：文档去重不是 `MD5` 内容去重，当前实现按 `kbId + 规范化后的完整文件路径` 去重（见 `rag-server/README.md` 与 `DocumentsServiceImpl`）；`model_config.json` 按文件 mtime/size 缓存，文件变化后下一次调用重读（见 `rag-llm/README.md`），不是常驻内存缓存。

## 环境与依赖

本机已验证工具链：

| 工具 | 位置 / 版本 | 说明 |
| --- | --- | --- |
| JDK | `D:\JDK-11`（11.0.25） | 后端编译、测试、打包必须使用；不要用默认的 JDK 21 |
| Maven | `D:\apache-maven-3.9.9\bin\mvn.cmd` | 仓库没有 Maven Wrapper，使用 `mvn` 而不是 `mvnw.cmd`；`-o` 离线可用 |
| Node / npm | v24.14.0 / 10.8.2 | 前端构建与 Node 内置测试运行器 |
| Python（默认） | 3.11.9 | 可运行 `rag-mcp` 测试；不要假设它有 `rag-llm` 的全部依赖 |
| Python（rag-llm 测试） | `D:\conda-envs\LLM-from-scratch\python.exe`（3.12.12） | 已具备 langchain / langchain_milvus / pymilvus / aio_pika 等依赖 |

运行完整系统需要 MySQL、Redis、RabbitMQ、MinIO、Milvus。数据库结构脚本是根目录的 `general_rag_database.sql`。

Windows / PowerShell 注意事项：

- 终端会话的工作目录在多次命令之间保持；跨模块操作时显式 `cd` 或使用绝对路径。
- PowerShell 不支持 bash heredoc（`python - <<'PY'` 会报错）；多行脚本先写成临时文件再执行。
- Maven 的 `-Dtest=A,B` 在 PowerShell 中要整体加引号，例如 `'-Dtest=ChatStreamLifecycleTest,AccessKeysServiceImplTest'`。
- 只为当前命令作用域设置 JDK，不要修改系统全局配置：

```powershell
$env:JAVA_HOME = 'D:\JDK-11'
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
```

## 启动服务

从仓库根目录开始，先启动基础设施，再按依赖顺序启动服务：

```bash
# 本地 Embedding：8890（知识库向量化与检索必需）
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
$env:JAVA_HOME = 'D:\JDK-11'
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
mvn spring-boot:run
```

```bash
# 前端：Vite 默认 5173
cd rag-client
npm install
npm run dev
```

```bash
# MCP：8858/mcp（可选）
cd rag-mcp
python -m pip install -r requirements.txt
python -m rag_mcp.server
```

- `rag-llm` 必须在 `rag-llm/` 目录内启动：模块使用扁平顶层导入（如 `from utils import ...`），不是包结构。
- `rag-mcp` 必须在 `rag-mcp/` 目录内启动或测试，否则 `import rag_mcp` 可能失败。
- `rag-server` 默认激活 `dev` profile；本机基础设施地址与凭据在未跟踪的 `application-*.yml` 中，仓库只提交 `application.yml` 与 `*.yml.example`。
- 前端 `API_BASE_URL` 是 `rag-client/src/consts.js` 中的源码常量，没有环境变量或代理；切换后端需要改这一行，不要误提交无关的地址变更。

## 测试与验证

### 前端 `rag-client`

```bash
cd rag-client
npm run build
node --test tests/markdownMath.test.js tests/search.test.js tests/chatTransport.test.js
```

- `npm run build` 是本模块的主要验证；`dist/assets` 超过 500 kB 的警告是已知现象，不算失败。
- 上述三个 Node 测试当前通过（2026-11-02 验证）。
- `tests/chat.test.js` 当前在 `beforeEach` 失败：测试仍导入 `vars.js` 已不再导出的 `contextMultiplier`。这是测试漂移，不要据此判断业务代码回归，使用前先更新测试。
- `tests/chat.browser.mjs` 是浏览器端到端脚本，需要 Vite 开发服务和 web-access CDP，不作为常规单测运行。
- 仓库没有配置前端 lint 脚本；不要依赖 `CONTRIBUTING.md` 中的 `npm run lint:fix`。

### 后端 `rag-server`

```powershell
cd rag-server
$env:JAVA_HOME = 'D:\JDK-11'
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
mvn -o test
```

- 完整 `mvn test` 会运行 `RagServerApplicationTests`，它是 `@SpringBootTest`，会连接共享开发环境的 MySQL/Redis/RabbitMQ，并启动 `@RabbitListener` 消费者，可能消费真实队列消息。运行前先征得用户确认。
- 日常提交可先跑可独立运行的单元测试（2026-11-02 验证 13 项通过）：

```powershell
mvn -o '-Dtest=ChatStreamLifecycleTest,AccessKeysServiceImplTest,OpenApiKnowledgeBaseServiceImplTest' test
```

- `KbPermissionServiceImplTest` 单独用 `-Dtest=` 跑会因 MyBatis-Plus lambda 缓存（`TableInfo`）未初始化而假失败（`can not find lambda cache for this entity`）。要么跑完整套件（先加载 `RagServerApplicationTests`），要么按 `ChatStreamLifecycleTest` 的方式在测试中预初始化 `TableInfo`。
- 新测试若直接使用 `LambdaQueryWrapper` / `LambdaUpdateWrapper`，先考虑改用字符串列名的 `UpdateWrapper`，或把逻辑抽成纯函数测试，避免同类假失败。
- 仓库没有配置 Checkstyle/Spotless 插件；不要把 `CONTRIBUTING.md` 中的 `mvn checkstyle:check` 当作验证步骤。

### AI 服务 `rag-llm`

```powershell
cd rag-llm
& 'D:\conda-envs\LLM-from-scratch\python.exe' -m pytest tests -q
```

- 当前基线（2026-11-02）：47 项中 40 通过、7 失败。7 项失败都来自 `tests/test_document_embedding.py`：该测试用 `types.ModuleType` 构造的 `utils` 桩缺少 `plain_text_fallback_split`，而生产代码 `mq/document_embedding.py` 已对未知扩展名增加纯文本兜底切分。这是测试桩漂移；更新测试桩/用例，不要回退兜底功能。
- 测试用桩替代 langchain / milvus / minio / aio_pika；没有完整依赖的机器也能运行，但涉及真实 Milvus/MinIO 的路径需要到具备完整依赖和基础设施的环境验证。
- 生产/集成验证应在具备完整依赖与基础设施的环境执行；本地改动可先用 `python -m py_compile <file>` 做语法检查。

### MCP 服务 `rag-mcp`

```powershell
cd rag-mcp
python -m pytest tests -q
```

- 当前 5 项全部通过（2026-11-02 验证）；测试用 `types.ModuleType` 桩替代 `aio_pika`。
- 必须从 `rag-mcp/` 目录运行，保证 `rag_mcp` 可导入；本机默认 Python 已安装 `fastmcp` 3.x、`httpx`、`pytest`。

### `embedding_rerank`

```powershell
cd embedding_rerank
python -m pytest test -q
```

- `test/` 不是纯单元测试：用例会请求运行中的 8890/8891 服务，`test_gpu_memory.py` 还依赖 `vllm` 与 GPU。没有对应环境时不要把它当作回归基线；作为服务启动后的冒烟测试使用。

### 通用规则

- 改动后至少运行目标模块的构建/测试；失败时先判断是代码回归还是已知测试漂移（见上）。
- 仓库没有 CI 配置（无 `.github/workflows`），所有验证都是本地手工执行，所以必须在交付前实际运行并报告命令和结果。
- 不要为了让测试通过而删除或跳过既有用例；先确认期望行为是否已被产品代码有意改变。

## 构建

```powershell
# 后端：必须 JDK 11；产物 target/rag-server-1.0.0.jar（Spring Boot 可执行 jar）
cd rag-server
$env:JAVA_HOME = 'D:\JDK-11'
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
mvn -o clean package -DskipTests
```

- 需要与源码严格对应的产物时务必加 `clean`；普通 `package` 在无变化时会跳过重建，jar 时间戳不能用于判断新旧。
- 前端 `npm run build` 输出到 `rag-client/dist/`。
- Python 服务无构建步骤；依赖清单是各模块的 `requirements.txt`。`rag-llm` 本地 requirements 使用非固定名称，生产两份副本使用 `==` 固定版本；新增依赖时同步更新本地与两台生产机的 requirements。

## 关键实现约束

### Agentic RAG（`rag-llm`）

- 知识库对话先由 RAG Gateway 判断是否需要检索；需要时只进入 Agentic RAG，不恢复已删除的传统单次 RAG。
- 固定六个工具：`keyword_search`、`read_file_chunks`、`expand_context`、`semantic_search`、`find_files`、`stop_search`；`stop_search` 由编排层处理，不作为普通检索结果执行。
- 默认最多 10 轮，`options.maxRounds` 会被夹紧到 1~20。轮次结束条件包括 `stop_search`、控制器无工具调用、达到上限、控制器失败或 token 超限。
- 文档定位统一使用数值 `documentId`，不能用文件名替代；关键词检索用 `document_ids`，连续读取/扩展上下文用 `document_id`。
- 关键词和语义检索排除已进入参考集合的 chunk；连续读取和上下文扩展保留原文连续性。最终引用按 chunk 主键去重，并以 `fileName` 归因。
- 工具结果摘要中的 `matchedTotal`、`truncated`、`topScore`、`scoreThreshold`、`failedQueries` 有明确语义，修改检索逻辑时必须保持这些信号。
- 生产 Milvus 为 `v2.6.11`；表达式解析器不支持 raw string，`LIKE` 中的 `%`/`_` 要按 `_escape_like` 的双反斜杠规则转义。旧 collection 可能缺 `documentId`/`fileName`/`text` 索引，使用 `scripts/ensure_milvus_indexes.py`（默认 dry-run）检查。

### 文档入库链路

```text
rag-server 上传 -> 权限与路径去重 -> MinIO -> RabbitMQ rag.document.process.queue
-> rag-llm 下载、按扩展名解析分块、Embedding -> Milvus
-> RabbitMQ 回传成功/失败 -> rag-server 持久化状态
```

- 上传以知识库为边界，单文件/单请求上限 100MB；文件名先把 `\` 规范为 `/`，再把空格替换为 `_`。
- 去重键是 `kbId + 规范化完整路径`（含文件名），不是文件内容 MD5；同内容不同路径允许上传。并发上传依赖 `FOR UPDATE` 锁定索引范围。
- 删除顺序：Milvus 向量 -> `document_chunks` -> 逻辑删除 `documents` -> MinIO 对象。跨存储无事务，必须考虑部分完成。
- 新增文件类型时同时检查 `rag-llm/mq/document_embedding.py`、`rag-llm/utils.py`、前端上传白名单/预览和 `python-docx` 等依赖。

### MCP（`rag-mcp`）

- 九个工具：`list_knowledge_bases`、`create_knowledge_base`、`search_knowledge_base_by_keywords`、`search_knowledge_base_by_semantics`、`find_knowledge_base_files`、`read_knowledge_base_chunks`、`expand_knowledge_base_context`、`upload_private_knowledge_base_file`、`delete_private_knowledge_base_file`。
- Access Key 由 Java 负责签发与 SHA-256 校验；`rag-mcp` 只做协议、参数校验和编排。创建只接受 `private`/`public`，上传/删除只支持本人创建且当前为 `private` 的知识库。
- 修改工具时同步维护 `rag_mcp/server.py` 的 `instructions`、工具描述、`annotations` 与 `output_schema`；参数上界为 `keywords` ≤ 20、`document_ids` ≤ 100、`name_pattern` ≤ 256。
- 二进制/大文件（PDF、docx、xlsx、图片）走客户端上传脚本 `skills/kb-operations/scripts/upload_kb_file.py`，不要把 base64 或文件正文塞进工具调用。
- 每个工具调用最终只发布一条 `SUCCESS`/`FAIL` 审计消息到 `rag.audit.exchange`；审计不得包含完整 Access Key、Authorization、检索正文或上传内容。
- 复用 `clients._get_http_client` 的共享 `httpx.AsyncClient`，不要按请求新建客户端；审计发布超时由 `MCP_AUDIT_PUBLISH_TIMEOUT` 控制，检索调用超时由 `MCP_RETRIEVAL_TIMEOUT` 控制。

### SSE 与前端流式

- Java 侧每次流式请求只创建一个 `AtomicBoolean saved`，正常完成、异常、客户端取消三条终止路径统一用 `compareAndSet(false, true)`，保证消息最多保存一次。
- 前端停止必须保留 `AbortController.abort()` 后立即调用幂等 `finalizeStopped()` 的组合，不能依赖中止回调。
- 自动跟随只在距底部 50px 以内；滚轮或 `touchstart` 暂停跟随；离开底部后显示“回到底部”。流式输出用即时滚动，明确点击操作才用平滑滚动。
- 会话搜索使用 300ms 防抖、重置游标，并防止旧请求覆盖新结果。
- 聊天/公告渲染禁止原始 HTML；不要给公告容器改 `.markdown-body`。

### 模型与回退

- 会话标题：`deepseek-v4-flash/deepseek`；RAG Gateway：`glm-5.2/z-ai`；检索控制器：`step-5-preview/stepfun`（`reasoning_effort=medium`）。
- 官方聊天 LLM 默认超时 60 秒；候选显式 `timeout` 时以候选值为准。
- Fallback 只允许发生在流开始之前；一旦输出有效流内容，不再切换候选。OpenAI 兼容 Gemini 路径在流式入口改用 `ainvoke`。
- `model_config.json` 按 mtime/size 缓存；修改后下一次调用自动重读。不要把真实 API Key 写入该文件或文档。

## 代码风格

### Java（`rag-server`）

- 基础包 `com.rag.ragserver`，按 `controller` / `service` / `service/impl` / `mapper` / `domain` / `dto` / `configuration` / `rabbit` / `aspect` / `exception` 分层。
- 使用 Lombok：`@RequiredArgsConstructor` 构造器注入、`@Slf4j`；不新增字段注入。
- 控制器请求必须用专用 DTO + Bean Validation，不能信任调用方传入用户编号、所有者或权限字段；对外响应使用用途明确的 VO。
- Mapper 跨表查询必须定义强类型 DTO/VO，禁止返回 `List<Map<String, Object>>`；MyBatis XML 放在 `src/main/resources/com/rag/ragserver/mapper/`，`namespace` 与接口全限定名一致。
- 新增表/字段时同步更新 `general_rag_database.sql`、实体和 Mapper。

### Python（`rag-llm` / `rag-mcp`）

- 注释、docstring 和用户可见错误信息使用中文；类型注解和 Pydantic 模型是现有惯例。
- `rag-llm` 是扁平模块结构（顶层 `from utils import ...`），不要随意改成包导入；`rag-mcp` 是标准包（`from rag_mcp...`）。
- 工具参数模型放在 `agentic_rag_toolkit.py`；工具描述和参数边界要能让外部调用方理解何时使用、何时不要编造 ID。
- 不要新增同步阻塞 I/O 到异步路径；耗时解析/向量化使用 `asyncio.to_thread` 或现有异步客户端。
- 环境配置集中在 `rag_mcp/config.py` 或模块入口；不在业务代码里散落硬编码地址。

### Vue（`rag-client`）

- 使用 Composition API + `<script setup>`；页面在 `src/views/`，共享组件在 `src/components/`，文件名使用 PascalCase（如 `SessionList.vue`）。
- 路由使用 `createWebHistory()`；生产部署必须把未知路径回退到 `index.html`。
- API 调用集中在 `src/api/`，通过 Axios 实例自动附加 `Authorization: Bearer <token>`；响应约定 `{ code, message, data }`，`code !== 200` 进入统一错误处理。
- 组件自动按需引入由 `unplugin-vue-components` 处理；不要为了单个页面引入整套样式或重复注册 Ant Design Vue。

## Git 与提交

- 使用 Conventional Commits，前缀后可用中文描述：`feat: ...`、`fix: ...`、`fix(rag-client): ...`、`perf(rag-client): ...`。
- 只显式暂存本次改动涉及的文件；不要使用 `git add .` / `git add -A`。`rag-client/src/consts.js` 的本地调试地址变更不要混入无关提交。
- 不要提交生成物和本地配置：`rag-client/dist/`、`rag-server/target/`、`__pycache__/`、`*.log`、`temp/`、`.env*`、`application-*.yml`、`model_config.json`、`rag-client/.env.deploy`。
- 仓库仍跟踪少量 `rag-llm/mq/__pycache__/*.pyc`；运行 Python 测试会改写这些文件，提交前检查 `git status` 并用 `git restore` 还原，不要把编译缓存混入提交。
- 当前工作区里 `rag-llm/services/chat.py`、`rag-llm/prompts.py` 未被 Git 跟踪但生产已部署；修改它们不会出现在默认 diff 中，提交前显式确认是否纳入版本控制。
- `.gitignore` 同时忽略 `rag-llm/main.py`，但它已被 Git 跟踪（ignore 对已跟踪文件无效）；不要因为它在 ignore 列表里就认为可以随意改。该文件用 `os.environ.setdefault` 写入明文开发环境凭据，生产必须用环境变量覆盖，不要把其中的值复制到文档、日志或提交信息。
- 不要新增真实密钥、密码、Access Key、JWT Secret 或生产地址到仓库；一旦发现泄露，先撤销/轮换再继续。

## 部署

部署、推送、创建 PR 等对外或共享环境操作必须先得到用户明确确认，并在部署前完成构建、测试和真实数据检查。

生产环境由两台 Linux 主机组成：

- 网关机：Nginx TLS、`rag-server`（host 网络，:8080）、`rag-llm`（:8848）、`rag-mcp`（:8858）。
- 中间件机：MySQL、Redis、RabbitMQ、MinIO、Milvus，另有 `rag-server`（容器映射 :8086）、`rag-llm`、`embedding_rerank`（:8890）、`rag-mcp`。
- 两台机器都用各自的 `start_services.sh` 管理服务；支持单个服务重启（`all|rag-llm|rag-mcp`，中间件机还支持 `embedding`/`embedding_rerank`）。不要重新引入基于继承 FD 的 `flock` 启动锁。
- 前端托管在阿里云 OSS；发布使用 `cd rag-client && npm run deploy:oss`（先构建，可选 `--skip-build` / `--dry-run` / `--delete`）。发布后必须刷新 `/` 与 `/index.html` 的 CDN 缓存，再验证新入口和带哈希的 assets。

Python 文件部署规范：

1. 不要直接覆盖生产文件。先在目标机创建 `/tmp/rag-deploy-stage-<TS>/`，放入待部署文件、`deploy*.sh`（复制后用 `cmp`/sha256 校验）、`rollback*.sh`、`DEPLOY_README.txt`、`MANIFEST.txt`。
2. 生产原文件备份到 `/tmp/rag-deploy-backup-<TS>/`。
3. 本地工作区可能是 CRLF；部署前归一为 LF，再比对 sha256，一致后才应用。
4. 应用后重启对应服务并做健康检查：`rag-llm` 200、`rag-mcp` 的 `GET /mcp` 返回 405、`rag-server` 无 token 访问 `/api/kb` 返回 401。
5. 部署脚本在目标机执行前先 `sed -i 's/\r$//'`，避免 CRLF 让 `set -euo pipefail` 报 `set: pipefail: invalid option name`。

Java jar 发布：

- 本地用 JDK 11 构建 `rag-server/target/rag-server-1.0.0.jar`，先备份目标机的 `app.jar` 为 `app.jar.bak-<TS>`，核对 sha256 后覆盖，再 `docker restart rag-spring-boot`；只换 jar 不需要重建容器。
- 网关机容器使用 host 网络监听 :8080；中间件机容器映射 :8086->:8080。用无 token 的 `GET /api/kb`（期望 401）和日志中的 `Started RagServerApplication` 验证启动。

回滚 = 恢复备份文件后重启同一服务；不要用删除生产文件或重建共享基础设施的方式绕过问题。

## 安全要点

- Access Key 是账户凭证：格式 `grs_ak_` + 32 字节安全随机数的无填充 Base64URL；明文只在创建响应中出现一次，服务端只保存 SHA-256。不得写入 URL、日志、审计消息、代码或文档。
- JWT 的 `jti` 必须存在于 Redis 登录状态；前端 401 统一拦截，但不要在日志中打印完整 token。
- `/rag/retrieval/*` 无公共鉴权，严禁暴露到公网；`rag-mcp` 公网入口由 Nginx 终止 TLS 并把 `/mcp` 原路径代理到 `8858/mcp`，不要配置 `MCP_PUBLIC_URL`。
- MCP 审计不得保存 Authorization、完整 Access Key、文档分块正文或上传文件内容；Nginx/应用日志也不得记录 `Authorization` 头。
- 上传/删除的权限判断必须在 Java 侧完成；`rag-mcp` 不得直接访问 MySQL/Milvus，也不得自行实现权限规则。
- 前端聊天与公告禁止原始 HTML 渲染；Access Key 完整值不得持久化到列表状态或 localStorage。

## 已知陷阱速查

- JDK 21 + 当前 Lombok/编译器组合会报 `JCTree$JCImport.qualid`；后端始终用 JDK 11。
- `-Dtest=` 单独跑 `KbPermissionServiceImplTest` 会假失败；见测试章节。
- `rag-llm` 测试的 `utils` 桩和前端 `chat.test.js` 有已知漂移；先区分测试问题与代码回归。
- `rag-llm` 测试/服务必须在 `rag-llm/` 目录内运行；`rag-mcp` 同理。
- PowerShell 中 `rg` 传通配符路径可能报 os error 123；先用 `Get-ChildItem` 列文件或传显式路径。
- 前端构建的大 chunk 警告是已知现象；不要为此盲目引入新的拆包依赖。
- 没有 CI、没有 lint 配置；`CONTRIBUTING.md` 里的 lint/format 命令不能直接照抄。
- 根 README 与模块 README 在去重键和模型配置缓存上表述不一致，以模块 README/代码为准。

## 参考文档

- `rag-client/README.md`：路由、HTTP 约定、SSE 交互、页面实现、构建。
- `rag-server/README.md`：接口、认证、权限、配置、数据库、RabbitMQ、Java 11 构建。
- `rag-llm/README.md`：聊天 API、六工具 Agentic RAG、内部检索接口、模型配置与回退。
- `rag-mcp/README.md`：九个 MCP 工具、环境变量、Access Key、审计、Nginx 部署。
- `embedding_rerank/README.md`：Embedding/Rerank API、配置与启动。
- `skills/`：仓库内置代理技能；`kb-operations` 提供 MCP 知识库操作与二进制文件上传脚本。
