# rag-llm：检索增强生成服务

`rag-llm` 是知识库文档处理、Agentic RAG 检索和 LLM 流式回答服务。服务基于 FastAPI、LangChain、Milvus、RabbitMQ 和 MinIO 实现。

## 服务概览

- **监听端口**：`8848`
- **FastAPI 根路径**：`/rag`
- **应用入口**：`main.py` 中的 `app`
- **聊天入口**：`POST /rag/chat/stream`
- **检索模式**：唯一的知识库检索模式是 Agentic RAG；RAG Gateway 判断需要检索时进入 Agentic RAG
- **检索工具**：5 个检索工具加 1 个停止工具，共 6 个
- **Embedding**：当前固定使用本地 Embedding 服务，地址为 `http://192.168.188.6:8890`，模型为 `Qwen/Qwen3-Embedding-0.6B`
- **Rerank**：`qwen3-rerank`，提供方为 `qwen`

`root_path=/rag` 是 FastAPI 的根路径设置。应用代码中的路由前缀仍是 `/chat` 和 `/retrieval`，部署后通过 `/rag` 前缀访问。

## 目录与职责

```text
rag-llm/
├── main.py                       FastAPI 应用入口、端口和根路径
├── dependencies.py               应用生命周期、RabbitMQ 消费者和 Milvus 释放任务
├── services/chat.py              会话标题和流式聊天入口
├── services/retrieval.py         内部检索 HTTP 接口
├── agentic_rag_utils.py          Agentic RAG 编排、轮次、上下文和引用
├── agentic_rag_controller.py     检索决策控制器
├── agentic_rag_toolkit.py        六个检索工具及 Milvus 查询
├── rag_gateway.py                判断聊天请求是否需要知识库检索
├── utils.py                      模型、Embedding、分块和流式处理工具
├── model_config.json             运行时模型配置，不应提交敏感内容
├── mq/connection.py              RabbitMQ 连接和消费
├── mq/document_embedding.py      文档下载、解析、分块、向量化和入库
├── milvus_utils.py               Milvus 客户端和集合生命周期
├── minio_utils.py                MinIO 文件读取
├── aiohttp_utils.py              Rerank HTTP 调用
└── requirements.txt              Python 依赖
```

## 安装与运行

### 安装依赖

在 `rag-llm` 目录执行：

```bash
pip install -r requirements.txt
```

### 启动

```bash
# 开发环境
uvicorn main:app --host 0.0.0.0 --port 8848 --reload

# 生产环境示例
uvicorn main:app --host 0.0.0.0 --port 8848 --workers 2

# 直接运行 main.py
python main.py
```

应用启动时会连接 RabbitMQ，并启动 `rag.document.process.queue` 消费者；同时启动 Milvus 连接释放任务。RabbitMQ、MinIO 和 Milvus 不可用时，应用生命周期初始化可能失败。

## 配置

### 基础设施连接

`mq/connection.py`、`minio_utils.py` 和 Milvus 初始化代码本身从以下环境变量读取连接信息：

```bash
RABBITMQ_HOST=<RabbitMQ 主机>
RABBITMQ_PORT=<RabbitMQ 端口>
RABBITMQ_USERNAME=<RabbitMQ 用户名>
RABBITMQ_PASSWORD=<RabbitMQ 密码>
```

MinIO 使用：

```bash
MINIO_ENDPOINT=<MinIO 地址>
MINIO_ACCESS_KEY=<MinIO 访问密钥>
MINIO_SECRET_KEY=<MinIO 私密密钥>
```

`MILVUS_URI` 和 `MILVUS_TOKEN` 用于 Milvus 连接：

```bash
MILVUS_URI=<Milvus URI>
MILVUS_TOKEN=<Milvus 令牌>
```

但是当前 `main.py` 在导入这些模块之前直接给上述变量赋值，会覆盖进程外传入的同名值。按现有代码部署时需要修改 `main.py` 中的连接值；若要由部署环境注入，则应先改造入口为仅在变量缺失时设置默认值。不要把真实密码、访问密钥或令牌写入 README、提交到 Git 或复制到日志中。

### 模型配置文件

聊天模型和 Rerank 配置从 `model_config.json` 读取。文件路径相对于进程工作目录，服务每次调用配置读取函数时都会从磁盘重新读取，不使用进程内缓存。请根据部署环境准备该文件；本 README 不提供密钥、内网地址或具体供应商凭据。

聊天配置的顶层结构为 `chat`，按提供方组织；每个提供方包含 `settings` 默认候选和可选的模型名候选。候选可以是对象或列表，列表项默认启用，也可以使用 `enabled: false` 禁用。候选通常包含以下字段：

```json
{
  "chat": {
    "<provider>": {
      "settings": [
        {
          "api_key": "由部署环境提供",
          "base_url": "由部署环境提供",
          "model_provider": "可选",
          "timeout": 60,
          "max_retries": 1,
          "enabled": true
        }
      ],
      "<model-name>": [
        {
          "api_key": "由部署环境提供",
          "base_url": "由部署环境提供",
          "enabled": true
        }
      ]
    }
  },
  "rerank": {
    "qwen": {
      "settings": {
        "api_key": "由部署环境提供"
      },
      "qwen3-rerank": {
        "endpoint": "由部署环境提供"
      }
    }
  }
}
```

实际使用的模型由请求中的 `model.name` 和 `model.provider` 选择。Agentic RAG 的检索决策控制器固定使用 `MiniMax-M3/minimax`；RAG Gateway 和会话标题生成固定使用 `glm-5.2/z-ai`，并关闭 thinking。官方聊天 LLM 的默认超时为 60 秒；候选自身配置的 `timeout` 可以覆盖该默认值。控制器使用 LangChain LLM，默认超时为 30 秒。

当前 Embedding 入口 `get_embedding_instance()` 不读取远程 Embedding 配置，而是固定返回本地服务实例。Rerank 从 `model_config.json` 的 `rerank.qwen.qwen3-rerank` 读取 endpoint 和 API 密钥。

## 聊天 API

### `POST /rag/chat/session/name`

根据请求体中的 `content` 生成会话标题。标题模型固定为 `glm-5.2/z-ai`。请求示例：

```json
{
  "content": "请查找知识库中的项目部署要求"
}
```

响应结构：

```json
{
  "title": "项目部署要求"
}
```

### `POST /rag/chat/stream`

唯一的聊天入口，返回 `text/event-stream`。请求体由 `history`、`model` 和 `options` 组成：

```json
{
  "history": [
    {"role": "user", "content": "项目的部署要求是什么？"}
  ],
  "model": {
    "name": "<已配置的模型名>",
    "provider": "<已配置的提供方>"
  },
  "options": {
    "kbId": 123,
    "userId": 456,
    "maxRounds": 10,
    "systemPrompt": "可选的系统提示词",
    "thinking": false,
    "webSearch": false
  }
}
```

`history` 的最后一项必须是当前用户问题。`options.kbId` 和 `options.userId` 同时存在时，服务先调用 RAG Gateway 判断是否需要检索；判断结果为 `use_rag` 时进入 Agentic RAG，检索轮次默认取 `maxRounds=10`。RAG Gateway 调用失败时默认进入 Agentic RAG。判断为 `direct_answer` 时才使用纯 LLM 模式。

SSE 事件使用 JSON 数据行：

```text
data: {"type":"process","payload":"..."}

data: {"type":"thinking","payload":"..."}

data: {"type":"content","payload":"..."}

data: {"type":"error","payload":"..."}

data: {"type":"rag_summary","payload":"..."}

data: {"type":"usage","payload":{...}}
```

`payload` 在聊天流中通常是 JSON 编码后的字符串；`usage` 包含 `prompt_tokens`、`completion_tokens`、`total_tokens`、`latency_ms`、`first_token_latency_ms` 和 `is_success`。服务不提供 README 中曾列出的独立 `/chat/agentic` 或 `/chat/agentic/stream` 路由。

## Agentic RAG

### 六个工具

`agentic_rag_toolkit.py` 注册以下六个工具：

1. `keyword_search`：按具体关键词检索正文，可用 `document_ids` 限定文档。
2. `read_file_chunks`：按 `document_id` 和起止 chunk 索引读取连续正文，单次最多 20 个 chunk。
3. `expand_context`：围绕已命中的 `document_id` 和 `chunk_index` 扩展前后上下文。
4. `semantic_search`：多个 query 并行召回，使用 Rerank 和相关性阈值筛选。
5. `find_files`：按文件名模式查找文件，只返回文件元信息，不返回正文。
6. `stop_search`：控制器判断信息足够、无法构造有效新查询或达到轮次上限时停止检索。

### 检索流程和约束

- RAG Gateway 的 `use_rag` 决策只进入 Agentic RAG，不恢复传统一次性 RAG 分支。
- 聊天入口默认最大检索轮次为 10，可通过 `options.maxRounds` 传入其他值。
- 检索控制器固定为 `MiniMax-M3/minimax`。
- 轮次结束条件包括：调用 `stop_search`、控制器没有工具调用、达到最大轮次、控制器调用失败，或控制器消息累计超过 256000 个 token。
- 控制器 token 使用 `o200k_base` 编码计算。
- `reference_docs` 以数值 chunk PK 去重；`all_docs` 以 `documentId` 聚合文件信息。
- 文档定位统一使用 `documentId`。关键词检索使用 `document_ids`；连续读取和上下文扩展使用 `document_id`，不使用文件名定位。
- 只有关键词检索和语义检索传递已命中的 chunk PK 以尽量避免重复；连续读取和上下文扩展不做排除过滤。
- Milvus 结果保留 `fileName`，用于展示和来源归因。
- Agentic RAG 使用 `text-embedding-v4/qwen` 作为配置标识，但当前 Embedding 实例实际固定连接本地 `8890` 服务。

最终回答阶段使用主聊天请求中的模型生成流式答案，并把检索过程、引用文档和使用统计转换为聊天 SSE 事件。

## 内部检索 API

以下五个接口挂载在 `/rag/retrieval` 下，供受信任的 `rag-mcp` 内部调用。它们没有公共认证层，调用前的 Java 授权和请求参数由上游负责；必须保持在受保护的内部网络中，不得直接暴露到公网。

五个接口都要求 `ownerUserId` 和 `knowledgeBaseId`。chunk 结果包含 `chunkId`、`documentId`、`fileName`、`chunkIndex`、`totalChunks`、`content`、`score` 和 `scoreType` 等字段。

| 方法 | 路径 | 主要请求字段 | 用途 |
|---|---|---|---|
| POST | `/rag/retrieval/keywords` | `keywords`、`matchMode`、`topK`、可选 `documentIds` | 关键词检索 |
| POST | `/rag/retrieval/semantic` | `queries`、`relevanceQuery`、`topK`、`relevanceThreshold` | 语义检索和 Rerank |
| POST | `/rag/retrieval/files` | `namePattern`、`offset`、`limit` | 文件元信息发现 |
| POST | `/rag/retrieval/chunks` | `documentId`、`startChunkIndex`、`endChunkIndex` | 读取连续 chunk |
| POST | `/rag/retrieval/context` | `documentId`、`chunkIndex`、`windowSize` | 扩展上下文 |

关键词请求示例：

```json
{
  "ownerUserId": 456,
  "knowledgeBaseId": 123,
  "keywords": ["部署", "配置"],
  "matchMode": "OR",
  "topK": 15,
  "documentIds": [789]
}
```

语义请求示例：

```json
{
  "ownerUserId": 456,
  "knowledgeBaseId": 123,
  "queries": ["应用部署配置", "服务启动要求"],
  "relevanceQuery": "部署方式、启动参数和运行环境要求",
  "topK": 10,
  "relevanceThreshold": 0.3
}
```

接口默认值和限制以 `services/retrieval.py` 的 Pydantic 请求模型为准：关键词 `topK` 默认 15、语义 `topK` 默认 10，二者最大均为 50；文件列表 `limit` 默认 30、最大 100；上下文 `windowSize` 默认 2、最大 9。

## 文档向量化

应用启动后消费 RabbitMQ 队列 `rag.document.process.queue`。任务包含文档、知识库、用户、MinIO 对象等信息，处理过程为：

1. 从 MinIO 读取文件。
2. 按扩展名解析 PDF、TXT、Markdown、JSON、代码、标记文本或图片。
3. 为每个 chunk 写入 `documentId`、`chunkIndex`、`maxChunkIndex` 和 `fileName` 元数据。
4. 使用当前本地 Embedding 服务生成向量，并按最多 32 条一批写入 Milvus。
5. 通过 RabbitMQ 发布处理成功或失败消息。

Milvus 客户端按用户和知识库获取，服务维护集合生命周期，并运行后台释放任务。检索和向量化都依赖 `MILVUS_URI`、`MILVUS_TOKEN` 以及相应知识库集合。

## 关键约束

- 端口必须以当前入口为准：`8848`；FastAPI `root_path` 为 `/rag`。
- 当前聊天 API 只有 `/chat/session/name` 和 `/chat/stream`；Agentic RAG 是聊天流内部模式，不是单独公开路由。
- Agentic RAG 固定使用六工具；`stop_search` 由编排层处理，不作为普通检索结果工具执行。
- 文件定位使用数值 `documentId`；不能用 `fileName` 替代文档定位。
- 五个 `/rag/retrieval/*` 接口没有公共认证，必须只允许受信任内部调用。
- `model_config.json` 每次访问都从磁盘读取；修改配置后无需依赖进程内缓存刷新。
- 官方 LLM 默认超时为 60 秒。候选已显式配置 `timeout` 时，以候选值为准。
- LLM fallback 只允许发生在流开始之前；已经输出有效流内容后，不能切换到下一个候选。`ainvoke` 失败时仍可按候选顺序 fallback。
- 当提供方为 `other` 且模型名以 `gemini` 开头时，流式入口改用 `ainvoke`，以兼容该类 OpenAI 兼容 Gemini 中继返回空流内容的情况。
- 不要把任何真实 API 密钥、RabbitMQ 密码、MinIO 密钥或 Milvus 令牌写入文档、示例、提交记录或日志。

## 常用文件

- `main.py`：服务端口、`root_path`、路由挂载和进程入口。
- `services/chat.py`：聊天请求、RAG Gateway 分流、Agentic RAG 默认轮次和 SSE 事件。
- `services/retrieval.py`：五个受信任内部检索接口及请求模型。
- `agentic_rag_controller.py`：`MiniMax-M3/minimax` 检索控制器和工具选择提示词。
- `agentic_rag_toolkit.py`：六工具定义及底层 Milvus 检索。
- `agentic_rag_utils.py`：Agentic RAG 多轮编排、token 限制、去重和引用。
- `utils.py`：模型配置读取、官方 LLM fallback、Embedding 入口和统一流式处理。
