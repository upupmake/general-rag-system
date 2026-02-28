# RAG LLM Service - AI 服务模块

LLM 服务层，负责文档解析、向量化、智能检索和问答生成。基于 FastAPI + LangChain + LangGraph 实现。

## 核心技术

- **FastAPI** - 异步 Web 框架（默认端口 8888，root_path="/rag"）
- **LangChain** - LLM 应用框架（langchain, langchain_core, langchain_community）
- **LangGraph** - 工作流编排（Agentic RAG 状态机）
- **Pydantic** - 数据验证和结构化输出
- **aio_pika** - 异步 RabbitMQ 客户端
- **miniopy_async** - 异步 MinIO 客户端（S3 兼容）
- **langchain_milvus** - Milvus 向量存储集成
- **PyMuPDF + pdfplumber** - PDF 解析
- **Tesseract OCR** - 图像文字识别（可选）
- **Uvicorn** - ASGI 服务器

## 项目结构

```
rag-llm/
├── main.py                          # FastAPI 应用入口（端口 8888）
├── dependencies.py                  # Lifespan 管理（RabbitMQ、Milvus 初始化）
├── requirements.txt                 # Python 依赖
├── model_config.json                # 模型配置（需创建，已在 .gitignore）
├── model_config.json.example        # 配置模板
│
├── services/                        # 业务服务
│   └── chat.py                      # 聊天服务路由（/chat）
│
├── mq/                              # RabbitMQ 消息队列
│   ├── connection.py                # 连接管理
│   ├── document_embedding.py        # 文档向量化消费者（rag.document.process.queue）
│   └── session_name.py              # 会话名称生成消费者（session.name.generate.producer.queue）
│
├── agentic_rag_controller.py        # LangGraph 状态机控制器（max 5 轮检索）
├── agentic_rag_toolkit.py           # 5 种检索工具 + PROMPT（TOOL_DEFINE/TOOL_SELECT）
├── agentic_rag_utils.py             # Agentic RAG 核心服务（generate_workflow_id 等）
│
├── rag_utils.py                     # 传统 RAG 工具函数
│   ├── 文档解析（PDF、TXT、Markdown）
│   ├── 文本分块（RecursiveCharacterTextSplitter）
│   ├── 向量化（Embedding）
│   └── 检索增强生成
│
├── milvus_utils.py                  # Milvus 向量数据库操作
│   ├── MilvusClientManager         # 客户端生命周期管理（30min 自动释放）
│   ├── 向量检索（语义检索、关键词过滤）
│   └── 异步锁（防并发冲突）
│
├── minio_utils.py                   # MinIO 对象存储操作（文件上传/下载）
├── utils.py                         # 通用工具函数（LLM 初始化、模型配置加载）
│
├── openai_utils.py                  # OpenAI API 封装
├── gemini_utils.py                  # Gemini API 封装
├── aiohttp_utils.py                 # 异步 HTTP 工具
├── wrapper.py                       # 装饰器和包装器
└── run.log                          # 运行日志
```

## API 路由

### 聊天服务 (`/rag/chat`)

| 方法 | 路径 | 功能 | 认证 |
|------|-----|------|------|
| POST | `/rag/chat/stream` | 流式 RAG 问答（SSE） | 否 |
| POST | `/rag/chat/agentic` | Agentic RAG 对话 | 否 |
| POST | `/rag/chat/agentic/stream` | Agentic RAG 流式对话（SSE） | 否 |

**请求参数**（以 `/rag/chat/agentic` 为例）
```json
{
  "query": "什么是RAG？",
  "kb_ids": [1, 2],
  "session_id": 123,
  "user_id": 456,
  "model_name": "gpt-4",
  "max_rounds": 3,
  "grade_score_threshold": 0.4
}
```

**流式响应格式**（SSE）
```
data: {"type": "workflow_id", "workflow_id": "abc123"}
data: {"type": "token", "content": "RAG"}
data: {"type": "token", "content": "是"}
data: {"type": "citations", "data": [...]}
data: {"type": "done"}
```

### 健康检查

| 方法 | 路径 | 功能 |
|------|-----|------|
| GET | `/health` | 服务健康状态 |
| GET | `/rag/` | 服务信息 |

### RabbitMQ 消费者（异步任务）

**文档向量化队列** (`rag.document.process.queue`)
- 监听文档上传事件
- 下载文档、解析、分块、向量化
- 存储到 Milvus
- 更新文档状态

**会话名称生成队列** (`session.name.generate.producer.queue`)
- 监听会话创建事件
- 根据首条消息生成会话标题
- 更新会话名称

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 配置文件

⚠️ **关键配置**：`main.py` 中硬编码了基础设施连接信息，生产环境建议使用环境变量。

**方式一：修改 main.py（开发环境）**
```python
# main.py 中的配置（第 30-40 行）
os.environ["RABBITMQ_HOST"] = "192.168.188.6"
os.environ["RABBITMQ_PORT"] = "5678"
os.environ["RABBITMQ_USERNAME"] = "make"
os.environ["RABBITMQ_PASSWORD"] = "make20260101"

os.environ["MINIO_ENDPOINT"] = "192.168.188.6:9002"
os.environ["MINIO_ACCESS_KEY"] = "make"
os.environ["MINIO_SECRET_KEY"] = "make20260101"

os.environ["MILVUS_URI"] = "http://192.168.188.6:19530"
os.environ["MILVUS_TOKEN"] = "make:make5211314"
```

**方式二：使用环境变量（生产环境推荐）**
```bash
export RABBITMQ_HOST=localhost
export RABBITMQ_PORT=5672
export MINIO_ENDPOINT=localhost:9000
export MILVUS_URI=http://localhost:19530
export MILVUS_TOKEN=username:password
```

### 模型配置

编辑 `model_config.json`（参考 `model_config.json.example`）：

```json
{
  "models": {
    "openai": {
      "api_key": "sk-your-openai-api-key",
      "base_url": "https://api.openai.com/v1",
      "model": "gpt-5.2"
    },
    "deepseek": {
      "api_key": "sk-your-deepseek-api-key",
      "base_url": "https://api.deepseek.com",
      "model": "deepseek-chat"
    },
    "qwen": {
      "api_key": "sk-your-qwen-api-key",
      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "model": "qwen3-max"
    }
  },
  "embedding": {
    "provider": "qwen",
    "model": "text-embedding-v4"
  }
}
```

⚠️ 不要将 `model_config.json` 提交到 Git！

### 运行服务
```bash
# 单进程（开发环境）
uvicorn main:app --host 0.0.0.0 --port 8888 --reload

# 多进程（生产环境）
uvicorn main:app --host 0.0.0.0 --port 8888 --workers 2

# 直接运行
python main.py
```

服务运行在 `http://localhost:8888`（注意：端口是 8888，不是 8000）

### 验证服务
```bash
curl http://localhost:8888/health
curl http://localhost:8888/rag/
```

## 核心功能说明

### 1. Agentic RAG 工作流

**5 种检索工具**（`agentic_rag_toolkit.py`）
1. **search_by_grep** - 关键词精确匹配（全库/单文件/多文件）
2. **search_by_filename_and_chunk_range** - 通过文件名获取指定 chunk
3. **extend_file_chunk_windows** - 快速扩展 chunk 上下文窗口
4. **search_by_multi_queries_in_database** - 多角度语义检索 + Rerank
5. **list_filename_by_like** - 文件名模糊匹配（SQL LIKE 语法）

**状态机控制**（`agentic_rag_controller.py`）
- 最多 5 轮检索（max_rounds=5）
- 自动评分机制（grade_score_threshold=0.4）
- 动态工具选择（TOOL_SELECT_PROMPT）
- 结果引用（citations）

### 2. 文档向量化流程

1. **RabbitMQ 接收任务** - 监听 `rag.document.process.queue`
2. **从 MinIO 下载文档** - 根据 documentId 下载文件
3. **文档解析** - PyMuPDFLoader / Tesseract OCR（图片文字识别）
4. **文本分块** - RecursiveCharacterTextSplitter（chunk_size=800, overlap=100）
5. **向量化** - 调用 Embedding API（Qwen text-embedding-v4）
6. **存储 Milvus** - 向量 + 元数据（documentId, chunkIndex, fileName 等）
7. **状态更新** - 通知 rag-server 处理完成

### 3. Milvus 集合生命周期管理

**MilvusClientManager** (`milvus_utils.py`)
- 按知识库（kbId）创建独立集合
- 30 分钟无访问自动释放连接
- 异步锁防止并发冲突
- 集合命名：`kb_{kbId}`

**向量检索优化**
- 自动分词（空格分隔）+ 关键词过滤
- 支持 Rerank 重排序（可选）
- TopK 限制（默认 10）

### 4. 支持的 LLM 模型

**model_config.json 配置**（16+ 模型）
- OpenAI: gpt-5.2, gpt-5.2-codex
- Anthropic: claude-4.5-sonnet, claude-4.5-opus
- Qwen: qwen3-max, qwen3-vl-plus, text-embedding-v4
- Gemini: gemini-3-flash-preview, gemini-3-pro-preview
- DeepSeek: deepseek-chat, deepseek-reasoner
- 其他: Moonshot, X-AI, Minimax, Xiaomi, ByteDance

## 开发指南

### 添加新的检索工具

在 `agentic_rag_toolkit.py` 中扩展：

```python
# 1. 在 RetrievalDecision 枚举中添加工具
class ToolEnum(str, Enum):
    SEARCH_BY_GREP = "search_by_grep"
    YOUR_NEW_TOOL = "your_new_tool"  # 新增

# 2. 实现工具函数
async def your_new_tool(param1: str, param2: int) -> List[Dict]:
    # 实现检索逻辑
    return results

# 3. 在 execute_tool 中注册
async def execute_tool(decision: RetrievalDecision):
    if decision.tool == ToolEnum.YOUR_NEW_TOOL:
        return await your_new_tool(**decision.parameters)

# 4. 更新 TOOL_DEFINE_PROMPT 和 TOOL_SELECT_PROMPT
```

### 调试技巧

**启用详细日志**
```python
# main.py
logging.basicConfig(level=logging.DEBUG)
```

**测试单个模块**
```bash
# 测试 Milvus 连接
python -c "from milvus_utils import MilvusClientManager; print('OK')"

# 测试 MinIO 连接
python -c "from minio_utils import test_connection; test_connection()"
```

**查看 RabbitMQ 队列状态**
```bash
# 登录 RabbitMQ 管理界面
http://localhost:15672
```

### 性能优化建议

1. **批量向量化** - 使用批处理减少 API 调用（batch_size=32）
2. **异步处理** - 充分利用 asyncio 和 aiohttp
3. **连接复用** - Milvus 客户端池化（MilvusClientManager）
4. **缓存策略** - 常见查询结果缓存（Redis）
5. **模型选择** - 根据场景选择合适的模型（速度 vs 质量）

## 注意事项

⚠️ **关键约束**

1. **端口配置**：默认端口 8888（不是 8000），root_path="/rag"
2. **硬编码配置**：`main.py` 中硬编码了 RabbitMQ、MinIO、Milvus 连接信息
3. **模型配置**：`model_config.json` 需手动创建（已在 .gitignore）
4. **Milvus 集合**：按 kbId 创建独立集合，30 分钟无访问自动释放
5. **文档分块**：chunk_size=800, overlap=100（RecursiveCharacterTextSplitter）
6. **异步任务**：RabbitMQ 消费者随应用启动，需确保队列已创建
7. **向量维度**：默认 1024 维（Qwen text-embedding-v4），需与 Milvus 集合一致

## 常见问题

**Q: RabbitMQ 连接失败？**  
A: 检查 RabbitMQ 是否启动，用户名密码是否正确，队列是否已创建

**Q: Milvus 插入向量失败？**  
A: 确认集合是否已创建，向量维度是否匹配（1024），是否有足够权限

**Q: 文档解析乱码？**  
A: 检查文档编码，PDF 可能需要 OCR（Tesseract），中文需正确字体支持

**Q: 流式响应中断？**  
A: 检查网络稳定性，增加超时时间，确保前端正确处理 SSE 连接

**Q: 为什么端口是 8888？**  
A: `main.py` 中明确配置为 8888，与 rag-server 的 LLM 服务地址配置一致

**Q: 如何切换 Embedding 模型？**  
A: 修改 `model_config.json` 中的 `embedding.provider` 和 `embedding.model`

---

**返回主文档**：[../README.md](../README.md)
