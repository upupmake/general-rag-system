# Embedding 与 Rerank 模型服务

`embedding_rerank` 提供两个相互独立的 FastAPI 模型服务：Embedding 服务负责把文本转换为向量，Rerank 服务负责计算查询与文档的相关性分数。两个服务都由 vLLM 加载模型，通过各自的启动脚本运行。

## 服务入口

| 服务 | 默认模型 | 默认地址 | 核心接口 | 启动脚本 |
| --- | --- | --- | --- | --- |
| Embedding | `Qwen/Qwen3-Embedding-0.6B` | `http://localhost:8890` | `POST /v1/embeddings` | `embedding_start.py` |
| Rerank | `Qwen/Qwen3-Reranker-0.6B` | `http://localhost:8891` | `POST /v1/rerank` | `rerank_start.py` |

Embedding 默认输出 1024 维向量。Rerank 的单条结果分数字段为 `relevance_score`。

## 文档导航

### Embedding

- [快速开始](EmbeddingQUICKSTART.md)：最短启动与请求验证步骤。
- [实现与配置](EmbeddingREADME.md)：模型加载、文本处理、配置项及实现约束。
- [API 契约](EmbeddingAPI.md)：端点、请求体、响应体和错误响应。

### Rerank

- [快速开始](RerankQUICKSTART.md)：最短启动与请求验证步骤。
- [实现与配置](RerankREADME.md)：模型打分流程、配置项及实现约束。
- [API 契约](RerankAPI.md)：端点、请求体、响应体和错误响应。

## 目录结构

```text
embedding_rerank/
├── config/
│   ├── embedding_config.py
│   └── rerank_config.py
├── service/
│   ├── embedding_service.py
│   └── rerank_service.py
├── embedding_start.py
├── rerank_start.py
└── README.md
```

## 配置约束

当前配置由 `config/embedding_config.py` 和 `config/rerank_config.py` 中的 Pydantic 配置类提供。需要调整配置时，直接修改对应类属性并重启服务；当前实现不从环境变量读取这些服务配置。

两个配置类都声明了 `model_path` 和 `batch_timeout_ms`，但当前服务初始化与请求处理没有使用这两个字段。模型实际由 `model_name` 指定，请不要把这两个保留字段当作已生效配置。

## 启动

在本目录分别启动两个服务：

```bash
python embedding_start.py
```

```bash
python rerank_start.py
```

启动过程会加载对应模型。服务可用后，可访问：

- Embedding 健康检查：`GET http://localhost:8890/health`
- Rerank 健康检查：`GET http://localhost:8891/health`
- Embedding 交互式接口文档：`http://localhost:8890/docs`
- Rerank 交互式接口文档：`http://localhost:8891/docs`
