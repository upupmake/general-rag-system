# General RAG MCP

独立的 FastMCP 3 知识检索服务。服务只负责 MCP 协议、工具参数校验和调用编排：

1. 使用调用者的 Access Key 请求 Java OpenAPI。
2. Java 判断用户是否可以访问目标知识库，并返回授权后的知识库路由信息。
3. MCP 服务调用内部 `rag-llm` 原子检索接口。

MCP 服务不直接访问 MySQL 或 Milvus，也不实现知识库权限规则。

## 工具

- `list_knowledge_bases`
- `search_knowledge_base_by_keywords`
- `search_knowledge_base_by_semantics`
- `find_knowledge_base_files`
- `read_knowledge_base_chunks`
- `expand_knowledge_base_context`

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `JAVA_OPENAPI_BASE_URL` | `https://starvpn.forwardforever.top:5616/api/openapi/v1` | Java Access Key 和知识库权限 OpenAPI |
| `RAG_LLM_BASE_URL` | `http://192.168.188.6:8848/rag` | 内部 RAG 检索服务地址 |
| `MCP_HOST` | `0.0.0.0` | MCP 监听地址 |
| `MCP_PORT` | `8858` | MCP 监听端口 |
| `RABBITMQ_HOST` | `192.168.188.6` | MCP 工具审计消息 RabbitMQ 地址 |
| `RABBITMQ_PORT` | `5678` | RabbitMQ 端口 |
| `RABBITMQ_USERNAME` | `make` | RabbitMQ 用户名 |
| `RABBITMQ_PASSWORD` | `make20260101` | RabbitMQ 密码 |

`rag-llm` 的检索接口必须部署在受控内部网络中，不能直接暴露给外部调用者。

## 运行

```powershell
python -m pip install -r requirements.txt
python -m rag_mcp.server
```

MCP Streamable HTTP 地址：

```text
http://127.0.0.1:8858/mcp
```

每次工具调用完成后，服务会将用户身份、工具名、输入摘要、结果摘要、状态和耗时异步发布到 RabbitMQ 的 `rag.audit.exchange`。Access Key 明文不会进入消息。

客户端使用 Access Key：

```text
Authorization: Bearer grs_ak_...
```
