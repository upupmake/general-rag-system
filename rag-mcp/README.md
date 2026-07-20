# General RAG MCP 服务

`rag-mcp` 是独立的 FastMCP 3 服务，通过 Streamable HTTP 提供知识库工具。它负责 MCP 协议、Bearer Access Key 鉴权入口、工具参数校验和调用编排，不直接访问 MySQL 或 Milvus，也不复制知识库权限规则。

一次工具调用的链路如下：

1. 客户端向 `/mcp` 发送 `Authorization: Bearer grs_ak_...`。
2. `rag-mcp` 调用 Java OpenAPI 的 `/auth/verify` 校验 Access Key，并使用 Java 返回的用户身份处理后续请求。
3. 知识库访问先由 Java OpenAPI 授权；创建知识库、上传文件和删除文件也由 Java OpenAPI 执行。
4. 检索工具将 Java 授权后的 `ownerUserId`、`knowledgeBaseId` 和工具参数发送到内网 `rag-llm` 的原子检索接口。

`rag-llm` 的检索接口没有公开鉴权层，必须只部署在受控内网，不能直接暴露给外部调用者。

## 工具

服务提供 9 个工具。文件名模式只用于发现文件；限定检索范围、读取片段和扩展上下文均使用 `documentId`，不使用文件名代替文档 ID。

### `list_knowledge_bases`

无参数。调用 Java `GET /knowledge-bases`，按当前 Access Key 可访问来源返回知识库列表，包括本人创建、工作空间共享、受邀和公开知识库分类。

### `create_knowledge_base`

调用 Java `POST /knowledge-bases` 创建知识库。

| 参数 | 类型 | 限制与默认值 |
| --- | --- | --- |
| `name` | `string` | 必填，长度 1 到 100 |
| `visibility` | `"private"` 或 `"public"` | 可选，默认 `"private"`；不支持工作空间共享 |
| `description` | `string` 或 `null` | 可选，最多 200 个字符，默认 `null` |

### `search_knowledge_base_by_keywords`

先调用 Java `GET /knowledge-bases/{knowledge_base_id}/access` 授权，再调用内网 `rag-llm` 的 `POST /rag/retrieval/keywords`。

| 参数 | 类型 | 限制与默认值 |
| --- | --- | --- |
| `knowledge_base_id` | `integer` | 必填，知识库 ID |
| `keywords` | `list[string]` | 必填，至少 1 个关键词 |
| `match_mode` | `"AND"` 或 `"OR"` | 可选，默认 `"OR"` |
| `top_k` | `integer` | 可选，1 到 50，默认 `15` |
| `document_ids` | `list[integer]` 或 `null` | 可选；提供后仅在指定文档中检索 |

### `search_knowledge_base_by_semantics`

先完成 Java 知识库授权，再调用内网 `rag-llm` 的 `POST /rag/retrieval/semantic`，执行多查询语义召回、关键词辅助召回、去重、Rerank 和相关性过滤。

| 参数 | 类型 | 限制与默认值 |
| --- | --- | --- |
| `knowledge_base_id` | `integer` | 必填，知识库 ID |
| `queries` | `list[string]` | 必填，1 到 10 条语义查询 |
| `relevance_query` | `string` | 必填，长度至少 1，用于 Rerank 和相关性判断 |
| `top_k` | `integer` | 可选，1 到 50，默认 `10` |
| `relevance_threshold` | `float` | 可选，0 到 1，默认 `0.3` |

### `find_knowledge_base_files`

先完成 Java 知识库授权，再调用内网 `rag-llm` 的 `POST /rag/retrieval/files`。

| 参数 | 类型 | 限制与默认值 |
| --- | --- | --- |
| `knowledge_base_id` | `integer` | 必填，知识库 ID |
| `name_pattern` | `string` | 必填，长度至少 1；可使用 `%` 匹配任意字符 |
| `offset` | `integer` | 可选，从 0 开始，默认 `0` |
| `limit` | `integer` | 可选，1 到 100，默认 `30` |

该工具返回文件的 `documentId`、文件名和总片段数，供后续检索范围、片段读取或上下文扩展使用。

### `read_knowledge_base_chunks`

先完成 Java 知识库授权，再调用内网 `rag-llm` 的 `POST /rag/retrieval/chunks`，按 `documentId` 读取连续原文片段。

| 参数 | 类型 | 限制 |
| --- | --- | --- |
| `knowledge_base_id` | `integer` | 必填，知识库 ID |
| `document_id` | `integer` | 必填，文档 ID |
| `start_chunk_index` | `integer` | 必填，从 0 开始 |
| `end_chunk_index` | `integer` | 必填，从 0 开始，不能小于起始索引；单次最多读取 20 个片段 |

### `expand_knowledge_base_context`

先完成 Java 知识库授权，再调用内网 `rag-llm` 的 `POST /rag/retrieval/context`，围绕命中的片段扩展上下文。

| 参数 | 类型 | 限制与默认值 |
| --- | --- | --- |
| `knowledge_base_id` | `integer` | 必填，知识库 ID |
| `document_id` | `integer` | 必填，文档 ID |
| `chunk_index` | `integer` | 必填，从 0 开始 |
| `window_size` | `integer` | 可选，0 到 9，默认 `2`；表示中心片段前后各扩展的片段数 |

### `upload_private_knowledge_base_file`

只允许操作当前 Access Key 用户本人创建、且可见性为 `private` 的个人知识库。服务先调用 Java `GET /knowledge-bases/{knowledge_base_id}/private-access` 校验，再调用 Java `POST /knowledge-bases/{knowledge_base_id}/documents` 上传。

| 参数 | 类型 | 限制 |
| --- | --- | --- |
| `knowledge_base_id` | `integer` | 必填，必须是本人创建的个人私有知识库 |
| `file_name` | `string` | 必填，长度至少 1；必须是安全相对路径，不能是绝对路径，不能包含 `.` 或 `..` 路径段或空字符 |
| `content` | `string` | 必填，UTF-8 文本内容 |

该工具适用于文本文件和代码文件，例如 `.txt`、`.md`；PDF 或其他类型文件请使用网页端上传。上传过程中使用临时文件，调用结束后会清理。

### `delete_private_knowledge_base_file`

只允许操作本人创建的个人私有知识库。服务先调用 Java 私有访问校验，再调用 Java `DELETE /knowledge-bases/{knowledge_base_id}/documents/{document_id}` 删除文档；Java 负责再次校验文档与知识库的绑定关系。

| 参数 | 类型 | 限制 |
| --- | --- | --- |
| `knowledge_base_id` | `integer` | 必填，必须是本人创建的个人私有知识库 |
| `document_id` | `integer` | 必填，必须属于指定知识库 |

## 配置

配置集中在 `rag_mcp/config.py`，环境变量存在时覆盖代码默认值。生产环境应显式设置敏感配置，密码不会在本说明中展开。

| 环境变量 | 代码默认值 | 说明 |
| --- | --- | --- |
| `JAVA_OPENAPI_BASE_URL` | `https://starvpn.forwardforever.top:5616/api/openapi/v1` | Java Access Key 鉴权、知识库授权和文档变更 OpenAPI |
| `RAG_LLM_BASE_URL` | `http://192.168.188.6:8848/rag` | 内网 `rag-llm` 基础地址；检索路径追加 `/retrieval/keywords`、`/semantic`、`/files`、`/chunks` 或 `/context` |
| `MCP_HOST` | `0.0.0.0` | MCP 监听地址 |
| `MCP_PORT` | `8858` | MCP 监听端口 |
| `RABBITMQ_HOST` | `192.168.188.6` | 审计消息 RabbitMQ 地址 |
| `RABBITMQ_PORT` | `5678` | RabbitMQ 端口 |
| `RABBITMQ_USERNAME` | `make` | RabbitMQ 用户名 |
| `RABBITMQ_PASSWORD` | 代码内默认值 | RabbitMQ 密码；生产环境应通过环境变量设置 |
| `MCP_AUDIT_EXCHANGE` | `rag.audit.exchange` | 审计 Direct 类型交换机名称 |
| `MCP_AUDIT_ROUTING_KEY` | `rag.mcp.tool.log.v1` | 审计消息路由键 |

`MCP_PUBLIC_URL` 不是当前配置项，不要设置或恢复它。客户端知道完整的 `/mcp` 地址并直接提供 Bearer Access Key。

## 安装与运行

在 `rag-mcp` 目录执行：

```powershell
python -m pip install -r requirements.txt
python -m rag_mcp.server
```

依赖版本范围来自 `requirements.txt`：`fastmcp>=3.2,<4`、`httpx>=0.27,<1`、`aio-pika>=9.4,<10`。

本地 Streamable HTTP 地址：

```text
http://127.0.0.1:8858/mcp
```

客户端请求头：

```text
Authorization: Bearer grs_ak_你的完整AccessKey
```

完整 Access Key 只应放在请求头中，不要写入 URL、代码仓库、日志或审计消息。

## RabbitMQ 审计

每次工具调用最终发布一条 `SUCCESS` 或 `FAIL` 审计消息。消息包含 Java 鉴权得到的 `userId`、`accessKeyId`、工具名、调用 ID、知识库或文档 ID、输入摘要、结果摘要、状态、错误信息和耗时；不包含完整 Access Key、检索片段正文、上传内容或知识库描述正文。发布使用持久化消息，并等待 RabbitMQ publisher confirm；失败会重试，工具调用不会因为审计发布异常而重新执行。

`rag-mcp` 负责声明并发布到持久化 Direct 类型交换机；`rag-server` 负责声明队列、绑定和消费：

| 类型 | 名称 |
| --- | --- |
| 审计交换机 | `rag.audit.exchange` |
| 审计路由键 | `rag.mcp.tool.log.v1` |
| 审计队列 | `rag.mcp.tool.log.queue` |
| 死信交换机（DLX） | `rag.audit.dlx` |
| 死信路由键 | `rag.mcp.tool.log.dead` |
| 死信队列 | `rag.mcp.tool.log.dead.queue` |

审计队列配置为持久化队列；消费失败时最多尝试 3 次且不重新入队，消息随后按上述 DLX 路由到死信队列。

## 公网 Nginx 代理

公网 TLS 由 Nginx 终止，FastMCP 只监听后端的 `8858` 端口，不绑定公网 TLS 端口，也不管理证书。客户端使用公网域名的完整 `/mcp` 地址；Nginx 将同一路径代理到后端 `/mcp`，不要设置 `MCP_PUBLIC_URL`。

下面是 `/mcp` 位置的关键配置。证书、域名和 `listen` 端口按实际部署补充：

```nginx
location /mcp {
    proxy_pass http://127.0.0.1:8858/mcp;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Authorization $http_authorization;
    proxy_buffering off;
    proxy_cache off;
    gzip off;
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
}
```

不要在 Nginx 日志或应用日志中记录 `Authorization` 请求头或完整 Access Key。

## 运行边界

- Java OpenAPI 是 Access Key 身份、知识库访问权限以及知识库和文档变更的唯一控制方。
- `rag-mcp` 不连接 MySQL、Milvus，不实现或绕过 Java 权限判断。
- `rag-llm` 的 `/rag/retrieval/*` 只供 `rag-mcp` 在内网调用，不能公开暴露。
- 上传和删除只支持本人创建的个人私有知识库；创建工具只接受 `private` 或 `public`。
- 访问日志、审计消息和错误信息都不得泄露完整 Access Key、上传内容或检索正文。
