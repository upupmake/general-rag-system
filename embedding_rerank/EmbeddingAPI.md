# Embedding API 契约

Embedding 服务默认地址为 `http://localhost:8890`，默认模型为 `Qwen/Qwen3-Embedding-0.6B`，默认输出 1024 维向量。请求和响应使用 JSON；接口当前不要求认证。

`POST /v1/embeddings` 的响应结构接近 OpenAI Embeddings API，但额外支持 `instruction`，且 token 用量为服务内部估算值，不应视为完整的 OpenAI API 兼容实现。

## 端点列表

### 1. GET / - 根路径

获取服务基本信息。

**请求**
```bash
curl http://localhost:8890/
```

**响应**
```json
{
  "service": "Embedding Service",
  "version": "1.0.0",
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "status": "running"
}
```

---

### `GET /health`：健康检查

检查服务健康状态和配置信息。

**请求**
```bash
curl http://localhost:8890/health
```

**响应**
```json
{
  "status": "healthy",
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "gpu_memory_utilization": 0.15,
  "max_model_len": 8192,
  "device": "cuda"
}
```

**状态码**
- `200 OK` - 服务正常
- `503 Service Unavailable` - 模型未加载

---

### `POST /v1/embeddings`：生成向量

接收单个文本或文本列表，并按输入顺序返回向量。

**请求**

```bash
curl -X POST http://localhost:8890/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "你的文本内容",
    "instruction": "可选的任务指令"
  }'
```

**请求体参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `input` | `string \| string[]` | 是 | 单个文本或文本列表；列表不可为空，每条文本不可为空白 |
| `instruction` | `string \| null` | 否 | 非空时，每条文本会被改写为 `Instruct: {instruction}\nQuery: {text}` |
| `model` | `string \| null` | 否 | 当前实现忽略该字段，实际模型由服务配置决定 |

**请求示例**

*单个文本*
```json
{
  "input": "What is the capital of China?"
}
```

*批量文本*
```json
{
  "input": [
    "What is the capital of China?",
    "Explain gravity",
    "What is machine learning?"
  ]
}
```

*带指令的文本*
```json
{
  "input": ["What is the capital of China?"],
  "instruction": "Given a web search query, retrieve relevant passages that answer the query"
}
```

**响应**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.123, -0.456, 0.789]
    }
  ],
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

**响应字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| `object` | `string` | 固定值 "list" |
| `data` | `array` | 向量数据列表 |
| `data[].object` | `string` | 固定值 "embedding" |
| `data[].index` | `integer` | 在输入列表中的索引位置 |
| `data[].embedding` | `float[]` | 向量数据；默认模型输出 1024 维，示例数组已省略 |
| `model` | `string` | 服务配置中的 `model_name` |
| `usage.prompt_tokens` | `integer` | 对原始输入文本按 `len(text) // 2` 得到的估算总数 |
| `usage.total_tokens` | `integer` | 与 `usage.prompt_tokens` 相同 |

**状态码**
- `200 OK`：成功
- `400 Bad Request`：`input` 是空列表，或其中存在空白文本
- `422 Unprocessable Entity`：缺少 `input`、字段类型不符合请求模型，或 JSON 无法校验
- `500 Internal Server Error`：向量生成过程中出现未处理异常
- `503 Service Unavailable`：模型尚未加载

---

### `POST /embeddings`：生成向量别名

请求模型、处理逻辑、响应和错误与 `/v1/embeddings` 相同。

**请求**
```bash
curl -X POST http://localhost:8890/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "你的文本"}'
```

---

## 使用示例

### Python 示例

```python
import requests

response = requests.post(
    "http://localhost:8890/v1/embeddings",
    json={"input": ["文本一", "文本二"]},
)
response.raise_for_status()
result = response.json()
embeddings = [item["embedding"] for item in result["data"]]
```

## 错误响应

业务错误使用如下结构：

```json
{
  "detail": "错误描述信息"
}
```

### 常见错误

**400 Bad Request**
```json
{
  "detail": "Input cannot be empty"
}
```

**503 Service Unavailable**
```json
{
  "detail": "Model not loaded"
}
```

**500 Internal Server Error**
```json
{
  "detail": "Internal error: [具体错误信息]"
}
```

FastAPI 请求模型校验失败时返回 422，`detail` 为校验错误数组，而不是上述字符串结构。

## 实现注意事项

- `data` 与输入顺序一致，`index` 从 `0` 开始。
- 默认模型的向量维度为 1024。
- 服务没有请求级模型切换能力。
- 服务没有在 API 层声明最大批量大小。
- 服务没有显式截断输入；模型长度配置见 [EmbeddingREADME.md](EmbeddingREADME.md)。

## 相关文档

- [模块总览](README.md)
- [Embedding 实现与配置](EmbeddingREADME.md)
- [Embedding 快速开始](EmbeddingQUICKSTART.md)
