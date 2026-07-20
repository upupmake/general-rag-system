# Rerank API 契约

Rerank 服务默认地址为 `http://localhost:8891`，默认模型为 `Qwen/Qwen3-Reranker-0.6B`。请求和响应使用 JSON；接口当前不要求认证。

接口返回查询与文档对的相关性分数，但不会按分数重新排列结果。调用方需要使用 `relevance_score` 自行排序。

## API端点

### `GET /`：服务信息

获取服务基本信息。

**端点**: `GET /`

**响应示例**:
```json
{
  "service": "Rerank Service",
  "version": "1.0.0",
  "model": "Qwen/Qwen3-Reranker-0.6B",
  "status": "running"
}
```

---

### `GET /health`：健康检查

检查服务健康状态和配置信息。

**端点**: `GET /health`

**响应示例**:
```json
{
  "status": "healthy",
  "model": "Qwen/Qwen3-Reranker-0.6B",
  "gpu_memory_utilization": 0.25,
  "max_model_len": 8192,
  "device": "cuda"
}
```

**状态码**:
- `200`: 服务健康
- `503`: 模型未加载

---

### `POST /v1/rerank`：计算相关性分数

对查询-文档对进行重排序，返回相关性分数。

**端点**: `POST /v1/rerank`

**请求参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `pairs` | `QueryDocPair[]` | 是 | 查询与文档对列表；不可为空 |
| `instruction` | `string \| null` | 否 | 任务指令；缺省、`null` 或空字符串时使用默认检索指令 |
| `model` | `string \| null` | 否 | 当前实现忽略该字段，实际模型由服务配置决定 |

**QueryDocPair 结构**:
```json
{
  "query": "查询文本",
  "document": "文档文本"
}
```

**完整请求示例**:
```json
{
  "pairs": [
    {
      "query": "What is the capital of China?",
      "document": "The capital of China is Beijing."
    },
    {
      "query": "What is the capital of China?",
      "document": "Shanghai is the largest city in China."
    },
    {
      "query": "What is the capital of China?",
      "document": "China has a long history."
    }
  ],
  "instruction": "Given a web search query, retrieve relevant passages that answer the query"
}
```

**响应示例**:
```json
{
  "results": [
    {
      "index": 0,
      "relevance_score": 0.9876,
      "query": "What is the capital of China?",
      "document": "The capital of China is Beijing."
    },
    {
      "index": 1,
      "relevance_score": 0.4321,
      "query": "What is the capital of China?",
      "document": "Shanghai is the largest city in China."
    },
    {
      "index": 2,
      "relevance_score": 0.1234,
      "query": "What is the capital of China?",
      "document": "China has a long history."
    }
  ],
  "model": "Qwen/Qwen3-Reranker-0.6B",
  "processing_time": 0.125
}
```

**响应字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `results` | `array` | 与输入顺序一致的结果列表 |
| `results[].index` | `integer` | 原始输入中的索引位置 |
| `results[].relevance_score` | `float` | `yes` 与 `no` 概率归一化后的相关性分数 |
| `results[].query` | `string` | 原始查询文本 |
| `results[].document` | `string` | 原始文档文本 |
| `model` | `string` | 服务配置中的 `model_name` |
| `processing_time` | `float` | 从输入处理到完成打分的耗时，单位为秒 |

**状态码**:
- `200`：成功
- `400`：`pairs` 是空列表，或任一 `query`、`document` 是空白文本
- `422`：缺少 `pairs`、字段类型不符合请求模型，或 JSON 无法校验
- `500`：输入处理或模型打分过程中出现未处理异常
- `503`：模型尚未加载

**错误响应示例**:
```json
{
  "detail": "Pairs cannot be empty"
}
```

---

### `POST /rerank`：相关性打分别名

请求模型、处理逻辑、响应和错误与 `/v1/rerank` 相同。

**端点**: `POST /rerank`

**请求/响应**: 与 `/v1/rerank` 相同

---

## 使用示例

### cURL

```bash
# 单个查询-文档对
curl -X POST http://localhost:8891/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "pairs": [
      {
        "query": "Python编程",
        "document": "Python是一种高级编程语言"
      }
    ]
  }'

# 批量查询-文档对
curl -X POST http://localhost:8891/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "pairs": [
      {"query": "Python编程", "document": "Python是一种编程语言"},
      {"query": "Python编程", "document": "Java也是一种编程语言"},
      {"query": "Python编程", "document": "机器学习很流行"}
    ],
    "instruction": "检索与编程相关的文档"
  }'
```

### Python

```python
import requests

query = "Python 编程语言"
documents = ["Python 是一种编程语言", "今天有雨"]
pairs = [{"query": query, "document": document} for document in documents]

response = requests.post(
    "http://localhost:8891/v1/rerank",
    json={"pairs": pairs},
)
response.raise_for_status()
results = sorted(
    response.json()["results"],
    key=lambda item: item["relevance_score"],
    reverse=True,
)
```

## 错误与处理约束

业务错误响应使用 `{"detail": "错误信息"}`。FastAPI 请求模型校验失败时返回 422，`detail` 为校验错误数组。

- 空 `pairs` 返回 `{"detail": "Pairs cannot be empty"}`。
- 空白查询返回 `{"detail": "Query at index {i} is empty"}`。
- 空白文档返回 `{"detail": "Document at index {i} is empty"}`。
- 模型未加载返回 `{"detail": "Model not loaded"}`。
- 打分异常返回 500，`detail` 以 `Internal error:` 开头。

## 实现注意事项

- 默认 instruction 为 `Given a web search query, retrieve relevant passages that answer the query`。
- `instruction` 传入 `null` 或空字符串时也会使用默认值。
- 输入 token 序列按配置的 `max_length` 截断，并为固定 suffix 保留空间。
- 服务没有请求级模型切换能力，也没有在 API 层声明最大批量大小。

## 相关文档

- [模块总览](README.md)
- [Rerank 实现与配置](RerankREADME.md)
- [Rerank 快速开始](RerankQUICKSTART.md)
