# Embedding 实现与配置

Embedding 服务位于 `service/embedding_service.py`，使用 vLLM 的 `LLM(task="embed")` 加载 `Qwen/Qwen3-Embedding-0.6B`，默认监听 `8890` 端口。模型输出为 1024 维向量。

本文说明服务实现、配置和运行约束。请求响应契约见 [EmbeddingAPI.md](EmbeddingAPI.md)，最短启动步骤见 [EmbeddingQUICKSTART.md](EmbeddingQUICKSTART.md)。

## 组件关系

- `config/embedding_config.py`：定义 `EmbeddingConfig`，并创建全局 `config` 实例。
- `service/embedding_service.py`：定义 FastAPI 应用、生命周期、健康检查和向量接口。
- `embedding_start.py`：读取 `config`，通过 Uvicorn 启动 `service.embedding_service:app`。

服务启动时加载模型；加载失败会抛出异常并阻止服务进入可用状态。服务关闭时释放模型引用，并在 CUDA 可用时调用 `torch.cuda.empty_cache()`。


## 实现流程

`POST /v1/embeddings` 和 `POST /embeddings` 都调用同一个处理函数：

1. 将单个字符串标准化为字符串列表。
2. 拒绝空列表，以及空字符串或去除首尾空白后为空的文本。
3. 如果请求提供 `instruction`，把每条文本转换为 `Instruct: {instruction}\nQuery: {text}`。
4. 调用 vLLM 的 `embedding_model.embed(processed_texts)`。
5. 返回每条向量、输入索引、模型名和 token 数估算值。

请求中的 `model` 字段会被 Pydantic 接收，但当前实现不使用它；实际模型由配置中的 `model_name` 决定。

## 运行依赖

代码直接导入 `torch`、`fastapi`、`pydantic`、`vllm`，因此运行环境需要提供这些依赖及其兼容版本。具体版本不在当前模块代码中声明。

## 运行方式

在 `embedding_rerank` 目录执行：

```bash
python embedding_start.py
```

启动脚本默认传入 `host=0.0.0.0`、`port=8890`、`workers=1`、`reload=False`。服务加载完成后，通过 `GET /health` 检查模型是否可用。

当前实现会检查 `torch.cuda.is_available()` 并把设备信息写入日志和健康响应；代码没有独立的 CPU 降级流程说明，实际模型能否在当前设备运行取决于 vLLM 和运行环境。

## 配置说明

配置方式：直接修改 `config/embedding_config.py` 中 `EmbeddingConfig` 的类属性，然后重启服务。当前实现不读取环境变量。

默认配置如下：

| 属性 | 默认值 | 运行时用途 |
| --- | --- | --- |
| `model_name` | `Qwen/Qwen3-Embedding-0.6B` | 传给 vLLM 的模型标识，也写入响应 |
| `model_path` | `None` | 当前服务未使用 |
| `gpu_memory_utilization` | `0.15` | 传给 vLLM |
| `max_model_len` | `8192` | 传给 vLLM，并写入健康响应 |
| `tensor_parallel_size` | `1` | 传给 vLLM |
| `dtype` | `float16` | 传给 vLLM |
| `host` | `0.0.0.0` | Uvicorn 监听地址 |
| `port` | `8890` | Uvicorn 监听端口 |
| `workers` | `1` | Uvicorn 工作进程数 |
| `batch_timeout_ms` | `10` | 当前服务未使用 |
| `log_level` | `INFO` | 日志配置和 Uvicorn 日志级别 |

`model_path` 虽然在配置类中声明，但模型初始化只读取 `config.model_name`。`batch_timeout_ms` 也没有连接到请求队列或批处理器，不能据此推断存在动态批处理。

### 模型初始化参数

服务创建 vLLM 实例时实际传入：

```python
LLM(
    model=config.model_name,
    task="embed",
    gpu_memory_utilization=config.gpu_memory_utilization,
    max_model_len=config.max_model_len,
    tensor_parallel_size=config.tensor_parallel_size,
    dtype=config.dtype,
    trust_remote_code=True,
    enable_chunked_prefill=False,
)
```

`enable_chunked_prefill=False` 是服务代码固定传入的参数，不是配置项。

## API 端点

| 端点 | 方法 | 处理函数 |
| --- | --- | --- |
| `/` | GET | 返回服务名、版本、模型名和运行状态 |
| `/health` | GET | 模型加载后返回健康信息，未加载时返回 503 |
| `/v1/embeddings` | POST | 生成向量 |
| `/embeddings` | POST | 调用同一向量处理逻辑 |

请求和响应字段以 [EmbeddingAPI.md](EmbeddingAPI.md) 为准。

## 关键约束

- 默认模型 `Qwen/Qwen3-Embedding-0.6B` 输出 1024 维向量。
- 批量响应的 `data` 顺序与输入顺序一致，`index` 从 `0` 开始。
- `instruction` 会应用到请求中的每条文本。
- `usage.prompt_tokens` 和 `usage.total_tokens` 都来自 `len(text) // 2` 的简化估算，不是 tokenizer 的精确统计；统计使用原始文本，不包含拼接后的指令。
- 当前接口没有鉴权、限流或请求批量上限配置。
- 服务代码没有为输入执行显式截断；可接受长度受 vLLM 的 `max_model_len` 及模型处理行为约束。

## 相关文档

- [模块总览](README.md)
- [Embedding 快速开始](EmbeddingQUICKSTART.md)
- [Embedding API 契约](EmbeddingAPI.md)
