# Rerank 实现与配置

Rerank 服务位于 `service/rerank_service.py`，使用 vLLM 加载 `Qwen/Qwen3-Reranker-0.6B`，默认监听 `8891` 端口。服务接收查询与文档对，返回字段名为 `relevance_score` 的相关性分数。

本文说明服务实现、配置和运行约束。请求响应契约见 [RerankAPI.md](RerankAPI.md)，最短启动步骤见 [RerankQUICKSTART.md](RerankQUICKSTART.md)。

## 组件关系

- `config/rerank_config.py`：定义 `RerankConfig`，并创建全局 `config` 实例。
- `service/rerank_service.py`：定义 tokenizer、模型、打分流程和 FastAPI 端点。
- `rerank_start.py`：读取 `config`，通过 Uvicorn 启动 `service.rerank_service:app`。

服务启动时先加载 tokenizer，再加载模型并创建采样参数。任一步骤失败都会抛出异常，服务不会进入可用状态。


## 实现流程

`POST /v1/rerank` 和 `POST /rerank` 都调用同一个处理函数：

1. 拒绝空 `pairs`，以及 `query` 或 `document` 为空白的条目。
2. 使用请求中的 `instruction`；未提供或传入 `null`、空字符串时使用默认检索指令。
3. 把每个查询与文档对格式化为 system/user 消息，并应用 tokenizer 的 chat template，且关闭 thinking。
4. 将 token 序列截取到 `max_length - len(suffix_tokens)`，再追加固定 suffix。
5. 调用 vLLM 生成，并只允许 `yes`、`no` 两个 token。
6. 对 `yes` 和 `no` 的概率归一化，得到 `relevance_score`。

请求中的 `model` 字段会被接收但不会参与模型选择，实际模型由配置中的 `model_name` 决定。结果保持请求顺序，服务不按分数自动排序。

## 运行依赖

代码直接导入 `torch`、`fastapi`、`pydantic`、`transformers` 和 `vllm`，因此运行环境需要提供这些依赖及其兼容版本。具体版本不在当前模块代码中声明。

## 运行方式

在 `embedding_rerank` 目录执行：

```bash
python rerank_start.py
```

启动脚本默认传入 `host=0.0.0.0`、`port=8891`、`workers=1`、`reload=False`。服务加载完成后，通过 `GET /health` 检查模型是否可用。

当前实现会检查 `torch.cuda.is_available()` 并把设备信息写入日志和健康响应；代码没有独立的 CPU 降级流程说明，实际模型能否在当前设备运行取决于 vLLM 和运行环境。

## 配置说明

配置方式：直接修改 `config/rerank_config.py` 中 `RerankConfig` 的类属性，然后重启服务。当前实现不读取环境变量。

默认配置如下：

| 属性 | 默认值 | 运行时用途 |
| --- | --- | --- |
| `model_name` | `Qwen/Qwen3-Reranker-0.6B` | tokenizer 与 vLLM 的模型标识，也写入响应 |
| `model_path` | `None` | 当前服务未使用 |
| `gpu_memory_utilization` | `0.25` | 传给 vLLM |
| `max_model_len` | `8192` | 传给 vLLM，并写入健康响应 |
| `tensor_parallel_size` | `1` | 传给 vLLM |
| `dtype` | `float16` | 传给 vLLM |
| `enable_prefix_caching` | `True` | 传给 vLLM |
| `max_length` | `8192` | 请求 token 序列的截断上限，包含固定 suffix |
| `temperature` | `0.0` | 传给 `SamplingParams` |
| `max_tokens` | `1` | 传给 `SamplingParams` |
| `logprobs` | `20` | 传给 `SamplingParams` |
| `host` | `0.0.0.0` | Uvicorn 监听地址 |
| `port` | `8891` | Uvicorn 监听端口 |
| `workers` | `1` | Uvicorn 工作进程数 |
| `batch_timeout_ms` | `10` | 当前服务未使用 |
| `log_level` | `INFO` | 日志配置和 Uvicorn 日志级别 |

`model_path` 虽然在配置类中声明，但 tokenizer 和模型初始化都只读取 `config.model_name`。`batch_timeout_ms` 没有连接到请求队列或批处理器。

## 模型与采样参数

模型初始化使用 `tensor_parallel_size`、`max_model_len`、`enable_prefix_caching`、`gpu_memory_utilization`、`dtype`，并固定传入 `trust_remote_code=True`。采样参数使用 `temperature`、`max_tokens`、`logprobs`，同时把 `allowed_token_ids` 限制为 tokenizer 得到的 `yes` 与 `no` token。

## API 端点

| 端点 | 方法 | 处理函数 |
| --- | --- | --- |
| `/` | GET | 返回服务名、版本、模型名和运行状态 |
| `/health` | GET | 模型加载后返回健康信息，未加载时返回 503 |
| `/v1/rerank` | POST | 计算查询与文档对的相关性分数 |
| `/rerank` | POST | 调用同一重排序处理逻辑 |

请求和响应字段以 [RerankAPI.md](RerankAPI.md) 为准。

## 关键约束

- 单条结果字段为 `relevance_score`，值由 `yes` 与 `no` 的模型概率归一化得到。
- 返回 `results` 与请求 `pairs` 顺序一致，`index` 是原始位置；如需按分数排序，由调用方完成。
- 默认 instruction 为 `Given a web search query, retrieve relevant passages that answer the query`。
- 超长 token 序列会在尾部截断后追加固定 suffix。
- 当前接口没有鉴权、限流或请求批量上限配置。

## 相关文档

- [模块总览](README.md)
- [Rerank 快速开始](RerankQUICKSTART.md)
- [Rerank API 契约](RerankAPI.md)
