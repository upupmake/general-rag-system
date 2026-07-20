# Rerank 服务快速开始

以下命令均在 `embedding_rerank` 目录执行。

## 1. 检查配置

默认配置位于 `config/rerank_config.py`：

- 模型：`Qwen/Qwen3-Reranker-0.6B`
- 监听地址：`0.0.0.0`
- 端口：`8891`

如需调整，直接修改 `RerankConfig` 的类属性并重启服务。当前服务配置不读取环境变量。

## 2. 启动服务

```bash
python rerank_start.py
```

启动脚本通过 Uvicorn 加载 `service.rerank_service:app`，并在启动阶段加载 tokenizer 和模型。

## 3. 验证健康状态

另开终端执行：

```bash
curl http://localhost:8891/health
```

模型加载完成时，接口返回 `status` 为 `healthy`，并包含模型名称、当前模型长度配置和设备类型。

## 4. 计算相关性分数

```bash
curl -X POST http://localhost:8891/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "pairs": [
      {
        "query": "什么是检索增强生成？",
        "document": "检索增强生成会先检索外部知识，再让模型生成回答。"
      }
    ]
  }'
```

检查响应中的 `results[0].relevance_score`。结果保持请求顺序，`results[0].index` 为 `0`；接口不会按分数自动排序。

## 下一步

- [Rerank 实现与配置](RerankREADME.md)
- [Rerank API 契约](RerankAPI.md)
- [模块总览](README.md)
