# Embedding 服务快速开始

以下命令均在 `embedding_rerank` 目录执行。

## 1. 检查配置

默认配置位于 `config/embedding_config.py`：

- 模型：`Qwen/Qwen3-Embedding-0.6B`
- 监听地址：`0.0.0.0`
- 端口：`8890`

如需调整，直接修改 `EmbeddingConfig` 的类属性并重启服务。当前服务配置不读取环境变量。

## 2. 启动服务

```bash
python embedding_start.py
```

启动脚本通过 Uvicorn 加载 `service.embedding_service:app`，并在启动阶段加载模型。

## 3. 验证健康状态

另开终端执行：

```bash
curl http://localhost:8890/health
```

模型加载完成时，接口返回 `status` 为 `healthy`，并包含模型名称、当前模型长度配置和设备类型。

## 4. 生成向量

```bash
curl -X POST http://localhost:8890/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":"什么是检索增强生成？"}'
```

检查响应中的 `data[0].embedding`。默认模型输出 1024 维向量，`data[0].index` 为 `0`。

批量输入可直接传入字符串数组：

```bash
curl -X POST http://localhost:8890/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":["文本一","文本二"]}'
```

## 下一步

- [Embedding 实现与配置](EmbeddingREADME.md)
- [Embedding API 契约](EmbeddingAPI.md)
- [模块总览](README.md)
