# rag-llm 模块说明

本目录是 `rag-llm` 的局部规则，完整项目规则见仓库根目录 `AGENTS.md`。同一主题冲突时以本文件（更接近的规则）为准，但不得据此放宽根文件中的安全、权限和部署约束；模块 API 细节以本目录 `README.md` 和当前代码为准。

## 运行与测试

- 必须在 `rag-llm/` 目录内运行，模块使用扁平顶层导入（`from utils import ...`、`from services.chat import ...`），不是包结构。
- 启动：`uvicorn main:app --host 0.0.0.0 --port 8848`；`root_path=/rag`，FastAPI 路由挂载见 `main.py`。
- 本机测试：

```powershell
cd rag-llm
& 'D:\conda-envs\LLM-from-scratch\python.exe' -m pytest tests -q
```

- 当前基线 47 项全部通过（2026-11-02）。修改 `mq/document_embedding.py` 新增对 `utils` 的调用时，同步更新测试的 `utils` 桩。

## 关键约束

- 知识库聊天只走 Agentic RAG；固定六工具，`stop_search` 由编排层处理，不恢复传统单次 RAG。
- `options.maxRounds` 默认 10，服务端夹紧到 1~20；轮次结束条件见根 `AGENTS.md` 与 `agentic_rag_utils.py`。
- 文档定位使用数值 `documentId`；关键词检索用 `document_ids`，连续读取/上下文扩展用 `document_id`。
- `main.py` 从同目录 `.env` 加载 RabbitMQ/MinIO/Milvus 凭据（`python-dotenv`）；缺失必需变量时启动失败。`.env` 不提交、文件权限 600，不要把值复制到文档、日志或提交信息，也不要新增硬编码密钥。
- `model_config.json` 按 mtime/size 缓存，修改后下一次调用重读；该文件被 `.gitignore` 忽略，只提交 `model_config.json.example`。
- `/rag/retrieval/*` 没有公共鉴权，只能被内网 `rag-mcp` 调用，不得暴露到公网。
- 生产 Milvus 为 v2.6.11，`LIKE` 转义和旧 collection 索引问题见根 `AGENTS.md`；索引脚本默认 dry-run，写操作需显式参数。
- 新增 Python 依赖时同步更新本目录 `requirements.txt` 与两台生产机的固定版本副本。
