---
name: kb-operations
description: 知识库操作：列出/创建知识库、上传本地文件（PDF、docx、xlsx、图片、代码等）、关键词与语义检索、读取原文片段、扩展上下文、删除文档。当用户说"把文件传到知识库""上传资料到知识库""在知识库里查…""知识库有哪些文档/文件"时使用。
license: Apache-2.0
---

# 知识库操作

MCP 工具负责结构化操作，本地文件的字节用脚本直传。**不要把文件内容塞进对话上下文**（base64 传 10MB 级 PDF 会直接炸上下文）。

## 分工

| 场景 | 用什么 |
| --- | --- |
| 列出可访问的知识库 | MCP `list_knowledge_bases` |
| 创建知识库 | MCP `create_knowledge_base` |
| 已知术语/错误码检索 | MCP `search_knowledge_base_by_keywords` |
| 探索性问题检索 | MCP `search_knowledge_base_by_semantics` |
| 找文件名（不确定 documentId） | MCP `find_knowledge_base_files` |
| 精读原文 | MCP `read_knowledge_base_chunks` |
| 补全命中片段前后文 | MCP `expand_knowledge_base_context` |
| 删除文档 | MCP `delete_private_knowledge_base_file` |
| 上传**文本/代码**（小文件） | MCP `upload_private_knowledge_base_file` 或脚本 |
| 上传**PDF/docx/xlsx/图片/大文件** | 脚本 `scripts/upload_kb_file.py`（必须） |

## 上传前先确认权限

上传接口只接受**本人创建、visibility=private 的个人私有知识库**。共享库、受邀库、公开库一律返回 403「仅支持操作本人创建的个人私有知识库」。

先用 `list_knowledge_bases` 看 `owned` 分组里 `visibility=private` 的库，拿到 `knowledgeBaseId`。

## 上传脚本

```bash
# <skill-dir> 是本 skill 的绝对目录（技能加载时注入），下面的命令都以它开头
# 列一下本账号创建且可上传的库
python <skill-dir>/scripts/upload_kb_file.py --list

# 上传单个文件（PDF/docx/xlsx/图片/任意二进制都行），<kb-id> 从上面的列表或 list_knowledge_bases 取
python <skill-dir>/scripts/upload_kb_file.py --kb-id <kb-id> --file "D:/docs/手册.pdf"

# 批量上传
python <skill-dir>/scripts/upload_kb_file.py --kb-id <kb-id> --file a.pdf --file b.docx

# 保留目录结构（同名文件冲突时也靠它区分）
python <skill-dir>/scripts/upload_kb_file.py --kb-id <kb-id> --file a.pdf --name "docs/规范/手册.pdf"

# 只组装请求不发网络，用来排查参数（不产生任何数据）
python <skill-dir>/scripts/upload_kb_file.py --kb-id <kb-id> --file a.pdf --dry-run
```

Windows PowerShell 里把路径用引号包起来即可，Python 3.9+ 标准库，无需安装依赖。

环境变量：

- `RAG_ACCESS_KEY` 必填，`grs_ak_` 开头的 Access Key（也可 `--access-key` 传入）。环境里没有时向用户索要——用户如果已经在 MCP 客户端里配过 kb 服务，同一把 key 就在那个服务的 `Authorization: Bearer` 头里，可以直接问用户要或让他确认后复用。**不要猜测或虚构 key**。
- `RAG_OPENAPI_BASE` 可选，默认 `https://starvpn.forwardforever.top:5616/api/openapi/v1`

脚本把文件以 multipart 流式 POST 到 `POST /openapi/v1/knowledge-bases/{kbId}/documents`，与站内上传走同一套服务端逻辑，字节从磁盘直达服务器。

## 上传后的状态

接口返回 `documentId`、`fileName`、`status`。文档要经过解析、切分、向量化后才可检索，刚上传时 `status` 不是就绪态。**检索不到不要立刻判定失败**，先用 `find_knowledge_base_files` 确认文档存在及片段数。

## 硬限制

- 单文件 100MB、单请求 100MB（`application.yml` 的 multipart 配置）
- 同一知识库内**同名文件会被拒绝**（400「同名文件已存在」），要么改名、要么先删再传
- 建议使用这些后缀：`.md .txt .pdf .docx .json .py .java .js .ts .vue .html .xml .yml .sh .rb .css .scss .jpg .jpeg .png .gif .bmp .webp`；服务端不做扩展名白名单校验，但保持一致，避免解析器不认识
- 不要臆造 `documentId` 或 `chunkIndex`，它们只能来自工具返回

## 常见报错对照

| 报错 | 原因 / 处理 |
| --- | --- |
| 仅支持操作本人创建的个人私有知识库 | 库里不是自己的，或 visibility 不是 private |
| 同名文件已存在 | 换 `--name`，或先删旧文档 |
| 没有权限操作该知识库 | Access Key 对应账号不是该库所有者 |
| 下游文档服务当前不可用 | Java 侧不可达 / 服务未启动，检查 base URL |
| 文件超过 100MB | multipart 上限，先拆分 |

## 典型流程

1. `list_knowledge_bases` → 取私有库 id
2. 上传：文本走 MCP，其余走脚本
3. `find_knowledge_base_files` 确认文档与片段数
4. `search_knowledge_base_by_keywords` / `search_knowledge_base_by_semantics` 命中
5. `read_knowledge_base_chunks` / `expand_knowledge_base_context` 精读
