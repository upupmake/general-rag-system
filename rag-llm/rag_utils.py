"""
RAG Service - 多角度查询与并行处理的异步RAG服务
封装了以下高级特性：
1. 多角度查询生成(Multi-Query Generation)
2. 并行检索(Parallel Retrieval)
3. Rerank文档重排序(Document Reranking)
4. 对话历史管理（最近5轮）
5. 流式答案生成
"""
import asyncio
import logging
import os
from typing import AsyncGenerator, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

from aiohttp_utils import rerank
from milvus_utils import MilvusClientManager
from utils import get_official_llm, get_embedding_instance, get_structured_data_agent, get_display_docs, \
    unified_llm_stream, get_langchain_llm, filter_grade_threshold, merge_consecutive_chunks

logger = logging.getLogger(__name__)


# ============= Pydantic Models =============
class MultiQueryList(BaseModel):
    """多角度查询列表"""
    queries: list[str] = Field(description="从不同角度生成的语义和关键词查询列表，涵盖至少6个查询")
    grade_query: str = Field(description="用于文档评分的查询，应该是结合上下文、解决指代消歧后的完整问题")
    reasoning: str = Field(description="生成这些查询的原因")


class RAGService:
    """异步RAG服务类"""

    async def generate_multi_queries(
            self,
            question: str,
            history: list,
            model_info: dict
    ) -> tuple[list[str], str]:
        """
        生成多角度查询

        Args:
            question: 当前用户问题
            history: 对话历史（LangChain消息格式）
            model_info: 模型配置信息

        Returns:
            (多角度查询列表, 评分用查询)
        """
        # 构建对话历史（最近4轮，即8条消息）
        history_context = ""
        if history:
            recent_history = history[-8:] if len(history) > 8 else history
            history_context = "\n".join([
                f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
                for m in recent_history
            ])

        system_prompt = f"""你是检索查询优化专家，负责将用户问题转换为多个高效检索查询。

## 任务要求

### 1. 生成 6-10 个查询，覆盖以下角度：
- **定义类**：概念、是什么
- **实现类**：如何做、技术栈、代码
- **对比类**：差异、优缺点
- **场景类**：应用、案例、最佳实践
- **关键词类**（3-4个）：专有名词、API、技术术语（空格分隔）

### 2. 查询质量要求：
✅ 语义明确、互补、包含足够上下文
✅ 技术问题需中英文查询（如涉及API/术语）
❌ 避免：过于宽泛、简单重复、过长啰嗦

### 3. grade_query 生成规则：
- 指代消歧（"它" → 具体对象）
- 补全省略信息（结合历史）
- 简洁陈述句（去掉疑问词）
- 特殊处理："继续" → 延续上轮主题

---

## 示例

**示例1：带上下文**
```
历史：
用户: RAG系统是什么？
助手: RAG是检索增强生成系统...
当前问题: 它的检索流程是怎样的？

输出：
{{
  "queries": [
    "RAG系统的完整检索流程和步骤",
    "多查询生成和并行检索的实现机制",
    "向量检索与关键词检索的协同方式",
    "RAG检索流程与传统搜索的对比",
    "RAG retrieval pipeline implementation",
    "检索流程 多查询 rerank 向量检索",
    "generate_multi_queries parallel_retrieve"
  ],
  "grade_query": "RAG系统的检索流程",
  "reasoning": "'它'指代RAG系统。生成了定义、实现、对比、技术术语查询，覆盖多查询、并行检索、rerank等关键概念。"
}}
```

**示例2：继续类问题**
```
历史：
用户: PDF是如何处理的？
助手: PDF分文本型和图片型...用PyMuPDF和Tesseract...
当前问题: 详细说说OCR部分

输出：
{{
  "queries": [
    "PDF文件OCR识别的实现步骤和流程",
    "Tesseract OCR的配置参数和优化方法",
    "图片型PDF识别准确率提升技巧",
    "OCR识别后的文本清理和预处理",
    "pytesseract Tesseract OCR DPI",
    "chi_sim+eng language configuration"
  ],
  "grade_query": "PDF文件处理中OCR识别的实现细节",
  "reasoning": "用户聚焦OCR部分。生成了实现步骤、工具配置、优化方法、后处理查询，提取pytesseract、Tesseract等关键工具。"
}}
```

---

## 当前任务

**对话历史**：
{history_context if history_context else '无对话历史'}

**当前问题**：{question}

**请输出**：严格JSON格式，包含 queries（6-10个）、grade_query、reasoning"""
        model_info = {
            'name': 'doubao-seed-2.0-pro',
            'provider': 'other'
        }
        generate_config = {
            "extra_body": {
                "thinking": {
                    "type": "disabled"
                }
            }
        }
        llm = get_langchain_llm(model_info, **generate_config)
        structured_agent = get_structured_data_agent(llm, MultiQueryList)
        # 使用异步调用
        result = await structured_agent.ainvoke({"messages": [{"role": "user", "content": system_prompt}]})

        logger.info(f"生成的多角度查询: {result['structured_response'].queries}")
        logger.info(f"生成的评分查询(grade_query): {result['structured_response'].grade_query}")
        return result['structured_response'].queries, result['structured_response'].grade_query

    async def parallel_retrieve(
            self,
            query_list: list[str],
            user_id: int,
            kb_id: int,
            top_k: int = 5
    ) -> list[Document]:
        """
        并行检索多个查询

        Args:
            query_list: 查询列表
            user_id: 用户ID
            kb_id: 知识库ID
            top_k: 每个查询返回的文档数量

        Returns:
            去重后的文档列表
        """
        milvus_uri = os.environ.get("MILVUS_URI")
        milvus_token = os.environ.get("MILVUS_TOKEN")

        embedding_config = {
            'name': 'text-embedding-v4',
            'provider': 'qwen'
        }
        embeddings = get_embedding_instance(embedding_config)

        all_docs = []
        doc_set = set()

        vector_store = None
        try:
            vector_store = await MilvusClientManager.get_instance(
                user_id, kb_id, milvus_uri, milvus_token, embeddings
            )
            if not vector_store:
                logger.warning(f"无法连接到知识库: {kb_id}")
                return []

            # 1. 向量检索器 (大幅提高Top-K以增加候选集)
            retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

            # 定义单个查询的异步检索函数（向量检索）
            async def retrieve_vector(query: str) -> list[Document]:
                try:
                    return await retriever.ainvoke(query)
                except Exception as e:
                    logger.error(f"向量检索出错: {e}")
                    return []

            # 定义关键词检索函数（基于Milvus scalar filtering）
            async def retrieve_keyword(query: str) -> list[Document]:
                """
                使用 text like '%keyword%' 进行模糊匹配
                注意：Milvus 的 like 性能较差，仅适合短关键词或作为辅助召回
                """
                # 简单的关键词提取策略：如果查询较短，直接用；否则按空格切分取最长的词
                keywords = []
                if len(query) < 10:
                    keywords.append(query)
                else:
                    # 简单分词，取长度大于2的词
                    words = [w for w in query.strip().split() if len(w) >= 2]
                    keywords.extend(words)
                if not keywords:
                    return []

                docs = []
                try:
                    # 获取集合对象
                    # langchain_milvus.Milvus 实例通常有 col 或 collection 属性
                    col = getattr(vector_store, "col", getattr(vector_store, "collection", None))
                    if not col:
                        logger.error("无法获取 Milvus 集合对象")
                        return []

                    # 检查 fileName 和 chunkIndex 字段是否存在，防止旧版本 collection 报错
                    has_filename = False
                    has_chunkindex = False
                    try:
                        if hasattr(col, 'schema') and hasattr(col.schema, 'fields'):
                            fields = col.schema.fields
                            has_filename = any(field.name == 'fileName' for field in fields)
                            has_chunkindex = any(field.name == 'chunkIndex' for field in fields)
                    except Exception as e:
                        logger.warning(f"检查 schema 失败: {e}")

                    # 准备输出字段
                    query_output_fields = ["text", "pk", "documentId"]
                    if has_chunkindex:
                        query_output_fields.append("chunkIndex")
                    if has_filename:
                        logger.info(f"包含文件名检索")
                        query_output_fields.append("fileName")

                    for kw in keywords:
                        # 转义特殊字符
                        safe_kw = kw.replace("'", "\\'").replace('"', '\\"')

                        # 构造表达式: text 字段包含关键词 或 fileName 包含关键词
                        if has_filename:
                            expr = f'text like "%{safe_kw}%" or fileName like "%{safe_kw}%"'
                        else:
                            expr = f'text like "%{safe_kw}%"'

                        # 使用 run_in_executor 避免阻塞

                        res = await vector_store.aclient.query(
                            collection_name=vector_store.collection_name,
                            filter=expr,
                            output_fields=query_output_fields,  # 确保获取必要字段
                            limit=5  # 限制关键词召回数量，避免过多
                        )

                        # 转换为 Document 对象
                        for item in res:
                            content = item.pop('text', '')
                            # 移除 milvus 内部字段
                            if 'pk' in item:
                                pk = item['pk']
                            else:
                                pk = str(item.get('id', ''))

                            docs.append(Document(
                                page_content=content,
                                metadata={**item, 'pk': pk, 'retrieval_source': 'keyword'}
                            ))

                except Exception as e:
                    # 关键词检索失败不影响主流程
                    logger.warning(f"关键词检索失败: {e}")

                return docs

            # 并行执行所有检索任务
            tasks = []
            for query in query_list:
                tasks.append(retrieve_vector(query))
                # 对每个查询也尝试关键词检索
                tasks.append(retrieve_keyword(query))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 统计不同来源的文档数量
            count_vector = 0
            count_keyword = 0

            vector_docs = []
            keyword_docs = []

            # 分类收集
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"检索任务失败: {result}")
                    continue
                for doc in result:
                    source = doc.metadata.get('retrieval_source', 'vector')
                    if source == 'keyword':
                        keyword_docs.append(doc)
                    else:
                        vector_docs.append(doc)

            # 优先处理向量检索结果
            for doc in vector_docs:
                pk = doc.metadata.get('pk') or doc.metadata.get('id')
                if pk:
                    pk = str(pk)
                    if pk not in doc_set:
                        doc_set.add(pk)
                        all_docs.append(doc)
                        count_vector += 1

            # 再处理关键词检索结果（补充）
            for doc in keyword_docs:
                pk = doc.metadata.get('pk') or doc.metadata.get('id')
                if pk:
                    pk = str(pk)
                    if pk not in doc_set:
                        doc_set.add(pk)
                        all_docs.append(doc)
                        count_keyword += 1

            logger.info(f"最终去重后共 {len(all_docs)} 个文档 (向量召回: {count_vector}, 关键词召回: {count_keyword})")

        finally:
            # 连接由 MilvusClientManager 管理，不需要在此关闭
            pass

        return all_docs

    async def parallel_grade_documents(
            self,
            documents: list[Document],
            query: str,
            model_info: dict = None,
            grade_top_n: int = 5,
            grade_score_threshold: float = 0.3
    ) -> list[Document]:
        """
        使用Rerank模型评估文档相关性（替代LLM评分）

        Args:
            documents: 文档列表
            query: 原始查询
            model_info: 模型配置信息（保留参数以兼容旧接口，但不使用）
            grade_top_n: 返回前N个最相关的文档，默认5个
            grade_score_threshold: 相关性分数阈值（斩杀线），默认0.3，低于此分数的文档将被过滤

        Returns:
            按相关性分数排序的文档列表
        """
        if not documents:
            return []

        try:
            # 提取文档内容，并添加文件名作为前缀以辅助Rerank模型判断上下文
            doc_contents = [
                f"{doc.page_content} [来源：{doc.metadata.get('fileName')}] "
                for doc in documents
            ]

            # 使用rerank进行文档排序，应用斩杀线
            rerank_result = await rerank(
                query=query,
                documents=doc_contents,
                grade_top_n=min(grade_top_n, len(documents)),
                return_documents=False,  # 不需要返回文档内容，节省带宽
                grade_score_threshold=grade_score_threshold  # 应用斩杀线
            )

            # 根据rerank结果重新排序文档
            ranked_docs = []
            for item in rerank_result.get("output", {}).get("results", []):
                original_idx = item['index']
                relevance_score = item['relevance_score']

                # 获取原始文档并添加rerank分数到metadata
                doc = documents[original_idx]
                doc.metadata['rerank_score'] = relevance_score
                ranked_docs.append(doc)

            logger.info(f"Rerank评分完成，返回 {len(ranked_docs)} 个相关文档（斩杀线: {grade_score_threshold}）")
            return ranked_docs

        except Exception as e:
            logger.error(f"Rerank评分失败: {e}，回退到返回所有文档")
            # 出错时返回原始文档列表（兜底策略）
            return documents[:grade_top_n] if len(documents) > grade_top_n else documents

    async def stream_rag_response_with_process(
            self,
            question: str,
            history: list,
            model_info: dict,
            kb_id: Optional[int] = None,
            user_id: Optional[int] = None,

            system_prompt: Optional[str] = None,
            options: dict = None,
            retrieve_k: int = 15,

            grade_top_n: int = 50,
            grade_score_threshold: float = 0.3,

            context_top_n: int = 10,
    ) -> AsyncGenerator[dict, None]:
        """
        带过程信息的流式RAG响应生成器

        Args:
            question: 用户问题
            history: 对话历史（LangChain消息格式）
            model_info: 模型配置信息
            kb_id: 知识库ID（可选）
            user_id: 用户ID（可选）
            system_prompt: 自定义系统提示词（可选）
            options: 其他选项（如启用Web搜索等）
            retrieve_k: 每个查询检索的文档数量
            context_top_n: 最终参数上下文构建的文档数量
            grade_top_n: Rerank评分时考虑的文档数量
            grade_score_threshold: Rerank相关性分数阈值

        Yields:
            包含type和payload的字典，type可以是"process"或"content"
        """
        context = "无相关文档"

        # 如果有知识库，执行RAG流程
        if kb_id and user_id:
            try:
                # 1. 生成多角度查询
                yield {
                    "type": "process",
                    "payload": {
                        "step": "query_generation",
                        "title": "生成多角度查询",
                        "description": "正在分析问题并生成多个检索角度...",
                        "status": "running"
                    }
                }

                logger.info("开始生成多角度查询...")
                query_list, grade_query = await self.generate_multi_queries(question, history, model_info)

                yield {
                    "type": "process",
                    "payload": {
                        "step": "query_generation",
                        "title": "生成多角度查询",
                        "description": f"已生成 {len(query_list)} 个检索查询",
                        "status": "completed",
                        "content": "\n".join(
                            [f"- {q}" for q in query_list] + [f"\n评分查询: {grade_query}"])
                    }
                }

                # 2. 并行检索
                yield {
                    "type": "process",
                    "payload": {
                        "step": "retrieval",
                        "title": "检索知识库",
                        "description": "正在从知识库中并行检索相关文档...",
                        "status": "running"
                    }
                }

                logger.info("开始并行检索...")
                all_docs = await self.parallel_retrieve(query_list, user_id, kb_id, top_k=retrieve_k)

                yield {
                    "type": "process",
                    "payload": {
                        "step": "retrieval",
                        "title": "检索知识库",
                        "description": f"检索到 {len(all_docs)} 个候选文档",
                        "status": "completed"
                    }
                }

                # 3. Rerank文档重排序
                graded_docs = []
                if all_docs:
                    yield {
                        "type": "process",
                        "payload": {
                            "step": "rerank",
                            "title": "评估文档相关性",
                            "description": "正在使用Rerank模型评分并排序文档...",
                            "status": "running"
                        }
                    }

                    logger.info("开始Rerank文档重排序...")
                    # 这里的 grade_top_n 传入较大值 (如 50)，以便获取更多候选文档用于后续拼接，
                    graded_docs = await self.parallel_grade_documents(
                        all_docs,
                        grade_query,
                        grade_top_n=grade_top_n,
                        grade_score_threshold=grade_score_threshold
                    )

                yield {
                    "type": "process",
                    "payload": {
                        "step": "rerank",
                        "title": "评估文档相关性",
                        "description": f"简单相关性：{grade_score_threshold:.2f}， 筛选出 {len(graded_docs)} 个相关文档",
                        "status": "completed"
                    }
                }

                # 4. 构建上下文
                consecutive_docs = 0
                merged_docs = []
                if graded_docs:
                    filter_result = filter_grade_threshold(graded_docs)
                    threshold = filter_result['threshold']
                    high_ratio = filter_result['high_ratio']
                    graded_docs = filter_result['documents']
                    kmeans_centers = filter_result.get('kmeans_centers', [])
                    min_high_score = filter_result.get('min_high_score', None)
                    max_low_score = filter_result.get('max_low_score', None)
                    yield {
                        "type": "process",
                        "payload": {
                            "step": "filter",
                            "title": "动态过滤文档",
                            "description": f"动态相关性：{threshold:.2f}，高分占比：{high_ratio * 100:.2f}%，筛选得到 {len(graded_docs)} 个文档。" +
                                           (
                                               f"高分区域边界：{min_high_score:.2f}，低分区域边界：{max_low_score:.2f}。"
                                               if min_high_score is not None and max_low_score is not None else ""
                                           ) +
                                           (
                                               f" 中心：[{', '.join([f'{c:.2f}' for c in kmeans_centers])}]。"
                                               if kmeans_centers else ""
                                           ),
                            "status": "completed"
                        }
                    }
                    # 合并同一文档的连续切片
                    merged_docs = merge_consecutive_chunks(graded_docs, True)
                    consecutive_docs = len(merged_docs)

                    # 最终应用 top_n 限制
                    merged_docs = merged_docs[:context_top_n]

                    context = "\n\n".join([
                        f"[文档{i + 1}]: {doc.page_content} (来源: {doc.metadata.get('fileName')})"
                        for i, doc in enumerate(merged_docs)
                    ])
                logger.info(
                    f"构建上下文完成，合并后共有 {consecutive_docs} 个文档，选择前 {len(merged_docs)} 个用于回答。")

                display_docs = get_display_docs(merged_docs)
                yield {
                    "type": "process",
                    "payload": {
                        "step": "reformat",
                        "title": "构建上下文",
                        "description": f"连续文档合并，得到 {consecutive_docs} 份文档， 选择前 {len(merged_docs)} 份用于上下文。",
                        "status": "completed",
                        "content": "\n\n---\n\n".join(
                            [f"## 部分检索到的信息如下（仅展示前 {len(display_docs)} 份）"] + [
                                f"```document\n[文档{i + 1}] [来源: {doc.metadata.get('fileName')}] [相关性：{doc.metadata.get('rerank_score', 0):.3f}]: {doc.page_content}\n```"
                                for i, doc in enumerate(display_docs)
                            ]
                        ) if display_docs else "无相关文档"
                    }
                }


            except Exception as e:
                logger.error(f"RAG流程出错: {e}")
                yield {
                    "type": "process",
                    "payload": {
                        "step": "error",
                        "title": "检索失败",
                        "description": f"检索过程出错: {str(e)}",
                        "status": "error"
                    }
                }
                context = f"检索过程出错: {str(e)}"

        # 5. 生成答案
        yield {
            "type": "process",
            "payload": {
                "step": "generation",
                "title": "生成答案",
                "description": "正在基于检索结果生成回答...",
                "status": "running"
            }
        }

        # 构建系统提示词
        if system_prompt:
            logger.info("使用自定义提示词")
            final_system_prompt = f"""{system_prompt}

参考文档：
{context}"""
        else:
            logger.info("使用系统内置提示词")
            final_system_prompt = f"""你是一个专业的AI助手。基于提供的文档和对话历史回答用户问题。

要求：
1. 文档中的信息仅供参考
2. 如果文档不足以完整回答，结合对话历史进行推理或明确说明
3. 文档中的信息为切片信息，可能语义并不连贯或存在错误，你需要抽取或推理相关信息

参考文档：
{context}"""

        # 发送系统提示词用于token统计
        yield {
            "type": "system_prompt",
            "payload": final_system_prompt
        }

        # 构建对话消息
        conversation = [{"role": "system", "content": final_system_prompt}]
        for msg in history:
            if isinstance(msg, HumanMessage):
                conversation.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                conversation.append({"role": "assistant", "content": msg.content})
        # 添加当前问题
        conversation.append({"role": "user", "content": question})

        llm = get_official_llm(
            model_info,
            enable_web_search=options.get('webSearch', False),
            enable_thinking=options.get('thinking', False)
        )
        # 使用异步流式生成
        async for item in unified_llm_stream(llm, conversation):
            yield item


# 全局RAG服务实例
rag_service = RAGService()
