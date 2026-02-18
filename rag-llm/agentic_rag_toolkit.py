"""
Agentic RAG 工具集
包含6个原子化检索工具，提供完全透明可控的检索能力
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional, Literal

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from aiohttp_utils import rerank
from utils import filter_grade_threshold

logger = logging.getLogger(__name__)


# ============= 决策模型 =============
class RetrievalDecision(BaseModel):
    """检索决策结构化输出"""
    action: Literal["continue", "stop"] = Field(description="是否继续检索: continue=继续, stop=停止")
    reason: str = Field(description="本次决策的原因说明")

    # 如果action=continue, 以下字段必填
    tool: Optional[Literal[
        "search_by_grep",  # 1. 关键词检索（grep），支持全库/单文件/多文件
        "search_by_document_and_chunk_range",  # 2. 按文档ID获取连续chunk范围
        "search_by_filename_and_chunk_range",  # 3. 按文件名获取连续chunk范围
        "search_by_multi_queries_in_database",  # 4. 全库语义检索(多query+rerank)
        "list_filename_by_like"  # 5. 根据模式匹配列出文件
    ]] = Field(default=None, description="下一步要使用的检索工具")

    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="工具参数,根据tool不同而不同"
    )

    missing_info: List[str] = Field(
        default_factory=list,
        description="当前缺失的关键信息"
    )


# ============= 检索工具集 (6个工具) =============
class RetrievalToolkit:
    """原子化检索工具集 - 完全透明可控"""

    def __init__(self, vector_store, retriever):
        self.vector_store = vector_store
        self.retriever = retriever

    @staticmethod
    def _escape(s: str) -> str:
        """转义特殊字符"""
        return s.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")

    async def _milvus_filter(
            self,
            filter_expr: str,
            offset: int = 0,
            limit: int = 20,
            output_fields: Optional[List[str]] = None
    ) -> List[Document]:
        """底层Milvus查询封装"""
        if output_fields is None:
            output_fields = ["text", "pk", "documentId", "chunkIndex", "fileName", "maxChunkIndex"]
        try:
            rows = await self.vector_store.aclient.query(
                collection_name=self.vector_store.collection_name,
                filter=filter_expr,
                output_fields=output_fields,
                offset=offset,
                limit=limit,
            )

            docs = []
            for r in rows:
                docs.append(Document(
                    page_content=r.get("text", ""),
                    metadata={k: v for k, v in r.items() if k != "text"}
                ))

            return docs

        except Exception as e:
            logger.error(f"❌ Milvus查询失败: {e}")
            return []

    async def _vector_search(
            self,
            query: str,
            top_k: int = 10,
    ) -> List[Document]:
        """底层向量检索封装"""
        try:
            # 向量检索
            search_kwargs = {"k": top_k}
            docs = await self.retriever.ainvoke(query, search_kwargs=search_kwargs)
            # 尝试将query切分为多个keywords进行过滤（如果query中包含空格）
            keywords = query.split(" ")
            if len(keywords) > 1:
                expr_filter = " OR ".join([f'text like "%{self._escape(kw)}%"' for kw in keywords])
                docs.extend(await self._milvus_filter(filter_expr=expr_filter, limit=top_k))
            return docs

        except Exception as e:
            logger.error(f"❌ 向量检索失败: {e}")
            return []

    # ============= 工具1: 关键词检索（grep风格）=============
    async def search_by_grep(
            self,
            keywords: List[str],
            match_type: Literal["AND", "OR"] = "OR",
            top_k: int = 5,
            file_names: Optional[List[str]] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """
        工具1: 关键词检索（grep风格），支持全库检索或指定文件范围

        Args:
            keywords: 关键词列表
            match_type: 匹配模式 "AND" 或 "OR"
            top_k: 返回条数限制
            file_names: 可选文件名列表，为空则全库检索，1个为单文件检索，多个为多文件检索
            **kwargs: 额外参数（容错，忽略大模型可能传递的其他参数）

        Returns:
            {"results": List[Document], "total_hits": int}
        """
        if kwargs:
            logger.debug(f"⚠️ 工具1收到额外参数（已忽略）: {kwargs}")

        scope = "全库" if not file_names else f"{len(file_names)}个文件"
        logger.info(
            f"🔍 [1.grep检索] keywords={keywords}, type={match_type}, top_k={top_k}, scope={scope}, files={file_names}")

        keyword_conditions = [f'text like "%{self._escape(kw)}%"' for kw in keywords]
        keyword_expr = f" {match_type} ".join(keyword_conditions)

        if file_names:
            file_conditions = [f'fileName == "{self._escape(fn)}"' for fn in file_names]
            file_expr = " OR ".join(file_conditions)
            filter_expr = f'({file_expr}) and ({keyword_expr})'
        else:
            filter_expr = keyword_expr

        docs = await self._milvus_filter(
            filter_expr=filter_expr,
            limit=top_k
        )

        logger.info(f"✅ grep检索结果: {len(docs)}条")

        return {
            "results": docs,
            "total_hits": len(docs)
        }

    # ============= 工具2: 按文档ID获取连续chunk范围 =============
    async def search_by_document_and_chunk_range(
            self,
            document_id: int,
            start_chunk_index: int,
            end_chunk_index: int,
            **kwargs
    ) -> Dict[str, Any]:
        """
        工具2: 按文档ID获取连续chunk范围

        Args:
            document_id: 文档ID（必须是整数）
            start_chunk_index: 起始chunk索引（包含）
            end_chunk_index: 结束chunk索引（包含）
            **kwargs: 额外参数（容错，忽略大模型可能传递的其他参数）

        Returns:
            {"results": List[Document], "total_hits": int}
        """
        if kwargs:
            logger.debug(f"⚠️ 工具2收到额外参数（已忽略）: {kwargs}")

        logger.info(f"🔍 [2.chunk范围] docId={document_id}, range=[{start_chunk_index}, {end_chunk_index}]")

        filter_expr = f'documentId == {document_id} and chunkIndex >= {start_chunk_index} and chunkIndex <= {end_chunk_index}'

        limit = end_chunk_index - start_chunk_index + 1

        docs = await self._milvus_filter(
            filter_expr=filter_expr,
            limit=limit
        )

        docs.sort(key=lambda d: d.metadata.get("chunkIndex", 0))

        logger.info(f"✅ chunk范围检索结果: {len(docs)}条")

        return {
            "results": docs,
            "total_hits": len(docs)
        }

    # ============= 工具3: 按文件名获取连续chunk范围 =============
    async def search_by_filename_and_chunk_range(
            self,
            file_name: str,
            start_chunk_index: int,
            end_chunk_index: int,
            **kwargs
    ) -> Dict[str, Any]:
        """
        工具3: 按文件名获取连续chunk范围

        Args:
            file_name: 文件名（精确匹配）
            start_chunk_index: 起始chunk索引（包含）
            end_chunk_index: 结束chunk索引（包含）
            **kwargs: 额外参数（容错，忽略大模型可能传递的其他参数）

        Returns:
            {"results": List[Document], "total_hits": int}
        """
        if kwargs:
            logger.debug(f"⚠️ 工具3收到额外参数（已忽略）: {kwargs}")

        logger.info(f"🔍 [3.文件chunk范围] file='{file_name}', range=[{start_chunk_index}, {end_chunk_index}]")

        filter_expr = f'fileName == "{self._escape(file_name)}" and chunkIndex >= {start_chunk_index} and chunkIndex <= {end_chunk_index}'

        limit = end_chunk_index - start_chunk_index + 1

        docs = await self._milvus_filter(
            filter_expr=filter_expr,
            limit=limit
        )

        docs.sort(key=lambda d: d.metadata.get("chunkIndex", 0))

        logger.info(f"✅ 文件chunk范围检索结果: {len(docs)}条")

        return {
            "results": docs,
            "total_hits": len(docs)
        }

    # ============= 工具4: 全库语义检索(多query+rerank) =============
    async def search_by_multi_queries_in_database(
            self,
            queries: List[str],
            grade_query: str,
            top_k: int = 10,
            grade_score_threshold: float = 0.4,
            **kwargs
    ) -> Dict[str, Any]:
        """
        工具4: 全库语义检索(多query+rerank)

        Args:
            queries: 多语义查询列表（用于向量检索召回）
            grade_query: 专门用于Rerank评分的查询（通常是解除歧义后的用户原始问题）
            top_k: 最终返回条数（在rerank和动态过滤后）
            grade_score_threshold: Rerank分数阈值（默认0.4），低于此分数的文档将被过滤
                                   0.3=弱相关，0.5=一般相关，0.6=强相关，由大模型决定
            **kwargs: 额外参数（容错，忽略大模型可能传递的其他参数）

        Returns:
            {"results": List[Document], "total_hits": int}
            
        流程:
            1. 并行向量检索所有queries（召回阶段）
            2. 合并去重
            3. 使用grade_query进行Rerank重排序评分（精排阶段）
            4. K-Means动态阈值过滤
            5. 按rerank_score排序并返回top_k
        """
        if kwargs:
            logger.debug(f"⚠️ 工具4收到额外参数（已忽略）: {kwargs}")

        logger.info(
            f"🔍 [4.全库语义] queries={queries}, grade_query={grade_query}, top_k={top_k}, threshold={grade_score_threshold}")

        all_docs = []
        seen_pks = set()

        retrieval_top_k = max(top_k * 3, 15)
        tasks = [
            asyncio.create_task(self._vector_search(query=query, top_k=retrieval_top_k))
            for query in queries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for docs in results:
            if isinstance(docs, Exception):
                logger.warning(f"⚠️ 某个query检索失败: {docs}")
                continue
            for doc in docs:
                pk = doc.metadata.get("pk")
                if pk and pk not in seen_pks:
                    seen_pks.add(pk)
                    all_docs.append(doc)

        logger.info(f"📊 并行检索完成: 总计{len(all_docs)}个独立文档")
        if not all_docs:
            logger.warning("⚠️ 并行检索未找到任何文档")
            return {
                "results": [],
                "total_hits": 0
            }

        # Rerank
        try:
            doc_contents = [doc.page_content for doc in all_docs]
            rerank_result = await rerank(
                query=grade_query,
                documents=doc_contents,
                grade_score_threshold=grade_score_threshold
            )

            # 根据 rerank 结果重新排序文档，并添加 rerank_score
            ranked_docs = []
            items = rerank_result.get("output", {}).get("results", [])
            for item in items:
                original_idx = item['index']
                relevance_score = item['relevance_score']

                # 获取原始文档并添加 rerank 分数到 metadata
                doc = all_docs[original_idx]
                doc.metadata['rerank_score'] = relevance_score
                ranked_docs.append(doc)

            all_docs = ranked_docs
            logger.info(f"✅ Rerank完成，返回 {len(all_docs)} 个文档")
        except Exception as e:
            logger.warning(f"⚠️ Rerank失败: {e}")
            raise e

        # 动态阈值过滤
        try:
            filter_result = filter_grade_threshold(all_docs)
            all_docs = filter_result['documents']
            threshold = filter_result.get('threshold', 0.0)
            logger.info(f"✅ 动态过滤后剩余: {len(all_docs)}个文档，阈值: {threshold:.2f}")
        except Exception as e:
            logger.warning(f"⚠️ 动态过滤失败: {e}")
            raise e

        all_docs.sort(key=lambda d: d.metadata.get("rerank_score", 0.0), reverse=True)
        all_docs = all_docs[:top_k]

        return {
            "results": all_docs,
            "total_hits": len(all_docs)
        }

    # ============= 工具5: 根据前缀列出文件 =============
    async def list_filename_by_like(
            self,
            pattern: str,
            offset: int = 0,
            limit: int = 30,
            **kwargs
    ) -> Dict[str, Any]:
        """
        工具5: 根据文件名模式匹配列出文件信息（仅返回元信息，不包含文档内容）
        
        Args:
            pattern: 文件名匹配模式（支持SQL LIKE语法，如"doc%"表示前缀，"%report%"表示包含）
            offset: 偏移量（用于分页）
            limit: 返回数量限制
            **kwargs: 额外参数（容错，忽略大模型可能传递的其他参数）
        
        Returns:
            {"results": List[Document], "total_hits": int}
        
        注意:
            - 使用 chunkIndex == 0 来获取每个文件的首个chunk
            - 避免重复获取同一文件的多个chunk
            - 首个chunk包含完整的文件元信息(fileName, documentId, maxChunkIndex等)
            - page_content为空字符串，因为此工具仅用于列出文件，不获取内容
            - 返回的Document不会累积到all_docs中
        """
        if kwargs:
            logger.debug(f"⚠️ 工具5收到额外参数（已忽略）: {kwargs}")

        logger.info(f"🔍 [5.文件名匹配列表] pattern='{pattern}', offset={offset}, limit={limit}")

        filter_expr = f'fileName like "{self._escape(pattern)}" and chunkIndex == 0'

        output_fields = ["pk", "documentId", "chunkIndex", "fileName", "maxChunkIndex"]

        try:
            rows = await self.vector_store.aclient.query(
                collection_name=self.vector_store.collection_name,
                filter=filter_expr,
                output_fields=output_fields,
                limit=limit,
                offset=offset,
                order_by="fileName"
            )

            all_docs = []
            for r in rows:
                all_docs.append(Document(
                    page_content="",
                    metadata={
                        **r
                    }
                ))

            logger.info(f"✅ 文件列表查询成功，结果数={len(all_docs)}")

        except Exception as e:
            logger.error(f"❌ Milvus查询失败: {e}")
            all_docs = []

        return {
            "results": all_docs,
            "total_hits": len(all_docs)
        }

    # ============= 工具执行统一入口 =============
    async def execute_tool(
            self,
            tool: str,
            params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        执行指定的检索工具（统一入口）
        
        Args:
            tool: 工具名称
            params: 工具参数
            
        Returns:
            {"results": List[Document], "total_hits": int}
        """
        logger.info(f"🔧 执行工具: {tool}, 参数: {params}")

        try:
            if tool == "search_by_grep":
                return await self.search_by_grep(**params)

            elif tool == "search_by_document_and_chunk_range":
                return await self.search_by_document_and_chunk_range(**params)

            elif tool == "search_by_filename_and_chunk_range":
                return await self.search_by_filename_and_chunk_range(**params)

            elif tool == "search_by_multi_queries_in_database":
                return await self.search_by_multi_queries_in_database(**params)

            elif tool == "list_filename_by_like":
                return await self.list_filename_by_like(**params)

            else:
                logger.warning(f"⚠️ 未知工具: {tool}")
                return {"results": [], "total_hits": 0}

        except Exception as e:
            logger.error(f"❌ 工具执行失败: {tool}, 错误: {e}")
            return {"results": [], "total_hits": 0}


# ============= 检索工具对应的prompt =============
TOOL_DEFINE_PROMPT = """## 可用工具 

(共5个可调用tool)
### 1. search_by_grep
**功能**: 关键词精确匹配检索（grep风格），支持全库检索或限定文件范围，适用于代码级精确搜索
**参数**: 
  - keywords: List[str], 必填，关键词列表（如["API", "token"]）
  - match_type: str, 可选，"AND"或"OR"，默认"OR"
    * "AND": 返回包含所有关键词的chunk
    * "OR": 返回包含任一关键词的chunk
  - top_k: int, 可选，默认5，返回结果数量
  - file_names: List[str], 可选，默认None（空表示全库检索）
    * **为空（None或[]）**: 全库检索，搜索整个知识库
    * **单个文件（["config.py"]）**: 单文件检索，仅在指定文件内查找
    * **多个文件（["app.py", "utils.py", "config.py"]）**: 多文件检索，在指定多个文件中查找
**核心优势**: 精确匹配，无歧义，适合查找：
  - **方法/函数名**: def calculate_price, function handleSubmit, async getUserData
  - **类名**: class UserService, interface IDatabase, struct Config
  - **关键变量**: API_KEY, MAX_RETRIES, connectionString, userId
  - **关键描述/注释**: "配置文件路径", "初始化数据库连接", "TODO: 优化性能"
  - **错误代码**: ERROR_404, TIMEOUT_EXCEPTION, StatusCode.BadRequest
  - **配置项**: timeout: 30, max_connections: 100, "port": 8080
  - **跨文件的方法调用链**: login() → authenticate() → verifyToken()
  - **全局类/接口**: 所有实现IRepository的类，所有继承BaseService的类
  - **API端点**: "/api/users", "POST /login", "GET /data"
**适用场景**: 
  - 查找特定方法/函数实现（如"search_by_grep"）
  - 查找类定义和变量声明（如"class AgenticRAG"）
  - 查找带特定注释的代码段（如"核心算法"）
  - 定位配置参数和常量（如"DATABASE_URL"）
  - 跨文件查找专有名词、API名称（如"OpenAI API"）
  - 查找错误代码、异常类型（如"TimeoutException"）
  - 定位特定术语、技术栈（如"Redis", "PostgreSQL"）
**典型用法**: 
  - 问题："工具类在哪个文件？" → keywords=["class", "ToolKit"], file_names=None（全库）
  - 问题："config.py文件端口是多少？" → keywords=["port"], file_names=["config.py"]（单文件）
  - 问题："数据库连接函数在app.py或utils.py？" → keywords=["def", "connect", "database"], file_names=["app.py", "utils.py"]（多文件）
  - 问题："哪些文件使用了Redis？" → keywords=["Redis"], file_names=None（全库）
  - 问题："API端点定义在controller文件中" → keywords=["@app.route", "POST"], file_names=["user_controller.py", "order_controller.py"]（多文件）

### 2. search_by_document_and_chunk_range
**功能**: 按文档ID获取连续chunk范围
**参数**: 
  - document_id: int, 必填，文档ID（必须是整数）
  - start_chunk_index: int, 必填，起始chunk索引（包含）
  - end_chunk_index: int, 必填，结束chunk索引（包含）
**注意**: 
  - 可结合已检索文档的maxChunkIndex字段验证范围有效性
  - 索引从0开始，如maxChunkIndex=29表示chunk范围是0-29
**适用场景**: 
  - 补全不连续chunk（如已有1,5,9，获取2-8补齐上下文）
  - 扩展上下文窗口（如已有chunk 5，获取3-7扩展上下文）
  - 获取完整段落、代码块等需要连续文本的场景

### 3. search_by_filename_and_chunk_range
**功能**: 按文件名获取连续chunk范围
**参数**: 
  - file_name: str, 必填，精确文件名（如"rag-system/README.md"）
  - start_chunk_index: int, 必填，起始chunk索引（包含）
  - end_chunk_index: int, 必填，结束chunk索引（包含）
**注意**: 
  - 可结合已检索文档的maxChunkIndex字段验证范围有效性
  - 索引从0开始，如maxChunkIndex=29表示chunk范围是0-29
**适用场景**: 
  - 已知文件名和chunk范围，直接获取指定部分
  - 补全不连续chunk
  - 扩展上下文窗口
  - 获取完整段落、代码块等需要连续文本的场景

### 4. search_by_multi_queries_in_database
**功能**: 全库多角度语义检索，使用Rerank评分和动态过滤返回高质量结果
**参数**: 
  - queries: List[str], 必填，多个查询语句（用于召回阶段，可以是改写、扩展的查询）
  - grade_query: str, 必填，用于Rerank评分的查询（可使用改写后的原始问题）
  - grade_score_threshold: float, 可选，默认0.4，Rerank分数阈值（斩杀线）
    * 0.3: 弱相关（宽松模式，召回更多文档）
    * 0.4: 有些相关（探索模式，用于前期探索）
    * 0.5: 一般相关（平衡模式）
    * 0.6: 强相关（严格模式，只保留高度相关文档）
    * 低于此分数的文档将被过滤，由你根据问题复杂度和召回情况灵活决定
  - top_k: int, 可选，默认10，最终返回的文档数量
**执行流程**: 
  1. 并行向量检索所有queries（每个query召回top_k*3个候选）
  2. 合并去重
  3. 使用grade_query调用Rerank API进行精排序
  4. 应用grade_score_threshold过滤低相关文档（斩杀线）
  5. K-Means动态阈值过滤低相关文档
  6. 按rerank_score排序返回top_k个
**适用场景**: 
  - 首轮检索（推荐作为首轮工具）
  - 多角度语义探索
  - 需要高质量排序结果的场景
**阈值选择建议**:
  - 简单问题（如"什么是XXX"）→ 0.3~0.4（宽松）
  - 中等问题（如"如何实现XXX功能"）→ 0.5（默认）
  - 复杂问题（如"XXX与YYY的区别和联系"）→ 0.6（严格）

### 5. list_filename_by_like
**功能**: 根据文件名模式匹配列出文件，按文件名排序（仅返回元信息，不包含文档内容）
**参数**: 
  - pattern: str, 必填，文件名匹配模式（支持SQL LIKE语法）
    * "doc%": 前缀匹配（以"doc"开头的文件）
    * "%report%": 包含匹配（包含"report"的文件）
    * "dir/.../%": 目录匹配（某目录下的文件）
    * "%%": 表示匹配所有文件
  - offset: int, 可选，默认0，分页偏移量
  - limit: int, 可选，默认30，每页返回数量
**说明**: 
  - 自动使用chunkIndex==0过滤，每个文件只返回第一个chunk（包含完整元信息）
  - 结果按fileName字母序排序
  - ⚠️ 注意：此工具只返回文件元信息（fileName, documentId, maxChunkIndex）
  - 如果limit和返回数量一致，表示可能还有更多文件，需同步增加offset和增加limit进行快速分页查询
**返回格式**:
  - {{"type": "file_list", "total_files": x, "files": [{{"fileName": "...", "documentId": x, "maxChunkIndex": x}}]}}
**适用场景**: 
  - 浏览/探索文件列表（了解知识库有哪些文件）
  - 不确定具体文件名时查找相关文件
  - 按目录结构或名称模式查找文档
  - 获取documentId和maxChunkIndex后，可使用search_by_document_and_chunk_range或search_by_filename_and_chunk_range获取实际内容"""

TOOL_SELECT_PROMPT = """## 工具选择策略

### 首轮检索（根据问题类型选择）

**1. 概念性/描述性问题（没有明确关键词）**:
- 使用 **search_by_multi_queries_in_database** 进行多角度语义检索，获得Rerank高质量结果
- 适合："什么是XXX"、"如何理解XXX"、"XXX的原理"等问题

**2. 结构化内容查找（有明确关键词）**:
- 使用 **search_by_grep** 进行代码级精确搜索，快速定位方法、类、变量、配置项等
- 适合："XXX方法在哪"、"XXX类的定义"、"XXX配置项"等问题

**3. 文件探索（不确定文件名）**:
- 使用 **list_filename_by_like** 列出文件列表，然后用chunk范围工具获取内容
- 适合："有哪些XXX文件"、"查看所有配置文件"等问题

### 后续轮次策略（根据检索情况选择）

#### 1. 代码级精确搜索场景
**工具**: `search_by_grep`

**grep核心优势**: 
- 精确匹配，无歧义
- 适合结构化内容（代码、配置、参数）
- 速度快，无向量化延迟

**使用时机**（强烈推荐grep的场景）:
1. **查找方法/函数定义**:
   - "calculate_price方法在哪？" → keywords=["def calculate_price"], file_names=None（全库）
   - "getUserData在user_service.py实现？" → keywords=["async getUserData", "function"], file_names=["user_service.py"]（单文件）
   
2. **查找类/接口定义**:
   - "UserService类在哪？" → keywords=["class UserService"], file_names=None
   - "所有继承BaseModel的类" → keywords=["BaseModel", "class"], file_names=None（全库）
   
3. **查找关键变量/常量**:
   - "API_KEY在config.py配置？" → keywords=["API_KEY"], file_names=["config.py"]
   - "数据库连接字符串" → keywords=["DATABASE_URL", "connection_string"], file_names=None
   
4. **查找带特定注释/描述的代码**:
   - "TODO项在哪里？" → keywords=["TODO:", "fixme"], file_names=None
   - "标注为核心算法的部分" → keywords=["核心算法", "core algorithm"], file_names=None
   
5. **查找错误码/异常类型**:
   - "TimeoutException在哪处理？" → keywords=["TimeoutException", "catch"], file_names=None
   - "ERROR_404定义在哪？" → keywords=["ERROR_404", "="], file_names=None
   
6. **查找配置项/环境变量**:
   - "端口配置在哪？" → keywords=["port", "8080"], file_names=None
   - "Redis配置在settings文件" → keywords=["redis", "host"], file_names=["settings.py", "config.yaml"]

7. **查找API端点/路由**:
   - "登录接口在controller文件？" → keywords=["/login", "POST"], file_names=["user_controller.py", "auth_controller.py"]
   - "所有GET接口" → keywords=["@app.get", "GET"], file_names=None（全库）

**使用指南**:
- **file_names为空（None）**: 全库检索，适合不确定位置或需要跨文件查找
- **file_names单个（["file.py"]）**: 单文件检索，已知文件时使用（更快更准）
- **file_names多个（["a.py", "b.py"]）**: 多文件检索，限定在几个相关文件中查找
- **match_type="AND"**: 精确查找，要求包含所有关键词（如["def", "calculate_price"]表示必须同时包含这两个词）
- **match_type="OR"**: 广泛探索，包含任一关键词即可（如["try", "catch", "except"]表示包含其中任意一个即可）
- **关键词建议**: 可以是单个完整短语（如["def calculate_price"]）或拆分多个词（如["def", "calculate_price"]），根据实际需要选择
- **组合策略**: 先grep定位文件/chunk，再用chunk_range获取完整上下文

**典型工作流**:
```
问题: "UserService类的login方法实现"
↓
第1步: search_by_grep(keywords=["class UserService"], file_names=None) 
       → 找到UserService在user_service.py
↓
第2步: search_by_grep(keywords=["def login", "async login"], file_names=["user_service.py"])
       → 找到login方法在chunk 5
↓
第3步: search_by_filename_and_chunk_range(file_name="user_service.py", start=4, end=7)
       → 获取完整方法实现
```

#### 2. 补全上下文场景
**工具**: `search_by_document_and_chunk_range` / `search_by_filename_and_chunk_range`

**使用时机**:
- 发现chunks不连续（如1,5,9）→ 补齐中间部分（如获取2-8）
- 扩展现有chunk的前后文（如只有chunk 5）→ 获取3-7形成完整段落
- 从文件列表中获取到documentId/fileName和maxChunkIndex → 按需获取指定chunk范围
- **语义搜索/关键词搜索检索不到**：当知道目标文档的文件名或documentId，但语义检索和grep都无法命中时，可直接通过chunk范围获取文档局部内容

**使用指南**:
- 结合已检索文档的maxChunkIndex字段确定有效范围
- 按文档ID或文件名灵活选择工具
- 注意chunk索引从0开始（如maxChunkIndex=29表示有30个chunk，范围0-29）
- **绕过检索失败**：当其他检索工具失效时，可通过已知的文件名或documentId直接获取文档局部内容

**典型场景**:
```
场景1: 补全不连续chunk
已检索到chunk [1, 5, 9]，需要完整上下文
→ search_by_document_and_chunk_range(document_id=123, start=1, end=9)

场景2: 基于已知信息扩展获取（检索失效时）
通过list_filename_by_like找到目标文档：README.md (documentId=456, maxChunkIndex=20)
语义检索和grep都无法命中，但推测文档开头可能有相关信息
→ search_by_filename_and_chunk_range(file_name="README.md", start=0, end=5)
   先获取前几个chunk，根据内容决定是否继续扩展
```

#### 3. 多角度语义场景
**工具**: `search_by_multi_queries_in_database`

**使用时机**:
- 检索结果不理想，需要换角度重新检索
- 问题有多个方面，需要从不同角度探索
- 需要高质量语义匹配和Rerank排序
- 概念性、描述性问题（grep不适用的场景）

#### 4. 文件探索场景
**工具**: `list_filename_by_like`

**使用时机**:
- 不确定具体文件名，需要浏览文件列表
- 按名称模式查找相关文档（前缀、包含、目录等）
- 探索知识库中有哪些文件
- 进一步探索是否还有更多相关文件

**使用指南**:
- 此工具仅返回元信息（fileName, documentId, maxChunkIndex），不包含文档内容
- 获取元信息后，使用chunk范围工具获取实际内容
- 支持灵活的LIKE模式：前缀("doc%")、包含("%report%")、目录("dir/%")等

### 参数设置建议
- **top_k**: 首轮检索10-15个，补充检索5-7个
- **match_type**: 精确查找用"AND"（要求全部匹配），广泛探索用"OR"（任一匹配即可），默认"OR"
- **keywords**: 可以是完整短语或拆分词语，根据匹配精度需求决定
- **document_id**: 必须是整数类型int（不能是字符串）
- **chunk范围**: 注意maxChunkIndex边界，避免超出范围（索引从0开始）"""
