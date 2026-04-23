"""
Agentic RAG 工具集
包含 5 个原子化检索工具 + 1 个停止工具，基于 LangChain StructuredTool 原生 Function Calling
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional, Literal

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from aiohttp_utils import rerank
from utils import filter_grade_threshold

logger = logging.getLogger(__name__)


# ============= 工具参数模型 =============

class GrepSearchInput(BaseModel):
    """search_by_grep: 基于关键词精确匹配检索"""
    keywords: List[str] = Field(description="关键词列表，必须具体，避免泛词")
    match_type: Literal["AND", "OR"] = Field(default="OR", description="匹配模式")
    top_k: int = Field(default=15, description="返回条数限制")
    file_names: Optional[List[str]] = Field(default=None, description="文件名列表，为空则全库检索")


class ChunkRangeInput(BaseModel):
    """search_by_filename_and_chunk_range: 按文件名获取chunk范围"""
    file_name: str = Field(description="精确文件名")
    start_chunk_index: int = Field(description="起始chunk索引，包含")
    end_chunk_index: int = Field(description="结束chunk索引，包含")


class ContextWindowInput(BaseModel):
    """extend_file_chunk_context_window: 扩展chunk上下文窗口"""
    file_name: str = Field(description="精确文件名")
    chunk_index: int = Field(description="中心chunk索引，必须来自已命中的chunk")
    window_size: int = Field(default=2, description="上下文窗口大小，前后各取window_size个chunk")


class SemanticSearchInput(BaseModel):
    """search_by_multi_queries_in_database: 全库语义检索"""
    queries: List[str] = Field(description="多语义查询列表，建议3~6条，从不同角度描述同一问题")
    grade_query: str = Field(description="用于rerank的核心问题，通常是用户原始问题")
    top_k: int = Field(default=10, description="最终返回条数")
    grade_score_threshold: float = Field(default=0.3, description="Rerank分数阈值，0.3~0.6")


class FileListInput(BaseModel):
    """list_filename_by_like: 文件名模式匹配"""
    pattern: str = Field(description="文件名匹配模式，SQL LIKE语法，如 doc% 或 %report%")
    offset: int = Field(default=0, description="偏移量")
    limit: int = Field(default=30, description="返回数量限制")


class StopSearchInput(BaseModel):
    """stop_search: 停止检索"""
    reason: str = Field(description="停止检索的理由")


# ============= 检索工具集 =============

class RetrievalToolkit:
    """原子化检索工具集 - 基于 LangChain StructuredTool"""

    def __init__(self, vector_store, retriever):
        self.vector_store = vector_store
        self.retriever = retriever
        self._tools = self._build_tools()
        # tool_map 不包含 stop_search（stop 在调用方处理）
        self._tool_map = {t.name: t for t in self._tools if t.name != "stop_search"}

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

    async def _search_by_grep(
            self,
            keywords: List[str],
            match_type: str = "OR",
            top_k: int = 15,
            file_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """关键词检索（grep风格），支持全库检索或指定文件范围"""
        if file_names:
            file_names = [fn.replace(" ", "") for fn in file_names if fn.strip()]
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

    # ============= 工具2: 按文件名获取连续chunk范围 =============

    async def _search_by_filename_and_chunk_range(
            self,
            file_name: str,
            start_chunk_index: int,
            end_chunk_index: int,
    ) -> Dict[str, Any]:
        """按文件名获取连续chunk范围"""
        file_name = file_name.replace(" ", "")
        logger.info(f"🔍 [2.文件chunk范围] file='{file_name}', range=[{start_chunk_index}, {end_chunk_index}]")

        filter_expr = f'fileName == "{self._escape(file_name)}" and chunkIndex >= {start_chunk_index} and chunkIndex <= {end_chunk_index}'

        limit = end_chunk_index - start_chunk_index + 1
        if limit > 21:
            raise RuntimeError(f"单次chunk范围不能超过20个，当前为{limit}个，请缩小范围或分多次调用")

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

    # ============= 工具3: 快速扩展chunk上下文窗口 =============

    async def _extend_file_chunk_context_window(
            self,
            file_name: str,
            chunk_index: int,
            window_size: int = 2,
    ) -> Dict[str, Any]:
        """围绕某个已命中的chunk，快速查看前后上下文"""
        file_name = file_name.replace(" ", "")
        logger.info(f"🔍 [3.扩展上下文] file='{file_name}', chunk_index={chunk_index}, window={window_size}")

        start_chunk_index = max(0, chunk_index - window_size)
        end_chunk_index = chunk_index + window_size

        return await self._search_by_filename_and_chunk_range(
            file_name=file_name,
            start_chunk_index=start_chunk_index,
            end_chunk_index=end_chunk_index
        )

    # ============= 工具4: 全库语义检索(多query+rerank) =============

    async def _search_by_multi_queries_in_database(
            self,
            queries: List[str],
            grade_query: str,
            top_k: int = 10,
            grade_score_threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """
        全库语义检索(多query+rerank)

        流程:
            1. 并行向量检索所有queries（召回阶段）
            2. 合并去重
            3. 使用grade_query进行Rerank重排序评分（精排阶段）
            4. K-Means动态阈值过滤
            5. 按rerank_score排序并返回top_k
        """

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

    # ============= 工具5: 根据模式列出文件 =============

    async def _list_filename_by_like(
            self,
            pattern: str,
            offset: int = 0,
            limit: int = 30,
    ) -> Dict[str, Any]:
        """
        根据文件名模式匹配列出文件信息（仅返回元信息，不包含文档内容）

        注意:
            - 使用 chunkIndex == 0 来获取每个文件的首个chunk
            - 返回的Document不会累积到reference_docs中
        """

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

    # ============= 工具构建与分发 =============

    def _build_tools(self) -> List[StructuredTool]:
        return [
            StructuredTool(
                name="search_by_grep",
                description="基于关键词精确匹配检索正文内容，支持全库/单文件/多文件范围检索。适合已知明确关键词：函数名、类名、配置项、错误码、术语。",
                args_schema=GrepSearchInput,
                coroutine=self._search_by_grep,
            ),
            StructuredTool(
                name="search_by_filename_and_chunk_range",
                description="已知精确文件名时，按chunk范围顺序读取正文。单次范围不超过20个chunk。返回结果按chunkIndex升序。",
                args_schema=ChunkRangeInput,
                coroutine=self._search_by_filename_and_chunk_range,
            ),
            StructuredTool(
                name="extend_file_chunk_context_window",
                description="围绕某个已命中的chunk，快速查看前后上下文。适合局部扩展，不适合大范围通读。",
                args_schema=ContextWindowInput,
                coroutine=self._extend_file_chunk_context_window,
            ),
            StructuredTool(
                name="search_by_multi_queries_in_database",
                description="全库语义检索：多query并行召回+rerank+动态阈值过滤。适合概念性探索、缺乏明确关键词的问题。高成本工具。",
                args_schema=SemanticSearchInput,
                coroutine=self._search_by_multi_queries_in_database,
            ),
            StructuredTool(
                name="list_filename_by_like",
                description="按文件名模式查找候选文件，仅返回元信息不返回正文。使用SQL LIKE语法，%为通配符。后续需用文件级工具读取正文。",
                args_schema=FileListInput,
                coroutine=self._list_filename_by_like,
            ),
            StructuredTool(
                name="stop_search",
                description="停止检索，表示已获取足够信息来回答用户问题，或继续检索已无价值。",
                args_schema=StopSearchInput,
                coroutine=self._stop_search,
            ),
        ]

    def get_tools(self) -> List[StructuredTool]:
        """返回所有工具（含 stop_search）"""
        return self._tools

    async def execute_tool(self, tool_name: str, args: dict) -> Dict[str, Any]:
        """
        执行指定的检索工具（不含 stop_search）

        Args:
            tool_name: 工具名称
            args: 工具参数

        Returns:
            {"results": List[Document], "total_hits": int}
        """
        logger.info(f"🔧 执行工具: {tool_name}, 参数: {args}")
        if tool_name not in self._tool_map:
            raise ValueError(f"未知工具: {tool_name}")
        return await self._tool_map[tool_name].ainvoke(args)

    # ============= 停止工具 =============

    @staticmethod
    async def _stop_search(reason: str) -> str:
        """停止检索"""
        return f"检索已停止: {reason}"

