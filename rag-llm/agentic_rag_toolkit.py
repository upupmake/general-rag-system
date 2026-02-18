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
====================
[工具1] search_by_grep
====================
功能:
- 关键词精确匹配检索（grep风格），支持全库或指定文件范围。

适用场景:
- 明确关键词/函数名/类名/配置项/错误码/API路径等精确定位。

不适用场景:
- 纯概念性、无明确关键词的问题（优先语义检索）。

参数:
- keywords: List[str]，必填，非空
- match_type: "AND" | "OR"，可选，默认"OR"
- top_k: int，可选，默认5
- file_names: List[str] | null，可选；null或空列表表示全库检索

====================
[工具2] search_by_document_and_chunk_range
====================
功能:
- 按document_id获取连续chunk范围。

适用场景:
- 已知document_id，需要补全/扩展连续上下文。

不适用场景:
- 不知道document_id时。

参数:
- document_id: int，必填
- start_chunk_index: int，必填
- end_chunk_index: int，必填，且 start_chunk_index <= end_chunk_index

====================
[工具3] search_by_filename_and_chunk_range
====================
功能:
- 按file_name获取连续chunk范围。

适用场景:
- 已知file_name，需要补全/扩展连续上下文。

不适用场景:
- 文件名不确定时（应先用list_filename_by_like）。

参数:
- file_name: str，必填（精确匹配）
- start_chunk_index: int，必填
- end_chunk_index: int，必填，且 start_chunk_index <= end_chunk_index

====================
[工具4] search_by_multi_queries_in_database
====================
功能:
- 多query语义召回 + rerank精排 + 过滤，返回高质量语义结果。

适用场景:
- 初期探索，无明确思路时；
- 概念性/描述性问题；
- 需要多角度探索；
- grep结果不理想时。

不适用场景:
- 已有明确关键词可直接精确定位时（优先grep）。

参数:
- queries: List[str]，必填，建议3~6条、避免重复
- grade_query: str，必填（用于rerank评分）
- top_k: int，可选，默认10
- grade_score_threshold: float，可选，默认0.4（建议0.3~0.6）

====================
[工具5] list_filename_by_like
====================
功能:
- 按文件名模式列出文件（仅返回元信息，不返回正文）。

适用场景:
- 不确定文件名时先探索文件；
- 按目录/前缀/包含模式查找文件。

不适用场景:
- 直接获取正文内容（此工具做不到）。

参数:
- pattern: str，必填（SQL LIKE语法，如"%report%"、"docs/%"、"%%"）
- offset: int，可选，默认0
- limit: int，可选，默认30

====================
[全局硬规则]
====================
1) 只能使用上述5个工具名与参数名。
2) 参数类型必须正确；不要猜测不存在字段。
3) chunk范围必须满足 start <= end。
4) list_filename_by_like 仅返回文件元信息；若要正文，必须再调用chunk范围工具。
5) 输出中若出现非法工具名/参数名，视为错误决策。"""

TOOL_SELECT_PROMPT = """## 工具选择策略

你必须只输出一个JSON对象（禁止Markdown、禁止解释性文字），格式严格如下：
{
  "action": "continue" | "stop",
  "reason": "string",
  "tool": "search_by_grep" | "search_by_document_and_chunk_range" | "search_by_filename_and_chunk_range" | "search_by_multi_queries_in_database" | "list_filename_by_like" | null,
  "params": {} | null,
  "missing_info": ["string", ...]
}

====================
[第1步：先判断是否停止]
====================
停止检索 (action="stop"):
- 对话上下文、已检索文档和工具调用历史这三类信息能完整回答问题
- 反复检索后发现检索的文档信息与问题跟本不相关
- 已达到最大轮次

继续检索 (action="continue"):
- 文档数量不足，信息量明显不够
- 信息覆盖不全，问题有多个方面但只检索到部分信息
- 发现与问题相关的新线索需要深入挖掘

====================
[第2步：若继续]
====================
A. 明确关键词/函数名/类名/配置项/错误码/API路径
   -> search_by_grep

B. 已知 file_name 且需要连续上下文
   -> search_by_filename_and_chunk_range

C. 已知 document_id 且需要连续上下文
   -> search_by_document_and_chunk_range

D. 文件名不确定，需要先找文件
   -> list_filename_by_like

E. 概念性/描述性问题，或grep效果差
   -> search_by_multi_queries_in_database

====================
[第3步：参数构造规则]
====================
通用：
- 参数必须最小且合法，不要传无关字段。

search_by_grep:
- keywords必须非空，优先使用精确短语；
- 精确收敛用AND，广泛探索用OR。

search_by_document_and_chunk_range / search_by_filename_and_chunk_range:
- start_chunk_index <= end_chunk_index；
- 范围尽量结合maxChunkIndex避免越界；
- document_id必须是int。

search_by_multi_queries_in_database:
- queries建议3~6条，避免同义重复；
- grade_query使用“消歧后的核心问题”；
- top_k：首轮建议10~15，补充轮次建议5~7；
- grade_score_threshold：0.3~0.6按严格度调整。

list_filename_by_like:
- pattern使用LIKE语法；
- 仅用于发现文件元信息，不返回正文。

====================
[防循环规则]
====================
1) 禁止重复调用“同一工具 + 完全相同参数”。
2) 如需同工具重试，参数必须显著变化（关键词、范围、pattern、threshold、top_k等）。
3) 若无法构造有效新调用，直接stop并在missing_info写清缺口。

====================
[输出一致性]
====================
- action="continue"：tool和params必须非空且合法。
- action="stop"：tool和params必须为null。
- reason简洁具体（1~2句）。
- missing_info仅列真正影响回答的关键信息缺口。"""
