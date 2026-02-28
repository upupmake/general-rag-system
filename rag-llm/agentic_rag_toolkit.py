"""
Agentic RAG 工具集
包含5个原子化检索工具，提供完全透明可控的检索能力
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
    reason: str = Field(
        description="如果选择工具，需说明选择该工具的理由；如果停止检索，需说明为什么当前信息已足够或无法继续。")

    # 如果action=continue, 以下字段必填
    tool: Optional[Literal[
        "search_by_grep",  # 1. 关键词检索（grep），支持全库/单文件/多文件
        "search_by_filename_and_chunk_range",  # 2. 按文件名获取连续chunk范围
        "extend_file_chunk_windows",  # 3. 快速扩展chunk上下文窗口
        "search_by_multi_queries_in_database",  # 4. 全库语义检索(多query+rerank)
        "list_filename_by_like"  # 5. 根据模式匹配列出文件
    ]] = Field(default=None, description="下一步要使用的检索工具")

    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="工具参数,根据tool不同而不同"
    )

    existing_info: List[str] = Field(
        default_factory=list,
        description="当前已检索到的有用信息（简要总结）"
    )
    missing_info: List[str] = Field(
        default_factory=list,
        description="当前缺失的关键信息"
    )


# ============= 检索工具集 (5个工具) =============
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

    # ============= 工具2: 按文件名获取连续chunk范围 =============
    async def search_by_filename_and_chunk_range(
            self,
            file_name: str,
            start_chunk_index: int,
            end_chunk_index: int,
            **kwargs
    ) -> Dict[str, Any]:
        """
        工具2: 按文件名获取连续chunk范围

        Args:
            file_name: 文件名（精确匹配）
            start_chunk_index: 起始chunk索引（包含）
            end_chunk_index: 结束chunk索引（包含）
            **kwargs: 额外参数（容错，忽略大模型可能传递的其他参数）

        Returns:
            {"results": List[Document], "total_hits": int}
        """
        if kwargs:
            logger.debug(f"⚠️ 工具2收到额外参数（已忽略）: {kwargs}")

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
    async def extend_file_chunk_windows(
            self,
            file_name: str,
            chunk_index: int,
            window_size: int = 2,
            **kwargs
    ) -> Dict[str, Any]:
        """
        工具3: 快速扩展chunk上下文窗口

        Args:
            file_name: 文件名（精确匹配）
            chunk_index: 中心chunk索引
            window_size: 上下文窗口大小（默认2，表示前后各2个chunk）
            **kwargs: 额外参数（容错，忽略大模型可能传递的其他参数）

        Returns:
            {"results": List[Document], "total_hits": int}
        """
        if kwargs:
            logger.debug(f"⚠️ 工具3收到额外参数（已忽略）: {kwargs}")

        logger.info(f"🔍 [3.扩展上下文] file='{file_name}', chunk_index={chunk_index}, window={window_size}")

        start_chunk_index = max(0, chunk_index - window_size)
        end_chunk_index = chunk_index + window_size

        return await self.search_by_filename_and_chunk_range(
            file_name=file_name,
            start_chunk_index=start_chunk_index,
            end_chunk_index=end_chunk_index
        )

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
        if tool and params:
            if tool == "search_by_grep":
                return await self.search_by_grep(**params)
            elif tool == "search_by_filename_and_chunk_range":
                return await self.search_by_filename_and_chunk_range(**params)
            elif tool == "extend_file_chunk_windows":
                return await self.extend_file_chunk_windows(**params)
            elif tool == "search_by_multi_queries_in_database":
                return await self.search_by_multi_queries_in_database(**params)

            elif tool == "list_filename_by_like":
                return await self.list_filename_by_like(**params)
            else:
                raise ValueError(f"未知工具: {tool}")
        else:
            raise ValueError("工具名称和参数不能为空")


# ============= 联合优化后的 Agentic RAG 提示词 =============

TOOL_DEFINE_PROMPT = """## Agentic RAG 检索工具集（5 个原子化工具）

所有工具返回格式统一：{"results": List[Document], "total_hits": int}

================================================================================
[工具 1] search_by_grep - 关键词精确匹配检索
================================================================================
功能：基于 Milvus filter 的关键词匹配（SQL LIKE 语法），支持全库或指定文件范围

适用场景：
✅ 明确关键词：函数名/类名/配置项/错误码/API 路径/特定术语
✅ 需要精确匹配而非语义相似
✅ 已知文件名范围，需缩小检索域

不适用场景：
❌ 纯概念性/描述性问题（优先工具 4）
❌ 关键词过于宽泛（如"怎么使用"）

参数规范：
┌─────────────────┬──────────┬────────────────────────────────────────┐
│ 参数名          │ 类型     │ 说明                                 │
├─────────────────┼──────────┼────────────────────────────────────────┤
│ keywords        │ List[str]│ 必填，非空；优先使用精确短语          │
│ match_type      │ "AND"/"OR"│ 可选，默认"OR"；精确收敛用 AND         │
│ top_k           │ int      │ 可选，默认 5；建议 5~20                 │
│ file_names      │ List[str]│ 可选；null/空列表=全库；1 个=单文件    │
└─────────────────┴──────────┴────────────────────────────────────────┘

示例：
{"tool": "search_by_grep", "params": {"keywords": ["ValueError"], "match_type": "OR", "top_k": 10}}

================================================================================
[工具 2] search_by_filename_and_chunk_range - 按文件名获取连续 chunk
================================================================================
功能：已知 file_name 时，获取指定 chunk 索引范围的文档内容

适用场景：
✅ 已知确切文件名，需要读取内容
✅ 需要按 chunk 索引顺序阅读文件
✅ 需要精确控制 chunk 范围（如读取 0-19）

不适用场景：
❌ 文件名不确定（先用工具 5 探索）
❌ 只需要文件列表（工具 5 更高效）

参数规范：
┌─────────────────────┬──────┬────────────────────────────────────────┐
│ 参数名              │ 类型 │ 说明                                 │
├─────────────────────┼──────┼────────────────────────────────────────┤
│ file_name           │ str  │ 必填，精确匹配（从已检索文档 meta 获取） │
│ start_chunk_index   │ int  │ 必填，起始索引（包含）                │
│ end_chunk_index     │ int  │ 必填，结束索引（包含）                │
└─────────────────────┴──────┴────────────────────────────────────────┘

⚠️ 硬约束：
- 单次范围 ≤ 20 个 chunk（end - start + 1 ≤ 20）
- 返回结果按 chunkIndex 升序排序

示例：
{"tool": "search_by_filename_and_chunk_range", "params": {"file_name": "src/utils.py", "start_chunk_index": 0, "end_chunk_index": 19}}

================================================================================
[工具 3] extend_file_chunk_windows - 快速扩展 chunk 上下文窗口
================================================================================
功能：以指定 chunk 为中心，快速获取前后 window_size 个 chunk（上下文窗口扩展）

适用场景：
✅ 已检索到关键 chunk，需要查看前后上下文
✅ 快速获取 chunk 周围内容，无需手动计算范围
✅ 适合补充阅读上下文

参数规范：
┌─────────────────────┬──────┬────────────────────────────────────────┐
│ 参数名              │ 类型 │ 说明                                 │
├─────────────────────┼──────┼────────────────────────────────────────┤
│ file_name           │ str  │ 必填，精确文件名（从已检索文档 meta 获取）│
│ chunk_index         │ int  │ 必填，中心 chunk 索引                   │
│ window_size         │ int  │ 可选，窗口大小（默认 2，前后各 2 个 chunk）│
└─────────────────────┴──────┴────────────────────────────────────────┘

⚠️ 说明：
- 实际返回范围：[chunk_index-window_size, chunk_index+window_size]
- 自动处理边界（start_chunk_index 自动 max(0, chunk_index-window_size)）
- 总 chunk 数 = window_size * 2 + 1（不超过 20 个）

示例：
{"tool": "extend_file_chunk_windows", "params": {"file_name": "src/utils.py", "chunk_index": 10, "window_size": 3}}
// 将返回 chunk 索引 [7, 8, 9, 10, 11, 12, 13] 共 7 个 chunk

================================================================================
[工具 4] search_by_multi_queries_in_database - 全库语义检索+rerank 精排
================================================================================
功能：多 query 并行向量召回 → Rerank 重排序 → 动态阈值过滤 → 返回 top_k

完整流程：
1. 并行向量检索所有 queries（每 query 召回 top_k*3 条）
2. 合并去重（基于 pk）
3. 使用 grade_query 进行 Rerank 评分（0~1 分）
4. 动态阈值过滤（K-Means 自动计算）
5. 按 rerank_score 降序排序，返回 top_k

适用场景：
✅ 首轮检索：概念性/描述性问题
✅ 多角度探索：同一问题的不同表述
✅ search_by_grep 结果不理想时的补充检索
✅ 需要高质量语义匹配结果

不适用场景：
❌ 已有明确关键词可精确定位（优先工具 1）
❌ 已知具体文件名（优先工具 2 或工具 3）

参数规范：
┌─────────────────────────┬──────────┬────────────────────────────────────┐
│ 参数名                  │ 类型     │ 说明                             │
├─────────────────────────┼──────────┼────────────────────────────────────┤
│ queries                 │ List[str]│ 必填，3~6 条，避免同义重复        │
│ grade_query             │ str      │ 必填，消歧后的核心问题（用于评分）│
│ top_k                   │ int      │ 可选，默认 10；首轮 10~15，补充 5~7 │
│ grade_score_threshold   │ float    │ 可选，默认 0.4；范围 0.3~0.6       │
└─────────────────────────┴──────────┴────────────────────────────────────┘

⚠️ 关键说明：
- queries：从不同角度描述同一问题
- grade_query：使用用户原始问题或消除歧义后的核心问题
- grade_score_threshold：0.3 弱相关/0.5 一般相关/0.6 强相关
- 返回的 Document.metadata 包含'rerank_score'字段

示例：
{
  "tool": "search_by_multi_queries_in_database",
  "params": {
    "queries": ["用户认证流程", "login 验证步骤", "如何登录系统"],
    "grade_query": "用户登录认证的完整流程是什么",
    "top_k": 10,
    "grade_score_threshold": 0.4
  }
}

================================================================================
[工具 5] list_filename_by_like - 文件名模式匹配列表
================================================================================
功能：按 SQL LIKE 语法匹配文件名，仅返回元信息（不含正文）

返回内容：
- page_content: 空字符串
- meta {pk, documentId, chunkIndex, fileName, maxChunkIndex}

适用场景：
✅ 不确定文件名，需要先探索文件列表
✅ 按目录/前缀/包含模式查找文件
✅ 获取文件的 fileName 和 maxChunkIndex（为工具 2 或工具 3 做准备）

不适用场景：
❌ 直接获取文档正文（需再用工具 2 或工具 3）
❌ 已知确切文件名（直接用工具 2 或工具 3）

参数规范：
┌─────────────────┬──────┬────────────────────────────────────────┐
│ 参数名          │ 类型 │ 说明                                 │
├─────────────────┼──────┼────────────────────────────────────────┤
│ pattern         │ str  │ 必填，SQL LIKE 语法（%通配符）         │
│ offset          │ int  │ 可选，默认 0；分页用，避免重复         │
│ limit           │ int  │ 可选，默认 30；单次最大返回数          │
└─────────────────┴──────┴────────────────────────────────────────┘

LIKE 语法示例：
- "src/%"：src 目录下的所有文件
- "%config%"：文件名包含 config 的文件
- "test_%.py"：以 test_开头的 py 文件
- "%.md"：所有 md 文件

⚠️ 关键说明：
- 使用 chunkIndex == 0 获取每个文件的首个 chunk（含完整元信息）
- 若返回数 < limit，说明该 pattern 的文件已基本列举完毕
- 获取正文需再用工具 2（search_by_filename_and_chunk_range）或工具 3（extend_file_chunk_windows）

示例：
{"tool": "list_filename_by_like", "params": {"pattern": "src/auth/%", "limit": 30}}

================================================================================
[全局硬规则]
================================================================================
1. 只能使用上述 5 个工具名与参数名，拼写必须完全一致
2. 必填参数必须携带，参数类型必须正确（int/str/List 等）
3. chunk 范围约束：单次检索 ≤ 20 个 chunk
4. 工具 5 仅返回文件元信息；若要正文，必须再调用工具 2 或工具 3
5. 禁止重复调用"同一工具 + 完全相同参数"
6. 如需同工具重试，参数必须显著变化（关键词/范围/pattern/threshold/top_k 等）
"""

TOOL_SELECT_PROMPT = """## 检索决策输出规范

你必须只输出一个 JSON 对象（禁止 Markdown、禁止解释性文字），格式严格如下：

{
  "action": "continue" | "stop",
  "reason": "string（1~2 句，简洁具体）",
  "tool": "工具名" | null,
  "params": {} | null,
  "existing_info": ["已获取的关键信息摘要 1", "摘要 2", ...],
  "missing_info": ["仍缺失的关键信息 1", "缺失 2", ...]
}

================================================================================
[决策流程 - 按顺序判断]
================================================================================

【第 1 步】判断是否停止检索
────────────────────────────────────────────────────────────────────────────────
选择 stop 的条件（满足任一即可）：
✓ 已检索文档能完整回答问题
✓ 反复检索后文档与问题无关（检索失败）
✓ 已达到最大轮次限制（current_round >= max_rounds）
✓ 无法构造有效的新调用参数

选择 continue 的条件（满足任一即可）：
✓ 文档数量不足，信息量明显不够
✓ 信息覆盖不全，问题有多方面但只检索到部分
✓ 发现新线索需要深入挖掘
✓ 需要验证/补充已获取的信息

【第 2 步】若 continue，根据场景选择工具（优先级策略）
────────────────────────────────────────────────────────────────────────────────
┌────────────┬────────────────────────────────────┬─────────────────────────────────────────────────┐
│ 轮次阶段   │ 场景特征                           │ 推荐工具                                      │
├────────────┼────────────────────────────────────┼─────────────────────────────────────────────────┤
│ 首轮 (1-2)  │ 概念性/描述性问题，无明确关键词    │ search_by_multi_queries_in_database           │
│ 首轮 (1-2)  │ 有明确关键词/函数名/错误码         │ search_by_grep                                │
│ 中期 (3-4)  │ 已知 fileName，需读取连续 chunk      │ search_by_filename_and_chunk_range            │
│ 中期 (3-4)  │ 已定位关键 chunk，需查看前后上下文  │ extend_file_chunk_windows                     │
│ 任意轮次   │ 文件名不确定，需先探索文件列表     │ list_filename_by_like                         │
│ 后期 (5+)   │ 信息缺口明确，针对性补充           │ 根据缺口选择最匹配工具                        │
└────────────┴────────────────────────────────────┴─────────────────────────────────────────────────┘

【第 3 步】利用已检索文档的元信息（关键！）
────────────────────────────────────────────────────────────────────────────────
从"已检索文档"中提取以下信息指导后续决策：

1. 利用 maxChunkIndex 规划 chunk 范围：
   - 例：maxChunkIndex=29 表示该文件有 30 个 chunk（索引 0-29）
   - 规划范围时确保 end_chunk_index ≤ maxChunkIndex

2. 利用 fileName + chunkIndex 快速扩展上下文（推荐）：
   - 从已检索文档的 metadata 中获取 fileName 和 chunkIndex
   - 优先使用 extend_file_chunk_windows 快速获取前后内容
   - 比手动计算范围更高效，自动处理边界

3. 利用 fileName 调用文件级范围检索：
   - 从已检索文档的 metadata 中获取 fileName
   - 用于 search_by_filename_and_chunk_range 精确控制范围
   - 适合需要读取大范围连续 chunk 的场景

4. 利用已检索 chunk 的连续性判断是否需要补充：
   - 例：已获取 chunk [0,1,2] 和 [15,16,17]，中间 [3-14] 缺失
   - 可调用 search_by_filename_and_chunk_range 补全中间部分

5. 利用文件分布判断检索覆盖度：
   - total_files=1 且问题涉及多模块 → 需要扩展文件范围
   - total_files>5 但内容重复 → 需要收敛聚焦

【第 4 步】分析工具调用历史（防循环）
────────────────────────────────────────────────────────────────────────────────
从"工具调用历史"中检查：

1. 禁止重复调用"同一工具 + 完全相同参数"
2. 同工具重试时，至少改变一个关键参数：
   - search_by_grep：换关键词、改 match_type、调整 file_names
   - search_by_filename_and_chunk_range：移动窗口位置、缩小范围
   - extend_file_chunk_windows：换 chunk_index、调整 window_size
   - search_by_multi_queries_in_database：换 queries 表述、调整 threshold
   - list_filename_by_like：改 pattern、用 offset 分页
3. 连续 2 次检索结果<3 条 → 考虑换工具或 stop
4. 连续 3 次检索无新信息 → 强制 stop

【第 5 步】参数构造检查清单
────────────────────────────────────────────────────────────────────────────────
□ search_by_grep：
  - keywords 非空且具体（避免"怎么""如何"等泛词）
  - match_type 选择合理（精确用 AND，探索用 OR）

□ search_by_filename_and_chunk_range：
  - 范围 ≤ 20 个 chunk（end - start + 1 ≤ 20）
  - 检查 maxChunkIndex 边界

□ extend_file_chunk_windows：
  - file_name 精确匹配（从已检索文档 meta 获取）
  - chunk_index 是已检索到的有效 chunk 索引
  - window_size 建议 1-5（默认 2，返回 5 个 chunk）
  - ⚠️ 总 chunk 数 = window_size * 2 + 1，确保不超过 20

□ search_by_multi_queries_in_database：
  - queries 3~6 条，角度不同但语义相关
  - grade_query 是消歧后的核心问题
  - grade_score_threshold 在 0.3~0.6 之间

□ list_filename_by_like：
  - pattern 使用 LIKE 语法（%通配符）
  - 分页时正确设置 offset 避免重复

================================================================================
[工具 2 vs 工具 3 选择指南]
================================================================================
┌────────────────────────────────────────────────────┬────────────────────────────────────────────────────┐
│ 使用 search_by_filename_and_chunk_range            │ 使用 extend_file_chunk_windows                     │
├────────────────────────────────────────────────────┼────────────────────────────────────────────────────┤
│ 需要精确控制 start/end 索引                          │ 以某 chunk 为中心，快速获取前后内容                  │
│ 需要读取大范围连续 chunk(>10)                       │ 只需查看局部上下文 (3-11 个 chunk)                    │
│ 需要补全两个 chunk 之间的缺口                        │ 已定位关键 chunk，想看看周围说了什么                │
│ 需要分页读取完整文件                               │ 快速探索，无需手动计算范围                         │
└────────────────────────────────────────────────────┴────────────────────────────────────────────────────┘

简单规则：
- "我想看 chunk 10 前后的内容" → extend_file_chunk_windows (chunk_index=10, window_size=2)
- "我想读取 chunk 0 到 19 的完整内容" → search_by_filename_and_chunk_range (start=0, end=19)
- "我想补全 chunk 5 到 15 之间的内容" → search_by_filename_and_chunk_range (start=5, end=15)

================================================================================
[输出一致性校验]
================================================================================
✓ action="continue" → tool 和 params 必须非空且合法
✓ action="stop" → tool 和 params 必须为 null
✓ reason 简洁具体（1~2 句，说明决策依据）
✓ existing_info：当前已获取的有用信息摘要（List[str]）
✓ missing_info：真正影响回答的关键信息缺口（List[str]）
✓ 禁止重复上一轮完全相同的 tool+params 组合

================================================================================
[示例输出]
================================================================================
示例 1（首轮 - 语义检索）：
{"action": "continue", "reason": "首轮检索，问题为概念性描述，需语义召回探索", "tool": "search_by_multi_queries_in_database", "params": {"queries": ["API 认证流程", "token 验证方法", "如何鉴权"], "grade_query": "API 请求的认证和鉴权流程", "top_k": 10, "grade_score_threshold": 0.4}, "existing_info": [], "missing_info": ["认证流程步骤", "所需参数", "返回格式"]}

示例 2（中期 - 快速扩展上下文）：
{"action": "continue", "reason": "已定位关键 chunk，需查看前后上下文理解完整逻辑", "tool": "extend_file_chunk_windows", "params": {"file_name": "src/auth.py", "chunk_index": 10, "window_size": 3}, "existing_info": ["auth.py 第 10 个 chunk 包含 authenticate 函数定义"], "missing_info": ["函数完整实现", "调用示例"]}

示例 3（中期 - 范围读取）：
{"action": "continue", "reason": "已知文件名，需读取文件开头部分内容", "tool": "search_by_filename_and_chunk_range", "params": {"file_name": "src/auth.py", "start_chunk_index": 0, "end_chunk_index": 19}, "existing_info": ["auth.py 文件存在，共 30 个 chunk"], "missing_info": ["认证函数实现细节", "参数说明"]}

示例 4（停止检索）：
{"action": "stop", "reason": "已获取完整认证流程代码和说明，信息充足", "tool": null, "params": null, "existing_info": ["authenticate_user 函数实现", "token 验证逻辑", "API 调用示例"], "missing_info": []}
"""

# ============= 决策控制器专用系统提示词 =============
CONTROLLER_SYSTEM_PROMPT = """你是 RAG 检索决策专家，负责基于三类信息决定下一步检索行动。

## 你的输入信息

1. **对话上下文**：用户问题 + 对话历史 → 理解问题意图和所需信息类型
2. **已检索文档**：按文件聚合的文档列表（含 chunkIndex、maxChunkIndex、fileName 等元信息）
3. **工具调用历史**：历次工具调用的参数和结果 → 避免重复、分析失败原因

## 你的核心任务

1. 评估当前信息是否足以回答问题 → 决定 stop 或 continue
2. 若 continue，选择最合适的检索工具 → 基于轮次、场景、已获取信息
3. 构造合法的工具参数 → 利用已检索文档的元信息（maxChunkIndex、fileName 等）
4. 总结 existing_info 和 missing_info → 帮助追踪检索进度

## 可用工具（5 个）

| 序号 | 工具名                              | 用途                     |
|------|-------------------------------------|--------------------------|
| 1    | search_by_grep                      | 关键词精确匹配检索       |
| 2    | search_by_filename_and_chunk_range  | 按文件名获取连续 chunk 范围 |
| 3    | extend_file_chunk_windows           | 快速扩展 chunk 上下文窗口  |
| 4    | search_by_multi_queries_in_database | 全库语义检索+rerank 精排  |
| 5    | list_filename_by_like               | 文件名模式匹配列表       |

## 决策原则

1. **轮次感知**：首轮探索→中期收敛→后期补充，不同阶段用不同策略
2. **信息复用**：充分利用已检索文档的元信息规划后续检索
3. **防循环**：不重复相同调用，连续失败及时止损
4. **效率优先**：
   - 能用 search_by_grep 不用 search_by_multi_queries_in_database
   - 能用 extend_file_chunk_windows 不用 search_by_filename_and_chunk_range（小范围上下文）
   - 能用范围检索不用全库扫描

## 输出要求

- 严格遵守 RetrievalDecision JSON Schema
- 只输出 JSON 对象，禁止 Markdown、禁止解释性文字
- params 必须是完整的 JSON 对象，包含所有必填参数
- reason 要清晰说明决策依据（基于哪些信息、为什么选择该工具）
- existing_info 和 missing_info 必须是字符串列表 List[str]
- tool 必须是 5 个可用工具名之一，拼写完全一致
"""
