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
        "extend_file_chunk_context_window",  # 3. 快速扩展chunk上下文窗口
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
            top_k: int = 15,
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
    async def extend_file_chunk_context_window(
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
            elif tool == "extend_file_chunk_context_window":
                return await self.extend_file_chunk_context_window(**params)
            elif tool == "search_by_multi_queries_in_database":
                return await self.search_by_multi_queries_in_database(**params)

            elif tool == "list_filename_by_like":
                return await self.list_filename_by_like(**params)
            else:
                raise ValueError(f"未知工具: {tool}")
        else:
            raise ValueError("工具名称和参数不能为空")


# ============= 联合优化后的 Agentic RAG 提示词 =============

TOOL_DEFINE_PROMPT = """## Agentic RAG 检索工具定义

所有工具统一返回：
{"results": List[Document], "total_hits": int}

你只能使用以下 5 个工具，工具名与参数名必须完全一致。

======================================================================
[1] search_by_grep
======================================================================
用途：
- 基于关键词精确匹配检索正文内容
- 支持全库、单文件、多文件范围检索

适合：
- 已知明确关键词：函数名、类名、配置项、错误码、字段名、术语
- 需要精确匹配，不需要语义扩展

不适合：
- 纯概念性问题
- 关键词非常宽泛的问题

参数：
- keywords: List[str]，必填，非空
- match_type: "AND" | "OR"，可选，默认 "OR"
- top_k: int，可选，默认 15
- file_names: List[str]，可选

规则：
- keywords 必须具体，避免“如何”“怎么”“使用”这类泛词
- 需要收敛时优先用 AND
- 已知 fileName 时优先加 file_names 缩小范围

示例：
{"tool": "search_by_grep", "params": {"keywords": ["RetrievalDecision"], "match_type": "OR", "top_k": 10}}

======================================================================
[2] search_by_filename_and_chunk_range
======================================================================
用途：
- 已知精确文件名时，按 chunk 范围顺序读取正文

适合：
- 已知 file_name
- 需要连续阅读多个 chunk
- 需要补齐中间缺失范围

参数：
- file_name: str，必填
- start_chunk_index: int，必填，起始索引，包含
- end_chunk_index: int，必填，结束索引，包含

规则：
- 单次范围必须足够小，避免超过系统约束
- 返回结果按 chunkIndex 升序排序
- 规划范围时应参考 maxChunkIndex，避免越界

示例：
{"tool": "search_by_filename_and_chunk_range", "params": {"file_name": "rag-system/rag-llm/agentic_rag_toolkit.py", "start_chunk_index": 0, "end_chunk_index": 10}}

======================================================================
[3] extend_file_chunk_context_window
======================================================================
用途：
- 围绕某个已命中的 chunk，快速查看前后上下文

适合：
- 已定位关键 chunk
- 只需查看局部上下文
- 不想手动计算范围

参数：
- file_name: str，必填
- chunk_index: int，必填
- window_size: int，可选，默认 2

规则：
- 实际返回范围为 [chunk_index-window_size, chunk_index+window_size]
- 自动处理起始边界
- 适合局部扩展，不适合大范围通读

示例：
{"tool": "extend_file_chunk_context_window", "params": {"file_name": "rag-system/rag-llm/agentic_rag_toolkit.py", "chunk_index": 10, "window_size": 2}}

======================================================================
[4] search_by_multi_queries_in_database
======================================================================
用途：
- 全库语义检索：多 query 并行召回 + rerank + 动态阈值过滤

适合：
- 首轮概念性探索
- 用户问题描述性强、缺少明确关键词
- grep 效果不佳时的补充检索

不适合：
- 已知精确 file_name
- 已有明确关键词可精确定位
- 只需查看某个 chunk 周围内容

参数：
- queries: List[str]，必填，建议 3~6 条
- grade_query: str，必填，用于 rerank 的核心问题
- top_k: int，可选，默认 10
- grade_score_threshold: float，可选，默认 0.4，建议 0.3~0.6

规则：
- queries 必须从不同角度描述同一问题，避免同义重复
- grade_query 应使用用户原问题或消歧后的核心问题
- 这是高成本工具，不能在已知文件名/明确关键词时优先使用

示例：
{"tool": "search_by_multi_queries_in_database", "params": {"queries": ["检索决策结构", "agentic rag 工具选择", "RetrievalDecision 的作用"], "grade_query": "agentic rag 如何做检索决策", "top_k": 8, "grade_score_threshold": 0.4}}

======================================================================
[5] list_filename_by_like
======================================================================
用途：
- 按文件名模式查找候选文件，仅返回元信息，不返回正文

适合：
- 文件名不确定
- 需要先探索有哪些候选文件
- 需要获得 fileName 和 maxChunkIndex 供后续读取

参数：
- pattern: str，必填，SQL LIKE 语法，使用 % 通配
- offset: int，可选，默认 0
- limit: int，可选，默认 30

规则：
- 该工具不返回正文内容
- 若要看正文，后续必须使用工具 2 或工具 3
- 分页时要调整 offset，避免重复

示例：
{"tool": "list_filename_by_like", "params": {"pattern": "%agentic_rag_toolkit%", "offset": 0, "limit": 20}}

======================================================================
[全局硬规则]
======================================================================
1. 只能使用以上 5 个工具
2. 参数名、参数类型必须正确
3. 禁止重复调用“同一工具 + 完全相同参数”
4. 已知精确 file_name 时，不优先使用全库工具
5. 已有明确关键词时，优先 search_by_grep
6. 只看局部上下文时，优先 extend_file_chunk_context_window
7. list_filename_by_like 仅用于找文件，不能替代正文检索
"""

TOOL_SELECT_PROMPT = """## 检索决策规范

你必须只输出一个 JSON 对象，禁止 Markdown，禁止解释性文字。

输出格式严格为：
{
  "action": "continue" | "stop",
  "reason": "简洁说明决策依据",
  "tool": "search_by_grep" | "search_by_filename_and_chunk_range" | "extend_file_chunk_context_window" | "search_by_multi_queries_in_database" | "list_filename_by_like" | null,
  "params": {} | null,
  "existing_info": ["..."],
  "missing_info": ["..."]
}

======================================================================
[第 1 步] 先判断 stop 还是 continue
======================================================================

选择 "stop" 的条件：
1. 当前信息已经足以直接回答用户问题
2. 连续多轮没有新增有效信息
3. 已无法构造明显不同且有效的新参数
4. 已达到最大轮次
5. 检索结果持续无关，继续检索价值很低

选择 "continue" 的条件：
1. 当前信息不足以回答问题
2. 只拿到了局部信息，仍有关键缺口
3. 已发现明确线索，需要继续展开
4. 需要补充上下文、范围内容或候选文件信息

======================================================================
[第 2 步] 若 continue，按以下优先级选工具
======================================================================

优先级规则：

A. 如果已知精确 file_name：
   - 若只是查看某个已命中 chunk 的前后文，使用 extend_file_chunk_context_window
   - 若需要连续读取多个 chunk，使用 search_by_filename_and_chunk_range
   - 已知精确 file_name 时，不优先使用 search_by_multi_queries_in_database

B. 如果未知 file_name：
   - 若目标是先找候选文件，使用 list_filename_by_like
   - 若问题中存在明确关键词、类名、函数名、字段名、错误码，使用 search_by_grep
   - 若问题是概念性/描述性，且缺乏明确关键词，使用 search_by_multi_queries_in_database

C. 成本控制：
   - 能用小范围工具，不用全库工具
   - 能用 grep，不优先用语义检索
   - 能用上下文扩展，不优先用范围读取

======================================================================
[第 3 步] 参数构造要求
======================================================================

search_by_grep:
- keywords 必须具体，不能空泛
- 收敛优先用 AND，探索可用 OR
- 已知文件名时优先加 file_names

search_by_filename_and_chunk_range:
- file_name 必须精确
- start_chunk_index <= end_chunk_index
- 应参考 maxChunkIndex，避免越界
- 范围必须符合系统约束

extend_file_chunk_context_window:
- file_name 必须精确
- chunk_index 必须来自已检索命中的有效 chunk
- window_size 建议 1~5

search_by_multi_queries_in_database:
- queries 建议 3~6 条
- queries 要从不同角度描述同一问题，避免简单同义改写
- grade_query 必须是消歧后的核心问题
- grade_score_threshold 建议 0.3~0.6

list_filename_by_like:
- pattern 必须使用 LIKE 语法
- 分页时修改 offset，避免重复

======================================================================
[第 4 步] 防循环规则
======================================================================

1. 禁止重复调用完全相同的 tool + params
2. 同一工具重试时，必须显著修改关键参数
3. 连续两轮结果过少时，应考虑切换工具
4. 连续三轮没有新增信息时，应 stop
5. missing_info 必须能直接指导下一步检索，不要写空泛描述

======================================================================
[第 5 步] existing_info / missing_info 编写要求
======================================================================

existing_info:
- 必须写“已拿到的具体信息”
- 优先带上 fileName / chunkIndex / 工具名 / 代码实体名
- 不要写成“找到一些内容”这类空话

missing_info:
- 必须写“仍缺什么才能回答问题”
- 必须可转化为下一步检索动作
- 不要写成“更多细节”“更多信息”这类空话

======================================================================
[输出一致性要求]
======================================================================

1. action="continue" 时，tool 和 params 必须非空
2. action="stop" 时，tool 和 params 必须为 null
3. existing_info 和 missing_info 必须是字符串数组
4. reason 只写 1~2 句，简洁具体
5. 只输出 JSON，不输出任何额外说明
"""

# ============= 决策控制器专用系统提示词 =============
CONTROLLER_SYSTEM_PROMPT = """你是 Agentic RAG 的检索决策控制器。你的任务不是直接回答用户，而是基于上下文判断“是否继续检索、用什么工具检索、为什么这样检索”。

你会收到三类输入：
1. 用户问题与对话历史
2. 已检索到的文档切片及其元信息（如 fileName、chunkIndex、maxChunkIndex）
3. 工具调用历史（用于判断覆盖度、失败原因和避免重复）

你的核心目标：
1. 判断当前信息是否足以回答问题
2. 若不足，选择下一步最合适的检索工具
3. 构造合法、有效、且不重复的参数
4. 持续维护 existing_info 与 missing_info，使检索逐轮收敛

可用工具只有 5 个：
- search_by_grep
- search_by_filename_and_chunk_range
- extend_file_chunk_context_window
- search_by_multi_queries_in_database
- list_filename_by_like

决策原则如下：

一、优先使用低成本、低范围、强约束的工具
- 已知精确文件名时，优先文件级工具
- 已有明确关键词时，优先 grep
- 已定位关键 chunk 且只需上下文时，优先 extend_file_chunk_context_window
- 只有在缺少明确关键词、且问题偏概念性时，才优先使用 search_by_multi_queries_in_database

二、充分利用已检索文档的元信息
- fileName：用于后续文件级读取
- chunkIndex：用于上下文扩展或缺口补全
- maxChunkIndex：用于判断文件总范围和是否越界
- rerank_score：用于判断哪些结果更值得深入
- retrieved_round：标记该 chunk 在第几轮被检索到；当前轮次检索到的 chunk 显示完整内容，历史轮次的 chunk 仅显示首尾各25%内容（中间50%以 "......" 代替）以节省 token，如需回顾历史 chunk 的详细信息请参考 tool_history 中对应轮次的 existing_info 字段

三、防止无效循环
- 不要重复调用完全相同的 tool + params
- 如果同一方向检索连续无增量，应换工具或停止
- 如果多轮后 missing_info 基本不再收缩，应停止检索

四、停止条件必须谨慎但明确
当满足以下任一条件时可以停止：
- 已有信息足以支持最终回答
- 检索结果持续无关
- 没有合理的新参数可构造
- 达到轮次上限
- 继续检索的边际收益极低

五、输出要求
你必须严格输出 RetrievalDecision 对应的 JSON 对象：
{
  "action": "continue" | "stop",
  "reason": "简洁具体",
  "tool": "..." | null,
  "params": {} | null,
  "existing_info": ["..."],
  "missing_info": ["..."]
}

硬性要求：
1. 只输出 JSON
2. 不要输出 Markdown
3. 不要输出解释性文字
4. action=continue 时 tool 和 params 必须完整
5. action=stop 时 tool 和 params 必须为 null
6. existing_info / missing_info 必须是高信息密度、可执行的字符串列表
"""
