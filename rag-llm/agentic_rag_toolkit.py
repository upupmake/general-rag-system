"""
Agentic RAG 工具集
包含 5 个原子化检索工具 + 1 个停止工具，基于 LangChain StructuredTool 原生 Function Calling
"""
import asyncio
import logging
import math
from typing import List, Dict, Any, Optional, Literal

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, model_validator

from utils import filter_grade_threshold

logger = logging.getLogger(__name__)

# 关键词检索候选上限：高频词不能只返回任意 top_k 个
KEYWORD_CANDIDATE_MAX = 200
KEYWORD_CANDIDATE_FACTOR = 5
# 语义检索在排除已读切片时的额外 over-fetch 上限
SEMANTIC_EXCLUDE_OVERFETCH_MAX = 200


# ============= 工具参数模型 =============

class KeywordSearchInput(BaseModel):
    """keyword_search: 基于关键词精确匹配检索"""
    keywords: List[str] = Field(
        min_length=1,
        description=(
            "关键词列表，必须是文档中可能出现的具体术语、函数名、配置项、错误码等。"
            "示例：['学籍管理', '注册流程', '休学复学', '转专业', '毕业要求', '学位授予'] "
            "或 ['def train_model', 'learning_rate', 'batch_size', 'optimizer', 'loss_function']"
        )
    )
    match_mode: Literal["AND", "OR"] = Field(
        default="OR",
        description=(
            "匹配模式：OR=任意关键词匹配（扩大范围），AND=所有关键词必须出现（缩小范围）。"
            "建议：探索性问题用OR，精确查找用AND"
        )
    )
    top_k: int = Field(default=15, ge=1, le=50, description="返回结果数量上限，默认15条")
    document_ids: Optional[List[int]] = Field(
        default=None,
        description="限定检索范围的 documentId 列表；为空则全库检索。"
    )


class ReadFileChunksInput(BaseModel):
    """read_file_chunks: 按 documentId 读取连续chunk范围"""
    document_id: int = Field(
        description="文档 ID，可通过 find_files 或其他检索结果获取"
    )
    start_chunk_index: int = Field(
        ge=0,
        description="起始chunk索引（包含），从0开始"
    )
    end_chunk_index: int = Field(
        ge=0,
        description="结束chunk索引（包含）。注意：单次范围不能超过20个chunk"
    )

    @model_validator(mode="after")
    def check_chunk_range(self):
        if self.end_chunk_index < self.start_chunk_index:
            raise ValueError("end_chunk_index 不能小于 start_chunk_index")
        return self


class ExpandContextInput(BaseModel):
    """expand_context: 按 documentId 扩展已命中chunk的上下文窗口"""
    document_id: int = Field(
        description="命中文档的 documentId，必须来自已检索到的文档"
    )
    chunk_index: int = Field(
        description="中心chunk索引，必须来自已命中的chunk"
    )
    window_size: int = Field(
        default=2,
        ge=0,
        le=9,
        description=(
            "上下文窗口大小，前后各取window_size个chunk。"
            "默认2，即中心chunk前后各取2个，共5个chunk"
        )
    )


class SemanticSearchInput(BaseModel):
    """semantic_search: 全库语义检索（多query并行召回，按向量相似度排序）"""
    queries: List[str] = Field(
        min_length=1,
        max_length=10,
        description=(
            "从多角度、多方面生成的相关查询列表（建议4-6条）。"
            "每个query可以用空格分隔多个关键词，系统会同时进行语义匹配和关键词匹配。"
            "示例：用户问'研究生学籍管理规定有哪些' → "
            "queries=['研究生 学籍管理 规定', '学生 注册 入学流程', '休学 复学 保留学籍', "
            "'转专业 转学 学籍变动', '毕业要求 学位授予条件', '纪律处分 学籍处理']"
        )
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="最终返回结果数量上限（动态阈值过滤后截断），默认10条"
    )


class FindFilesInput(BaseModel):
    """find_files: 根据文件名模式查找文件"""
    pattern: str = Field(
        min_length=1,
        description=(
            "文件名匹配模式，使用 % 作为通配符。"
            "示例：'研究生%' 匹配以'研究生'开头的文件，'%手册%' 匹配包含'手册'的文件"
        )
    )
    offset: int = Field(default=0, ge=0, description="分页偏移量")
    limit: int = Field(default=30, ge=1, le=100, description="单次返回数量上限，默认30")


class StopSearchInput(BaseModel):
    """stop_search: 停止检索"""
    reason: str = Field(
        description=(
            "停止检索的理由，必须明确说明。"
            "示例：'已找到完整的学籍管理规定' 或 '知识库中没有相关内容'"
        )
    )


# ============= 检索工具集 =============

class RetrievalToolkit:
    """原子化检索工具集 - 基于 LangChain StructuredTool"""

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self._tools = self._build_tools()
        # tool_map 不包含 stop_search（stop 在调用方处理）
        self._tool_map = {t.name: t for t in self._tools if t.name != "stop_search"}

    @staticmethod
    def _escape_like(value: str, escape_wildcards: bool = True) -> str:
        """转义 Milvus LIKE 表达式中的字面量。

        escape_wildcards=True 时把 % 和 _ 也当字面量，用于正文关键词检索；
        find_files 的文件名模式需要保留用户传入的 %/_ 通配符，因此传 False。
        Milvus 2.6 的表达式解析器不接受 raw string（r"..."），普通字符串会先处理
        一次转义，所以 LIKE 转义需要写成双反斜杠（例如 \\% / \\_），否则会解析失败。
        """
        escaped = value.replace("\\", "\\\\\\\\").replace('"', '\\"')
        if escape_wildcards:
            escaped = escaped.replace("%", "\\\\%").replace("_", "\\\\_")
        return escaped

    @staticmethod
    def _filter_excluded_pks(docs: List[Document], exclude_pks: Optional[set]) -> List[Document]:
        if not exclude_pks:
            return docs
        return [doc for doc in docs if doc.metadata.get("pk") not in exclude_pks]

    @staticmethod
    def _build_exclude_pks_expr(exclude_pks: Optional[set]) -> str:
        if not exclude_pks:
            return ""
        numeric_pks = []
        for pk in exclude_pks:
            try:
                numeric_pks.append(int(pk))
            except (TypeError, ValueError):
                continue
        if not numeric_pks:
            return ""
        return f"not (pk in [{', '.join(str(pk) for pk in numeric_pks)}])"

    async def _milvus_filter(
            self,
            filter_expr: str,
            offset: int = 0,
            limit: int = 20,
            output_fields: Optional[List[str]] = None,
            exclude_pks: Optional[set] = None,
    ) -> List[Document]:
        """底层Milvus查询封装"""
        if output_fields is None:
            output_fields = ["text", "pk", "documentId", "chunkIndex", "fileName", "maxChunkIndex"]
        exclude_expr = self._build_exclude_pks_expr(exclude_pks)
        if exclude_expr:
            filter_expr = f"({filter_expr}) and {exclude_expr}"
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
            raise

    async def _vector_search(
            self,
            query: Optional[str] = None,
            top_k: int = 10,
            exclude_pks: Optional[set] = None,
            query_embedding: Optional[List[float]] = None,
    ) -> List[Document]:
        """底层向量检索封装（向量检索 + 关键词补召回，Milvus分数写入 metadata["score"]）。

        query_embedding 可由调用方批量计算后传入；向量检索和关键词补召回复用同一向量，
        避免同一个 query 被 embedding 两次。
        """
        if query is None and query_embedding is None:
            raise ValueError("query 和 query_embedding 至少提供一个")

        exclude_expr = self._build_exclude_pks_expr(exclude_pks)

        if query_embedding is None:
            embedding_function = (
                getattr(self.vector_store, "embedding_function", None)
                or getattr(self.vector_store, "embedding_func", None)
            )
            if embedding_function is None:
                raise RuntimeError("向量存储缺少 embedding_func")
            query_embedding = await embedding_function.aembed_query(query)

        # 向量检索（保留Milvus返回的相似度分数；已进入参考文档的chunk在检索侧排除）
        scored_docs = await self.vector_store.asimilarity_search_with_score_by_vector(
            query_embedding, k=top_k, expr=exclude_expr or None
        )
        docs = []
        for doc, score in scored_docs:
            doc.metadata["score"] = score
            docs.append(doc)

        # 尝试将query切分为多个keywords进行过滤（如果query中包含多个词）
        keywords = query.split() if query else []
        keywords = [kw for kw in keywords if kw.strip()]
        if len(keywords) > 1:
            keyword_expr = " OR ".join(
                [f'text like "%{self._escape_like(kw)}%"' for kw in keywords]
            )
            if exclude_expr:
                keyword_expr = f"({keyword_expr}) and {exclude_expr}"
            # 关键词补召回在关键词命中范围内再做一次向量检索，同样按相似度打分；
            # 补召回失败时降级为仅使用向量召回结果
            try:
                keyword_docs = await self.vector_store.asimilarity_search_with_score_by_vector(
                    query_embedding, k=top_k, expr=keyword_expr
                )
            except Exception as e:
                logger.error(f"❌ 关键词补召回失败，仅使用向量召回结果: {e}")
            else:
                for doc, score in keyword_docs:
                    doc.metadata["score"] = score
                    docs.append(doc)

        return docs

    # ============= 工具1: 关键词检索（grep风格）=============

    async def _count_matches(self, filter_expr: str) -> Optional[int]:
        """统计 filter 的真实命中总数；统计失败时返回 None，不影响检索。"""
        try:
            rows = await self.vector_store.aclient.query(
                collection_name=self.vector_store.collection_name,
                filter=filter_expr,
                output_fields=["count(*)"],
            )
        except Exception as e:
            logger.warning(f"⚠️ 统计命中总数失败: {e}")
            return None
        if not rows:
            return 0
        try:
            value = rows[0].get("count(*)", rows[0].get("count", 0))
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _lexical_score(doc: Document, keywords: List[str]) -> float:
        """轻量词法相关性：关键词覆盖率 + 词频 + 文件名命中。"""
        text = (doc.page_content or "").lower()
        file_name = str(doc.metadata.get("fileName", "")).lower()
        if not keywords:
            return 0.0
        matched = 0
        occurrences = 0
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if not keyword_lower:
                continue
            count = text.count(keyword_lower)
            if count:
                matched += 1
                occurrences += count
        coverage = matched / len(keywords)
        filename_bonus = 1.0 if any(k.lower() in file_name for k in keywords if k) else 0.0
        return coverage * 100.0 + math.log1p(occurrences) * 10.0 + filename_bonus * 5.0

    async def _search_by_grep(
            self,
            keywords: List[str],
            match_mode: str = "OR",
            top_k: int = 15,
            document_ids: Optional[List[int]] = None,
            exclude_pks: Optional[set] = None,
    ) -> Dict[str, Any]:
        """关键词检索（grep风格），支持全库检索或指定文档范围。

        Milvus query 不保证无 order_by 时的返回顺序，因此不能直接 limit=top_k：
        先 over-fetch 候选，再用 Python 词法分数排序，并返回真实命中总数。
        """
        keywords = [kw.strip() for kw in keywords if kw and kw.strip()]
        if not keywords:
            raise ValueError("keywords 不能为空")

        scope = "全库" if not document_ids else f"{len(document_ids)}个文件"
        logger.info(
            f"🔍 [1.grep检索] keywords={keywords}, mode={match_mode}, top_k={top_k}, scope={scope}, document_ids={document_ids}")

        keyword_conditions = [f'text like "%{self._escape_like(kw)}%"' for kw in keywords]
        keyword_expr = f" {match_mode} ".join(keyword_conditions)

        if document_ids:
            document_conditions = [f"documentId == {document_id}" for document_id in document_ids]
            document_expr = " OR ".join(document_conditions)
            scope_expr = f'({document_expr}) and ({keyword_expr})'
        else:
            scope_expr = keyword_expr

        exclude_count = len(exclude_pks or ())
        candidate_limit = min(
            max(top_k * KEYWORD_CANDIDATE_FACTOR, top_k + exclude_count),
            KEYWORD_CANDIDATE_MAX,
        )

        matched_total = await self._count_matches(scope_expr)
        docs = await self._milvus_filter(
            filter_expr=scope_expr,
            limit=candidate_limit,
            exclude_pks=exclude_pks
        )
        docs = self._filter_excluded_pks(docs, exclude_pks)

        # 同一 pk 只保留一次，再按词法分数降序；同分时按文档/切片顺序保持稳定。
        docs_by_pk = {}
        for doc in docs:
            pk = doc.metadata.get("pk")
            if pk is not None:
                docs_by_pk.setdefault(pk, doc)
        docs = list(docs_by_pk.values())
        candidate_count = len(docs)
        docs.sort(key=lambda d: (
            d.metadata.get("documentId") or 0,
            d.metadata.get("chunkIndex") or 0,
        ))
        docs.sort(key=lambda d: self._lexical_score(d, keywords), reverse=True)

        docs = docs[:top_k]
        truncated = (
            matched_total > len(docs)
            if matched_total is not None
            else candidate_count >= candidate_limit
        )

        logger.info(
            f"✅ grep检索结果: 返回{len(docs)}条, 命中总数={matched_total}, "
            f"候选={candidate_count}, 截断={truncated}"
        )

        return {
            "results": docs,
            "total_hits": len(docs),
            "matched_total": matched_total,
            "truncated": truncated,
        }

    # ============= 工具2: 按 documentId 获取连续chunk范围 =============

    async def _search_by_document_id_and_chunk_range(
            self,
            document_id: int,
            start_chunk_index: int,
            end_chunk_index: int,
    ) -> Dict[str, Any]:
        """按 documentId 获取连续chunk范围"""
        logger.info(f"🔍 [2.文件chunk范围] documentId='{document_id}', range=[{start_chunk_index}, {end_chunk_index}]")

        if end_chunk_index < start_chunk_index:
            raise ValueError("end_chunk_index 不能小于 start_chunk_index")

        filter_expr = f'documentId == {document_id} and chunkIndex >= {start_chunk_index} and chunkIndex <= {end_chunk_index}'

        limit = end_chunk_index - start_chunk_index + 1
        if limit > 20:
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

    async def _extend_document_chunk_context_window(
            self,
            document_id: int,
            chunk_index: int,
            window_size: int = 2,
    ) -> Dict[str, Any]:
        """围绕某个已命中的chunk，按 documentId 查看前后上下文"""
        logger.info(f"🔍 [3.扩展上下文] documentId='{document_id}', chunk_index={chunk_index}, window={window_size}")

        start_chunk_index = max(0, chunk_index - window_size)
        end_chunk_index = chunk_index + window_size

        return await self._search_by_document_id_and_chunk_range(
            document_id=document_id,
            start_chunk_index=start_chunk_index,
            end_chunk_index=end_chunk_index
        )

    async def read_document_chunks(
            self,
            document_id: int,
            start_chunk_index: int,
            end_chunk_index: int,
    ) -> Dict[str, Any]:
        """按文档ID读取连续chunk范围"""
        limit = end_chunk_index - start_chunk_index + 1
        if limit > 20:
            raise ValueError(f"单次chunk范围不能超过20个，当前为{limit}个")
        docs = await self._milvus_filter(
            filter_expr=(
                f"documentId == {document_id} and "
                f"chunkIndex >= {start_chunk_index} and chunkIndex <= {end_chunk_index}"
            ),
            limit=limit,
        )
        docs.sort(key=lambda doc: doc.metadata.get("chunkIndex", 0))
        return {"results": docs, "total_hits": len(docs)}

    async def expand_document_context(
            self,
            document_id: int,
            chunk_index: int,
            window_size: int = 2,
    ) -> Dict[str, Any]:
        """按文档ID扩展命中chunk的前后上下文"""
        return await self.read_document_chunks(
            document_id=document_id,
            start_chunk_index=max(0, chunk_index - window_size),
            end_chunk_index=chunk_index + window_size,
        )

    # ============= 工具4: 全库语义检索(多query并行) =============

    async def _search_by_multi_queries_in_database(
            self,
            queries: List[str],
            top_k: int = 10,
            exclude_pks: Optional[set] = None,
    ) -> Dict[str, Any]:
        """
        全库语义检索(多query并行)

        流程:
            1. 并行向量检索所有queries（召回阶段，保留Milvus向量分数）
            2. 合并去重（同一chunk保留最高分）
            3. 排除已进入参考文档的chunk
            4. K-Means双簇动态阈值过滤，按向量分数降序返回（top_k为上限）
        """

        queries = [query.strip() for query in queries if query and query.strip()]
        queries = list(dict.fromkeys(queries))
        if not queries:
            raise ValueError("queries 不能为空")

        logger.info(f"🔍 [4.全库语义] queries={queries}, top_k={top_k}")

        docs_by_pk: Dict[Any, Document] = {}

        exclude_count = len(exclude_pks or ())
        retrieval_top_k = (
            max(top_k * 3, 15)
            + min(exclude_count, SEMANTIC_EXCLUDE_OVERFETCH_MAX)
        )

        # 批量计算 query 向量；失败时回退到每条 query 单独 embedding。
        embeddings = None
        embedding_function = (
            getattr(self.vector_store, "embedding_function", None)
            or getattr(self.vector_store, "embedding_func", None)
        )
        if embedding_function is not None:
            try:
                embeddings = await embedding_function.aembed_documents(queries)
                if len(embeddings) != len(queries):
                    logger.warning("⚠️ 批量query向量化数量不一致，回退逐条检索")
                    embeddings = None
            except Exception as e:
                logger.warning(f"⚠️ 批量query向量化失败，回退逐条检索: {e}")

        if embeddings is None:
            tasks = [
                asyncio.create_task(
                    self._vector_search(query=query, top_k=retrieval_top_k, exclude_pks=exclude_pks)
                )
                for query in queries
            ]
        else:
            tasks = [
                asyncio.create_task(
                    self._vector_search(
                        query=query,
                        query_embedding=embedding,
                        top_k=retrieval_top_k,
                        exclude_pks=exclude_pks,
                    )
                )
                for query, embedding in zip(queries, embeddings)
            ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        failed_queries = 0
        for docs in results:
            if isinstance(docs, Exception):
                failed_queries += 1
                logger.warning(f"⚠️ 某个query检索失败: {docs}")
                continue
            for doc in docs:
                pk = doc.metadata.get("pk")
                if not pk:
                    continue
                # 同一chunk可能被多个query或语义/关键词两路同时命中，保留最高分
                existing = docs_by_pk.get(pk)
                if existing is None or doc.metadata.get("score", 0.0) > existing.metadata.get("score", 0.0):
                    docs_by_pk[pk] = doc

        if failed_queries:
            logger.warning(f"⚠️ {failed_queries}/{len(queries)} 个query检索失败")
        if failed_queries == len(queries):
            raise RuntimeError(f"语义检索失败: {len(queries)}个query全部检索失败，请检查Milvus连接或检索表达式")

        all_docs = list(docs_by_pk.values())
        all_docs = self._filter_excluded_pks(all_docs, exclude_pks)
        logger.info(f"📊 并行检索完成: 总计{len(all_docs)}个独立新文档")
        if not all_docs:
            logger.warning("⚠️ 并行检索未找到任何新文档")
            return {
                "results": [],
                "total_hits": 0,
                "kept_before_top_k": 0,
                "top_score": 0.0,
                "threshold": 0.0,
                "truncated": False,
                "failed_queries": failed_queries,
            }

        # 动态阈值过滤：K-Means 双簇按分数动态决定保留数量（top_k 仅作为上限）
        filter_result = filter_grade_threshold(all_docs)
        filtered_docs = filter_result["documents"]
        kept_before_top_k = len(filtered_docs)
        threshold = float(filter_result.get("threshold", 0.0))
        score_source = filtered_docs or all_docs
        top_score = float(max(
            (doc.metadata.get("score", 0.0) for doc in score_source),
            default=0.0,
        ))
        logger.info(
            f"✅ 动态阈值过滤后保留 {kept_before_top_k} 个文档，"
            f"top_score={top_score:.4f}, 阈值: {threshold:.4f}"
        )

        filtered_docs.sort(key=lambda d: d.metadata.get("score", 0.0), reverse=True)
        results = filtered_docs[:top_k]
        truncated = kept_before_top_k > top_k

        return {
            "results": results,
            "total_hits": len(results),
            "kept_before_top_k": kept_before_top_k,
            "top_score": top_score,
            "threshold": threshold,
            "truncated": truncated,
            "failed_queries": failed_queries,
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

        filter_expr = (
            f'fileName like "{self._escape_like(pattern, escape_wildcards=False)}" '
            f'and chunkIndex == 0'
        )

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
            raise

        return {
            "results": all_docs,
            "total_hits": len(all_docs)
        }

    # ============= 工具构建与分发 =============

    def _build_tools(self) -> List[StructuredTool]:
        return [
            StructuredTool(
                name="keyword_search",
                description=(
                    "基于关键词精确匹配检索正文内容，并按关键词覆盖率/词频/文件名做轻量相关性排序。"
                    "【适用场景】已知明确的术语、函数名、配置项、错误码等。"
                    "【特点】精确匹配，无法处理同义词或近义词；返回 matchedTotal 和 truncated 表示范围内总命中数及是否截断。"
                    "【典型用法】搜索'学籍管理规定'、'def train_model'、'MySQL连接超时'等。"
                ),
                args_schema=KeywordSearchInput,
                coroutine=self._search_by_grep,
            ),
            StructuredTool(
                name="read_file_chunks",
                description=(
                    "按 documentId 和chunk范围顺序读取正文内容。"
                    "【适用场景】已知 documentId，需要读取连续段落或章节。"
                    "【优点】按顺序读取，适合连续阅读。"
                    "【限制】单次最多20个chunk，超出需分次调用。"
                ),
                args_schema=ReadFileChunksInput,
                coroutine=self._search_by_document_id_and_chunk_range,
            ),
            StructuredTool(
                name="expand_context",
                description=(
                    "围绕某个已命中的chunk，按 documentId 扩展查看前后上下文。"
                    "【适用场景】已找到关键chunk，需要查看其上下文。"
                    "【限制】只能基于已命中的chunk扩展，不适合大范围通读。"
                ),
                args_schema=ExpandContextInput,
                coroutine=self._extend_document_chunk_context_window,
            ),
            StructuredTool(
                name="semantic_search",
                description=(
                    "全库语义检索：多query并行召回，按向量相似度排序并经K-Means动态阈值过滤。"
                    "【适用场景】概念性、探索性问题，可在无明确关键词时根据问题语义进行检索。"
                    "【特点】能发现语义相关的内容，覆盖面广；返回 topScore、threshold、keptBeforeTopK 和 truncated 供判断结果强弱与是否截断。"
                ),
                args_schema=SemanticSearchInput,
                coroutine=self._search_by_multi_queries_in_database,
            ),
            StructuredTool(
                name="find_files",
                description=(
                    "根据文件名模式查找文件，仅返回元信息（文件名、文档ID、总chunk数），不返回正文。"
                    "【适用场景】不确定精确文件名时，先查找文件列表。"
                    "【后续操作】找到文件后，需用 keyword_search 或 read_file_chunks 读取正文。"
                    "【典型用法】查找所有包含'手册'的文件：pattern='%手册%'。"
                ),
                args_schema=FindFilesInput,
                coroutine=self._list_filename_by_like,
            ),
            StructuredTool(
                name="stop_search",
                description=(
                    "停止检索，表示已获取足够信息来回答用户问题，或继续检索已无价值。"
                    "【触发条件】信息足够、结果无关、无法构造新查询、达到轮次上限。"
                ),
                args_schema=StopSearchInput,
                coroutine=self._stop_search,
            ),
        ]

    def get_tools(self) -> List[StructuredTool]:
        """返回所有工具（含 stop_search）"""
        return self._tools

    async def execute_tool(self, tool_name: str, args: dict, exclude_pks: Optional[set] = None) -> Dict[str, Any]:
        """
        执行指定的检索工具（不含 stop_search）

        Args:
            tool_name: 工具名称
            args: 工具参数
            exclude_pks: 已进入参考文档的chunk pk集合，仅用于搜索类工具过滤重复chunk

        Returns:
            {"results": List[Document], "total_hits": int}
        """
        logger.info(f"🔧 执行工具: {tool_name}, 参数: {args}")
        if tool_name not in self._tool_map:
            raise ValueError(f"未知工具: {tool_name}")
        tool = self._tool_map[tool_name]
        if tool_name == "keyword_search":
            validated_args = tool.args_schema.model_validate(args).model_dump()
            return await self._search_by_grep(**validated_args, exclude_pks=exclude_pks)
        if tool_name == "semantic_search":
            validated_args = tool.args_schema.model_validate(args).model_dump()
            return await self._search_by_multi_queries_in_database(**validated_args, exclude_pks=exclude_pks)
        return await tool.ainvoke(args)

    # ============= 停止工具 =============

    @staticmethod
    async def _stop_search(reason: str) -> str:
        """停止检索"""
        return f"检索已停止: {reason}"

