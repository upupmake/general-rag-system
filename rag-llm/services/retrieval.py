import os
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agentic_rag_toolkit import RetrievalToolkit
from milvus_utils import MilvusClientManager
from utils import get_embedding_instance

retrieval_service = APIRouter(prefix="/retrieval", tags=["retrieval"])


class RetrievalTarget(BaseModel):
    ownerUserId: int
    knowledgeBaseId: int


class KeywordSearchRequest(RetrievalTarget):
    keywords: list[str] = Field(min_length=1)
    matchMode: Literal["AND", "OR"] = "OR"
    topK: int = Field(default=15, ge=1, le=50)
    documentIds: Optional[list[int]] = None


class SemanticSearchRequest(RetrievalTarget):
    queries: list[str] = Field(min_length=1, max_length=10)
    relevanceQuery: str = Field(min_length=1)
    topK: int = Field(default=10, ge=1, le=50)
    relevanceThreshold: float = Field(default=0.3, ge=0.0, le=1.0)


class FindFilesRequest(RetrievalTarget):
    namePattern: str = Field(min_length=1)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=30, ge=1, le=100)


class ReadChunksRequest(RetrievalTarget):
    documentId: int
    startChunkIndex: int = Field(ge=0)
    endChunkIndex: int = Field(ge=0)


class ExpandContextRequest(RetrievalTarget):
    documentId: int
    chunkIndex: int = Field(ge=0)
    windowSize: int = Field(default=2, ge=0, le=9)


async def _get_toolkit(owner_user_id: int, knowledge_base_id: int) -> RetrievalToolkit:
    embeddings = get_embedding_instance({"name": "text-embedding-v4", "provider": "qwen"})
    vector_store = await MilvusClientManager.get_instance(
        owner_user_id,
        knowledge_base_id,
        os.environ.get("MILVUS_URI"),
        os.environ.get("MILVUS_TOKEN"),
        embeddings,
    )
    if vector_store is None:
        raise HTTPException(status_code=503, detail="无法连接到知识库")
    return RetrievalToolkit(
        vector_store,
        vector_store.as_retriever(search_kwargs={"k": 10}),
    )


def _format_chunks(knowledge_base_id: int, result: dict[str, Any]) -> dict[str, Any]:
    chunks = []
    for document in result["results"]:
        metadata = document.metadata
        score = metadata.get("rerank_score")
        chunks.append({
            "chunkId": metadata.get("pk"),
            "documentId": metadata.get("documentId"),
            "fileName": metadata.get("fileName"),
            "chunkIndex": metadata.get("chunkIndex"),
            "totalChunks": (
                metadata.get("maxChunkIndex") + 1
                if metadata.get("maxChunkIndex") is not None
                else None
            ),
            "content": document.page_content,
            "score": score,
            "scoreType": "rerank" if score is not None else None,
        })
    return {
        "knowledgeBaseId": knowledge_base_id,
        "results": chunks,
        "totalHits": result["total_hits"],
    }


@retrieval_service.post("/keywords")
async def search_by_keywords(body: KeywordSearchRequest) -> dict[str, Any]:
    toolkit = await _get_toolkit(body.ownerUserId, body.knowledgeBaseId)
    result = await toolkit.execute_tool("keyword_search", {
        "keywords": body.keywords,
        "match_mode": body.matchMode,
        "top_k": body.topK,
        "document_ids": body.documentIds,
    })
    return _format_chunks(body.knowledgeBaseId, result)


@retrieval_service.post("/semantic")
async def search_by_semantics(body: SemanticSearchRequest) -> dict[str, Any]:
    toolkit = await _get_toolkit(body.ownerUserId, body.knowledgeBaseId)
    result = await toolkit.execute_tool("semantic_search", {
        "queries": body.queries,
        "grade_query": body.relevanceQuery,
        "top_k": body.topK,
        "grade_score_threshold": body.relevanceThreshold,
    })
    return _format_chunks(body.knowledgeBaseId, result)


@retrieval_service.post("/files")
async def find_files(body: FindFilesRequest) -> dict[str, Any]:
    toolkit = await _get_toolkit(body.ownerUserId, body.knowledgeBaseId)
    result = await toolkit.execute_tool("find_files", {
        "pattern": body.namePattern,
        "offset": body.offset,
        "limit": body.limit,
    })
    files = [{
        "documentId": document.metadata.get("documentId"),
        "fileName": document.metadata.get("fileName"),
        "totalChunks": (
            document.metadata.get("maxChunkIndex") + 1
            if document.metadata.get("maxChunkIndex") is not None
            else None
        ),
    } for document in result["results"]]
    return {
        "knowledgeBaseId": body.knowledgeBaseId,
        "results": files,
        "totalHits": result["total_hits"],
    }


@retrieval_service.post("/chunks")
async def read_chunks(body: ReadChunksRequest) -> dict[str, Any]:
    if body.endChunkIndex < body.startChunkIndex:
        raise HTTPException(status_code=400, detail="结束chunk索引不能小于起始索引")
    toolkit = await _get_toolkit(body.ownerUserId, body.knowledgeBaseId)
    result = await toolkit.read_document_chunks(
        body.documentId,
        body.startChunkIndex,
        body.endChunkIndex,
    )
    return _format_chunks(body.knowledgeBaseId, result)


@retrieval_service.post("/context")
async def expand_context(body: ExpandContextRequest) -> dict[str, Any]:
    toolkit = await _get_toolkit(body.ownerUserId, body.knowledgeBaseId)
    result = await toolkit.expand_document_context(
        body.documentId,
        body.chunkIndex,
        body.windowSize,
    )
    return _format_chunks(body.knowledgeBaseId, result)
