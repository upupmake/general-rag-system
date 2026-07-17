from typing import Annotated, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import AccessToken, TokenVerifier
from fastmcp.server.dependencies import get_access_token
from pydantic import Field

from rag_mcp.clients import (
    ServiceError,
    authorize_knowledge_base,
    java_get,
    rag_retrieve,
)
from rag_mcp.config import MCP_HOST, MCP_PORT, MCP_PUBLIC_URL


class AccessKeyVerifier(TokenVerifier):
    def __init__(self) -> None:
        super().__init__(base_url=MCP_PUBLIC_URL)

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token.startswith("grs_ak_"):
            return None
        try:
            await java_get("/auth/verify", token)
        except ServiceError:
            return None
        return AccessToken(token=token, client_id="access-key", scopes=[])


mcp = FastMCP(
    "General RAG Retrieval",
    instructions=(
        "提供可组合的知识库检索工具。先调用 list_knowledge_bases 获取 knowledge_base_id，"
        "再根据任务选择关键词检索、语义检索、文件查找、连续片段读取或上下文扩展。"
    ),
    auth=AccessKeyVerifier(),
)


def _access_key() -> str:
    token = get_access_token()
    if token is None:
        raise ToolError("缺少 Access Key")
    return token.token


async def _target(knowledge_base_id: int) -> dict[str, int]:
    try:
        owner_user_id = await authorize_knowledge_base(knowledge_base_id, _access_key())
    except ServiceError as exc:
        raise ToolError(str(exc)) from exc
    return {
        "ownerUserId": owner_user_id,
        "knowledgeBaseId": knowledge_base_id,
    }


async def _retrieve(path: str, body: dict) -> dict:
    try:
        return await rag_retrieve(path, body)
    except ServiceError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
async def list_knowledge_bases() -> dict:
    """按个人创建、各工作空间共享、受邀请和公开分类列出当前用户可访问的知识库。"""
    try:
        return await java_get("/knowledge-bases", _access_key())
    except ServiceError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
async def search_knowledge_base_by_keywords(
    knowledge_base_id: int,
    keywords: Annotated[list[str], Field(min_length=1)],
    match_mode: Literal["AND", "OR"] = "OR",
    top_k: Annotated[int, Field(ge=1, le=50)] = 15,
    file_names: list[str] | None = None,
) -> dict:
    """使用明确术语、配置项、错误码或原文短语在知识库中执行关键词精确检索。"""
    body = await _target(knowledge_base_id)
    body.update({
        "keywords": keywords,
        "matchMode": match_mode,
        "topK": top_k,
        "fileNames": file_names,
    })
    return await _retrieve("/retrieval/keywords", body)


@mcp.tool
async def search_knowledge_base_by_semantics(
    knowledge_base_id: int,
    queries: Annotated[list[str], Field(min_length=1, max_length=10)],
    relevance_query: Annotated[str, Field(min_length=1)],
    top_k: Annotated[int, Field(ge=1, le=50)] = 10,
    relevance_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.3,
) -> dict:
    """执行多查询语义召回、关键词辅助召回、去重、Rerank 和相关性过滤。"""
    body = await _target(knowledge_base_id)
    body.update({
        "queries": queries,
        "relevanceQuery": relevance_query,
        "topK": top_k,
        "relevanceThreshold": relevance_threshold,
    })
    return await _retrieve("/retrieval/semantic", body)


@mcp.tool
async def find_knowledge_base_files(
    knowledge_base_id: int,
    name_pattern: Annotated[str, Field(min_length=1)],
    offset: Annotated[int, Field(ge=0)] = 0,
    limit: Annotated[int, Field(ge=1, le=100)] = 30,
) -> dict:
    """按文件名模式查找知识库文件，只返回 documentId、文件名和总片段数。使用 % 作为通配符。"""
    body = await _target(knowledge_base_id)
    body.update({"namePattern": name_pattern, "offset": offset, "limit": limit})
    return await _retrieve("/retrieval/files", body)


@mcp.tool
async def read_knowledge_base_chunks(
    knowledge_base_id: int,
    document_id: int,
    start_chunk_index: Annotated[int, Field(ge=0)],
    end_chunk_index: Annotated[int, Field(ge=0)],
) -> dict:
    """按 documentId 顺序读取一段连续原文，单次最多读取20个片段。"""
    if end_chunk_index < start_chunk_index:
        raise ToolError("结束片段索引不能小于起始片段索引")
    if end_chunk_index - start_chunk_index + 1 > 20:
        raise ToolError("单次最多读取20个片段")
    body = await _target(knowledge_base_id)
    body.update({
        "documentId": document_id,
        "startChunkIndex": start_chunk_index,
        "endChunkIndex": end_chunk_index,
    })
    return await _retrieve("/retrieval/chunks", body)


@mcp.tool
async def expand_knowledge_base_context(
    knowledge_base_id: int,
    document_id: int,
    chunk_index: Annotated[int, Field(ge=0)],
    window_size: Annotated[int, Field(ge=0, le=9)] = 2,
) -> dict:
    """围绕一个已命中的文档片段读取前后上下文。"""
    body = await _target(knowledge_base_id)
    body.update({
        "documentId": document_id,
        "chunkIndex": chunk_index,
        "windowSize": window_size,
    })
    return await _retrieve("/retrieval/context", body)


app = mcp.http_app(path="/mcp", stateless_http=True, json_response=True)


def main() -> None:
    mcp.run(
        transport="http",
        host=MCP_HOST,
        port=MCP_PORT,
        path="/mcp",
        stateless_http=True,
        json_response=True,
    )


if __name__ == "__main__":
    main()
