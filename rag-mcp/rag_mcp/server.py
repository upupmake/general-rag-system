import logging
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import AccessToken, TokenVerifier
from fastmcp.server.dependencies import get_access_token
from pydantic import Field

from rag_mcp.audit import new_invocation_id, now_millis, publish_tool_log
from rag_mcp.clients import (
    ServiceError,
    authorize_knowledge_base,
    java_get,
    rag_retrieve,
    verify_access_key,
)
from rag_mcp.config import MCP_HOST, MCP_PORT

logger = logging.getLogger(__name__)


class AccessKeyVerifier(TokenVerifier):
    def __init__(self) -> None:
        super().__init__()

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token.startswith("grs_ak_"):
            return None
        try:
            identity = await verify_access_key(token)
        except ServiceError:
            return None
        return AccessToken(
            token=token,
            client_id="access-key",
            scopes=[],
            claims=identity,
        )


mcp = FastMCP(
    "General RAG Retrieval",
    instructions=(
        "提供可组合的知识库检索工具。先调用 list_knowledge_bases 获取 knowledge_base_id，"
        "再根据任务选择关键词检索、语义检索、文件查找、连续片段读取或上下文扩展。"
    ),
    auth=AccessKeyVerifier(),
)


def _access_token() -> AccessToken:
    token = get_access_token()
    if token is None:
        raise ToolError("缺少 Access Key")
    return token


def _access_key() -> str:
    return _access_token().token


def _identity() -> tuple[int, int]:
    claims = _access_token().claims
    user_id = claims.get("userId")
    access_key_id = claims.get("accessKeyId")
    if user_id is None or access_key_id is None:
        raise ToolError("Access Key 鉴权信息不完整")
    return int(user_id), int(access_key_id)


async def _target(knowledge_base_id: int) -> dict[str, Any]:
    try:
        access = await authorize_knowledge_base(knowledge_base_id, _access_key())
    except ServiceError as exc:
        raise ToolError(str(exc)) from exc
    return {
        "ownerUserId": access["ownerUserId"],
        "knowledgeBaseId": knowledge_base_id,
    }


async def _retrieve(path: str, body: dict) -> dict:
    try:
        return await rag_retrieve(path, body)
    except ServiceError as exc:
        raise ToolError(str(exc)) from exc


def _result_summary(result: Any) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    summary: dict[str, Any] = {}
    for key in ("knowledgeBaseId", "totalHits"):
        if key in result:
            summary[key] = result[key]
    if isinstance(result.get("results"), list):
        items = []
        for item in result["results"]:
            if not isinstance(item, dict):
                continue
            keys = (
                ("chunkId", "documentId", "fileName", "chunkIndex", "totalChunks", "score", "scoreType")
                if "content" in item
                else ("documentId", "fileName", "totalChunks")
            )
            items.append({key: item[key] for key in keys if key in item})
        summary["returnedCount"] = len(result["results"])
        summary["items"] = items
    else:
        for key in ("owned", "workspaceShared", "invited", "public"):
            value = result.get(key)
            if isinstance(value, list):
                summary[key + "Count"] = (
                    sum(
                        len(group.get("knowledgeBases", []))
                        for group in value
                        if isinstance(group, dict) and isinstance(group.get("knowledgeBases"), list)
                    )
                    if key == "workspaceShared"
                    else len(value)
                )
    return summary


async def _audit_tool(
    tool_name: str,
    request_summary: dict[str, Any],
    operation: Callable[[], Awaitable[dict]],
    knowledge_base_id: int | None = None,
    document_id: int | None = None,
) -> dict:
    invocation_id = new_invocation_id()
    started_at = now_millis()
    user_id, access_key_id = _identity()
    result = None
    status = "SUCCESS"
    error_message = None
    try:
        result = await operation()
        return result
    except (ToolError, ServiceError) as exc:
        status = "FAIL"
        error_message = str(exc)
        raise ToolError(str(exc)) from exc
    except Exception as exc:
        status = "FAIL"
        error_message = "检索服务当前不可用"
        logger.exception("MCP tool failed: %s", tool_name)
        raise ToolError(error_message) from exc
    finally:
        message = {
            "invocationId": invocation_id,
            "userId": user_id,
            "accessKeyId": access_key_id,
            "toolName": tool_name,
            "knowledgeBaseId": knowledge_base_id,
            "documentId": document_id,
            "requestSummary": request_summary,
            "resultSummary": _result_summary(result),
            "status": status,
            "errorMessage": error_message,
            "durationMs": now_millis() - started_at,
            "createdAt": started_at,
        }
        await _publish_audit_safely(message, tool_name)


async def _publish_audit_safely(message: dict[str, Any], tool_name: str) -> None:
    try:
        await publish_tool_log(message)
    except Exception:
        logger.exception("Failed to publish MCP tool log: %s", tool_name)


@mcp.tool
async def list_knowledge_bases() -> dict:
    """按个人创建、各工作空间共享、受邀请和公开分类列出当前用户可访问的知识库。"""
    return await _audit_tool(
        "list_knowledge_bases",
        {},
        lambda: java_get("/knowledge-bases", _access_key()),
    )


@mcp.tool
async def search_knowledge_base_by_keywords(
    knowledge_base_id: Annotated[int, Field(description="由 list_knowledge_bases 返回的知识库 ID。")],
    keywords: Annotated[
        list[str],
        Field(description="用于精确匹配的关键词列表，至少提供一个术语、错误码或原文短语。", min_length=1),
    ],
    match_mode: Annotated[
        Literal["AND", "OR"],
        Field(description="关键词组合方式：AND 要求同时匹配全部关键词，OR 匹配任一关键词。"),
    ] = "OR",
    top_k: Annotated[int, Field(description="最多返回的匹配片段数，取值范围 1 到 50。", ge=1, le=50)] = 15,
    file_names: Annotated[
        list[str] | None,
        Field(description="可选的文件名列表；提供后仅在这些文件中检索。"),
    ] = None,
) -> dict:
    """使用明确术语、配置项、错误码或原文短语在知识库中执行关键词精确检索。"""
    request_summary = {"keywords": keywords, "matchMode": match_mode, "topK": top_k, "fileNames": file_names}

    async def operation() -> dict:
        body = await _target(knowledge_base_id)
        body.update(request_summary)
        return await _retrieve("/retrieval/keywords", body)

    return await _audit_tool("search_knowledge_base_by_keywords", request_summary, operation, knowledge_base_id)


@mcp.tool
async def search_knowledge_base_by_semantics(
    knowledge_base_id: Annotated[int, Field(description="由 list_knowledge_bases 返回的知识库 ID。")],
    queries: Annotated[
        list[str],
        Field(description="用于扩大召回范围的语义查询列表，提供 1 到 10 条不同表述。", min_length=1, max_length=10),
    ],
    relevance_query: Annotated[
        str,
        Field(description="用于 Rerank 和最终相关性判断的完整问题或检索目标。", min_length=1),
    ],
    top_k: Annotated[int, Field(description="最多返回的相关片段数，取值范围 1 到 50。", ge=1, le=50)] = 10,
    relevance_threshold: Annotated[
        float,
        Field(description="最低相关性阈值，取值范围 0 到 1；值越高，过滤越严格。", ge=0.0, le=1.0),
    ] = 0.3,
) -> dict:
    """执行多查询语义召回、关键词辅助召回、去重、Rerank 和相关性过滤。"""
    request_summary = {
        "queries": queries,
        "relevanceQuery": relevance_query,
        "topK": top_k,
        "relevanceThreshold": relevance_threshold,
    }

    async def operation() -> dict:
        body = await _target(knowledge_base_id)
        body.update(request_summary)
        return await _retrieve("/retrieval/semantic", body)

    return await _audit_tool("search_knowledge_base_by_semantics", request_summary, operation, knowledge_base_id)


@mcp.tool
async def find_knowledge_base_files(
    knowledge_base_id: Annotated[int, Field(description="由 list_knowledge_bases 返回的知识库 ID。")],
    name_pattern: Annotated[
        str,
        Field(description="文件名匹配模式，可使用 % 匹配任意字符，例如 %.md。", min_length=1),
    ],
    offset: Annotated[int, Field(description="分页起始偏移量，从 0 开始。", ge=0)] = 0,
    limit: Annotated[int, Field(description="本次最多返回的文件数，取值范围 1 到 100。", ge=1, le=100)] = 30,
) -> dict:
    """按文件名模式查找知识库文件，只返回 documentId、文件名和总片段数。使用 % 作为通配符。"""
    request_summary = {"namePattern": name_pattern, "offset": offset, "limit": limit}

    async def operation() -> dict:
        body = await _target(knowledge_base_id)
        body.update(request_summary)
        return await _retrieve("/retrieval/files", body)

    return await _audit_tool("find_knowledge_base_files", request_summary, operation, knowledge_base_id)


@mcp.tool
async def read_knowledge_base_chunks(
    knowledge_base_id: Annotated[int, Field(description="由 list_knowledge_bases 返回的知识库 ID。")],
    document_id: Annotated[int, Field(description="由 find_knowledge_base_files 或检索结果返回的 documentId。")],
    start_chunk_index: Annotated[int, Field(description="要读取的起始片段索引，包含该片段且从 0 开始。", ge=0)],
    end_chunk_index: Annotated[int, Field(description="要读取的结束片段索引，包含该片段；单次最多读取 20 个片段。", ge=0)],
) -> dict:
    """按 documentId 顺序读取一段连续原文，单次最多读取20个片段。"""
    request_summary = {"startChunkIndex": start_chunk_index, "endChunkIndex": end_chunk_index}

    async def operation() -> dict:
        if end_chunk_index < start_chunk_index:
            raise ToolError("结束片段索引不能小于起始索引")
        if end_chunk_index - start_chunk_index + 1 > 20:
            raise ToolError("单次最多读取20个片段")
        body = await _target(knowledge_base_id)
        body.update({
            "documentId": document_id,
            "startChunkIndex": start_chunk_index,
            "endChunkIndex": end_chunk_index,
        })
        return await _retrieve("/retrieval/chunks", body)

    return await _audit_tool("read_knowledge_base_chunks", request_summary, operation, knowledge_base_id, document_id)


@mcp.tool
async def expand_knowledge_base_context(
    knowledge_base_id: Annotated[int, Field(description="由 list_knowledge_bases 返回的知识库 ID。")],
    document_id: Annotated[int, Field(description="检索命中片段所属的 documentId。")],
    chunk_index: Annotated[int, Field(description="作为上下文中心的命中片段索引，从 0 开始。", ge=0)],
    window_size: Annotated[
        int,
        Field(description="中心片段前后各扩展的片段数，取值范围 0 到 9。", ge=0, le=9),
    ] = 2,
) -> dict:
    """围绕一个已命中的文档片段读取前后上下文。"""
    request_summary = {"chunkIndex": chunk_index, "windowSize": window_size}

    async def operation() -> dict:
        body = await _target(knowledge_base_id)
        body.update({
            "documentId": document_id,
            "chunkIndex": chunk_index,
            "windowSize": window_size,
        })
        return await _retrieve("/retrieval/context", body)

    return await _audit_tool("expand_knowledge_base_context", request_summary, operation, knowledge_base_id, document_id)


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
