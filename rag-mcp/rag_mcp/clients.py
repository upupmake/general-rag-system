from typing import Any

import httpx

from rag_mcp.config import JAVA_OPENAPI_BASE_URL, RAG_LLM_BASE_URL


class ServiceError(RuntimeError):
    pass


async def _request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.request(method, url, headers=headers, json=json)
            response.raise_for_status()
            return response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ServiceError("下游检索服务当前不可用") from exc


async def java_get(path: str, access_key: str) -> Any:
    payload = await _request(
        "GET",
        f"{JAVA_OPENAPI_BASE_URL}{path}",
        headers={"Authorization": f"Bearer {access_key}"},
    )
    if payload.get("code") != 200:
        raise ServiceError(payload.get("message") or "Access Key 无效")
    return payload.get("data")


async def authorize_knowledge_base(knowledge_base_id: int, access_key: str) -> int:
    access = await java_get(
        f"/knowledge-bases/{knowledge_base_id}/access",
        access_key,
    )
    if not access or not access.get("accessible") or access.get("ownerUserId") is None:
        raise ServiceError("没有权限访问该知识库")
    return int(access["ownerUserId"])


async def rag_retrieve(path: str, body: dict[str, Any]) -> dict[str, Any]:
    return await _request("POST", f"{RAG_LLM_BASE_URL}{path}", json=body)
