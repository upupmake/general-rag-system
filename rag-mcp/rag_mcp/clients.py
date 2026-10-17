from typing import Any

import httpx

from rag_mcp.config import JAVA_OPENAPI_BASE_URL, RAG_LLM_BASE_URL


class ServiceError(RuntimeError):
    pass


_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _http_client


async def close_http_client() -> None:
    global _http_client
    client = _http_client
    _http_client = None
    if client is not None and not client.is_closed:
        await client.aclose()


async def _request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        response = await _get_http_client().request(
            method,
            url,
            headers=headers,
            json=json,
            timeout=30,
        )
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


async def java_post(path: str, access_key: str, body: dict[str, Any]) -> Any:
    payload = await _request(
        "POST",
        f"{JAVA_OPENAPI_BASE_URL}{path}",
        headers={"Authorization": f"Bearer {access_key}"},
        json=body,
    )
    if payload.get("code") != 200:
        raise ServiceError(payload.get("message") or "请求处理失败")
    return payload.get("data")


async def java_upload_file(path: str, access_key: str, file_name: str, content: bytes) -> Any:
    try:
        response = await _get_http_client().post(
            f"{JAVA_OPENAPI_BASE_URL}{path}",
            headers={"Authorization": f"Bearer {access_key}"},
            files={"files": (file_name, content, "application/octet-stream")},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ServiceError("下游文档服务当前不可用") from exc
    if payload.get("code") != 200:
        raise ServiceError(payload.get("message") or "文档上传失败")
    return payload.get("data")


async def java_delete(path: str, access_key: str) -> Any:
    payload = await _request(
        "DELETE",
        f"{JAVA_OPENAPI_BASE_URL}{path}",
        headers={"Authorization": f"Bearer {access_key}"},
    )
    if payload.get("code") != 200:
        raise ServiceError(payload.get("message") or "文档删除失败")
    return payload.get("data")


async def verify_access_key(access_key: str) -> dict[str, int]:
    identity = await java_get("/auth/verify", access_key)
    if not identity or not identity.get("valid"):
        raise ServiceError("Access Key 无效")
    if identity.get("userId") is None or identity.get("accessKeyId") is None:
        raise ServiceError("Access Key 鉴权信息不完整")
    return {
        "userId": int(identity["userId"]),
        "accessKeyId": int(identity["accessKeyId"]),
    }


async def authorize_knowledge_base(knowledge_base_id: int, access_key: str) -> dict[str, int | str | None]:
    access = await java_get(
        f"/knowledge-bases/{knowledge_base_id}/access",
        access_key,
    )
    if not access or not access.get("accessible") or access.get("ownerUserId") is None:
        raise ServiceError("没有权限访问该知识库")
    return {
        "ownerUserId": int(access["ownerUserId"]),
        "accessSource": access.get("accessSource"),
    }


async def rag_retrieve(path: str, body: dict[str, Any]) -> dict[str, Any]:
    return await _request("POST", f"{RAG_LLM_BASE_URL}{path}", json=body)
