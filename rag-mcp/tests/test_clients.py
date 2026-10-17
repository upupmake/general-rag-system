import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_mcp import clients


def test_http_client_is_reused_and_closed_on_shutdown():
    async def run():
        first = clients._get_http_client()
        second = clients._get_http_client()
        assert first is second

        await clients.close_http_client()
        return first

    client = asyncio.run(run())

    assert client.is_closed
    assert clients._http_client is None


def test_rag_retrieve_uses_configured_timeout(monkeypatch):
    captured = {}

    async def fake_request(method, url, *, headers=None, json=None, timeout=30.0):
        captured.update({
            "method": method,
            "url": url,
            "json": json,
            "timeout": timeout,
        })
        return {"ok": True}

    monkeypatch.setattr(clients, "_request", fake_request)

    result = asyncio.run(clients.rag_retrieve("/retrieval/semantic", {"queries": ["q"]}))

    assert result == {"ok": True}
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/retrieval/semantic")
    assert captured["timeout"] == clients.MCP_RETRIEVAL_TIMEOUT
