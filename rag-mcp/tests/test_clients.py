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
