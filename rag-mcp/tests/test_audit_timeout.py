import asyncio
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _install_aio_pika_stub():
    aio_pika = types.ModuleType("aio_pika")
    aio_pika.abc = types.ModuleType("aio_pika.abc")
    aio_pika.abc.AbstractIncomingMessage = object

    class DeliveryMode:
        NOT_PERSISTENT = 1
        PERSISTENT = 2

    class Message:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class ExchangeType:
        DIRECT = "direct"

    aio_pika.DeliveryMode = DeliveryMode
    aio_pika.Message = Message
    aio_pika.ExchangeType = ExchangeType
    aio_pika.RobustConnection = object
    aio_pika.RobustChannel = object

    async def connect_robust(**kwargs):
        raise RuntimeError("RabbitMQ is not available in tests")

    aio_pika.connect_robust = connect_robust

    sys.modules.setdefault("aio_pika", aio_pika)
    sys.modules.setdefault("aio_pika.abc", aio_pika.abc)


_install_aio_pika_stub()

from rag_mcp import server


def test_audit_publish_timeout_does_not_block_tool_result(monkeypatch):
    async def slow_publish(message):
        await asyncio.sleep(10)

    monkeypatch.setattr(server, "publish_tool_log", slow_publish)
    monkeypatch.setattr(server, "MCP_AUDIT_PUBLISH_TIMEOUT", 0.05)

    started_at = time.monotonic()
    asyncio.run(server._publish_audit_safely({}, "test_tool"))
    elapsed = time.monotonic() - started_at

    assert elapsed < 1


def test_lifespan_closes_the_shared_http_client():
    from rag_mcp import clients

    async def run():
        client = clients._get_http_client()
        async with server._app_lifespan(server.mcp):
            pass
        return client

    client = asyncio.run(run())

    assert client.is_closed
    assert clients._http_client is None
