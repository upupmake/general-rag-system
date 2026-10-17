import asyncio
import importlib.util
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "mq" / "connection.py"


class DeliveryMode:
    NOT_PERSISTENT = 1
    PERSISTENT = 2


class FakeMessage:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeExchange:
    def __init__(self):
        self.published = []

    async def publish(self, message, routing_key):
        self.published.append((message, routing_key))


class FakeChannel:
    def __init__(self):
        self.exchange = FakeExchange()

    async def get_exchange(self, name):
        return self.exchange


def _load_module_under_test():
    saved = {}
    stub_names = ("aio_pika", "aio_pika.abc")
    for name in stub_names:
        saved[name] = sys.modules.get(name)

    aio_pika = types.ModuleType("aio_pika")
    aio_pika.abc = types.ModuleType("aio_pika.abc")
    aio_pika.abc.AbstractIncomingMessage = object
    aio_pika.DeliveryMode = DeliveryMode
    aio_pika.Message = FakeMessage
    aio_pika.RobustConnection = object
    aio_pika.RobustChannel = object

    os.environ.setdefault("RABBITMQ_HOST", "localhost")
    os.environ.setdefault("RABBITMQ_PORT", "5672")
    os.environ.setdefault("RABBITMQ_USERNAME", "guest")
    os.environ.setdefault("RABBITMQ_PASSWORD", "guest")

    try:
        sys.modules["aio_pika"] = aio_pika
        sys.modules["aio_pika.abc"] = aio_pika.abc
        spec = importlib.util.spec_from_file_location(
            "rag_llm_mq_connection_under_test",
            MODULE_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


MODULE = _load_module_under_test()


def test_publish_uses_persistent_delivery_mode():
    client = MODULE.AsyncRabbitMQClient()
    channel = FakeChannel()
    client.channel = channel

    asyncio.run(client.publish("test.exchange", "test.key", {"documentId": 1}))

    message, routing_key = channel.exchange.published[0]
    assert routing_key == "test.key"
    assert message.kwargs["delivery_mode"] == DeliveryMode.PERSISTENT
    assert message.kwargs["content_type"] == "application/json"
