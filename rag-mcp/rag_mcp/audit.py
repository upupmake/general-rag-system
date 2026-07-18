import asyncio
import json
import logging
import time
from typing import Any
from uuid import uuid4

import aio_pika
from aio_pika import DeliveryMode, Message

from rag_mcp.config import (
    MCP_AUDIT_EXCHANGE,
    MCP_AUDIT_ROUTING_KEY,
    RABBITMQ_HOST,
    RABBITMQ_PASSWORD,
    RABBITMQ_PORT,
    RABBITMQ_USERNAME,
)

logger = logging.getLogger(__name__)

_connection: aio_pika.RobustConnection | None = None
_channel: aio_pika.RobustChannel | None = None


async def publish_tool_log(message: dict[str, Any]) -> None:
    global _connection, _channel
    host = RABBITMQ_HOST
    port = RABBITMQ_PORT
    username = RABBITMQ_USERNAME
    password = RABBITMQ_PASSWORD
    if not host or not username or not password:
        raise RuntimeError("RabbitMQ audit publisher is not configured")

    for attempt in range(3):
        try:
            if _connection is None or _connection.is_closed or _channel is None or _channel.is_closed:
                _connection = await aio_pika.connect_robust(
                    host=host,
                    port=port,
                    login=username,
                    password=password,
                    heartbeat=60,
                    reconnect_interval=5,
                )
                _channel = await _connection.channel(publisher_confirms=True)
                await _channel.set_qos(prefetch_count=10)

            exchange = await _channel.declare_exchange(
                MCP_AUDIT_EXCHANGE,
                aio_pika.ExchangeType.DIRECT,
                durable=True,
            )
            await exchange.publish(
                Message(
                    body=json.dumps(message, ensure_ascii=False).encode("utf-8"),
                    content_type="application/json",
                    delivery_mode=DeliveryMode.PERSISTENT,
                ),
                routing_key=MCP_AUDIT_ROUTING_KEY,
            )
            return
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(2 ** attempt)


def new_invocation_id() -> str:
    return str(uuid4())


def now_millis() -> int:
    return int(time.time() * 1000)
