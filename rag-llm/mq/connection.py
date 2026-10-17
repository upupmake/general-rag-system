import json
import logging
import os
from typing import Optional, Callable, Awaitable, Any, Union, Dict

import aio_pika
from aio_pika import DeliveryMode
from aio_pika.abc import AbstractIncomingMessage

logger = logging.getLogger(__name__)


class AsyncRabbitMQError(Exception):
    """自定义 RabbitMQ 客户端异常"""
    pass


class AsyncRabbitMQClient:
    def __init__(
            self,
            prefetch_count: int = 2,
            heartbeat: int = 60,
            reconnect_interval: int = 5,
    ):
        self.host: Optional[str] = os.environ.get('RABBITMQ_HOST')
        port_str: Optional[str] = os.environ.get('RABBITMQ_PORT')
        self.username: Optional[str] = os.environ.get('RABBITMQ_USERNAME')
        self.password: Optional[str] = os.environ.get('RABBITMQ_PASSWORD')

        if not all([self.host, port_str, self.username, self.password]):
            raise AsyncRabbitMQError(
                "RabbitMQ connection details missing in environment variables (RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_USERNAME, RABBITMQ_PASSWORD)"
            )
        try:
            self.port: int = int(port_str)
        except ValueError:
            raise AsyncRabbitMQError(f"Invalid RABBITMQ_PORT: '{port_str}'. Must be an integer.")

        self.prefetch_count: int = prefetch_count
        self.heartbeat: int = heartbeat
        self.reconnect_interval: int = reconnect_interval
        self.connection: Optional[aio_pika.RobustConnection] = None
        self.channel: Optional[aio_pika.RobustChannel] = None

    async def connect(self):
        """连接到 RabbitMQ 服务器"""
        try:
            self.connection = await aio_pika.connect_robust(
                host=self.host,
                port=self.port,
                login=self.username,
                password=self.password,
                heartbeat=self.heartbeat,
                reconnect_interval=self.reconnect_interval,
                connection_attempts=3,
                retry_delay=2,
            )
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=self.prefetch_count)
            logger.info("Connected to RabbitMQ server.")
        except Exception as e:
            raise AsyncRabbitMQError(f"Failed to connect to RabbitMQ: {e}")
        return self

    async def __aenter__(self):
        """异步上下文管理器进入时连接到 RabbitMQ"""
        if not self.connection or self.connection.is_closed:
            await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出时关闭连接"""
        await self.close()
        if exc_type:
            logger.error(f"Exception occurred: {exc_val}")

    async def close(self):
        """异步关闭通道和连接"""
        try:
            if self.channel and not self.channel.is_closed:
                await self.channel.close()
                logger.info("RabbitMQ channel closed.")
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
                logger.info("RabbitMQ connection closed.")
        except Exception as e:
            logger.error(f"Error closing RabbitMQ connection/channel: {e}")
            # 记录错误，但允许程序继续

    async def consume(
            self,
            queue_name: str,
            callback: Callable[[AbstractIncomingMessage], Awaitable[Any]],
            no_ack: bool = False,
    ):
        """开始消费指定队列的消息"""
        if not self.channel:
            raise AsyncRabbitMQError("Channel is not available.")
        try:
            logger.info(f"Starting consuming messages from queue '{queue_name}'...")
            # 开始消费，consume 会在后台运行
            # callback 必须是一个 async 函数
            queue = await self.channel.get_queue(queue_name)
            await queue.consume(callback, no_ack=no_ack)
            logger.info(f"Consumer set up for queue '{queue_name}'. Waiting for messages.")
        except Exception as e:
            logger.error(f"Error starting consumer for queue '{queue_name}': {e}")
            raise AsyncRabbitMQError(f"Error starting consumer: {e}") from e

    async def publish(
            self,
            exchange_name: str,
            routing_key: str,
            message: Union[str, bytes, Dict[str, Any]]
    ):
        if not self.channel:
            raise AsyncRabbitMQError("Channel is not available.")
        if isinstance(message, (dict, list)):
            message_body = json.dumps(message).encode('utf-8')
            content_type = 'application/json'
        elif isinstance(message, str):
            message_body = message.encode('utf-8')
            content_type = 'text/plain'
        elif isinstance(message, bytes):
            message_body = message
            content_type = 'application/octet-stream'
        else:
            raise TypeError("Message must be dict, list, str, or bytes")

        try:
            logger.info(f"Publishing message to exchange '{exchange_name}' with routing key '{routing_key}'...")
            exchange = await self.channel.get_exchange(exchange_name)
            await exchange.publish(
                aio_pika.Message(
                    body=message_body,
                    content_type=content_type,
                    delivery_mode=DeliveryMode.PERSISTENT,
                ),
                routing_key=routing_key
            )
            logger.info(f"Message published to exchange '{exchange_name}' with routing key '{routing_key}'.")
        except Exception as e:
            logger.error(f"Error publishing message: {e}")
            raise AsyncRabbitMQError(f"Error publishing message: {e}") from e


rabbit_async_client = AsyncRabbitMQClient()
