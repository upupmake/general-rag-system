import json
import logging

from aio_pika.abc import AbstractIncomingMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from mq.connection import rabbit_async_client
from utils import get_langchain_llm, get_structured_data_agent


# 配置模型参数 (复用 services/chat.py 中的配置)

class SessionTitle(BaseModel):
    """请根据用户的输入内容，生成一个简短的标题。"""
    title: str = Field(..., description="会话标题")


class SessionNameGenerator:
    async def generate_session_name(self, content: str, llm) -> str:
        """
        使用 LLM 根据用户输入生成简短的会话标题
        """
        if not content or not llm:
            return "新的对话"
        try:
            messages = [
                {
                    'role': 'user',
                    'content': f"用户输入内容: {content}\n\n"
                               "1. 请根据用户的输入内容，生成一个简短的标题。\n"
                               "2. 直接返回标题，不要包含引号或其他多余文字。\n"
                               "3. 注意不是回答用户内容，而是生成一个概括性的标题。\n"
                },
            ]

            response = await llm.ainvoke({"messages": messages})
            return response['structured_response'].title.strip()
        except Exception as e:
            logger.error(f"Error generating session name: {e}")
            return "新的对话"

    async def on_receive_message(self, message: AbstractIncomingMessage):
        """
        RabbitMQ 消息回调函数
        监听队列: session.name.generate.producer.queue
        """
        async with message.process():
            try:
                body = message.body.decode()
                data = json.loads(body)
                logger.info(f"Received session name generation request: {body}")
                user_id = data.get("userId")
                session_id = data.get("sessionId")
                # 兼容不同的字段名，假设 Java 端发送的是 content 或 message
                content = data.get("firstMessage") or data.get("content") or data.get("message")
                model = data.get(
                    "model")  # {'id': 7, 'name': 'gemini-2.5-flash', 'provider': 'gemini', 'metadata': '{}'}
                model = {
                    'name': 'qwen3-max-2026-01-23',
                    'provider': 'other'
                }
                llm = get_langchain_llm(model)
                llm = get_structured_data_agent(llm, SessionTitle)
                # 调用 LLM 生成标题
                session_key = await self.generate_session_name(content, llm)

                # 构造响应消息，匹配 Java 端 SessionNameGenerateResponse 结构
                response_data = {
                    "userId": user_id,
                    "sessionId": session_id,
                    "sessionKey": session_key
                }

                # 发送回 Java 端监听的 Exchange 和 Routing Key
                # Exchange: session.name.generate.exchange
                # Routing Key: session.name.generate.consumer.key
                logger.info(f"Publishing generated session name: {response_data}")
                await rabbit_async_client.publish(
                    exchange_name="server.interact.llm.exchange",
                    routing_key="session.name.generate.consumer.key",
                    message=response_data
                )
            except json.JSONDecodeError | KeyError | TypeError:
                logger.error("Failed to decode JSON message body")
            except Exception as e:
                logger.error(f"Unexpected error in session name consumer: {e}")
                raise e


session_name_generator = SessionNameGenerator()
