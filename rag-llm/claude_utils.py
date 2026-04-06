import logging
from typing import AsyncGenerator

from anthropic import AsyncAnthropic

from wrapper import ResponseWrapper

logger = logging.getLogger(__name__)


class ClaudeInstance:
    def __init__(
            self,
            model_name: str,
            api_key: str,
            base_url: str,
            timeout: int = 30,
            max_retries: int = 2,
            enable_thinking: bool = False,
            enable_web_search: bool = False,
    ):
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.enable_web_search = enable_web_search

        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout
        )

    def _parse_messages(self, messages: list):
        system_prompt = ""
        claude_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                if system_prompt:
                    system_prompt += "\n" + content
                else:
                    system_prompt = content
            else:
                # Anthropic expects 'user' or 'assistant' roles
                # If there are any other roles, map them or ignore
                if role not in ["user", "assistant"]:
                    role = "user"  # fallback
                claude_messages.append({"role": role, "content": content})

        return system_prompt, claude_messages

    async def ainvoke(self, messages: list) -> ResponseWrapper:
        system_prompt, claude_messages = self._parse_messages(messages)
        try:
            kwargs = {
                "model": self.model_name,
                "max_tokens": 4096,
                "messages": claude_messages,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = await self.client.messages.create(**kwargs)
            
            # Anthropic response content is a list of blocks
            content_text = ""
            for block in response.content:
                if block.type == 'text':
                    content_text += block.text
            
            return ResponseWrapper(content=content_text)
        except Exception as e:
            logger.error(f"Claude ainvoke error: {e}")
            raise e

    async def astream(self, messages: list) -> AsyncGenerator[ResponseWrapper, None]:
        system_prompt, claude_messages = self._parse_messages(messages)
        try:
            kwargs = {
                "model": self.model_name,
                "max_tokens": 4096,
                "messages": claude_messages,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            async with self.client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield ResponseWrapper(content=text)

        except Exception as e:
            logger.error(f"Claude astream error: {e}")
            yield ResponseWrapper(content=f"Error: {str(e)}")


if __name__ == '__main__':
    import asyncio

    async def aaa():
        # 请替换为真实的 api_key 和 base_url 进行测试
        api_key = "sk-ece129556f2ee5d6251e242529840b40e49f51ebb578c10b4cb65cefa07cfb22"
        base_url = "https://openai.sufy.com/bypass/anthropic"
        model_name = "claude-4.5-haiku"

        instance = ClaudeInstance(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, who are you?"}
        ]

        print("Testing ainvoke...")
        try:
            response = await instance.ainvoke(messages)
            print(f"ainvoke response: {response.content}")
        except Exception as e:
            print(f"ainvoke failed: {e}")

        print("\nTesting astream...")
        try:
            async for chunk in instance.astream(messages):
                print(chunk.content, end="", flush=True)
            print()
        except Exception as e:
            print(f"astream failed: {e}")

    asyncio.run(aaa())
