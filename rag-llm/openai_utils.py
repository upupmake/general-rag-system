import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI

from wrapper import ResponseWrapper

logger = logging.getLogger(__name__)


class OpenAIInstance:
    def __init__(
            self,
            model_name: str,
            api_key: str,
            base_url: str,
            timeout: int = 30,
            max_retries: int = 2,
            enable_thinking: bool = False,
            enable_web_search: bool = False,
            provider: str = "openai",
    ):
        self.model_name = model_name
        self.provider = provider
        self.enable_thinking = enable_thinking
        self.enable_web_search = enable_web_search

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers={
                "User-Agent": "RAG Agent",
            },
        )

    def _use_responses_api(self):
        return self.model_name.startswith("gpt-")

    def _responses_extract(self, event):
        event_type = getattr(event, "type", None)
        if event_type == "response.output_text.delta":
            return ResponseWrapper(content=event.delta)
        if event_type in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
            return ResponseWrapper(content=[{"type": "reasoning", "text": event.delta}])
        return None

    def chat_api_extract(self, chunk):
        delta = chunk.choices[0].delta

        # Handle reasoning content
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
            return ResponseWrapper(content=[{"type": "reasoning", "text": delta.reasoning_content}])

        if delta.content:
            return ResponseWrapper(content=delta.content)

        return None

    async def ainvoke(self, messages: list) -> ResponseWrapper:
        try:
            if self._use_responses_api():
                response = await self.client.responses.create(
                    model=self.model_name,
                    input=messages,
                    **self.get_generate_config()
                )
                return ResponseWrapper(content=response.output_text)
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=False,
            )
            content = response.choices[0].message.content
            return ResponseWrapper(content=content)
        except Exception as e:
            logger.error(f"OpenAI ainvoke error: {e}")
            return ResponseWrapper(
                [
                    {
                        "type": "error",
                        "text": f"Error: {str(e)}"
                    }
                ]
            )

    def get_generate_config(self):
        # 包含extra_body, thinking, reasoning等配置
        extra_body = {}
        reasoning = {}
        reasoning_effort = {}
        tools = []

        if self.provider == "qwen":
            # 配置思考
            if self.enable_thinking:
                extra_body['enable_thinking'] = True
            else:
                extra_body['enable_thinking'] = False
            # 配置网页搜索
            if self.enable_web_search:
                extra_body['enable_code_interpreter'] = True
                extra_body['enable_thinking'] = True
                extra_body['enable_search'] = True
                extra_body['search_options'] = {"search_strategy": "agent_max"}
        elif self.provider == "minimax":
            extra_body["reasoning_split"] = True
        elif self.provider == "xiaomi":
            # 配置思考
            if self.enable_thinking:
                extra_body['thinking'] = {"type": "enabled"}
            else:
                extra_body['thinking'] = {"type": "disabled"}
            if self.enable_web_search:
                tools.append({
                    "type": "web_search",
                    "max_keyword": 3,
                    "force_search": False,
                    "limit": 1
                })
        elif (
                self.provider == "deepseek"
                or self.provider == "z-ai"
                or self.provider == "moonshotai"
                or self.provider == "bytedance"
        ):
            # 配置思考
            if self.enable_thinking:
                extra_body['thinking'] = {"type": "enabled"}
            else:
                extra_body['thinking'] = {"type": "disabled"}
        elif self.model_name.startswith("gpt-"):
            # gpt-* 走 Responses API
            reasoning = {"effort": "medium"}
            if self.enable_web_search:
                tools.append({"type": "web_search"})
        elif "grok-4" in self.model_name:
            if self.enable_thinking:
                reasoning_effort = "high"
            else:
                reasoning_effort = "low"
        elif "anthropic" == self.provider:
            if self.enable_thinking:
                extra_body['thinking'] = {
                    "type": "enabled",
                    "budget_tokens": 4096
                }
            else:
                extra_body['thinking'] = {
                    "type": "disabled"
                }
        r = {}
        if extra_body:
            r['extra_body'] = extra_body
        if reasoning:
            r['reasoning'] = reasoning
        if reasoning_effort:
            r['reasoning_effort'] = reasoning_effort
        if tools:
            r["tools"] = tools
        return r

    async def astream(self, messages: list) -> AsyncGenerator[ResponseWrapper, None]:
        try:
            if self._use_responses_api():
                stream = await self.client.responses.create(
                    model=self.model_name,
                    input=messages,
                    stream=True,
                    **self.get_generate_config()
                )
                async for event in stream:
                    item = self._responses_extract(event)
                    if item:
                        yield item
                return

            generate_config = self.get_generate_config()
            logger.info(f"generate_config: {generate_config}")
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                **generate_config
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                item = self.chat_api_extract(chunk)
                if item:
                    yield item


        except Exception as e:
            logger.error(f"OpenAI astream error: {e}")
            yield ResponseWrapper(
                content=[
                    {
                        "type": "error",
                        "text": f"Error: {str(e)}"
                    }
                ]
            )
