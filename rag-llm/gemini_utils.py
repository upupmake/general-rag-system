import json
import logging
from typing import AsyncGenerator

import aiohttp
from google import genai
from google.genai import types
from google.genai.types import ThinkingConfig

from wrapper import ResponseWrapper

logger = logging.getLogger(__name__)


class GeminiInstance:
    def __init__(
            self,
            model_name: str,
            api_key: str,
            base_url: str,
            enable_web_search: bool = False,
            enable_thinking: bool = True,
            timeout: int = 30,
            max_retries: int = 2,
    ):
        """
        初始化 Gemini 实例
        :param api_key: API Key
        :param model_name: 模型名称，例如 "gemini-2.0-flash", "gemini-3-pro-preview"
        :param enable_web_search: 是否开启谷歌搜索 (Grounding)
        :param base_url: API 基础地址
        """
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.enable_web_search = enable_web_search

        # 保存原始配置供 astream 手动请求使用
        self.api_key = api_key
        # 确保 base_url 不以 / 结尾，方便后续拼接
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # --- SDK 初始化 (保留给 ainvoke 使用) ---
        # 配置 HTTP 选项
        retry_options = types.HttpRetryOptionsDict(attempts=max_retries)
        http_options = types.HttpOptionsDict(
            base_url=base_url,
            retry_options=retry_options,
            timeout=timeout * 1000,  # milliseconds
        )

        # 初始化客户端
        self.client = genai.Client(
            api_key=api_key,
            http_options=http_options
        )

        # 预定义搜索工具
        self.grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

    def _parse_messages(self, messages: list):
        """
        SDK 专用的消息解析 (供 ainvoke 使用)
        """
        system_instruction = None
        contents = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                if system_instruction:
                    system_instruction += "\n" + content
                else:
                    system_instruction = content
            elif role == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=content)]
                ))
            elif role == "assistant":
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(text=content)]
                ))

        return system_instruction, contents

    def _get_config(self, system_instruction: str = None) -> types.GenerateContentConfig:
        """SDK 专用配置生成 (供 ainvoke 使用)"""
        tools = [self.grounding_tool] if self.enable_web_search else None
        return types.GenerateContentConfig(
            tools=tools,
            system_instruction=system_instruction,
            thinking_config=ThinkingConfig(
                include_thoughts=True
            ) if self.enable_thinking else None
        )

    async def ainvoke(self, messages: list) -> ResponseWrapper:
        """
        一次性异步返回 (保持原样，使用 SDK)
        """
        system_instruction, contents = self._parse_messages(messages)
        config = self._get_config(system_instruction)
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            text = response.text if response.text else ""
            return ResponseWrapper(content=text)
        except Exception as e:
            logger.error(f"Gemini ainvoke error: {e}")
            return ResponseWrapper([
                {
                    "type": "error",
                    "text": f"Error: {str(e)}"
                }
            ])

    async def astream(self, messages: list) -> AsyncGenerator[ResponseWrapper, None]:
        # 1. 构建 URL
        endpoint = f"/v1beta/models/{self.model_name}:streamGenerateContent?alt=sse"
        url = f"{self.base_url}{endpoint}"

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        # 2. 手动构建 Payload
        contents_payload = []
        system_instruction = {}

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                system_instruction["parts"] = [{"text": content}]
            elif role == "user":
                contents_payload.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                contents_payload.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })

        payload: dict = {
            "contents": contents_payload
        }

        # 添加 System Instruction
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        # 添加 Tools
        if self.enable_web_search:
            payload["tools"] = [{"googleSearch": {}}]

        # 添加 Generation Config
        generation_config = {}
        if self.enable_thinking:
            generation_config["thinkingConfig"] = {
                "includeThoughts": True,
            }

        if generation_config:
            payload["generationConfig"] = generation_config

        # 3. 发起请求并处理 SSE
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=self.timeout) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Gemini Request Failed [Status: {response.status}]: {error_text}")
                        yield ResponseWrapper(
                            [
                                {
                                    "type": "error",
                                    "text": f"Request failed with status {response.status}: {error_text}"
                                }
                            ]
                        )
                        return

                    # 高性能流式读取
                    async for line in response.content:
                        line = line.strip()
                        if not line:
                            continue

                        decoded_line = line.decode('utf-8')

                        # 处理 SSE 数据行
                        if decoded_line.startswith("data: "):
                            json_str = decoded_line[6:]  # 去掉 "data: " 前缀
                            try:
                                data = json.loads(json_str)

                                # 提取 candidates
                                if "candidates" in data and data["candidates"]:
                                    candidate = data["candidates"][0]

                                    # 检查是否有内容
                                    if "content" in candidate and "parts" in candidate["content"]:
                                        parts = candidate["content"]["parts"]
                                        for part in parts:
                                            text = part.get("text", "")
                                            if text == "":
                                                continue
                                            is_thought = part.get("thought", False)
                                            if is_thought:
                                                # 封装思考内容
                                                yield ResponseWrapper(content=[{"type": "reasoning", "text": text}])
                                            else:
                                                # 普通内容
                                                yield ResponseWrapper(content=text)

                            except json.JSONDecodeError:
                                logger.warning(f"Failed to decode JSON chunk: {json_str}")
                                continue
                            except Exception as e:
                                logger.error(f"Error parsing chunk: {e}")
                                continue

        except aiohttp.ClientError as e:
            logger.error(f"Network error in astream: {e}")
            yield ResponseWrapper(
                [
                    {
                        "type": "error",
                        "text": f"Network error: {str(e)}"
                    }
                ]
            )
        except Exception as e:
            logger.error(f"Error in astream: {e}")
            yield ResponseWrapper(
                [
                    {
                        "type": "error",
                        "text": f"Error: {str(e)}"
                    }
                ]
            )
