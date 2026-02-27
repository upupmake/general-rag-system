"""
RAG Gateway - 判断是否需要使用RAG检索

在实际进行RAG检索之前，先通过LLM判断：
1. 问题是否可以仅凭模型自身知识和对话上下文回答
2. 或者需要从知识库中检索相关文档来辅助回答

这样可以避免不必要的检索，节省时间和资源。
"""

import logging
from typing import List, Optional

from pydantic import BaseModel, Field

from utils import get_langchain_llm, get_structured_data_agent

logger = logging.getLogger(__name__)


class RAGGatewayDecision(BaseModel):
    """RAG网关决策结果"""
    action: str = Field(description="决策动作: 'direct_answer' 或 'use_rag'")
    reason: str = Field(description="决策理由")


class RAGGateway:
    """
    RAG网关：判断是否需要使用RAG检索
    
    使用qwen3-max-2026-01-23模型进行判断，分析问题和对话历史，
    决定是否需要从知识库检索文档。
    """

    def __init__(self):
        """初始化RAG网关，使用qwen3-max-2026-01-23模型"""
        self.llm = None
        self.structured_agent = None
        self.model_info = {
            "name": "xiaomi/mimo-v2-flash",
            "provider": "other"
        }

    async def initialize(self):
        """初始化决策模型和结构化输出agent"""
        if not self.llm:
            logger.info(f"初始化RAG Gateway，使用模型: {self.model_info['name']}")
            self.llm = get_langchain_llm(
                model_info=self.model_info,
                enable_thinking=False,
                timeout=60,
                max_retries=3
            )
            self.structured_agent = get_structured_data_agent(
                llm=self.llm,
                data_type=RAGGatewayDecision
            )

    async def decide(
            self,
            current_question: str,
            history: Optional[List] = None,
    ) -> RAGGatewayDecision:
        """
        判断是否需要使用RAG检索
        
        Args:
            current_question: 当前用户问题
            history: 对话历史（字典格式：[{role, content}, ...]）
            
        Returns:
            RAGGatewayDecision对象，包含action和reason
        """
        if not self.llm:
            await self.initialize()

        history = history or []

        # 构建system prompt
        system_prompt = """你是一个智能助手的决策模块，负责判断是否需要从知识库中检索文档来回答用户问题。

**你的任务：**
分析用户的问题和对话历史，判断是否需要使用RAG（检索增强生成）。

**判断标准：**

1. **直接回答 (direct_answer)** - 以下情况无需检索：
   - 通用知识问题（如：什么是机器学习、Python基础语法）
   - 闲聊、问候、情感交流
   - 元问题（如：你能做什么、你是谁）

2. **使用RAG检索 (use_rag)** - 以下情况需要检索：
   - 询问特定文档、报告、政策、规范的内容
   - 需要引用具体数据、统计信息、事实依据
   - 询问企业内部信息、项目细节、技术文档
   - 询问时间敏感的信息（近期事件、最新政策）
   - 用户明确提到"文档中"、"资料里"、"根据XX"等
   - 问题涉及专业领域且需要准确引用

请以结构化的方式输出决策，包含action（"direct_answer"或"use_rag"）和reason（简短的决策理由，1-2句话）。

现在请分析以下用户问题并做出决策。"""

        # 构建消息列表（字典格式）
        messages = [{"role": "system", "content": system_prompt}]

        messages.extend(history)

        # 添加当前问题
        messages.append({"role": "user", "content": f"用户问题：{current_question}"})

        try:
            # 使用structured agent进行判断
            response = await self.structured_agent.ainvoke({"messages": messages})
            response = response['structured_response']
            logger.info(f"RAG Gateway决策: {response.action} - {response.reason}")
            return response

        except Exception as e:
            logger.error(f"RAG Gateway决策失败: {e}")
            # 默认使用RAG（安全策略）
            return RAGGatewayDecision(
                action="use_rag",
                reason=f"决策失败，默认使用RAG检索: {str(e)}"
            )


# 全局单例
_gateway_instance = None


async def get_rag_gateway() -> RAGGateway:
    """获取RAG网关单例"""
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = RAGGateway()
        await _gateway_instance.initialize()
    return _gateway_instance
