import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, Body, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage

from agentic_rag_utils import AgenticRAGService
from rag_gateway import get_rag_gateway
from rag_utils import rag_service
from utils import get_official_llm, cut_history, get_token_count, unified_llm_stream

logger = logging.getLogger(__name__)

chat_service = APIRouter(prefix="/chat", tags=["chat"])


async def process_rag_stream_events(stream_iterator, prompt_tokens: int = 0):
    """
    处理RAG流式事件的通用逻辑
    
    处理所有类型的事件：process, thinking, content, system_prompt
    并在结束时发送汇总和使用统计
    
    Args:
        stream_iterator: 异步迭代器，产生各种类型的事件
        prompt_tokens: 初始的prompt tokens数
        
    Yields:
        SSE格式的流式数据
    """
    full_content = ""
    cot_content = ""
    rag_process_data = []
    start_time = time.time()

    # 处理流式事件
    async for item in stream_iterator:
        if item["type"] == "process":
            # 检索过程信息
            rag_process_data.append(item["payload"])
            # 对payload进行json.dumps包裹，防止特殊字符导致JSON解析错误
            process_data = {
                "type": "process",
                "payload": json.dumps(item["payload"], ensure_ascii=False)
            }
            yield f"data: {json.dumps(process_data, ensure_ascii=False)}\n\n"
        elif item["type"] == "thinking":
            # 思考内容
            content = item["payload"]
            if content:
                cot_content += content
                # 对content进行json.dumps包裹，防止特殊字符导致JSON解析错误
                data = {
                    "type": "thinking",
                    "payload": json.dumps(content, ensure_ascii=False)
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        elif item["type"] == "content":
            # 答案内容
            content = item["payload"]
            if content:
                full_content += content
                # 对content进行json.dumps包裹，防止特殊字符导致JSON解析错误
                data = {
                    "type": "content",
                    "payload": json.dumps(content, ensure_ascii=False)
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        elif item["type"] == "system_prompt":
            # 接收系统提示词，用于计算token数
            system_prompt_text = item["payload"]
            # 将系统提示词的token数计入prompt_tokens
            prompt_tokens += get_token_count(system_prompt_text)

    # 发送RAG过程汇总
    if rag_process_data:
        # 对payload进行json.dumps包裹，防止特殊字符导致JSON解析错误
        rag_summary = {
            "type": "rag_summary",
            "payload": json.dumps(rag_process_data, ensure_ascii=False)
        }
        yield f"data: {json.dumps(rag_summary, ensure_ascii=False)}\n\n"

    # 发送使用统计
    end_time = time.time()
    latency_ms = int((end_time - start_time) * 1000)
    # 计算输出token：包括思考内容和回答内容
    completion_tokens = get_token_count(cot_content) + get_token_count(full_content)
    usage_data = {
        "type": "usage",
        "payload": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": latency_ms
        }
    }
    yield f"data: {json.dumps(usage_data)}\n\n"


async def stream_generator(model_instance, messages, prompt_tokens: int = 0, options: dict = None):
    """纯LLM流式响应生成器"""
    cot_content = ""
    full_content = ""
    start_time = time.time()  # Start timing

    async for item in unified_llm_stream(model_instance, messages):
        content = item["payload"]
        if item["type"] == "thinking":
            cot_content += content
        elif item["type"] == "content":
            full_content += content

        # 对content进行json.dumps包裹，防止特殊字符导致JSON解析错误
        data = {
            "type": item["type"],
            "payload": json.dumps(content, ensure_ascii=False)
        }
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    end_time = time.time()
    latency_ms = int((end_time - start_time) * 1000)  # Calculate latency
    # 计算输出token：包括思考内容和回答内容
    completion_tokens = get_token_count(cot_content) + get_token_count(full_content)
    usage_data = {
        "type": "usage",
        "payload": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": latency_ms  # Add latency_ms
        }
    }
    yield f"data: {json.dumps(usage_data)}\n\n"


async def rag_stream_generator(
        question: str,
        history: list,
        model_info: dict,
        kb_id: Optional[int] = None,
        user_id: Optional[int] = None,
        system_prompt: Optional[str] = None,
        prompt_tokens: int = 0,
        options: dict = None,
):
    """
    RAG流式响应生成器

    流程：
    1. 生成多角度查询
    2. 并行检索知识库
    3. 并行评估文档相关性
    4. 流式生成答案
    """
    stream_iterator = rag_service.stream_rag_response_with_process(
        question=question,
        history=history,
        model_info=model_info,
        kb_id=kb_id,
        user_id=user_id,
        system_prompt=system_prompt,
        options=options,
        retrieve_k=30,
        grade_top_n=50,
        grade_score_threshold=0.35,
        context_top_n=25,
    )

    # 使用通用的事件处理逻辑
    async for item in process_rag_stream_events(stream_iterator, prompt_tokens):
        yield item


async def agentic_rag_stream_generator(
        question: str,
        history: list,
        model_info: dict,
        kb_id: Optional[int] = None,
        user_id: Optional[int] = None,
        max_rounds: int = 15,
        system_prompt: Optional[str] = None,
        prompt_tokens: int = 0,
        options: dict = None,
):
    """
    Agentic RAG流式响应生成器
    
    流程：
    1. 初始化AgenticRAGService
    2. 执行Agentic检索流程（多轮次智能检索）
    3. 流式生成答案
    
    Args:
        question: 用户问题
        history: 对话历史
        model_info: 模型配置信息
        kb_id: 知识库ID
        user_id: 用户ID
        max_rounds: 最大检索轮次
        system_prompt: 自定义系统提示词
        prompt_tokens: 已有的prompt tokens数
        options: 其他选项
        
    Yields:
        SSE格式的流式数据
    """
    # 初始化Agentic RAG服务
    agentic_rag = AgenticRAGService(
        user_id=user_id,
        kb_id=kb_id
    )

    # 创建流式迭代器
    stream_iterator = agentic_rag.stream_agentic_rag_response_with_process(
        question=question,
        history=history,
        model_info=model_info,
        system_prompt=system_prompt,
        options=options,
        max_rounds=max_rounds
    )

    # 使用通用的事件处理逻辑
    async for item in process_rag_stream_events(stream_iterator, prompt_tokens):
        yield item


def build_langchain_messages(history: list) -> list:
    """
    将历史消息转换为LangChain消息格式

    Args:
        history: 原始历史消息列表

    Returns:
        LangChain消息列表
    """
    langchain_messages = []
    for msg in history:
        role = msg.get('role')
        content = msg.get('content')
        if role == 'user':
            langchain_messages.append(HumanMessage(content=content))
        elif role == 'assistant':
            langchain_messages.append(AIMessage(content=content))
    return langchain_messages


"""
history:
[{'id': 1, 'role': 'user|assistant', 'content': '...', 'ragContext': '...'}]
model:
{'id': 7, 'name': 'gemini-2.5-flash', 'provider': 'gemini', 'metadata': '{}'}
options:
{'kbId': 123, 'userId': 456, 'systemPrompt': '...'}
"""


@chat_service.post("/stream")
async def chat_stream(
        request: Request,
        history: list = Body(default=[]),
        model: dict = Body(),
        options: dict = Body(default={})
):
    logger.info(f"Received chat stream request: model={model}, options={options}")
    """
    流式对话接口

    支持两种模式：
    1. 纯LLM模式：当options中没有kbId时，直接使用LLM生成回复
    2. RAG模式：当options中有kbId时，执行多角度查询、并行检索、评分后生成回复

    流程（RAG模式）：
    1. 根据用户问题生成3-5个不同角度的查询（调用LLM）
    2. 并行检索知识库获取相关文档
    3. 并行评估文档相关性并打分
    4. 合并连续的文档切片
    5. 汇总相关文档并流式生成答案
    """
    # 从options中提取参数
    user_id = options.get('userId')  # 注意这个userId是指知识库持有者的ID，不是当前提问用户的ID
    kb_id = options.get('kbId')
    system_prompt = options.get('systemPrompt')

    # 截断策略：保留最新用户问题，其余历史按(user, assistant)成组，总token数<20480
    prompt_tokens = 0
    if history:
        history, prompt_tokens = cut_history(history, model)

    logger.info(f"保留历史对话消息数: {len(history) // 2} + 1，输入Token数: {prompt_tokens}")
    # 构建LangChain消息列表（不包含最后一条用户消息）
    langchain_messages = build_langchain_messages(history[:-1] if history else [])

    # 提取当前用户问题
    current_question = history[-1]['content']

    if not current_question:
        # 如果没有用户问题，返回错误
        async def error_generator():
            error_data = {
                "type": "error",
                "payload": "No user question found in history"
            }
            yield f"data: {json.dumps(error_data)}\n\n"

        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream"
        )

    # 如果有知识库ID，先使用RAG Gateway判断是否需要检索
    use_rag = False  # 标记是否使用RAG
    if kb_id and user_id:
        # 调用RAG Gateway进行判断
        try:
            gateway = await get_rag_gateway()
            decision = await gateway.decide(
                current_question=current_question,
                history=history[:-1]  # 不包含当前问题
            )
            logger.info(f"RAG Gateway决策: {decision.action} - {decision.reason}")
            # 根据决策结果设置是否使用RAG
            if decision.action == "use_rag":
                use_rag = True
            else:
                logger.info(f"跳过RAG检索，直接使用LLM回答。原因: {decision.reason}")
                use_rag = False
        except Exception as e:
            logger.error(f"RAG Gateway判断失败: {e}，默认使用RAG检索")
            use_rag = True  # 默认使用RAG

    # 根据判断结果选择模式
    if use_rag and kb_id and user_id:
        # 使用RAG模式
        # 检查是否使用 Agentic RAG 模式
        use_agentic_rag = options.get('agenticRag', True)  # 默认使用Agentic RAG模式，除非明确设置为False
        max_rounds = options.get('maxRounds', 15)  # Agentic RAG的最大轮次

        if use_agentic_rag:
            logger.info(f"使用Agentic RAG模式，知识库ID: {kb_id}, 用户ID: {user_id}, 最大轮次: {max_rounds}")
            return StreamingResponse(
                agentic_rag_stream_generator(
                    question=current_question,
                    history=langchain_messages,
                    model_info=model,
                    kb_id=kb_id,
                    user_id=user_id,
                    max_rounds=max_rounds,
                    system_prompt=system_prompt,
                    prompt_tokens=prompt_tokens,
                    options=options
                ),
                media_type="text/event-stream"
            )
        else:
            logger.info(f"使用传统RAG模式，知识库ID: {kb_id}, 用户ID: {user_id}")
            return StreamingResponse(
                rag_stream_generator(
                    question=current_question,
                    history=langchain_messages,
                    model_info=model,
                    kb_id=kb_id,
                    user_id=user_id,
                    system_prompt=system_prompt,
                    prompt_tokens=prompt_tokens,
                    options=options
                ),
                media_type="text/event-stream"
            )

    # 否则使用纯LLM模式
    else:
        logger.info("使用纯LLM模式")
        llm = get_official_llm(
            model,
            enable_web_search=options.get('webSearch', False),
            enable_thinking=options.get('thinking', False)
        )
        # 添加当前问题到消息列表
        all_messages = langchain_messages + [HumanMessage(content=current_question)]

        messages = []
        for msg in all_messages:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        return StreamingResponse(
            stream_generator(llm, messages, prompt_tokens=prompt_tokens, options=options),
            media_type="text/event-stream"
        )
