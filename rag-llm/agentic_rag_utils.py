"""
完全可控的 Agentic RAG 系统主类
- 基于 LangChain 原生 Function Calling (bind_tools)
- 自定义工具集(完全可控)
- 自定义决策流程(无黑盒Agent)
- 全程可追踪、可调试
"""
import json
import logging
import os
from typing import List, Dict, Any, Optional

import tiktoken

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agentic_rag_controller import RetrievalController
from agentic_rag_toolkit import RetrievalToolkit
from milvus_utils import MilvusClientManager
from rag_utils import merge_consecutive_chunks
from utils import get_embedding_instance
from utils import get_official_llm, unified_llm_stream

logger = logging.getLogger(__name__)


# ============= 核心Agentic RAG类 =============

class AgenticRAGService:
    """
    基于原生 Function Calling 的 Agentic RAG 系统
    - 自定义工具执行
    - 自定义决策流程
    - 全程日志追踪
    """

    def __init__(
            self,
            user_id: int,
            kb_id: int,
    ):
        self.user_id = user_id
        self.kb_id = kb_id

        # Milvus配置
        self.milvus_uri = os.environ.get("MILVUS_URI")
        self.milvus_token = os.environ.get("MILVUS_TOKEN")

        # Embedding配置
        self.embedding_config = {
            "name": "text-embedding-v4",
            "provider": "qwen",
        }
        # 初始化状态
        self.vector_store = None
        self.toolkit = None
        self.controller = None

        logger.info(f"🚀 AgenticRAG初始化: user_id={user_id}, kb_id={kb_id}")

    async def initialize(self):
        """异步初始化"""
        # 获取embedding实例
        embeddings = get_embedding_instance(self.embedding_config)

        # 获取Milvus vector store
        self.vector_store = await MilvusClientManager.get_instance(
            self.user_id,
            self.kb_id,
            self.milvus_uri,
            self.milvus_token,
            embeddings
        )

        if not self.vector_store:
            raise RuntimeError("无法连接到Milvus知识库")

        # 初始化retriever
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 10}
        )

        # 初始化工具集
        self.toolkit = RetrievalToolkit(self.vector_store, retriever)

        # 初始化决策控制器
        self.controller = RetrievalController()

        logger.info("✅ AgenticRAG初始化完成")

    @staticmethod
    def _deduplicate_docs(docs: List[Document]) -> List[Document]:
        """文档去重(基于pk)"""
        seen = set()
        unique_docs = []

        for doc in docs:
            pk = doc.metadata.get("pk")
            if pk and pk not in seen:
                seen.add(pk)
                unique_docs.append(doc)
            elif not pk:
                unique_docs.append(doc)

        return unique_docs

    @staticmethod
    def _format_all_docs_table(all_docs: List[Document]) -> str:
        """
        将Document列表格式化为紧凑的Markdown表格（节省token）
        """
        if not all_docs:
            return "无文档信息"

        lines = [
            "| DocID | 文件名 | 总Chunks |",
            "|-------|--------|----------|"
        ]

        for doc in all_docs:
            doc_id = doc.metadata.get("documentId", "")
            file_name = doc.metadata.get("fileName", "")
            max_chunk = doc.metadata.get("maxChunkIndex", 0)
            total_chunks = max_chunk + 1

            lines.append(f"| {doc_id} | {file_name} | {total_chunks} |")

        return "\n".join(lines)

    @staticmethod
    def _format_tool_message_content(
            tool_name: str, tool_result: Dict[str, Any], new_added: int, accumulated: int
    ) -> str:
        """
        格式化工具执行结果为 ToolMessage 内容字符串

        Args:
            tool_name: 工具名称
            tool_result: 工具原始返回结果
            new_added: 去重后新增的文档切片数
            accumulated: 累计文档切片总数

        Returns:
            用于 ToolMessage.content 的字符串
        """
        if tool_name == "list_filename_by_like":
            results = tool_result.get("results", [])
            if not results:
                return "未找到匹配的文件"
            # 格式为 markdown 表格
            table_lines = [
                "| 文件名 | DocID | 总Chunks |",
                "|--------|-------|----------|"
            ]
            for doc in results:
                doc_id = doc.metadata.get("documentId", "")
                file_name = doc.metadata.get("fileName", "")
                max_chunk = doc.metadata.get("maxChunkIndex", 0)
                table_lines.append(f"| {file_name} | {doc_id} | {max_chunk + 1} |")
            return f"找到 {len(results)} 个文件:\n" + "\n".join(table_lines)
        else:
            retrieved = tool_result.get("total_hits", 0)
            return f"检索到 {retrieved} 个文档切片，新增 {new_added} 个，累计 {accumulated} 个"

    @staticmethod
    def _estimate_tool_messages_tokens(tool_messages: List) -> int:
        """
        估算 tool_messages（AIMessage + ToolMessage 序列）的总 token 数
        使用 tiktoken cl100k_base 编码，一次性拼接后计算
        """
        parts = []
        for msg in tool_messages:
            if isinstance(msg, AIMessage):
                if msg.content:
                    parts.append(str(msg.content))
                if msg.tool_calls:
                    parts.append(json.dumps(msg.tool_calls, ensure_ascii=False, default=str))
            elif isinstance(msg, ToolMessage):
                if msg.content:
                    parts.append(str(msg.content))
        if not parts:
            return 0
        combined = "\n".join(parts)
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(combined))

    async def retrieve_with_process(
            self,
            question: str,
            history: Optional[list] = None,
            max_rounds: int = 20
    ):
        """
        Agentic检索主流程（生成器版本，yield process信息）

        基于原生 Function Calling 的多轮检索循环：
        1. 调用 controller.decide_next_action 获取 AIMessage（含 tool_calls）
        2. 如果有 tool_calls，依次执行工具并通过 ToolMessage 回传结果
        3. 如果无 tool_calls 或调用 stop_search，则停止检索

        Args:
            question: 用户问题
            history: 对话历史
            max_rounds: 最大检索轮次

        Yields:
            process信息字典或最终结果
        """
        history = history or []

        if not self.toolkit or not self.controller:
            await self.initialize()

        all_docs: Dict[int, Document] = {}  # 所有遇到的文档，key为documentId
        reference_docs: Dict[str, Document] = {}  # 用于回答的参考文档，key为pk
        tool_messages: List = []  # 累积的 AIMessage 和 ToolMessage
        total_rounds = 0
        should_stop = False

        for round_no in range(1, max_rounds + 1):
            logger.info(f"\n📍 第{round_no}轮")
            total_rounds = round_no

            response: AIMessage = await self.controller.decide_next_action(
                question=question,
                history=history,
                reference_docs=list(reference_docs.values()),
                tool_messages=tool_messages,
                tools=self.toolkit.get_tools(),
                current_round=round_no,
                max_rounds=max_rounds,
            )

            # 无工具调用 = LLM 决定停止
            if not response.tool_calls:
                logger.info(f"⏹️ LLM 未调用工具，停止检索")
                yield {
                    "type": "process",
                    "payload": {
                        "step": f"round_{round_no}",
                        "title": "停止检索",
                        "description": "LLM 判断已获取足够信息",
                        "content": response.content or "检索完成",
                        "status": "completed"
                    }
                }
                break

            # 将 AIMessage 加入历史
            tool_messages.append(response)

            # 记录本轮调用的工具名（用于 yield）
            round_tool_calls = []

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]

                # 处理 stop_search
                if tool_name == "stop_search":
                    stop_reason = tool_args.get("reason", "")
                    logger.info(f"⏹️ stop_search: {stop_reason}")
                    yield {
                        "type": "process",
                        "payload": {
                            "step": f"round_{round_no}",
                            "title": "停止检索",
                            "description": stop_reason or "已获取足够信息",
                            "content": f"**停止检索**\n\n理由: {stop_reason}" if stop_reason else "**停止检索**\n\n已获取足够信息",
                            "status": "completed"
                        }
                    }
                    # 用标志位退出外层循环
                    should_stop = True
                    break

                # 执行工具
                try:
                    tool_result = await self.toolkit.execute_tool(tool_name, tool_args)
                    new_docs = tool_result["results"]
                    before_count = len(reference_docs)

                    # list_filename_by_like 只返回元信息，不加入reference_docs
                    if tool_name != "list_filename_by_like":
                        for doc in new_docs:
                            pk = doc.metadata.get("pk")
                            if pk and pk not in reference_docs:
                                doc.metadata["retrieved_round"] = round_no
                                reference_docs[pk] = doc

                    # 记录所有遇到的文档信息
                    for doc in new_docs:
                        doc_id = doc.metadata.get("documentId")
                        if doc_id and doc_id not in all_docs:
                            all_docs[doc_id] = Document(
                                page_content="",
                                metadata={
                                    "fileName": doc.metadata.get("fileName"),
                                    "documentId": doc.metadata.get("documentId"),
                                    "maxChunkIndex": doc.metadata.get("maxChunkIndex"),
                                }
                            )

                    after_count = len(reference_docs)
                    new_added = after_count - before_count

                    logger.info(f"📊 {tool_name}: 本轮新增 {new_added}, 累积总数 {after_count}")
                    if new_added == 0:
                        logger.warning(f"⚠️ 本轮无新增文档")

                    # 格式化结果为 ToolMessage 内容
                    tool_message_content = self._format_tool_message_content(
                        tool_name, tool_result, new_added, after_count
                    )
                    tool_messages.append(ToolMessage(
                        content=tool_message_content,
                        tool_call_id=tool_call_id
                    ))

                    round_tool_calls.append({
                        "name": tool_name,
                        "args": tool_args,
                        "retrieved": tool_result.get("total_hits", 0),
                        "new_added": new_added,
                        "accumulated": after_count,
                    })

                except Exception as e:
                    logger.error(f"❌ 工具 {tool_name} 执行失败: {e}")
                    # 将错误信息作为 ToolMessage 回传给 LLM
                    tool_messages.append(ToolMessage(
                        content=f"工具执行出错: {str(e)}",
                        tool_call_id=tool_call_id
                    ))
                    round_tool_calls.append({
                        "name": tool_name,
                        "args": tool_args,
                        "error": str(e),
                    })

            # 如果 stop_search 已触发，退出外层循环
            if should_stop:
                break

            # yield 每一轮的 process 信息（合并本轮所有工具调用）
            if round_tool_calls:
                description_parts = []
                content_parts = []
                for tc in round_tool_calls:
                    args_json = json.dumps(tc["args"], ensure_ascii=False, indent=2)
                    if "error" in tc:
                        description_parts.append(f"{tc['name']}: 出错")
                        content_parts.append(
                            f"**{tc['name']}**\n"
                            f"```json\n{args_json}\n```\n"
                            f"❌ 出错: {tc['error']}"
                        )
                    else:
                        description_parts.append(
                            f"{tc['name']}: 检索{tc['retrieved']} 新增{tc['new_added']} 累计{tc['accumulated']}"
                        )
                        content_parts.append(
                            f"**{tc['name']}**\n"
                            f"```json\n{args_json}\n```\n"
                            f"检索 {tc['retrieved']} 个切片，新增 {tc['new_added']} 个，累计 {tc['accumulated']} 个"
                        )

                tool_names = ", ".join(tc["name"] for tc in round_tool_calls)

                yield {
                    "type": "process",
                    "payload": {
                        "step": f"round_{round_no}",
                        "title": tool_names,
                        "description": " | ".join(description_parts),
                        "content": "\n\n".join(content_parts),
                        "status": "completed"
                    }
                }

            # 检查累积 token 是否超限（180k），超限则提前终止
            total_tokens = self._estimate_tool_messages_tokens(tool_messages)
            logger.info(f"📏 当前 tool_messages 累积 token: {total_tokens}")
            if total_tokens > 180_000:
                logger.warning(f"⚠️ token 超限 ({total_tokens} > 180000)，提前终止检索")
                yield {
                    "type": "process",
                    "payload": {
                        "step": f"round_{round_no}",
                        "title": "提前终止",
                        "description": f"检索上下文 token 已达 {total_tokens}，超过 180k 限制，停止检索并生成答案",
                        "content": f"**提前终止**\n\n累积 token: **{total_tokens:,}** / 180,000\n\n已获取足够上下文，停止检索并生成答案",
                        "status": "completed"
                    }
                }
                break

        # 最终结果
        yield {
            "type": "final_result",
            "payload": {
                "reference_documents": list(reference_docs.values()),
                "total_rounds": total_rounds,
                "all_documents": sorted(all_docs.values(), key=lambda d: d.metadata.get("fileName", ""))
            }
        }

    async def stream_agentic_rag_response_with_process(
            self,
            question: str,
            history: list,
            model_info: dict,
            system_prompt: Optional[str] = None,
            options: dict = None,
            max_rounds: int = 20,
    ):
        """
        带过程信息的流式Agentic RAG响应生成器

        Args:
            question: 用户问题
            history: 对话历史（LangChain消息格式）
            model_info: 模型配置信息
            system_prompt: 自定义系统提示词（可选）
            options: 其他选项（如启用Web搜索等）
            max_rounds: 最大检索轮次

        Yields:
            包含type和payload的字典，type可以是"process"或"content"
        """

        context = "具体内容为空"
        reference_documents = []
        all_documents = []
        all_docs_table = ""

        # 执行Agentic RAG流程
        try:
            logger.info("开始Agentic检索流程...")
            yield {
                "type": "process",
                "payload": {
                    "step": "begin",
                    "title": "开始检索",
                    "description": "使用知识库进行检索",
                    "status": "running"
                }
            }
            final_item = None
            async for item in self.retrieve_with_process(
                    question=question,
                    history=history,
                    max_rounds=max_rounds
            ):
                if item.get("type") == "final_result":
                    final_item = item
                else:
                    yield item

            if final_item:
                retrieval_result = final_item["payload"]
                all_documents = retrieval_result.get("all_documents", [])
                reference_documents = retrieval_result.get("reference_documents", [])
                total_rounds = retrieval_result["total_rounds"]
                all_docs_table = self._format_all_docs_table(all_documents)
                logger.info(f"Agentic检索完成: {total_rounds}轮, {len(reference_documents)}个文档")
            else:
                raise RuntimeError("Agentic检索流程未返回最终结果")

            # 合并连续切片并构建上下文
            if reference_documents:
                merged_docs = merge_consecutive_chunks(reference_documents, False)

                context = "\n\n".join([
                    f"[来源: {doc.metadata.get('fileName')}]: {doc.page_content}"
                    for i, doc in enumerate(merged_docs)
                ])

                merged_docs_table_lines = [
                    "| 文件名 | 最大索引 | 索引范围 |",
                    "|----------------------|--------|--------|"
                ]
                for doc in merged_docs:
                    file_name = doc.metadata.get("fileName", "未知")
                    max_chunk_index = doc.metadata.get("maxChunkIndex", 0)
                    chunk_index = doc.metadata.get("chunkIndex", 0)
                    last_chunk_index = doc.metadata.get("last_chunk_index", chunk_index)
                    index_range = f"{chunk_index}-{last_chunk_index}"
                    merged_docs_table_lines.append(f"| {file_name} | {max_chunk_index} | {index_range} |")

                merged_docs_table = "\n".join(merged_docs_table_lines)

                yield {
                    "type": "process",
                    "payload": {
                        "step": "context_construction",
                        "title": "构建上下文",
                        "description": f"基于检索结果构建了回答上下文，包含 {len(merged_docs)} 个切片",
                        "content": merged_docs_table,
                        "status": "completed"
                    }
                }

            else:
                yield {
                    "type": "process",
                    "payload": {
                        "step": "no_documents",
                        "title": "未检索到文档",
                        "description": "Agentic检索未找到相关文档",
                        "status": "completed"
                    }
                }

        except Exception as e:
            logger.error(f"Agentic RAG流程出错: {e}")
            yield {
                "type": "process",
                "payload": {
                    "step": "error",
                    "title": "检索失败",
                    "description": f"Agentic检索过程出错: {str(e)}",
                    "status": "error"
                }
            }
            context = f"检索过程出错: {str(e)}"

        # 生成答案
        yield {
            "type": "process",
            "payload": {
                "step": "generation",
                "title": "生成答案",
                "description": "正在基于检索结果生成回答...",
                "status": "running"
            }
        }

        # 构建系统提示词
        if system_prompt:
            logger.info("使用自定义提示词")
            final_system_prompt = f"""{system_prompt}

检索过程中涉及到的全部文档列表（元信息表格）：
{all_docs_table}

可能与问题有关的参考文档中的内容：
{context}"""
        else:
            logger.info("使用系统内置提示词")
            final_system_prompt = f"""你是一个专业的AI助手。基于提供的文档和对话历史回答用户问题。

要求：
1. 优先基于文档内容作答，文档是主要信息来源
2. 如果文档不足以完整回答，结合对话历史进行推理或明确说明
3. 文档中的信息为切片信息，可能语义并不连贯或存在错误，你需要抽取或推理相关信息

检索过程中涉及到的全部文档列表（元信息表格）：
{all_docs_table}

可能与问题有关的参考文档中的内容：
{context}"""

        yield {
            "type": "system_prompt",
            "payload": final_system_prompt
        }

        # 构建对话消息
        conversation = [{"role": "system", "content": final_system_prompt}]
        for msg in history:
            if isinstance(msg, HumanMessage):
                conversation.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                conversation.append({"role": "assistant", "content": msg.content})
        conversation.append({"role": "user", "content": question})

        llm = get_official_llm(
            model_info,
            enable_web_search=options.get('webSearch', False) if options else False,
            enable_thinking=options.get('thinking', False) if options else False
        )

        async for item in unified_llm_stream(llm, conversation):
            yield item
