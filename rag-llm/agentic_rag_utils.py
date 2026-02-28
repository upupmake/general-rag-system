"""
完全可控的 Agentic RAG 系统主类
- 自定义工具集(完全可控)
- 自定义决策流程(无黑盒Agent)
- LangChain仅提供辅助(LLM调用、Embedding、文档处理)
- 全程可追踪、可调试
"""
import json
import logging
import os
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

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
    完全可控的Agentic RAG系统
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

        # 默认模型配置
        self.model_info = {
            "name": "qwen3.5-plus",
            "provider": "qwen"
        }

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
        self.controller = RetrievalController(self.model_info)

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
        
        Args:
            all_docs: List[Document] - Document对象列表，保持传入的顺序
            
        Returns:
            Markdown表格字符串
        """
        if not all_docs:
            return "无文档信息"

        # 构建Markdown表格（不排序，保持传入顺序）
        lines = [
            "| DocID | 文件名 | 总Chunks |",
            "|-------|--------|----------|"
        ]

        for doc in all_docs:
            doc_id = doc.metadata.get("documentId", "")
            file_name = doc.metadata.get("fileName", "")
            max_chunk = doc.metadata.get("maxChunkIndex", 0)
            total_chunks = max_chunk + 1  # 0-based转为总数

            lines.append(f"| {doc_id} | {file_name} | {total_chunks} |")

        return "\n".join(lines)

    @staticmethod
    def _format_tool_result(
            tool: str, tool_result: Dict[str, Any], retrieved: int, new_added: int, accumulated: int
    ) -> Dict[str, Any]:
        """
        格式化工具执行结果
        
        Args:
            tool: 工具名称
            tool_result: 工具原始返回结果
            retrieved: 本轮检索到的文档数
            new_added: 去重后新增的文档数
            accumulated: 累计文档总数
            
        Returns:
            格式化后的结果描述（dict）
        """
        if tool == "list_filename_by_like":
            results = tool_result.get("results", [])
            if not results:
                return {
                    "type": "file_list",
                    "files_table": "无文件",
                    "total_files": 0,
                }
            files_table = AgenticRAGService._format_all_docs_table(results)

            return {
                "type": "file_list",
                "files_table": files_table,
                "total_files": len(results),
            }
        else:
            return {
                "type": "document_retrieval",
                "retrieved": retrieved,
                "new_added": new_added,
                "accumulated": accumulated,
                "description": f"使用工具检索到了{retrieved}个文档，去重后新增{new_added}个文档，当前累计{accumulated}个文档"
            }

    async def retrieve_with_process(
            self,
            question: str,
            history: Optional[list] = None,
            max_rounds: int = 20
    ):
        """
        Agentic检索主流程（生成器版本，yield process信息）

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

        all_docs: Dict[int, Document] = {}  # 收集检索过程中遇到的所有文档，key为documentId
        reference_docs: Dict[str, Document] = {}  # 收集用于回答的参考文档，key为pk
        trace = []

        for round_no in range(1, max_rounds + 1):
            logger.info(f"\n📍 第{round_no}轮")

            decision = await self.controller.decide_next_action(
                question=question,
                history=history,
                reference_docs=list(reference_docs.values()),
                trace=trace,
                current_round=round_no,
                max_rounds=max_rounds,
            )
            logger.info(f"🤖 决策结果: {decision.model_dump()}")

            if decision.action == "stop":
                logger.info(f"⏹️ 决策停止: {decision.reason}")
                trace.append({
                    "round": round_no,
                    "result": None,
                    "decision": decision.model_dump()
                })

                # yield停止信息
                content_parts = []
                if decision.reason:
                    content_parts.append(f"理由: {decision.reason}")
                if decision.missing_info:
                    content_parts.append(f"缺失信息: {decision.missing_info}")

                content_parts = ["```"] + content_parts + ["```"] if content_parts else []

                yield {
                    "type": "process",
                    "payload": {
                        "step": f"round_{round_no}",
                        "title": "停止检索",
                        "description": "已获取足够信息",
                        "content": "\n".join(content_parts) if content_parts else "检索完成",
                        "status": "completed"
                    }
                }
                break
            try:
                tool_result = await self.toolkit.execute_tool(decision.tool, decision.params)
                new_docs = tool_result["results"]
                before_count = len(reference_docs)

                # list_filename_by_like 只返回元信息，不加入reference_docs
                if decision.tool != "list_filename_by_like":
                    for doc in new_docs:
                        pk = doc.metadata.get("pk")
                        if pk and pk not in reference_docs:
                            reference_docs[pk] = doc
                # 无论是否加入reference_docs，都记录所有遇到的文档信息（直接存Document对象）
                for doc in new_docs:
                    doc_id = doc.metadata.get("documentId")
                    if doc_id and doc_id not in all_docs:
                        # 创建一个轻量级Document对象（只保留metadata，不保留page_content节省内存）
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

                logger.info(f"📊 本轮新增: {new_added}, 累积总数: {after_count}")

                formatted_result = self._format_tool_result(
                    tool=decision.tool,
                    tool_result=tool_result,
                    retrieved=len(new_docs),
                    new_added=new_added,
                    accumulated=after_count
                )
                if new_added == 0:
                    logger.warning(f"⚠️ 本轮无新增文档")
            except Exception as e:
                formatted_result = {
                    "type": "error",
                    "description": str(e)
                }

            trace.append({
                "round": round_no,
                "result": formatted_result,
                "decision": decision.model_dump()
            })

            # yield每一轮的process信息
            # 统一使用content_parts列表构建content
            content_parts = []

            # 1. 决策理由
            if decision.reason:
                content_parts.append(f"理由: {decision.reason}")
            # 2. 已有信息
            if decision.existing_info:
                content_parts.append(f"已有信息: {decision.existing_info}")
            # 3. 缺失信息
            if decision.missing_info:
                content_parts.append(f"缺失信息: {decision.missing_info}")
            # 4. 工具名称
            content_parts.append(f"工具: {decision.tool}")
            # 5. 调用参数
            params_str = json.dumps(decision.params, ensure_ascii=False, indent=2)
            content_parts.append(f"参数: {params_str}")
            # 6. 执行结果（根据工具类型格式化）
            if formatted_result.get("type") == "file_list":
                # 文件列表工具
                total_files = formatted_result.get('total_files', 0)
                description = f"列出 {total_files} 个文件"
                content_parts.append(f"结果: 列出 {total_files} 个文件")
            elif formatted_result.get("type") == "document_retrieval":
                # 文档检索工具
                retrieved = formatted_result.get('retrieved', 0)
                new_added = formatted_result.get('new_added', 0)
                accumulated = formatted_result.get('accumulated', 0)
                description = f"检索 {retrieved} 个，新增 {new_added} 个，累计 {accumulated} 个"
                content_parts.append(f"结果: 检索 {retrieved} 个，新增 {new_added} 个，累计 {accumulated} 个")
            elif formatted_result.get("type") == "error":
                # 错误情况
                description = "error"
                content_parts.append(f"执行工具时出错: {formatted_result.get('description')}")
            else:
                description = "执行完成"
                content_parts.append(f"工具执行结果: {json.dumps(formatted_result, ensure_ascii=False)}")

            content_parts = ["```"] + content_parts + ["```"] if content_parts else []

            yield {
                "type": "process",
                "payload": {
                    "step": f"round_{round_no}",
                    "title": decision.tool,
                    "description": description,
                    "content": "\n".join(content_parts),
                    "status": "completed"
                }
            }

        # 最终结果
        yield {
            "type": "final_result",
            "payload": {
                "reference_documents": list(reference_docs.values()),
                "trace": trace,
                "total_rounds": len(trace),
                "all_documents": sorted(all_docs.values(), key=lambda d: d.metadata.get("fileName", ""))  # 按文件名排序
            }
        }

    async def stream_agentic_rag_response_with_process(
            self,
            # 问题和上下文
            question: str,
            history: list,
            # 用于最终回答的模型、system prompt和模型配置
            model_info: dict,
            system_prompt: Optional[str] = None,
            options: dict = None,
            # Agentic参数
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
                    "description": f"使用知识库进行检索",
                    "status": "running"
                }
            }
            final_item = None
            # 使用 retrieve_with_process 获取检索过程的实时反馈
            async for item in self.retrieve_with_process(
                    question=question,
                    history=history,
                    max_rounds=max_rounds
            ):
                # 直接转发 retrieve_with_process 产生的 process 事件

                if item.get("type") == "final_result":
                    final_item = item
                else:
                    yield item

                # 最后一个 process_item 包含完整结果
            if final_item:
                retrieval_result = final_item["payload"]
                all_documents = retrieval_result.get("all_documents", [])
                reference_documents = retrieval_result.get("reference_documents", [])
                trace = retrieval_result["trace"]
                total_rounds = retrieval_result["total_rounds"]
                logger.info(f"Agentic检索完成: {total_rounds}轮, {len(reference_documents)}个文档")
            else:
                raise RuntimeError("Agentic检索流程未返回最终结果")
            # 合并连续切片并构建上下文
            if reference_documents:
                # 合并同一文档的连续切片（不需要分数，所以传False）
                merged_docs = merge_consecutive_chunks(reference_documents, False)

                # 构建上下文
                context = "\n\n".join([
                    f"[来源: {doc.metadata.get('fileName')}]: {doc.page_content}"
                    for i, doc in enumerate(merged_docs)
                ])

                all_docs_table = self._format_all_docs_table(all_documents)

                logger.info(f"构建上下文完成，使用 {len(merged_docs)} 个文档")

                # 构建merged_docs表格
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
                        "content": f"\n{merged_docs_table}\n",
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

        # 4. 生成答案
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
1. 文档中的信息仅供参考
2. 如果文档不足以完整回答，结合对话历史进行推理或明确说明
3. 文档中的信息为切片信息，可能语义并不连贯或存在错误，你需要抽取或推理相关信息


检索过程中涉及到的全部文档列表（元信息表格）：
{all_docs_table}

可能与问题有关的参考文档中的内容：
{context}"""

        # 发送系统提示词用于token统计
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
        # 添加当前问题
        conversation.append({"role": "user", "content": question})

        llm = get_official_llm(
            model_info,
            enable_web_search=options.get('webSearch', False) if options else False,
            enable_thinking=options.get('thinking', False) if options else False
        )

        # 使用异步流式生成
        async for item in unified_llm_stream(llm, conversation):
            yield item
