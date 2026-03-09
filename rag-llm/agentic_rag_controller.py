"""
Agentic RAG 决策控制器
负责调用LLM进行检索决策，但保持完全可控
"""
import json
import logging
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from agentic_rag_toolkit import RetrievalDecision, TOOL_DEFINE_PROMPT, TOOL_SELECT_PROMPT, CONTROLLER_SYSTEM_PROMPT
from utils import get_langchain_llm

logger = logging.getLogger(__name__)


# ============= 决策控制器 =============
class RetrievalController:
    """检索决策控制器 - 调用LLM但完全可控"""

    def __init__(self):
        model_info = {
            "name": "qwen3-max-2026-01-23",
            "provider": "other"
        }
        generate_config = {
            "extra_body": {
                "thinking": {
                    "type": "disabled"
                },
            }
        }
        self.llm = get_langchain_llm(model_info, **generate_config)
        self.llm = self.llm.with_structured_output(RetrievalDecision)

    @staticmethod
    def _format_history(history: list) -> str:
        """格式化对话历史"""
        if not history:
            return "无对话历史"

        lines = []
        for msg in history:
            if isinstance(msg, HumanMessage):
                role, content = "用户", msg.content
            elif isinstance(msg, AIMessage):
                role, content = "助手", msg.content
            elif isinstance(msg, dict):
                role_key = msg.get("role", "")
                role = "用户" if role_key == "user" else "助手" if role_key == "assistant" else role_key
                content = msg.get("content", "")
            else:
                role, content = "未知", str(msg)
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    @staticmethod
    def _aggregate_docs_by_file(docs: List[Document]) -> Dict[str, List[Document]]:
        """
        按文件聚合文档，并按chunkIndex排序
        
        Returns:
            {
                "file1.pdf": [doc1, doc2, ...],  # 按chunkIndex排序
                "file2.pdf": [doc3, doc4, ...],
            }
        """
        file_docs = {}
        for doc in docs:
            file_name = doc.metadata.get("fileName")
            if file_name not in file_docs:
                file_docs[file_name] = []
            file_docs[file_name].append(doc)

        # 按chunkIndex排序每个文件的文档
        for file_name in file_docs:
            file_docs[file_name].sort(key=lambda d: d.metadata.get("chunkIndex", 0))

        return file_docs

    @staticmethod
    def _compress_old_content(content: str) -> str:
        """旧 chunk 内容压缩：保留前25%和后25%，中间50%替换为省略号"""
        length = len(content)
        if length <= 200:
            return content
        quarter = length // 4
        return content[:quarter] + "......" + content[-quarter:]

    @staticmethod
    def _format_docs_by_file(docs: List[Document], current_round: int) -> Dict[str, Any]:
        """
        格式化文档：按文件聚合并显示内容。
        当前轮次检索到的 chunk 显示完整内容，历史轮次的 chunk 仅显示首尾各25%，
        中间50%以 "......" 代替以节省 token，详细内容可参考 tool_history 中的 existing_info。
        
        Returns:
            {
                "total_files": int,
                "total_chunks": int,
                "files": [
                    {
                        "fileName": str,
                        "documentId": int,
                        "maxChunkIndex": int,
                        "chunk_count": int,
                        "chunks": [
                            {
                                "chunkIndex": int,
                                "retrieved_round": int,
                                "content": str
                            }
                        ]
                    }
                ]
            }
        """
        file_docs = RetrievalController._aggregate_docs_by_file(docs)

        result = {
            "total_files": len(file_docs),
            "total_chunks": len(docs),
            "files": []
        }

        for file_name, file_chunks in sorted(file_docs.items()):
            if not file_chunks:
                continue

            first_chunk = file_chunks[0]

            # 按chunkIndex排序chunk
            sorted_chunks = sorted(file_chunks, key=lambda d: d.metadata.get("chunkIndex", 0))

            chunks_data = []
            for chunk in sorted_chunks:
                retrieved_round = chunk.metadata.get("retrieved_round")
                is_old = retrieved_round is not None and retrieved_round < current_round
                content = (
                    RetrievalController._compress_old_content(chunk.page_content)
                    if is_old else chunk.page_content
                )
                chunks_data.append({
                    "chunkIndex": chunk.metadata.get("chunkIndex", 0),
                    "retrieved_round": retrieved_round,
                    "content": content
                })

            file_info = {
                "fileName": file_name,
                "documentId": first_chunk.metadata.get("documentId"),
                "maxChunkIndex": first_chunk.metadata.get("maxChunkIndex"),
                "chunk_count": len(file_chunks),
                "chunks": chunks_data
            }

            result["files"].append(file_info)

        return result

    @staticmethod
    def _format_tool_call_history(trace: List[Dict]) -> List[Dict]:
        """
        格式化工具调用历史
        
        Args:
            trace: 执行轨迹列表
        
        Returns:
            [
                {
                    "round": 1,
                    "tool": "search_by_multi_queries_in_database",
                    "params": {...},
                    "result": "..." or {...}
                }
            ]
        """
        history = []
        for item in trace:
            decision = item.get("decision", {})
            result_data = item.get("result")

            call_info = {
                "round": item.get("round"),
                "tool": decision.get("tool"),
                "params": decision.get("params", {}),
            }

            if result_data is None:
                call_info["result"] = f"停止: {decision.get('reason', '未知原因')}"
            elif isinstance(result_data, dict):
                if result_data.get("type") == "file_list":
                    call_info["result"] = result_data
                elif result_data.get("type") == "document_retrieval":
                    call_info["result"] = result_data.get("description", "检索完成")
                else:
                    call_info["result"] = str(result_data)
            else:
                call_info["result"] = str(result_data)

            history.append(call_info)

        return history

    async def decide_next_action(
            self,
            question: str,
            history: list,
            current_round: int,
            max_rounds: int,
            reference_docs: List[Document],
            trace: List[Dict],
    ) -> RetrievalDecision:
        """
        决策下一步行动

        Args:
            question: 用户问题
            history: 对话历史
            current_round: 当前轮次
            max_rounds: 最大轮次
            reference_docs: 所有累积的文档
            trace: 执行轨迹（工具调用历史）

        Returns:
            RetrievalDecision: 决策结果
        """

        # 1. 对话上下文（用于判断能否回答）
        history = history.copy()
        history.append({"role": "user", "content": question})
        conversation_context = {
            "current_question": question,
            "history": self._format_history(history),
        }

        # 2. RAG检索信息（按文件聚合并排序，旧chunk内容压缩）
        docs_info = self._format_docs_by_file(reference_docs, current_round)

        # 3. 工具调用历史（包含参数和结果）
        tool_history = self._format_tool_call_history(trace)

        # system: 静态指令，可被缓存
        system_prompt = f"{CONTROLLER_SYSTEM_PROMPT}\n\n{TOOL_DEFINE_PROMPT}\n\n{TOOL_SELECT_PROMPT}"

        # user: 动态数据，每轮不同
        user_prompt = f"""## 一、对话上下文

{json.dumps(conversation_context, ensure_ascii=False, indent=2)}

## 二、已检索的文档和对应切片（按文件聚合，按chunkIndex排序）

{json.dumps(docs_info, ensure_ascii=False, indent=2)}

## 三、工具调用历史

{json.dumps(tool_history, ensure_ascii=False, indent=2)}

---

## 当前轮次: {current_round}/{max_rounds}

请基于以上三类信息和决策策略，输出结构化决策（仅JSON对象）。"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            decision = await self.llm.ainvoke(messages)
            return decision

        except Exception as e:
            logger.error(f"❌ 决策失败: {e}, 默认停止")
            return RetrievalDecision(
                action="stop",
                reason=f"决策出错: {str(e)}"
            )
