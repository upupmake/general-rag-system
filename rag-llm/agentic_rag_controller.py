"""
Agentic RAG 决策控制器
使用 LangChain 原生 Function Calling (bind_tools) 进行检索决策
"""
import json
import logging
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import StructuredTool

from utils import get_langchain_llm

logger = logging.getLogger(__name__)


# ============= 决策控制器系统提示词 =============

CONTROLLER_SYSTEM_PROMPT = """你是 Agentic RAG 的检索决策控制器。你的任务不是直接回答用户，而是基于上下文选择最合适的检索工具来收集信息。

## 决策原则

1. **优先低成本工具**
   - 已知精确文件名时，优先文件级工具（search_by_filename_and_chunk_range / extend_file_chunk_context_window）
   - 已有明确关键词时，优先 search_by_grep
   - 只有概念性问题且缺乏明确关键词时，才用 search_by_multi_queries_in_database

2. **充分利用已检索信息**
   - 关注文档的 fileName、chunkIndex、maxChunkIndex
   - 避免重复检索相同内容
   - 已定位关键 chunk 且只需上下文时，优先 extend_file_chunk_context_window

3. **防止无效循环**
   - 不要重复调用完全相同的工具和参数
   - 连续无增量时应换工具或停止
   - 同一方向检索连续无增量，应换工具或 stop_search

4. **停止条件**（调用 stop_search）
   - 当前信息足以回答问题
   - 检索结果持续无关
   - 没有合理的新参数可构造
   - 达到轮次上限
   - 继续检索的边际收益极低

5. **参数构造要求**
   - search_by_grep: keywords 必须具体，避免泛词；收敛用 AND，探索用 OR
   - search_by_filename_and_chunk_range: 参考 maxChunkIndex 避免越界，单次不超过20个chunk
   - extend_file_chunk_context_window: chunk_index 必须来自已命中的chunk
   - search_by_multi_queries_in_database: queries 3~6条，从不同角度描述，grade_query 用核心问题
   - list_filename_by_like: 使用 SQL LIKE 语法，%为通配符
"""


# ============= 决策控制器 =============

class RetrievalController:
    """检索决策控制器 - 使用原生 Function Calling"""

    def __init__(self):
        model_info = {
            "name": "qwen3.5-397b-a17b",
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
        """
        file_docs = {}
        for doc in docs:
            file_name = doc.metadata.get("fileName")
            if file_name not in file_docs:
                file_docs[file_name] = []
            file_docs[file_name].append(doc)

        for file_name in file_docs:
            file_docs[file_name].sort(key=lambda d: d.metadata.get("chunkIndex", 0))

        return file_docs

    @staticmethod
    def _format_docs_by_file(docs: List[Document]) -> Dict[str, Any]:
        """
        格式化文档：按文件聚合并显示完整信息（包含内容）
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

            sorted_chunks = sorted(file_chunks, key=lambda d: d.metadata.get("chunkIndex", 0))

            chunks_data = []
            for chunk in sorted_chunks:
                chunks_data.append({
                    "chunkIndex": chunk.metadata.get("chunkIndex", 0),
                    "retrieved_round": chunk.metadata.get("retrieved_round"),
                    "content": chunk.page_content
                })

            file_info = {
                "fileName": file_name,
                "documentId": first_chunk.metadata.get("documentId"),
                "maxChunkIndex": first_chunk.metadata.get("maxChunkIndex"),
                "retrieved_chunk_count": len(file_chunks),
                "chunks": chunks_data
            }

            result["files"].append(file_info)

        return result

    async def decide_next_action(
            self,
            question: str,
            history: list,
            current_round: int,
            max_rounds: int,
            reference_docs: List[Document],
            tool_messages: List,
            tools: List[StructuredTool],
    ) -> AIMessage:
        """
        决策下一步行动（原生 Function Calling）

        Args:
            question: 用户问题
            history: 对话历史
            current_round: 当前轮次
            max_rounds: 最大轮次
            reference_docs: 所有累积的文档
            tool_messages: 累积的 AIMessage 和 ToolMessage 列表
            tools: StructuredTool 列表（从 toolkit.get_tools() 获取）

        Returns:
            AIMessage: 包含 tool_calls 属性的响应消息
        """

        # 1. 对话上下文
        history_copy = history.copy()
        history_copy.append({"role": "user", "content": question})
        conversation_context = {
            "current_question": question,
            "history": self._format_history(history_copy),
        }

        # 2. RAG检索信息（按文件聚合并排序）
        docs_info = self._format_docs_by_file(reference_docs)

        # 3. 构建 system prompt（静态，可被缓存）
        system_prompt = CONTROLLER_SYSTEM_PROMPT

        # 4. 构建 user prompt
        round_hint = f"## 当前轮次: {current_round}/{max_rounds}" + (
            " ⚠️ 最后一轮，若信息仍不足请停止并基于现有内容作答" if current_round == max_rounds else ""
        )

        user_prompt = f"""## 一、对话上下文

{json.dumps(conversation_context, ensure_ascii=False, indent=2)}

## 二、已检索的文档和对应切片（按文件聚合，按chunkIndex排序）

{json.dumps(docs_info, ensure_ascii=False, indent=2)}

---

{round_hint}

请基于以上信息和决策策略，选择最合适的工具进行下一步检索，或调用 stop_search 停止检索。"""

        # 5. 构建消息列表：system + user + tool_messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        # 追加之前的工具调用历史（AIMessage + ToolMessage）
        messages.extend(tool_messages)

        logger.debug(
            f"\n{'=' * 60}\n"
            f"[Round {current_round}] decide_next_action INPUT\n"
            f"{'=' * 60}\n"
            f"[SYSTEM]\n{system_prompt}\n\n"
            f"[USER]\n{user_prompt}\n"
            f"[TOOL_MESSAGES count: {len(tool_messages)}]\n"
            f"{'=' * 60}"
        )

        try:
            # 使用 bind_tools 创建带工具的 LLM
            llm_with_tools = self.llm.bind_tools(tools)
            response: AIMessage = await llm_with_tools.ainvoke(messages)

            logger.debug(
                f"\n{'=' * 60}\n"
                f"[Round {current_round}] decide_next_action OUTPUT\n"
                f"{'=' * 60}\n"
                f"tool_calls: {json.dumps([tc for tc in (response.tool_calls or [])], ensure_ascii=False, default=str, indent=2)}\n"
                f"content: {response.content}\n"
                f"{'=' * 60}"
            )
            return response

        except Exception as e:
            logger.error(f"❌ 决策失败: {e}")
            raise
