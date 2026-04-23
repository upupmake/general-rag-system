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

## 工具列表

1. **keyword_search** - 关键词精确匹配检索
2. **read_file_chunks** - 按文件名读取连续chunk范围
3. **expand_context** - 扩展已命中chunk的上下文
4. **semantic_search** - 全库语义检索（多query并行+rerank+动态过滤）
5. **find_files** - 根据文件名模式查找文件
6. **stop_search** - 停止检索

## 决策原则

### 1. 根据场景选择工具
- 已知明确关键词或术语时，使用 keyword_search
- 已知文件名需要连续阅读时，使用 read_file_chunks
- 需要探索性检索或缺乏明确关键词时，使用 semantic_search
- 已找到关键chunk需要上下文时，使用 expand_context

### 2. 充分利用已检索信息
- 关注文档的 fileName、chunkIndex、maxChunkIndex
- 避免重复检索相同内容
- 已定位关键 chunk 且只需上下文时，优先 expand_context

### 3. 防止无效循环
- 不要重复调用完全相同的工具和参数
- 连续无增量时应换工具或 stop_search
- 同一方向检索连续无增量，应换工具或 stop_search

### 4. 停止条件（调用 stop_search）
- 当前信息足以回答问题
- 检索结果持续无关
- 没有合理的新参数可构造
- 达到轮次上限
- 继续检索的边际收益极低

## 工具组合策略

### 策略1: 文件定位 → 精确检索
适用场景：用户问题涉及特定文件
```
find_files → 确认文件存在 → keyword_search 或 read_file_chunks
```

### 策略2: 语义探索 → 上下文扩展
适用场景：概念性、探索性问题
```
semantic_search → 找到关键chunk → expand_context 扩展上下文
```

### 策略3: 关键词检索 → 失败降级
适用场景：关键词明确但可能不精确
```
keyword_search → 无结果或结果不足 → semantic_search
```

### 策略4: 范围读取 → 细节补充
适用场景：需要连续阅读某段内容
```
read_file_chunks → 找到关键内容 → expand_context 补充上下文
```

## 失败处理策略

| 失败情况 | 应对策略 |
|----------|----------|
| keyword_search 无结果 | 1. 换关键词 2. 改用 semantic_search |
| semantic_search 结果少 | 1. 调整 queries 2. 降低 grade_score_threshold |
| 文件 chunk 越界 | 1. 检查 maxChunkIndex 2. 缩小范围 |
| 连续无增量 | 1. 换工具 2. stop_search |

## 参数构造要求

### keyword_search
- keywords 必须具体，避免泛词（如"方法"、"流程"太泛）
- 收敛用 AND，探索用 OR

### read_file_chunks
- 参考 maxChunkIndex 避免越界
- 单次不超过20个chunk

### expand_context
- chunk_index 必须来自已命中的chunk

### semantic_search
- queries 建议4~6条，从多角度、多方面生成，可用空格分隔多个关键词
- grade_query 是用户本次问题或意图的完整改写
- grade_score_threshold 根据问题类型调整

### find_files
- 使用 SQL LIKE 语法，%为通配符
"""


# ============= 决策控制器 =============

class RetrievalController:
    """检索决策控制器 - 使用原生 Function Calling"""

    def __init__(self):
        model_info = {
            # "name": "qwen3.5-397b-a17b",
            # "provider": "other"
            "name": "MiniMax-M2.7-highspeed",
            "provider": "minimax"
        }
        generate_config = {
            # "extra_body": {
            #     "thinking": {
            #         "type": "disabled"
            #     },
            # }
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
