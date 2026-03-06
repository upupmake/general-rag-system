import base64
import io
import json
import logging
import os
import re
from functools import lru_cache
from typing import List

import fitz
import numpy as np
import tiktoken
from PIL import Image
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
    MarkdownHeaderTextSplitter
)
from sklearn.cluster import KMeans

from gemini_utils import GeminiInstance
from openai_utils import OpenAIInstance

# 尝试导入pytesseract，如果不存在则标记为不可用
try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

logger = logging.getLogger(__name__)


# 统一返回结构


@lru_cache(maxsize=1)
def _load_config_cached():
    """Cache the configuration to avoid blocking I/O on every request"""
    config_path = "model_config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_model_setting(model_info: dict):
    """根据模型信息加载配置并初始化 LLM"""
    provider = model_info.get("provider")
    model_name = model_info.get("name")

    if not provider or not model_name:
        raise ValueError("Model provider and name are required")

    config = _load_config_cached()
    config = config['chat']

    provider_config = config.get(provider)
    if not provider_config:
        raise ValueError(f"Provider '{provider}' not found in configuration")

    # 合并配置：公共配置 < 模型特定配置
    settings = provider_config.get("settings", {}).copy()
    model_specific_settings = provider_config.get(model_name, {})
    settings.update(model_specific_settings)

    return settings


def get_official_llm(
        model_info: dict,
        enable_web_search: bool = False,
        enable_thinking: bool = False,
        timeout: int = 60,
        max_retries: int = 5,
):
    """根据模型信息加载配置并初始化 LLM"""
    settings = _get_model_setting(model_info)
    provider = model_info.get("provider")
    model_name = model_info.get("name")

    api_key = settings.get("api_key")
    base_url = settings.get("base_url")

    if provider == "gemini":
        return GeminiInstance(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            enable_web_search=enable_web_search,
            enable_thinking=True,
            timeout=timeout,
            max_retries=max_retries,
        )
    return OpenAIInstance(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        enable_web_search=enable_web_search,
        enable_thinking=enable_thinking,
        provider=model_info['provider']
    )


def get_embedding_instance(embedding_info: dict):
    """根据嵌入模型信息加载配置并初始化 Embedding"""
    provider = embedding_info.get("provider")
    model_name = embedding_info.get("name")

    if not provider or not model_name:
        raise ValueError("Embedding provider and name are required")

    config = _load_config_cached()
    config = config['embedding']

    provider_config = config.get(provider)
    if not provider_config:
        raise ValueError(f"Provider '{provider}' not found in configuration")

    # 合并配置：公共配置 < 模型特定配置
    settings = provider_config.get("settings", {}).copy()
    model_specific_settings = provider_config.get(model_name, {})
    settings.update(model_specific_settings)

    api_key = settings.get("api_key")
    base_url = settings.get("base_url")

    # 初始化 LangChain Embedding
    return init_embeddings(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        provider=settings['provider'] if "provider" in settings else None,
        dimensions=settings['dimensions'] if "dimensions" in settings else None,
        check_embedding_ctx_length=settings.get('check_embedding_ctx_length', False)
    )


def get_langchain_llm(
        model_info: dict,
        timeout: int = 60,
        max_retries: int = 5,
        **kwargs
):
    # 初始化langchain类型的LLM
    settings = _get_model_setting(model_info)
    model_name = model_info.get("name")
    api_key = settings.get("api_key")
    base_url = settings.get("base_url")

    llm = init_chat_model(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        model_provider=settings['model_provider'] if "model_provider" in settings else None,
        timeout=timeout,
        max_retries=max_retries,
        **kwargs
    )
    return llm


def get_structured_data_agent(
        llm: BaseChatModel,
        data_type,
):
    return create_agent(
        model=llm,
        response_format=data_type
    )


def markdown_split(
        markdown_text: str,
        headers_to_split_on: list = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
):
    if headers_to_split_on is None:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3")
        ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    markdown_splits = markdown_splitter.split_text(markdown_text)
    separators = [
        "\n\n", "\n",
        "。", "！", "？",
        ".", "!", "?",
        "，", ",", " "
    ]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        add_start_index=True
    )
    return text_splitter.split_documents(markdown_splits)


def json_split(json_data: dict, min_chunk_size: int = 100, max_chunk_size: int = 1536):
    json_splitter = RecursiveJsonSplitter(min_chunk_size=min_chunk_size, max_chunk_size=max_chunk_size)
    return json_splitter.split_json(json_data)


def code_split(code_text: str, language: str, chunk_size: int = 1024, chunk_overlap: int = 100):
    language = Language(language)
    code_splitter = RecursiveCharacterTextSplitter.from_language(
        language=language, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    if language == Language.PYTHON:
        # 针对Python代码，增加特殊的切分逻辑
        code_splitter._separators = [
            # First, try to split along class definitions
            "\nclass ",
            "\nasync def ",
            "\n\tasync def ",
            "\ndef ",
            "\n\tdef ",
            # Now split by the normal type of lines
            "\n\n",
            "\n",
            " ",
            "",
        ]
    return code_splitter.split_text(code_text)


def plain_text_split(
        plain_text: str,
        chunk_size: int = 1024, chunk_overlap: int = 100,
        separators: list = None, force_split: bool = False,
        add_start_index: bool = True
):
    pattern = r'(?<=[\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef])\s+(?=[\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef])'
    plain_text = re.sub(pattern, '', plain_text)
    if separators is None:
        separators = [
            "\n\n", "\n",
            "。", "！", "？",
            ".", "!", "?",
            "，", ",", " "
        ]
    if force_split:
        if "" not in separators:
            separators.append("")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        add_start_index=add_start_index
    )
    return text_splitter.split_text(plain_text)


def _extract_text_with_ocr(pdf_path: str, language: str = 'chi_sim+eng'):
    """
    使用OCR从图片型PDF中提取文本

    Args:
        pdf_path: PDF文件路径
        language: OCR识别语言，默认中英文 (chi_sim+eng)

    Returns:
        提取的文本内容
    """
    if not TESSERACT_AVAILABLE:
        raise ImportError("pytesseract not installed. Install with: pip install pytesseract")

    doc = fitz.open(pdf_path)
    all_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # 将页面转换为图片（使用较高DPI以提高OCR准确率）
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2倍放大
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))

        # 使用pytesseract进行OCR识别
        text = pytesseract.image_to_string(img, lang=language)
        all_text.append(text)

        logger.info(f"OCR处理进度: {page_num + 1}/{len(doc)}")

    doc.close()
    return "".join(all_text)


def pdf_split(
        file_path: str,
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        text_threshold: int = 20,
        ocr_language: str = 'chi_sim+eng',
):
    """
    对PDF进行完美划分：全文合并后切分，解决跨页段落问题。
    自动检测图片型PDF并使用OCR提取文本。

    Args:
        file_path: PDF文件路径
        chunk_size: 文本块大小
        chunk_overlap: 文本块重叠大小
        text_threshold: 文本长度阈值，低于此值则认为是图片型PDF
        ocr_language: OCR识别语言，默认中英文 (chi_sim+eng)
    Returns:
        切分后的文本块列表
    """
    # 首先尝试常规文本提取
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    logger.info(f"PDF加载完成，页数：{len(docs)}")

    # 合并所有页面的文本
    text = "".join([doc.page_content for doc in docs])

    # 检测是否为图片型PDF（文本内容过少）
    if len(text.strip()) < text_threshold:
        logger.info(f"检测到图片型PDF（文本长度: {len(text)}），启动OCR识别...")
        if not TESSERACT_AVAILABLE:
            logger.warning("警告: pytesseract未安装，无法进行OCR识别")
            logger.warning("安装方法: pip install pytesseract")
            logger.warning("还需要安装Tesseract-OCR: https://github.com/tesseract-ocr/tesseract")
            return []

        try:
            text = _extract_text_with_ocr(file_path, ocr_language)
            logger.info(f"OCR识别完成，提取文本长度: {len(text)}")
        except Exception as e:
            logger.error(f"OCR识别失败: {str(e)}")
            return []
    else:
        logger.info(f"文本型PDF，直接提取文本（长度: {len(text)}）")

    # 切分文本
    return plain_text_split(
        plain_text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


async def image_split(
        file_input,
        chunk_size: int = 4096,
        chunk_overlap: int = 150,
):
    # 1. 读取图片数据并转Base64
    image_data = None
    if isinstance(file_input, str):
        with open(file_input, "rb") as f:
            image_data = f.read()
    else:
        # 假设是 file-like object
        if hasattr(file_input, 'seek'):
            file_input.seek(0)
        image_data = file_input.read()

    base64_image = base64.b64encode(image_data).decode('utf-8')

    # 2. 获取配置
    model_info = {
        'name': 'qwen3-vl-flash',
        'provider': 'qwen'
    }
    config = _load_config_cached()
    config = config['chat']
    provider_config = config.get(model_info['provider'])
    if not provider_config:
        raise ValueError(f"Provider '{model_info['provider']}' not found in configuration")
    # 合并配置：公共配置 < 模型特定配置
    settings = provider_config.get("settings", {}).copy()
    model_specific_settings = provider_config.get(model_info['name'], {})
    settings.update(model_specific_settings)

    api_key = settings.get('api_key', None)
    base_url = settings.get('base_url', None)

    if not api_key or not base_url:
        raise ValueError("API key and base URL must be provided in configuration for Qwen models.")

    # 3. 初始化 LLM
    llm = OpenAIInstance(
        model_name=model_info['name'],
        api_key=api_key,
        base_url=base_url,
        provider=model_info['provider']
    )

    # 4. 构造 Prompt
    prompt = (
        "你是一个用于知识库构建的图像内容解析模型。"
        "你的任务是将图片内容转换为结构化、客观、可用于向量检索的文本信息。\n\n"
        "请严格遵循以下规则：\n"
        "1. 如果图片包含文字，请完整、准确地转录所有可识别文字。\n"
        "2. 如果图片是架构图、流程图、系统图或表格，请按以下结构输出信息。\n"
        "3. 使用简洁、技术化、去修饰的语言，不要加入主观评价。\n"
        "4. 不要使用“这张图片”“图中可以看到”等描述性开头。\n"
        "5. 如果存在明确的技术名词、接口名、组件名，请原样保留。\n"
        "6. 仅输出解析后的文本内容，不要输出任何解释说明。\n\n"
        "请按照以下固定格式输出（即使某一项为空也要保留）：\n"
        "【IMAGE_TYPE】\n"
        "【TEXT_CONTENT】\n"
        "【ENTITIES】\n"
        "【RELATIONSHIPS】\n"
        "【PROCESS_FLOW】\n"
        "【KEY_TERMS】\n"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

    # 5. 调用模型
    try:
        response = await llm.ainvoke(messages)
        text = response.content
        logger.info(f"Image description generated, length: {len(text)}")
    except Exception as e:
        logger.error(f"Failed to generate image description: {e}")
        # 降级处理：如果调用失败，尝试返回空或者报错，这里选择返回空列表
        return []
    return markdown_split(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def get_token_count(text: str, encoding_name: str = "cl100k_base") -> int:
    """计算文本的token数量"""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception:
        # Fallback to cl100k_base if specific encoding not found
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def cut_history(history: list, model: dict):
    current_msg = history[-1]
    previous_msgs = history[:-1]

    processed_context = []
    current_token_count = get_token_count(current_msg.get('content') or "")
    n = len(previous_msgs)
    model_name = model.get("name", "")

    base_token = 10240  # 10k

    max_tokens = base_token * 8
    if model_name.startswith("gpt-"):
        if "codex" in model_name.lower():
            max_tokens = base_token * 6
        else:
            max_tokens = base_token * 4

    elif model_name.startswith("gemini-"):
        if "flash" in model_name.lower():
            max_tokens = base_token * 6
        elif "pro" in model_name.lower():
            max_tokens = base_token * 4

    elif model_name.startswith("claude-"):
        if "haiku" in model_name.lower():
            max_tokens = base_token * 6
        elif "sonnet" in model_name.lower():
            max_tokens = base_token * 4
        elif "opus" in model_name.lower():
            max_tokens = base_token * 2

    for i in range(n, 1, -2):
        pair = previous_msgs[i - 2: i]
        pair_tokens = sum(get_token_count(m.get('content') or "") for m in pair)

        if current_token_count + pair_tokens < max_tokens:
            current_token_count += pair_tokens
            processed_context = pair + processed_context
        else:
            logger.info(f"截断历史对话触发")
            break
    return processed_context + [current_msg], current_token_count


def content_extractor(content):
    """提取content中的文本和推理内容"""
    think_content = ""
    text_content = ""
    if isinstance(content, str):
        text_content = content
    elif isinstance(content, list):
        item = content[0]
        if item['type'] == 'text':
            if isinstance(item, str):
                text_content += item
            elif isinstance(item, dict) and "text" in item:
                text_content += item["text"]
        elif item['type'] == 'reasoning':
            if "text" in item:
                think_content += item["text"]
            elif "summary" in item:
                summary = item["summary"]
                if len(summary) > 0:
                    summary = summary[0]
                    if "text" in summary:
                        think_content += summary["text"]
    return think_content, text_content


def get_display_docs(documents: list, max_tokens: int = 2048, min_docs: int = 1):
    """根据token限制筛选展示的文档"""
    if len(documents) <= min_docs:
        return documents
    display_docs = [documents[i] for i in range(min_docs)]
    total_tokens = sum(get_token_count(documents[i].page_content) for i in range(min_docs))
    if total_tokens >= max_tokens:
        return display_docs
    for doc in documents[min_docs:]:
        content = doc.page_content
        doc_tokens = get_token_count(content)
        if total_tokens + doc_tokens <= max_tokens:
            display_docs.append(doc)
            total_tokens += doc_tokens
        else:
            break
    return display_docs


def reasoning_content_wrapper(chunk):
    if chunk.response_metadata:
        response_metadata = chunk.response_metadata
        if response_metadata.get("model_provider", ""):
            additional_kwargs = chunk.additional_kwargs
            reasoning_content = additional_kwargs.get("reasoning_content", "")
            if reasoning_content:
                return [{"type": "reasoning", "text": reasoning_content}]
    return ""


async def unified_llm_stream(model_instance, messages):
    """统一的LLM流式生成器"""
    try:
        async for chunk in model_instance.astream(messages):
            content = chunk.content or reasoning_content_wrapper(chunk)
            if content:
                think_content, text_content = content_extractor(content)
                if think_content:
                    yield {
                        "type": "thinking",
                        "payload": think_content
                    }
                if text_content:
                    yield {
                        "type": "content",
                        "payload": text_content
                    }
    except Exception as e:
        logger.error(f"LLM streaming error: {e}")


def filter_grade_threshold(
        docs: List[Document],  # 修正类型提示，兼容 Document
        high_score_threshold: float = 0.7,
        possible_search_ratio: float = 0.15
) -> dict:
    # 1. 提取分数
    scores = []
    valid_docs = []
    for doc in docs:
        score = doc.metadata.get('rerank_score', 0.0)
        if isinstance(score, (int, float)):
            scores.append(float(score))
            valid_docs.append(doc)

    if not scores:
        return {
            "high_ratio": 0,
            "threshold": 0.0,
            "documents": []
        }

    # 2. 降序排序
    scores = np.array(scores)
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_docs = [valid_docs[i] for i in order]
    n = len(sorted_scores)

    # 3. 数量过少直接返回
    if n < 2:
        return {
            "high_ratio": 1,
            "threshold": sorted_scores[0],
            "documents": sorted_docs
        }

    # 4. 高分直通车
    if sorted_scores.min() >= high_score_threshold:
        return {
            "high_ratio": 1,
            "threshold": sorted_scores.min(),
            "documents": sorted_docs
        }

    # 5. K-Means聚类
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(sorted_scores.reshape(-1, 1))

    # 获取标签与中心
    labels = kmeans.labels_
    centers_raw = kmeans.cluster_centers_.flatten()

    # 按中心值排序 (Low, High)
    sorted_idx = np.argsort(centers_raw)
    sorted_centers = centers_raw[sorted_idx]
    low_center = sorted_centers[0]

    # 计算高分占比
    sorted_counts = np.array([np.sum(labels == label) for label in sorted_idx])
    high_ratio = sorted_counts[1] / n
    high_score_label = sorted_idx[1]

    # 步骤 A: 先获取该簇的所有分数，不要直接 [-1]，防止列表为空
    high_cluster_subset = sorted_scores[labels == high_score_label]
    min_high_score = 0.0
    if len(high_cluster_subset) == 0:
        # 防御性逻辑：如果高分簇为空（极罕见），降级为低分中心或全保留
        kmeans_threshold = low_center
    else:
        # 步骤 B: 取高分簇的最小值（因为是降序排列，所以是最后一个）
        min_high_score = high_cluster_subset[-1]
        # 步骤 C: 计算阈值
        kmeans_threshold = min_high_score - (min_high_score - low_center) * possible_search_ratio

    # 步骤 D: 安全兜底 防止传入 possible_search_ratio > 1
    # 防止 buffer 太大导致阈值低于低分中心，这会导致把噪音全放进来
    kmeans_threshold = max(kmeans_threshold, low_center)

    filtered_docs = [doc for s, doc in zip(sorted_scores, sorted_docs) if s >= kmeans_threshold]

    # 获取低分区的最大边界值
    low_score_label = sorted_idx[0]
    low_cluster_subset = sorted_scores[labels == low_score_label]
    max_low_score = low_cluster_subset[0] if len(low_cluster_subset) > 0 else 0.0

    return {
        "high_ratio": high_ratio,
        "min_high_score": min_high_score,
        "max_low_score": max_low_score,
        "threshold": kmeans_threshold,
        "documents": filtered_docs,
        "kmeans_centers": sorted_centers.tolist()
    }


def merge_consecutive_chunks(
        docs: list[Document],
        contain_score: bool = False
) -> list[Document]:
    """合并同一文档的连续切片，去除重叠部分"""
    if not docs:
        return []

    # 按documentId分组
    docs_by_id = {}
    for doc in docs:
        # 优先使用 metadata 中的 documentId，如果没有则跳过
        doc_id = doc.metadata.get('documentId')
        if doc_id not in docs_by_id:
            docs_by_id[doc_id] = []
        docs_by_id[doc_id].append(doc)

    merged_results = []

    # 对每组进行排序和合并
    for doc_id, group in docs_by_id.items():
        # 按 chunkIndex 排序
        group.sort(key=lambda x: x.metadata.get('chunkIndex'))

        current_merged_doc = group[0]
        # 初始化 last_chunk_index
        current_merged_doc.metadata['last_chunk_index'] = current_merged_doc.metadata.get('chunkIndex')

        for i in range(1, len(group)):
            next_doc = group[i]

            last_chunk_idx = current_merged_doc.metadata.get('last_chunk_index')
            curr_chunk_idx = next_doc.metadata.get('chunkIndex')

            if last_chunk_idx is not None and curr_chunk_idx is not None and curr_chunk_idx == last_chunk_idx + 1:
                # 连续切片，进行合并
                text1 = current_merged_doc.page_content
                text2 = next_doc.page_content

                # 尝试去除重叠
                # 寻找 text1 的后缀与 text2 的前缀的最长匹配
                overlap_found = False
                # 限制最大检测长度，提高性能，通常重叠在 100-200 字符
                max_overlap_check = min(len(text1), len(text2), 500)
                min_overlap = 10

                if max_overlap_check >= min_overlap:
                    # 优化算法：使用 find 替代枚举
                    # 取 text2 的前缀作为种子（长度为 min_overlap）
                    seed = text2[:min_overlap]

                    # 在 text1 的末尾区域搜索种子
                    # 搜索范围从 len(text1) - max_overlap_check 开始
                    start_search = len(text1) - max_overlap_check
                    search_region = text1[start_search:]

                    # 在区域内查找种子
                    pos = search_region.find(seed)
                    while pos != -1:
                        # 计算在 text1 中的绝对位置
                        abs_pos = start_search + pos
                        # 潜在的重叠部分是 text1[abs_pos:]
                        # 检查 text2 是否以这段文本开头
                        potential_overlap = text1[abs_pos:]
                        if text2.startswith(potential_overlap):
                            current_merged_doc.page_content = text1 + text2[len(potential_overlap):]
                            overlap_found = True
                            break
                        # 继续查找下一个匹配
                        pos = search_region.find(seed, pos + 1)

                if not overlap_found:
                    current_merged_doc.page_content = text1 + text2  # 直接拼接

                # 更新元数据
                current_merged_doc.metadata['last_chunk_index'] = curr_chunk_idx
                # 更新分数为两者的最大值
                if contain_score:
                    current_merged_doc.metadata['rerank_score'] = max(
                        current_merged_doc.metadata.get('rerank_score', 0),
                        next_doc.metadata.get('rerank_score', 0)
                    )

            else:
                # 不连续，保存当前，开始新的
                merged_results.append(current_merged_doc)
                current_merged_doc = next_doc
                current_merged_doc.metadata['last_chunk_index'] = current_merged_doc.metadata.get('chunkIndex')

        merged_results.append(current_merged_doc)

    # 重新按 rerank_score 排序
    if contain_score:
        merged_results.sort(key=lambda x: x.metadata.get('rerank_score', 0), reverse=True)

    return merged_results
