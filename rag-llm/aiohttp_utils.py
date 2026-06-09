import asyncio
import logging
from typing import Optional

import aiohttp

from utils import _load_config

logger = logging.getLogger(__name__)


async def rerank(
        query: str,
        documents: list[str],
        provider: str = "qwen",
        model_name: str = "qwen3-rerank",
        grade_top_n: Optional[int] = None,
        return_documents: bool = True,
        grade_score_threshold: Optional[float] = None
) -> dict:
    """
    文档重排序服务

    Args:
        query: 查询文本
        documents: 待排序的文档列表
        provider: 服务提供商，默认"qwen"
        model_name: 模型名称，默认"qwen3-rank"
        grade_top_n: 返回前N个文档，None则返回全部
        return_documents: 是否返回文档内容
        grade_score_threshold: 相关性分数阈值（斩杀线），低于此分数的文档将被过滤，默认None（不过滤）

    Returns:
        {
            "results": [
                {
                    "index": 0,
                    "relevance_score": 0.95,
                    "document": "文档内容"  # 如果return_documents=True
                },
                ...
            ]
        }

    Example:
        # 基础使用
        result = await rerank(
            query="什么是文本排序模型",
            documents=[
                "文本排序模型广泛用于搜索引擎和推荐系统中",
                "量子计算是计算科学的一个前沿领域"
            ],
            top_n=1
        )

        # 使用斩杀线
        result = await rerank(
            query="什么是机器学习",
            documents=doc_list,
            top_n=5,
            score_threshold=0.3  # 分数低于0.3的文档将被过滤
        )
    """
    if not documents:
        logger.warning("文档列表为空，跳过重排序")
        return {"results": []}

    config = _load_config()

    try:
        rerank_config = config['rerank'][provider]
        model_config = rerank_config.get(model_name, {})
        settings = rerank_config.get('settings', {})

        endpoint = model_config.get('endpoint')
        api_key = settings.get('api_key')

        if not endpoint:
            raise ValueError(f"未找到模型 {model_name} 的 endpoint 配置")
        if not api_key:
            raise ValueError(f"未找到 {provider} 的 api_key 配置")

    except KeyError as e:
        logger.error(f"配置错误: {e}")
        raise ValueError(f"无效的rerank配置: provider={provider}, model={model_name}")

    # 构建请求体 - 请求所有文档的排序结果
    payload = {
        "model": model_name,
        "input": {
            "query": query,
            "documents": documents
        },
        "parameters": {
            "return_documents": return_documents
            # 不在这里设置top_n，让API返回所有文档的排序结果
        }
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    endpoint,
                    json=payload,
                    headers=headers,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Rerank API 错误: {response.status}, {error_text}")
                    raise RuntimeError(f"Rerank API 请求失败: {response.status}")

                result = await response.json()

                # 在本地应用过滤逻辑
                all_results = result.get("output", {}).get("results", [])
                filtered_results = all_results

                # 1. 应用分数阈值过滤（斩杀线）
                if grade_score_threshold is not None:
                    original_count = len(all_results)
                    filtered_results = [
                        item for item in all_results
                        if item.get("relevance_score", 0) >= grade_score_threshold
                    ]

                    if len(filtered_results) < original_count:
                        logger.info(
                            f"应用斩杀线 {grade_score_threshold}：过滤掉 {original_count - len(filtered_results)} 个低分文档，"
                            f"保留 {len(filtered_results)} 个高质量文档"
                        )

                # 2. 应用top_n限制
                if grade_top_n is not None and len(filtered_results) > grade_top_n:
                    filtered_results = filtered_results[:grade_top_n]
                    logger.info(f"应用top_n={grade_top_n}：返回前 {grade_top_n} 个文档")

                # 更新结果
                result["output"]["results"] = filtered_results

                logger.info(
                    f"重排序完成，处理了 {len(documents)} 个文档，"
                    f"返回 {len(filtered_results)} 个结果"
                )
                return result
    except asyncio.TimeoutError:
        logger.error("Rerank API 请求超时")
        raise RuntimeError("Rerank API 请求超时")
    except aiohttp.ClientError as e:
        logger.error(f"Rerank API 网络错误: {e}")
        raise RuntimeError(f"Rerank API 网络错误: {str(e)}")
    except Exception as e:
        logger.error(f"Rerank 处理失败: {e}")
        raise
