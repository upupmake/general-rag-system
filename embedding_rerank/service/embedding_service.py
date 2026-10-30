"""
Embedding Service - FastAPI服务
基于vLLM的高性能Embedding推理服务
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Union, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from vllm import LLM

from config.embedding_config import config

# 配置日志
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================= 配置参数 =================
# 使用config对象替代环境变量


# ================= 数据模型 =================
class EmbeddingRequest(BaseModel):
    """Embedding请求模型"""
    input: Union[str, List[str]] = Field(..., description="输入文本，支持单条或批量")
    instruction: Optional[str] = Field(None, description="可选任务指令")
    model: Optional[str] = Field(None, description="模型名称（当前版本忽略此字段）")

    class Config:
        json_schema_extra = {
            "example": {
                "input": ["What is the capital of China?", "Explain gravity"],
                "instruction": "Given a web search query, retrieve relevant passages that answer the query"
            }
        }


class EmbeddingData(BaseModel):
    """单个Embedding数据"""
    object: str = Field(default="embedding", description="对象类型")
    index: int = Field(..., description="索引位置")
    embedding: List[float] = Field(..., description="向量数据")


class UsageInfo(BaseModel):
    """Token使用统计"""
    prompt_tokens: int = Field(..., description="输入token数")
    total_tokens: int = Field(..., description="总token数")


class EmbeddingResponse(BaseModel):
    """Embedding响应模型"""
    object: str = Field(default="list", description="对象类型")
    data: List[EmbeddingData] = Field(..., description="Embedding数据列表")
    model: str = Field(..., description="使用的模型名称")
    usage: UsageInfo = Field(..., description="使用统计")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model: str
    gpu_memory_utilization: float
    max_model_len: int
    device: str


# ================= 全局变量 =================
embedding_model: Optional[LLM] = None


# ================= 工具函数 =================
def get_detailed_instruct(task_description: str, query: str) -> str:
    """为查询添加任务指令"""
    return f'Instruct: {task_description}\nQuery: {query}'


def count_tokens(text: str) -> int:
    """
    估算token数（简化版）
    实际应使用tokenizer，这里简单按字符数估算
    """
    # 简化估算：中文约1.5字符/token，英文约4字符/token
    # 取平均值约2.5字符/token
    return len(text) // 2


# ================= 生命周期管理 =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global embedding_model

    logger.info("=" * 60)
    logger.info("Embedding Service Starting...")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"GPU Memory Utilization: {config.gpu_memory_utilization}")
    logger.info(f"Max Model Length: {config.max_model_len}")
    logger.info(f"Tensor Parallel Size: {config.tensor_parallel_size}")
    logger.info(f"DType: {config.dtype}")

    try:
        # 检查GPU可用性
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"GPU Available: {gpu_count} GPU(s) detected")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
                logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        else:
            logger.warning("No GPU detected, will use CPU (slower)")

        # 初始化模型
        logger.info("Loading embedding model...")
        start_time = time.time()

        embedding_model = LLM(
            model=config.model_name,
            task="embed",
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype=config.dtype,
            trust_remote_code=True,
            enable_chunked_prefill=False,  # embed task 不支持 chunked prefill，必须关闭
        )

        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f}s")
        logger.info("=" * 60)
        logger.info("Embedding Service Ready!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        raise

    yield

    # 清理资源
    logger.info("Shutting down Embedding Service...")
    embedding_model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Embedding Service stopped")


# ================= FastAPI应用 =================
app = FastAPI(
    title="Embedding Service",
    description="基于vLLM的高性能Embedding推理服务",
    version="1.0.0",
    lifespan=lifespan
)


# ================= API端点 =================
@app.get("/", response_model=dict)
async def root():
    """根路径"""
    return {
        "service": "Embedding Service",
        "version": "1.0.0",
        "model": config.model_name,
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    return HealthResponse(
        status="healthy",
        model=config.model_name,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        device=device
    )


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    生成文本向量
    兼容OpenAI Embeddings API格式
    """
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 标准化输入为列表
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input
        if not texts:
            raise HTTPException(status_code=400, detail="Input cannot be empty")

        # 检查输入长度
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"Text at index {i} is empty"
                )

        # 添加指令（如果提供）
        if request.instruction:
            processed_texts = [
                get_detailed_instruct(request.instruction, text)
                for text in texts
            ]
        else:
            processed_texts = texts

        logger.info(f"Processing {len(processed_texts)} texts for embedding")

        # 生成向量
        start_time = time.time()
        # 超长文本截断到模型上限，避免请求失败后残留的输出污染下一个请求
        outputs = embedding_model.embed(
            processed_texts,
            truncate_prompt_tokens=config.max_model_len,
        )
        inference_time = time.time() - start_time

        # 提取向量
        embeddings = [output.outputs.embedding for output in outputs]

        if len(embeddings) != len(processed_texts):
            logger.error(
                f"Embedding count mismatch: got {len(embeddings)} outputs "
                f"for {len(processed_texts)} texts"
            )
            raise HTTPException(
                status_code=500,
                detail=f"embedding count mismatch: {len(embeddings)}/{len(processed_texts)}",
            )

        # 计算token使用量
        total_tokens = sum(count_tokens(text) for text in texts)

        # 构建响应
        data = [
            EmbeddingData(
                object="embedding",
                index=i,
                embedding=embedding
            )
            for i, embedding in enumerate(embeddings)
        ]

        logger.info(
            f"Generated {len(embeddings)} embeddings in {inference_time:.3f}s "
            f"({len(embeddings) / inference_time:.1f} texts/s)"
        )

        return EmbeddingResponse(
            object="list",
            data=data,
            model=config.model_name,
            usage=UsageInfo(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings_simple(request: EmbeddingRequest):
    """
    生成文本向量（简化路径）
    与 /v1/embeddings 功能相同
    """
    return await create_embeddings(request)


# ================= 启动说明 =================
if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on {config.host}:{config.port}")

    uvicorn.run(
        "app:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        access_log=True
    )
