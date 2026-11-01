import logging
import os
from pathlib import Path

from dotenv import load_dotenv
import uvicorn

# uvicorn --workers 2 --port 8848 --host 0.0.0.0 main:app

load_dotenv(Path(__file__).with_name(".env"))

os.environ["NUMEXPR_MAX_THREADS"] = "2"

# 中间件统一部署在 .6；非敏感连接地址保留默认值
os.environ.setdefault("RABBITMQ_HOST", "192.168.188.6")
os.environ.setdefault("RABBITMQ_PORT", "5678")

os.environ.setdefault("MINIO_ENDPOINT", "192.168.188.6:9002")

os.environ.setdefault("MILVUS_URI", "http://192.168.188.6:19530")

# 凭据只能来自 rag-llm/.env 或进程环境，禁止写入仓库
_REQUIRED_ENV = (
    "RABBITMQ_USERNAME",
    "RABBITMQ_PASSWORD",
    "MINIO_ACCESS_KEY",
    "MINIO_SECRET_KEY",
    "MILVUS_TOKEN",
)
_missing_env = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
if _missing_env:
    raise RuntimeError(
        "缺少必需环境变量: "
        + ", ".join(_missing_env)
        + "；请在 rag-llm/.env 或进程环境中配置"
    )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='run.log',
    encoding='utf-8'
)

from fastapi import FastAPI
from services.chat import chat_service
from services.retrieval import retrieval_service
from dependencies import app_lifespan

app = FastAPI(root_path="/rag", lifespan=app_lifespan)
app.include_router(chat_service)
app.include_router(retrieval_service)

if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8848,
        workers=1,
        log_level="info",
        reload=False
    )
