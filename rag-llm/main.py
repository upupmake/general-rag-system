import logging
import os

import uvicorn

# uvicorn --workers 2 --port 8848 --host 0.0.0.0 main:app

os.environ["NUMEXPR_MAX_THREADS"] = "2"

# 中间件统一部署在 .6；setdefault 允许外部环境变量覆盖
os.environ.setdefault("RABBITMQ_HOST", "192.168.188.6")
os.environ.setdefault("RABBITMQ_PORT", "5678")
os.environ.setdefault("RABBITMQ_USERNAME", "make")
os.environ.setdefault("RABBITMQ_PASSWORD", "make20260101")

os.environ.setdefault("MINIO_ENDPOINT", "192.168.188.6:9002")
os.environ.setdefault("MINIO_ACCESS_KEY", "make")
os.environ.setdefault("MINIO_SECRET_KEY", "make20260101")

os.environ.setdefault("MILVUS_URI", "http://192.168.188.6:19530")
os.environ.setdefault("MILVUS_TOKEN", "make:make5211314")

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
