import logging
import os

import uvicorn

# uvicorn --workers 2 --port 8848 --host 0.0.0.0 main:app

os.environ["NUMEXPR_MAX_THREADS"] = "2"

# os.environ["RABBITMQ_HOST"] = "127.0.0.1"
os.environ["RABBITMQ_HOST"] = "192.168.188.6"
os.environ["RABBITMQ_PORT"] = "5678"
os.environ["RABBITMQ_USERNAME"] = "make"
os.environ["RABBITMQ_PASSWORD"] = "make20260101"

# os.environ["MINIO_ENDPOINT"] = "127.0.0.1:9002"
os.environ["MINIO_ENDPOINT"] = "192.168.188.6:9002"
os.environ["MINIO_ACCESS_KEY"] = "make"
os.environ["MINIO_SECRET_KEY"] = "make20260101"

# os.environ["MILVUS_URI"] = "http://127.0.0.1:19530"
os.environ["MILVUS_URI"] = "http://192.168.188.6:19530"
os.environ["MILVUS_TOKEN"] = "make:make5211314"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='run.log',
    encoding='utf-8'
)

from fastapi import FastAPI
from services.chat import chat_service
from dependencies import app_lifespan

app = FastAPI(root_path="/rag", lifespan=app_lifespan)
app.include_router(chat_service)

if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8848,
        workers=2,
        log_level="info",
        reload=False
    )
