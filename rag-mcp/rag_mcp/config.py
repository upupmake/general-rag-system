import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

JAVA_OPENAPI_BASE_URL = os.getenv(
    "JAVA_OPENAPI_BASE_URL",
    "https://starvpn.forwardforever.top:5616/api/openapi/v1",
).rstrip("/")
RAG_LLM_BASE_URL = os.getenv(
    "RAG_LLM_BASE_URL",
    "http://192.168.188.14:6667/rag",
).rstrip("/")
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8858"))

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "192.168.188.6")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5678"))
RABBITMQ_USERNAME = os.getenv("RABBITMQ_USERNAME")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD")
if not RABBITMQ_USERNAME or not RABBITMQ_PASSWORD:
    raise RuntimeError(
        "缺少 RABBITMQ_USERNAME/RABBITMQ_PASSWORD；请在 rag-mcp/.env 或进程环境中配置"
    )
MCP_AUDIT_EXCHANGE = os.getenv("MCP_AUDIT_EXCHANGE", "rag.audit.exchange")
MCP_AUDIT_ROUTING_KEY = os.getenv("MCP_AUDIT_ROUTING_KEY", "rag.mcp.tool.log.v1")
MCP_AUDIT_PUBLISH_TIMEOUT = float(os.getenv("MCP_AUDIT_PUBLISH_TIMEOUT", "3"))
MCP_RETRIEVAL_TIMEOUT = float(os.getenv("MCP_RETRIEVAL_TIMEOUT", "60"))
