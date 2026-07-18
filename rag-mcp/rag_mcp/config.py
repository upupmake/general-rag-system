import os

JAVA_OPENAPI_BASE_URL = os.getenv(
    "JAVA_OPENAPI_BASE_URL",
    "https://starvpn.forwardforever.top:5616/api/openapi/v1",
    # "http://127.0.0.1:8080/api/openapi/v1",
).rstrip("/")
RAG_LLM_BASE_URL = os.getenv(
    "RAG_LLM_BASE_URL",
    "http://192.168.188.6:8848/rag",
).rstrip("/")
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8858"))
# MCP_PUBLIC_URL = os.getenv("MCP_PUBLIC_URL", f"http://127.0.0.1:{MCP_PORT}")
MCP_PUBLIC_URL = os.getenv("MCP_PUBLIC_URL", f"https://starvpn.forwardforever.top:7777")

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "192.168.188.6")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5678"))
RABBITMQ_USERNAME = os.getenv("RABBITMQ_USERNAME", "make")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "make20260101")
MCP_AUDIT_EXCHANGE = os.getenv("MCP_AUDIT_EXCHANGE", "rag.audit.exchange")
MCP_AUDIT_ROUTING_KEY = os.getenv("MCP_AUDIT_ROUTING_KEY", "rag.mcp.tool.log.v1")
