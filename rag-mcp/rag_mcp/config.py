import os

JAVA_OPENAPI_BASE_URL = os.getenv(
    "JAVA_OPENAPI_BASE_URL",
    "http://127.0.0.1:5616/api/openapi/v1",
).rstrip("/")
RAG_LLM_BASE_URL = os.getenv(
    "RAG_LLM_BASE_URL",
    "http://192.168.188.6:8848/rag",
).rstrip("/")
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8858"))
MCP_PUBLIC_URL = os.getenv("MCP_PUBLIC_URL", f"http://127.0.0.1:{MCP_PORT}/mcp")
