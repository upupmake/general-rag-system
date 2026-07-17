import os

JAVA_OPENAPI_BASE_URL = os.getenv(
    "JAVA_OPENAPI_BASE_URL",
    "http://127.0.0.1:8080/api/openapi/v1",
).rstrip("/")
RAG_LLM_BASE_URL = os.getenv(
    "RAG_LLM_BASE_URL",
    "http://127.0.0.1:8848/rag",
).rstrip("/")
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", "8850"))
MCP_PUBLIC_URL = os.getenv("MCP_PUBLIC_URL", f"http://127.0.0.1:{MCP_PORT}/mcp")
