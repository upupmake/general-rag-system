"""
Run a real Agentic RAG retrieval and print every intermediate stream event.

This script intentionally does not write model_config.json and does not contain
real API keys. Provide credentials through environment variables or CLI args.

Example:
    python rag-llm/debug_agentic_rag_real.py \
        --user-id 1 \
        --kb-id 1 \
        --question "这个知识库主要讲了什么？" \
        --controller-provider minimax \
        --controller-model MiniMax-M3 \
        --controller-api-key "%MINIMAX_API_KEY%" \
        --controller-base-url "%MINIMAX_BASE_URL%" \
        --answer-provider qwen \
        --answer-model qwen3-max \
        --answer-api-key "%QWEN_API_KEY%" \
        --answer-base-url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
        --rerank-api-key "%QWEN_API_KEY%" \
        --milvus-uri "http://127.0.0.1:19530" \
        --milvus-token "username:password"

Required runtime dependencies/services:
    - Milvus collection for the given user_id/kb_id already exists.
    - Embedding service matches the vectors stored in Milvus.
    - Chat model config for controller and final answer model.
    - Rerank API config for semantic_search.
"""

import argparse
import asyncio
import json
import logging
import os
from copy import deepcopy


def _env(name: str, default=None):
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _mask(value: str):
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"


def _build_config(args):
    config = {
        "chat": {},
        "embedding": {
            "qwen": {
                "text-embedding-v4": {
                    "provider": "openai",
                    "check_embedding_ctx_length": False,
                    "dimensions": args.embedding_dimensions,
                },
                "settings": {
                    "api_key": args.embedding_api_key,
                    "base_url": args.embedding_base_url,
                }
            }
        },
        "rerank": {
            args.rerank_provider: {
                args.rerank_model: {
                    "endpoint": args.rerank_endpoint,
                },
                "settings": {
                    "api_key": args.rerank_api_key,
                }
            }
        }
    }

    for provider, model, api_key, base_url, model_provider in [
        (
            args.controller_provider,
            args.controller_model,
            args.controller_api_key,
            args.controller_base_url,
            args.controller_model_provider,
        ),
        (
            args.answer_provider,
            args.answer_model,
            args.answer_api_key,
            args.answer_base_url,
            args.answer_model_provider,
        ),
    ]:
        provider_config = config["chat"].setdefault(provider, {"settings": {}})
        provider_config["settings"].update({
            "api_key": api_key,
            "base_url": base_url,
        })
        provider_config[model] = {}
        if model_provider:
            provider_config[model]["model_provider"] = model_provider

    return config


def _load_config_from_path(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _patch_config_loader(config: dict):
    import utils
    import aiohttp_utils

    def load_config_for_debug():
        return deepcopy(config)

    utils._load_config = load_config_for_debug
    aiohttp_utils._load_config = load_config_for_debug


def _patch_embedding(args):
    if not args.embedding_base_url:
        return

    from langchain.embeddings import init_embeddings
    import agentic_rag_utils

    def get_debug_embedding_instance(_embedding_info: dict):
        return init_embeddings(
            model=args.embedding_model,
            api_key=args.embedding_api_key or "local",
            base_url=args.embedding_base_url,
            provider=args.embedding_provider,
            dimensions=args.embedding_dimensions,
            check_embedding_ctx_length=False,
        )

    agentic_rag_utils.get_embedding_instance = get_debug_embedding_instance


def _print_config_summary(args):
    print("\n========== Debug Config ==========")
    print(f"user_id={args.user_id}, kb_id={args.kb_id}, max_rounds={args.max_rounds}")
    print(f"milvus_uri={args.milvus_uri}")
    print(f"milvus_token={_mask(args.milvus_token)}")
    print(f"controller={args.controller_provider}/{args.controller_model}, base_url={args.controller_base_url}, api_key={_mask(args.controller_api_key)}")
    print(f"answer={args.answer_provider}/{args.answer_model}, base_url={args.answer_base_url}, api_key={_mask(args.answer_api_key)}")
    print(f"rerank={args.rerank_provider}/{args.rerank_model}, endpoint={args.rerank_endpoint}, api_key={_mask(args.rerank_api_key)}")
    if args.embedding_base_url:
        print(f"embedding={args.embedding_provider}/{args.embedding_model}, base_url={args.embedding_base_url}, api_key={_mask(args.embedding_api_key)}")
    else:
        print("embedding=project default get_embedding_instance()")
    print("==================================\n")


def _print_event(index: int, item: dict, answer_parts: list, thinking_parts: list):
    event_type = item.get("type")
    payload = item.get("payload")

    print(f"\n========== EVENT #{index}: {event_type} ==========")

    if event_type == "process":
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    elif event_type == "system_prompt":
        print(payload)
    elif event_type == "content":
        answer_parts.append(str(payload))
        print(payload, end="", flush=True)
    elif event_type == "thinking":
        thinking_parts.append(str(payload))
        print(payload, end="", flush=True)
    elif event_type == "error":
        print(payload)
    else:
        print(json.dumps(item, ensure_ascii=False, indent=2, default=str))


async def _run(args):
    if args.model_config:
        config = _load_config_from_path(args.model_config)
    else:
        config = _build_config(args)
    _patch_config_loader(config)

    os.environ["MILVUS_URI"] = args.milvus_uri
    os.environ["MILVUS_TOKEN"] = args.milvus_token

    from agentic_rag_utils import AgenticRAGService

    _patch_embedding(args)
    _print_config_summary(args)

    service = AgenticRAGService(user_id=args.user_id, kb_id=args.kb_id)
    model_info = {
        "provider": args.answer_provider,
        "name": args.answer_model,
    }

    answer_parts = []
    thinking_parts = []
    event_index = 0

    async for item in service.stream_agentic_rag_response_with_process(
            question=args.question,
            history=[],
            model_info=model_info,
            system_prompt=args.system_prompt,
            options={
                "webSearch": args.web_search,
                "thinking": args.thinking,
            },
            max_rounds=args.max_rounds,
    ):
        event_index += 1
        _print_event(event_index, item, answer_parts, thinking_parts)

    print("\n\n========== FINAL THINKING ==========")
    print("".join(thinking_parts) or "<empty>")
    print("\n========== FINAL ANSWER ==========")
    print("".join(answer_parts) or "<empty>")


def _parse_args():
    parser = argparse.ArgumentParser(description="Run a real Agentic RAG retrieval and print every stream event.")
    parser.add_argument("--user-id", type=int, default=int(_env("RAG_DEBUG_USER_ID", "1")))
    parser.add_argument("--kb-id", type=int, default=int(_env("RAG_DEBUG_KB_ID", "1")))
    parser.add_argument("--question", default=_env("RAG_DEBUG_QUESTION", "这个知识库主要讲了什么？"))
    parser.add_argument("--max-rounds", type=int, default=int(_env("RAG_DEBUG_MAX_ROUNDS", "10")))
    parser.add_argument("--system-prompt", default=_env("RAG_DEBUG_SYSTEM_PROMPT"))
    parser.add_argument("--model-config", default=_env("RAG_DEBUG_MODEL_CONFIG"), help="Optional path to an existing model_config.json. Secrets are not printed.")

    parser.add_argument("--milvus-uri", default=_env("MILVUS_URI", "http://127.0.0.1:19530"))
    parser.add_argument("--milvus-token", default=_env("MILVUS_TOKEN", ""))

    parser.add_argument("--controller-provider", default=_env("RAG_DEBUG_CONTROLLER_PROVIDER", "minimax"))
    parser.add_argument("--controller-model", default=_env("RAG_DEBUG_CONTROLLER_MODEL", "MiniMax-M3"))
    parser.add_argument("--controller-api-key", default=_env("RAG_DEBUG_CONTROLLER_API_KEY", _env("MINIMAX_API_KEY", "")))
    parser.add_argument("--controller-base-url", default=_env("RAG_DEBUG_CONTROLLER_BASE_URL", _env("MINIMAX_BASE_URL", "")))
    parser.add_argument("--controller-model-provider", default=_env("RAG_DEBUG_CONTROLLER_MODEL_PROVIDER", "openai"))

    parser.add_argument("--answer-provider", default=_env("RAG_DEBUG_ANSWER_PROVIDER", "qwen"))
    parser.add_argument("--answer-model", default=_env("RAG_DEBUG_ANSWER_MODEL", "qwen3-max"))
    parser.add_argument("--answer-api-key", default=_env("RAG_DEBUG_ANSWER_API_KEY", _env("QWEN_API_KEY", "")))
    parser.add_argument("--answer-base-url", default=_env("RAG_DEBUG_ANSWER_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--answer-model-provider", default=_env("RAG_DEBUG_ANSWER_MODEL_PROVIDER"))

    parser.add_argument("--rerank-provider", default=_env("RAG_DEBUG_RERANK_PROVIDER", "qwen"))
    parser.add_argument("--rerank-model", default=_env("RAG_DEBUG_RERANK_MODEL", "qwen3-rerank"))
    parser.add_argument("--rerank-api-key", default=_env("RAG_DEBUG_RERANK_API_KEY", _env("QWEN_API_KEY", "")))
    parser.add_argument("--rerank-endpoint", default=_env("RAG_DEBUG_RERANK_ENDPOINT", "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"))

    parser.add_argument("--embedding-provider", default=_env("RAG_DEBUG_EMBEDDING_PROVIDER", "openai"))
    parser.add_argument("--embedding-model", default=_env("RAG_DEBUG_EMBEDDING_MODEL", "text-embedding-v4"))
    parser.add_argument("--embedding-api-key", default=_env("RAG_DEBUG_EMBEDDING_API_KEY", _env("QWEN_API_KEY", "")))
    parser.add_argument("--embedding-base-url", default=_env("RAG_DEBUG_EMBEDDING_BASE_URL"), help="Optional. If omitted, uses project default get_embedding_instance().")
    parser.add_argument("--embedding-dimensions", type=int, default=int(_env("RAG_DEBUG_EMBEDDING_DIMENSIONS", "1024")))

    parser.add_argument("--web-search", action="store_true", default=_env("RAG_DEBUG_WEB_SEARCH", "false").lower() == "true")
    parser.add_argument("--thinking", action="store_true", default=_env("RAG_DEBUG_THINKING", "false").lower() == "true")
    parser.add_argument("--debug-logs", action="store_true", default=_env("RAG_DEBUG_LOGS", "false").lower() == "true")
    return parser.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug_logs else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
