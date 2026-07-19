import asyncio
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


_INIT_CHAT_MODEL_CALLS = []


def _install_dependency_stubs():
    langchain = types.ModuleType("langchain")
    langchain_agents = types.ModuleType("langchain.agents")
    langchain_agents.create_agent = lambda *args, **kwargs: None
    langchain_chat_models = types.ModuleType("langchain.chat_models")

    def init_chat_model(*args, **kwargs):
        _INIT_CHAT_MODEL_CALLS.append({"args": args, "kwargs": kwargs})
        return {"args": args, "kwargs": kwargs}

    langchain_chat_models.init_chat_model = init_chat_model
    langchain_embeddings = types.ModuleType("langchain.embeddings")
    langchain_embeddings.init_embeddings = lambda *args, **kwargs: None

    langchain_community = types.ModuleType("langchain_community")
    langchain_community_document_loaders = types.ModuleType("langchain_community.document_loaders")
    langchain_community_document_loaders.PyMuPDFLoader = object

    langchain_core = types.ModuleType("langchain_core")
    langchain_core_documents = types.ModuleType("langchain_core.documents")
    langchain_core_documents.Document = object
    langchain_core_language_models = types.ModuleType("langchain_core.language_models")
    langchain_core_language_models.BaseChatModel = object

    langchain_text_splitters = types.ModuleType("langchain_text_splitters")
    langchain_text_splitters.Language = object
    langchain_text_splitters.RecursiveCharacterTextSplitter = object
    langchain_text_splitters.RecursiveJsonSplitter = object
    langchain_text_splitters.MarkdownHeaderTextSplitter = object

    numpy = types.ModuleType("numpy")
    tiktoken = types.ModuleType("tiktoken")
    sklearn = types.ModuleType("sklearn")
    sklearn_cluster = types.ModuleType("sklearn.cluster")
    sklearn_cluster.KMeans = object

    gemini_utils = types.ModuleType("gemini_utils")
    openai_utils = types.ModuleType("openai_utils")
    wrapper = types.ModuleType("wrapper")

    class ResponseWrapper:
        def __init__(self, content):
            self.content = content

    gemini_utils.GeminiInstance = object
    openai_utils.OpenAIInstance = object
    wrapper.ResponseWrapper = ResponseWrapper

    modules = {
        "langchain": langchain,
        "langchain.agents": langchain_agents,
        "langchain.chat_models": langchain_chat_models,
        "langchain.embeddings": langchain_embeddings,
        "langchain_community": langchain_community,
        "langchain_community.document_loaders": langchain_community_document_loaders,
        "langchain_core": langchain_core,
        "langchain_core.documents": langchain_core_documents,
        "langchain_core.language_models": langchain_core_language_models,
        "langchain_text_splitters": langchain_text_splitters,
        "numpy": numpy,
        "tiktoken": tiktoken,
        "sklearn": sklearn,
        "sklearn.cluster": sklearn_cluster,
        "gemini_utils": gemini_utils,
        "openai_utils": openai_utils,
        "wrapper": wrapper,
    }
    sys.modules.update(modules)


_install_dependency_stubs()
import utils


class FakeLLM:
    def __init__(self, events=None, invoke_response=None, invoke_error=None):
        self.events = events or []
        self.invoke_response = invoke_response
        self.invoke_error = invoke_error

    async def ainvoke(self, messages):
        if self.invoke_error:
            raise self.invoke_error
        return utils.ResponseWrapper(self.invoke_response)

    async def astream(self, messages):
        for event in self.events:
            if isinstance(event, Exception):
                raise event
            yield utils.ResponseWrapper(event)


def test_old_dict_config_merges_to_single_candidate():
    original_load_config = utils._load_config
    try:
        utils._load_config = lambda: {
            "chat": {
                "deepseek": {
                    "settings": {"api_key": "default-key", "base_url": "default-url", "model_provider": "openai"},
                    "deepseek-v4-flash": {"base_url": "model-url"}
                }
            }
        }
        candidates = utils._get_model_candidates({"provider": "deepseek", "name": "deepseek-v4-flash"})
        assert candidates == [{"api_key": "default-key", "base_url": "model-url", "model_provider": "openai"}]
    finally:
        utils._load_config = original_load_config


def test_list_config_uses_model_candidates_before_settings_fallback():
    original_load_config = utils._load_config
    try:
        utils._load_config = lambda: {
            "chat": {
                "deepseek": {
                    "settings": [
                        {"api_key": "default-1", "base_url": "u1"},
                        {"api_key": "default-disabled", "base_url": "u2", "enabled": False}
                    ],
                    "deepseek-v4-flash": [
                        {"api_key": "model-1", "base_url": "mu1"}
                    ]
                }
            }
        }
        candidates = utils._get_model_candidates({"provider": "deepseek", "name": "deepseek-v4-flash"})
        assert [item["api_key"] for item in candidates] == ["model-1"]
    finally:
        utils._load_config = original_load_config


async def _collect_stream(llm):
    chunks = []
    async for chunk in llm.astream([{"role": "user", "content": "hello"}]):
        chunks.append(chunk.content)
    return chunks


def test_stream_fallback_before_first_chunk():
    attempts = []

    class TestFallback(utils.FallbackLLMInstance):
        def _build_llm(self, settings):
            attempts.append(settings["api_key"])
            if settings["api_key"] == "bad":
                return FakeLLM(events=[[{"type": "error", "text": "connect failed"}]])
            return FakeLLM(events=["ok"])

    llm = TestFallback(
        model_info={"provider": "deepseek", "name": "deepseek-v4-flash"},
        candidates=[{"api_key": "bad"}, {"api_key": "good"}],
    )
    assert asyncio.run(_collect_stream(llm)) == ["ok"]
    assert attempts == ["bad", "good"]


def test_stream_does_not_fallback_after_first_chunk():
    attempts = []

    class TestFallback(utils.FallbackLLMInstance):
        def _build_llm(self, settings):
            attempts.append(settings["api_key"])
            if settings["api_key"] == "first":
                return FakeLLM(events=["started", RuntimeError("mid-stream failed")])
            return FakeLLM(events=["should-not-run"])

    llm = TestFallback(
        model_info={"provider": "deepseek", "name": "deepseek-v4-flash"},
        candidates=[{"api_key": "first"}, {"api_key": "second"}],
    )
    chunks = asyncio.run(_collect_stream(llm))
    assert chunks[0] == "started"
    assert chunks[1][0]["type"] == "error"
    assert "mid-stream failed" in chunks[1][0]["text"]
    assert attempts == ["first"]


def test_ainvoke_fallbacks_on_error_response():
    attempts = []

    class TestFallback(utils.FallbackLLMInstance):
        def _build_llm(self, settings):
            attempts.append(settings["api_key"])
            if settings["api_key"] == "bad":
                return FakeLLM(invoke_response=[{"type": "error", "text": "failed"}])
            return FakeLLM(invoke_response="ok")

    llm = TestFallback(
        model_info={"provider": "deepseek", "name": "deepseek-v4-flash"},
        candidates=[{"api_key": "bad"}, {"api_key": "good"}],
    )
    response = asyncio.run(llm.ainvoke([{"role": "user", "content": "hello"}]))
    assert response.content == "ok"
    assert attempts == ["bad", "good"]


def test_stream_all_candidates_fail():
    attempts = []

    class TestFallback(utils.FallbackLLMInstance):
        def _build_llm(self, settings):
            attempts.append(settings["api_key"])
            return FakeLLM(events=[[{"type": "error", "text": f"{settings['api_key']} failed"}]])

    llm = TestFallback(
        model_info={"provider": "deepseek", "name": "deepseek-v4-flash"},
        candidates=[{"api_key": "bad-1"}, {"api_key": "bad-2"}],
    )
    chunks = asyncio.run(_collect_stream(llm))
    assert attempts == ["bad-1", "bad-2"]
    assert chunks == [[{"type": "error", "text": "All LLM candidates failed: bad-2 failed"}]]


def test_official_llm_default_timeout_is_60():
    original_load_config = utils._load_config
    try:
        utils._load_config = lambda: {
            "chat": {
                "deepseek": {
                    "settings": [{"api_key": "key", "base_url": "url", "model_provider": "openai"}]
                }
            }
        }
        llm = utils.get_official_llm({"provider": "deepseek", "name": "deepseek-v4-flash"})
        assert llm.timeout == 60
    finally:
        utils._load_config = original_load_config


def test_langchain_default_timeout_is_30():
    original_load_config = utils._load_config
    _INIT_CHAT_MODEL_CALLS.clear()
    try:
        utils._load_config = lambda: {
            "chat": {
                "deepseek": {
                    "settings": [{"api_key": "key", "base_url": "url", "model_provider": "openai"}]
                }
            }
        }
        utils.get_langchain_llm({"provider": "deepseek", "name": "deepseek-v4-flash"})
        assert _INIT_CHAT_MODEL_CALLS[-1]["kwargs"]["timeout"] == 30
    finally:
        utils._load_config = original_load_config


def test_langchain_llm_uses_first_candidate():
    original_load_config = utils._load_config
    _INIT_CHAT_MODEL_CALLS.clear()
    try:
        utils._load_config = lambda: {
            "chat": {
                "deepseek": {
                    "settings": [{"api_key": "default-key", "base_url": "default-url", "model_provider": "openai"}],
                    "deepseek-v4-flash": [{"api_key": "model-key", "base_url": "model-url", "model_provider": "openai"}]
                }
            }
        }
        utils.get_langchain_llm({"provider": "deepseek", "name": "deepseek-v4-flash"})
        kwargs = _INIT_CHAT_MODEL_CALLS[-1]["kwargs"]
        assert kwargs["model"] == "deepseek-v4-flash"
        assert kwargs["api_key"] == "model-key"
        assert kwargs["base_url"] == "model-url"
        assert kwargs["model_provider"] == "openai"
    finally:
        utils._load_config = original_load_config


def test_langchain_llm_falls_back_to_request_provider():
    original_load_config = utils._load_config
    _INIT_CHAT_MODEL_CALLS.clear()
    try:
        utils._load_config = lambda: {
            "chat": {
                "gemini": {
                    "settings": [{"api_key": "key", "base_url": "url"}]
                }
            }
        }
        utils.get_langchain_llm({"provider": "gemini", "name": "gemini-2.5-flash"})
        kwargs = _INIT_CHAT_MODEL_CALLS[-1]["kwargs"]
        assert kwargs["model_provider"] == "gemini"
    finally:
        utils._load_config = original_load_config


def test_openai_candidate_build_uses_provider_and_candidate_settings():
    calls = []

    class FakeOpenAIInstance:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    original_openai = utils.OpenAIInstance
    try:
        utils.OpenAIInstance = FakeOpenAIInstance
        llm = utils.FallbackLLMInstance(
            model_info={"provider": "deepseek", "name": "deepseek-v4-flash"},
            candidates=[{"api_key": "key", "base_url": "url", "timeout": 11, "max_retries": 1}],
            enable_web_search=True,
            enable_thinking=True,
            timeout=30,
            max_retries=3,
        )
        llm._build_llm(llm.candidates[0])
        assert calls == [{
            "model_name": "deepseek-v4-flash",
            "api_key": "key",
            "base_url": "url",
            "timeout": 11,
            "max_retries": 1,
            "enable_web_search": True,
            "enable_thinking": True,
            "provider": "deepseek"
        }]
    finally:
        utils.OpenAIInstance = original_openai


def test_gemini_candidate_build_uses_gemini_instance():
    calls = []

    class FakeGeminiInstance:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    original_gemini = utils.GeminiInstance
    try:
        utils.GeminiInstance = FakeGeminiInstance
        llm = utils.FallbackLLMInstance(
            model_info={"provider": "gemini", "name": "gemini-2.5-flash"},
            candidates=[{"api_key": "key", "base_url": "url"}],
            enable_web_search=True,
            enable_thinking=False,
            timeout=30,
            max_retries=3,
        )
        llm._build_llm(llm.candidates[0])
        assert calls == [{
            "model_name": "gemini-2.5-flash",
            "api_key": "key",
            "base_url": "url",
            "enable_web_search": True,
            "enable_thinking": True,
            "timeout": 30,
            "max_retries": 3
        }]
    finally:
        utils.GeminiInstance = original_gemini


async def _collect_stream_text(llm):
    chunks = []
    async for chunk in llm.astream([{"role": "user", "content": "hello"}]):
        chunks.append(chunk.content)
    return chunks


def test_other_gemini_stream_uses_ainvoke_once():
    calls = []

    class FakeLLM:
        async def ainvoke(self, messages):
            calls.append(("ainvoke", messages))
            return utils.ResponseWrapper("final-text")

        async def astream(self, messages):
            calls.append(("astream", messages))
            yield utils.ResponseWrapper("stream-text")

    class TestFallback(utils.FallbackLLMInstance):
        def _build_llm(self, settings):
            return FakeLLM()

    llm = TestFallback(
        model_info={"provider": "other", "name": "gemini-2.5-flash"},
        candidates=[{"api_key": "key"}],
    )
    assert asyncio.run(_collect_stream_text(llm)) == ["final-text"]
    assert calls == [("ainvoke", [{"role": "user", "content": "hello"}])]


async def _collect_unified_stream(llm):
    return [item async for item in utils.unified_llm_stream(llm, [])]


def test_unified_stream_ignores_empty_response_wrapper_chunks():
    llm = FakeLLM(events=["", "answer"])
    assert asyncio.run(_collect_unified_stream(llm)) == [
        {"type": "content", "payload": "answer"}
    ]


def _run_tests():
    tests = [
        test_old_dict_config_merges_to_single_candidate,
        test_list_config_uses_model_candidates_before_settings_fallback,
        test_stream_fallback_before_first_chunk,
        test_stream_does_not_fallback_after_first_chunk,
        test_ainvoke_fallbacks_on_error_response,
        test_stream_all_candidates_fail,
        test_official_llm_default_timeout_is_60,
        test_langchain_default_timeout_is_30,
        test_langchain_llm_uses_first_candidate,
        test_langchain_llm_falls_back_to_request_provider,
        test_openai_candidate_build_uses_provider_and_candidate_settings,
        test_gemini_candidate_build_uses_gemini_instance,
        test_other_gemini_stream_uses_ainvoke_once,
        test_unified_stream_ignores_empty_response_wrapper_chunks,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    os.chdir(ROOT)
    _run_tests()
