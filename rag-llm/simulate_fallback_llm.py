import asyncio
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _install_dependency_stubs():
    langchain = types.ModuleType("langchain")
    langchain_agents = types.ModuleType("langchain.agents")
    langchain_agents.create_agent = lambda *args, **kwargs: None
    langchain_chat_models = types.ModuleType("langchain.chat_models")
    langchain_chat_models.init_chat_model = lambda *args, **kwargs: None
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

    sys.modules.update({
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
    })


_install_dependency_stubs()
import utils


class SimulatedLLM:
    def __init__(self, candidate, attempts):
        self.candidate = candidate
        self.attempts = attempts

    async def astream(self, messages):
        self.attempts.append(self.candidate["api_key"])
        if self.candidate.get("fail"):
            yield utils.ResponseWrapper([{"type": "error", "text": self.candidate["fail"]}])
            return
        yield utils.ResponseWrapper(self.candidate["content"])


class SimulatedFallback(utils.FallbackLLMInstance):
    def __init__(self, candidates):
        super().__init__(
            model_info={"provider": "deepseek", "name": "deepseek-v4-flash"},
            candidates=candidates,
            timeout=30,
        )
        self.attempts = []
        self.timeouts = []

    def _build_llm(self, settings):
        timeout = settings.get("timeout", self.timeout)
        self.timeouts.append(timeout)
        return SimulatedLLM(settings, self.attempts)


async def run_case(name, candidates):
    llm = SimulatedFallback(candidates)
    outputs = []
    async for chunk in llm.astream([{"role": "user", "content": "hello"}]):
        outputs.append(chunk.content)
    return {
        "case": name,
        "attempts": llm.attempts,
        "timeouts": llm.timeouts,
        "outputs": outputs,
    }


async def main():
    cases = [
        (
            "first_success",
            [{"api_key": "candidate-1", "base_url": "url-1", "content": "success from candidate-1"}],
        ),
        (
            "fail_then_success",
            [
                {"api_key": "candidate-1", "base_url": "url-1", "fail": "candidate-1 connect failed"},
                {"api_key": "candidate-2", "base_url": "url-2", "content": "success from candidate-2"},
            ],
        ),
        (
            "all_failed",
            [
                {"api_key": "candidate-1", "base_url": "url-1", "fail": "candidate-1 connect failed"},
                {"api_key": "candidate-2", "base_url": "url-2", "fail": "candidate-2 connect failed"},
            ],
        ),
    ]

    for name, candidates in cases:
        result = await run_case(name, candidates)
        print(f"CASE {result['case']}")
        print(f"  attempts: {result['attempts']}")
        print(f"  timeouts: {result['timeouts']}")
        print(f"  outputs: {result['outputs']}")


if __name__ == "__main__":
    asyncio.run(main())
