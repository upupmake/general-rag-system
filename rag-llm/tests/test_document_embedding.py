import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "mq" / "document_embedding.py"


class FakeDocument:
    def __init__(self, page_content):
        self.page_content = page_content
        self.metadata = {}


class FakeMinioResponse:
    def __init__(self, data):
        self.data = data

    async def read(self):
        return self.data


class FakeMinioClient:
    def __init__(self, data=b"file-content"):
        self.data = data
        self.calls = []

    async def get_object(self, bucket_name, object_name):
        self.calls.append((bucket_name, object_name))
        if self.data is None:
            return None
        return FakeMinioResponse(self.data)


class FakeRabbitClient:
    def __init__(self):
        self.messages = []

    async def publish(self, exchange_name, routing_key, message):
        self.messages.append({
            "exchange_name": exchange_name,
            "routing_key": routing_key,
            "message": message,
        })


class FakeMilvusClient:
    def __init__(self, existing_pks=None):
        self.rows = [{"pk": pk} for pk in (existing_pks or [])]
        self.query_calls = []
        self.deleted = []

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return self.rows

    def delete(self, **kwargs):
        self.deleted.append(kwargs)


class FakeVectorStore:
    def __init__(self, existing_pks=None, fail_after=None):
        self.client = FakeMilvusClient(existing_pks)
        self.collection_name = "kb_1"
        self.fail_after = fail_after
        self.added_batches = []
        self._next_id = 100

    def add_documents(self, documents):
        if self.fail_after is not None and len(self.added_batches) >= self.fail_after:
            raise RuntimeError("milvus write failed")
        ids = []
        for _ in documents:
            ids.append(self._next_id)
            self._next_id += 1
        self.added_batches.append(list(documents))
        return ids


class EmptyIdsVectorStore(FakeVectorStore):
    def add_documents(self, documents):
        self.added_batches.append(list(documents))
        return []


class FakeMilvusManager:
    store = None

    @classmethod
    async def get_instance(cls, *args, **kwargs):
        return cls.store


class FakeProcessContext:
    def __init__(self):
        self.exited = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True
        return False


class FakeMessage:
    def __init__(self, body):
        self.body = body
        self.process_kwargs = None
        self.process_context = None

    def process(self, **kwargs):
        self.process_kwargs = kwargs
        self.process_context = FakeProcessContext()
        return self.process_context


def _load_module_under_test():
    saved = {}
    stub_names = (
        "aio_pika",
        "aio_pika.abc",
        "langchain_core",
        "langchain_core.documents",
        "utils",
        "milvus_utils",
        "minio_utils",
        "mq",
        "mq.connection",
    )
    for name in stub_names:
        saved[name] = sys.modules.get(name)

    aio_pika = types.ModuleType("aio_pika")
    aio_pika_abc = types.ModuleType("aio_pika.abc")
    aio_pika_abc.AbstractIncomingMessage = object
    aio_pika.abc = aio_pika_abc

    langchain_core = types.ModuleType("langchain_core")
    langchain_documents = types.ModuleType("langchain_core.documents")
    langchain_documents.Document = FakeDocument
    langchain_core.documents = langchain_documents

    utils = types.ModuleType("utils")
    utils.pdf_split = lambda *args, **kwargs: []
    utils.plain_text_split = lambda text, *args, **kwargs: []
    utils.markdown_split = lambda *args, **kwargs: []
    utils.json_split = lambda *args, **kwargs: []
    utils.code_split = lambda *args, **kwargs: []

    async def image_split(*args, **kwargs):
        return []

    utils.image_split = image_split
    utils.get_embedding_instance = lambda *args, **kwargs: object()

    milvus_utils = types.ModuleType("milvus_utils")
    milvus_utils.MilvusClientManager = FakeMilvusManager

    minio_utils = types.ModuleType("minio_utils")
    minio_utils.minio_client = FakeMinioClient()

    mq = types.ModuleType("mq")
    mq.__path__ = []
    mq_connection = types.ModuleType("mq.connection")
    mq_connection.rabbit_async_client = FakeRabbitClient()
    mq.connection = mq_connection

    stubs = {
        "aio_pika": aio_pika,
        "aio_pika.abc": aio_pika_abc,
        "langchain_core": langchain_core,
        "langchain_core.documents": langchain_documents,
        "utils": utils,
        "milvus_utils": milvus_utils,
        "minio_utils": minio_utils,
        "mq": mq,
        "mq.connection": mq_connection,
    }

    try:
        sys.modules.update(stubs)
        spec = importlib.util.spec_from_file_location(
            "rag_llm_document_embedding_under_test",
            MODULE_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


MODULE = _load_module_under_test()


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    rabbit = FakeRabbitClient()
    minio = FakeMinioClient()
    monkeypatch.setattr(MODULE, "rabbit_async_client", rabbit)
    monkeypatch.setattr(MODULE, "minio_client", minio)
    monkeypatch.setattr(MODULE.utils, "plain_text_split", lambda text, *args, **kwargs: [])
    monkeypatch.setattr(MODULE.utils, "markdown_split", lambda *args, **kwargs: [])
    monkeypatch.setattr(MODULE.utils, "pdf_split", lambda *args, **kwargs: [])
    monkeypatch.setattr(MODULE.utils, "json_split", lambda *args, **kwargs: [])
    monkeypatch.setattr(MODULE.utils, "code_split", lambda *args, **kwargs: [])
    monkeypatch.setattr(MODULE.utils, "get_embedding_instance", lambda *args, **kwargs: object())

    async def default_image_split(*args, **kwargs):
        return []

    monkeypatch.setattr(MODULE.utils, "image_split", default_image_split)
    monkeypatch.setattr(MODULE.MilvusClientManager, "store", None, raising=False)

    return {
        "rabbit": rabbit,
        "minio": minio,
        "tmp_path": tmp_path,
    }


def _body(**overrides):
    data = {
        "documentId": 42,
        "kbId": 7,
        "userId": 3900,
        "filePath": "docs/demo.txt",
        "fileName": "demo.txt",
        "bucketName": "general-rag",
    }
    data.update(overrides)
    return json.dumps(data).encode("utf-8")


def test_unsupported_file_type_reports_failure(env):
    message = FakeMessage(_body(filePath="docs/demo.csv", fileName="demo.csv"))

    asyncio.run(MODULE.document_embedding_consumer.on_receive_message(message))

    assert len(env["rabbit"].messages) == 1
    payload = env["rabbit"].messages[0]["message"]
    assert payload["status"] == "failed"
    assert "不支持的文件类型" in payload["message"]
    assert message.process_kwargs == {
        "requeue": True,
        "reject_on_redelivered": True,
    }
    assert message.process_context.exited


def test_empty_splits_reports_failure(env):
    message = FakeMessage(_body())

    asyncio.run(MODULE.document_embedding_consumer.on_receive_message(message))

    assert len(env["rabbit"].messages) == 1
    payload = env["rabbit"].messages[0]["message"]
    assert payload["status"] == "failed"
    assert "未解析出任何有效内容" in payload["message"]


def test_reprocess_deletes_stale_vectors(env, monkeypatch):
    splits = ["first", "second"]
    monkeypatch.setattr(
        MODULE.utils,
        "plain_text_split",
        lambda text, *args, **kwargs: splits,
    )
    store = FakeVectorStore(existing_pks=[1, 2])
    monkeypatch.setattr(MODULE.MilvusClientManager, "store", store, raising=False)

    message = FakeMessage(_body())

    asyncio.run(MODULE.document_embedding_consumer.on_receive_message(message))

    assert len(env["rabbit"].messages) == 1
    payload = env["rabbit"].messages[0]["message"]
    assert payload["status"] == "success"
    assert payload["chunksCount"] == 2
    assert store.client.deleted == [{
        "collection_name": "kb_1",
        "ids": [1, 2],
    }]
    assert [len(batch) for batch in store.added_batches] == [2]


def test_partial_write_failure_rolls_back_new_vectors(env, monkeypatch):
    splits = [f"chunk-{index}" for index in range(40)]
    monkeypatch.setattr(
        MODULE.utils,
        "plain_text_split",
        lambda text, *args, **kwargs: splits,
    )
    store = FakeVectorStore(existing_pks=[1, 2], fail_after=1)
    monkeypatch.setattr(MODULE.MilvusClientManager, "store", store, raising=False)

    message = FakeMessage(_body())

    asyncio.run(MODULE.document_embedding_consumer.on_receive_message(message))

    assert len(env["rabbit"].messages) == 1
    payload = env["rabbit"].messages[0]["message"]
    assert payload["status"] == "failed"
    assert "milvus write failed" in payload["message"]
    assert len(store.added_batches) == 1
    assert len(store.client.deleted) == 1
    assert store.client.deleted[0]["ids"] == list(range(100, 132))


def test_empty_vector_ids_keep_old_vectors(env, monkeypatch):
    monkeypatch.setattr(
        MODULE.utils,
        "plain_text_split",
        lambda text, *args, **kwargs: ["only-chunk"],
    )
    store = EmptyIdsVectorStore(existing_pks=[1, 2])
    monkeypatch.setattr(MODULE.MilvusClientManager, "store", store, raising=False)

    message = FakeMessage(_body())

    asyncio.run(MODULE.document_embedding_consumer.on_receive_message(message))

    assert len(env["rabbit"].messages) == 1
    payload = env["rabbit"].messages[0]["message"]
    assert payload["status"] == "failed"
    assert "向量写入数量与切片数量不一致" in payload["message"]
    assert store.client.deleted == []


def test_invalid_json_is_acked_without_failure_message(env):
    message = FakeMessage(b"{not-json")

    asyncio.run(MODULE.document_embedding_consumer.on_receive_message(message))

    assert env["rabbit"].messages == []
    assert message.process_kwargs == {
        "requeue": True,
        "reject_on_redelivered": True,
    }
    assert message.process_context.exited


def test_missing_document_id_is_dropped_without_failure_message(env):
    data = json.loads(_body())
    data.pop("documentId")
    message = FakeMessage(json.dumps(data).encode("utf-8"))

    asyncio.run(MODULE.document_embedding_consumer.on_receive_message(message))

    assert env["rabbit"].messages == []
    assert message.process_context.exited


def test_image_uses_matching_mime_type(env, monkeypatch):
    captured = {}

    async def fake_image_split(file_input, *args, **kwargs):
        captured["mime_type"] = kwargs.get("mime_type")
        return [FakeDocument("image description")]

    monkeypatch.setattr(MODULE.utils, "image_split", fake_image_split)
    store = FakeVectorStore()
    monkeypatch.setattr(MODULE.MilvusClientManager, "store", store, raising=False)

    message = FakeMessage(_body(filePath="docs/demo.PNG", fileName="demo.PNG"))

    asyncio.run(MODULE.document_embedding_consumer.on_receive_message(message))

    assert captured["mime_type"] == "image/png"
    assert len(env["rabbit"].messages) == 1
    assert env["rabbit"].messages[0]["message"]["status"] == "success"
