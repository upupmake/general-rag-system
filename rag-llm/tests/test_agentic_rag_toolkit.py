import asyncio
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import agentic_rag_toolkit as toolkit_module
from agentic_rag_toolkit import (
    ExpandContextInput,
    KeywordSearchInput,
    ReadFileChunksInput,
    RetrievalToolkit,
    SemanticSearchInput,
)


class FakeEmbeddingFunction:
    def __init__(self):
        self.document_calls = []
        self.query_calls = []

    async def aembed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [[float(index + 1)] for index in range(len(texts))]

    async def aembed_query(self, text):
        self.query_calls.append(text)
        return [1.0]


class FakeAsyncMilvus:
    def __init__(self, rows=None, count=0):
        self.rows = rows or []
        self.count = count
        self.calls = []

    async def query(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("output_fields") == ["count(*)"]:
            return [{"count(*)": self.count}]
        return self.rows


class FakeVectorStore:
    def __init__(self, rows=None, count=0, search_results=None):
        self.collection_name = "kb_1"
        self.aclient = FakeAsyncMilvus(rows=rows, count=count)
        self.embedding_func = FakeEmbeddingFunction()
        self.search_results = search_results or []
        self.search_calls = []

    async def asimilarity_search_with_score_by_vector(self, embedding, k=4, expr=None, **kwargs):
        self.search_calls.append({"embedding": embedding, "k": k, "expr": expr})
        return self.search_results


def _row(pk, text, document_id=1, chunk_index=0, file_name="a.md"):
    return {
        "pk": pk,
        "text": text,
        "documentId": document_id,
        "chunkIndex": chunk_index,
        "fileName": file_name,
        "maxChunkIndex": 3,
    }


def _doc(pk, text, score, document_id=1, chunk_index=0, file_name="a.md"):
    return Document(
        page_content=text,
        metadata={
            "pk": pk,
            "documentId": document_id,
            "chunkIndex": chunk_index,
            "fileName": file_name,
            "maxChunkIndex": 3,
            "score": score,
        },
    )


def test_escape_like_escapes_wildcards_for_content_search():
    assert "learning\\\\_rate" == RetrievalToolkit._escape_like("learning_rate")
    assert "50\\\\%\\\\_x" == RetrievalToolkit._escape_like("50%_x")
    escaped = RetrievalToolkit._escape_like('a"b\\c')
    assert '\\"' in escaped
    assert "\\\\\\\\" in escaped
    assert RetrievalToolkit._escape_like("%手册%", escape_wildcards=False) == "%手册%"


def test_lexical_score_prefers_keyword_coverage_and_frequency():
    both = Document(
        page_content="alpha alpha beta",
        metadata={"fileName": "alpha.md"},
    )
    one = Document(
        page_content="alpha alpha alpha alpha",
        metadata={"fileName": "other.md"},
    )
    assert RetrievalToolkit._lexical_score(both, ["alpha", "beta"]) > RetrievalToolkit._lexical_score(
        one, ["alpha", "beta"]
    )


def test_keyword_search_overfetches_sorts_and_reports_total():
    rows = [
        _row(1, "alpha beta", file_name="alpha.md"),
        _row(2, "alpha"),
        _row(3, "alpha alpha alpha"),
    ]
    store = FakeVectorStore(rows=rows, count=3)
    toolkit = RetrievalToolkit(store)

    result = asyncio.run(toolkit._search_by_grep(keywords=["alpha", "beta"], top_k=2))

    assert result["matched_total"] == 3
    assert result["total_hits"] == 2
    assert result["truncated"] is True
    assert result["results"][0].metadata["pk"] == 1
    milvus_calls = [
        call for call in store.aclient.calls
        if call.get("output_fields") != ["count(*)"]
    ]
    assert milvus_calls[-1]["limit"] == 10
    assert 'text like "%alpha%"' in milvus_calls[-1]["filter"]


def test_keyword_search_escapes_like_wildcards():
    store = FakeVectorStore(rows=[], count=0)
    toolkit = RetrievalToolkit(store)

    asyncio.run(toolkit._search_by_grep(keywords=["learning_rate"], top_k=1))

    milvus_calls = [
        call for call in store.aclient.calls
        if call.get("output_fields") != ["count(*)"]
    ]
    assert "learning\\\\_rate" in milvus_calls[-1]["filter"]


def test_keyword_search_builds_document_scope_and_and_mode():
    store = FakeVectorStore(rows=[], count=0)
    toolkit = RetrievalToolkit(store)

    asyncio.run(toolkit._search_by_grep(
        keywords=["alpha", "beta"],
        match_mode="AND",
        document_ids=[3, 4],
        top_k=1,
    ))

    milvus_calls = [
        call for call in store.aclient.calls
        if call.get("output_fields") != ["count(*)"]
    ]
    assert "(documentId == 3 OR documentId == 4) and" in milvus_calls[-1]["filter"]
    assert 'text like "%alpha%" AND text like "%beta%"' in milvus_calls[-1]["filter"]


def test_find_files_preserves_user_wildcards():
    store = FakeVectorStore(rows=[], count=0)
    toolkit = RetrievalToolkit(store)

    asyncio.run(toolkit._list_filename_by_like("%手册%"))

    call = store.aclient.calls[-1]
    assert 'fileName like "%手册%"' in call["filter"]
    assert call["order_by"] == "fileName"


def test_semantic_search_batches_queries_reuses_vectors_and_reports_signals(monkeypatch):
    docs = [
        _doc(1, "alpha", 0.8),
        _doc(2, "beta", 0.4),
    ]

    def fake_filter(documents):
        return {"documents": list(documents), "threshold": 0.5}

    monkeypatch.setattr(toolkit_module, "filter_grade_threshold", fake_filter)
    store = FakeVectorStore(search_results=[(docs[0], 0.8), (docs[1], 0.4)])
    toolkit = RetrievalToolkit(store)

    result = asyncio.run(
        toolkit._search_by_multi_queries_in_database(
            queries=["alpha beta", "gamma delta"],
            top_k=1,
            exclude_pks={99, 100, 101},
        )
    )

    assert store.embedding_func.document_calls == [["alpha beta", "gamma delta"]]
    assert store.embedding_func.query_calls == []
    # 2 queries × (向量召回 + 关键词补召回)
    assert len(store.search_calls) == 4
    # max(top_k * 3, 15) + min(exclude_count, 200) = 15 + 3
    assert all(call["k"] == 18 for call in store.search_calls)
    assert result["kept_before_top_k"] == 2
    assert result["top_score"] == 0.8
    assert result["threshold"] == 0.5
    assert result["truncated"] is True
    assert result["total_hits"] == 1
    assert result["failed_queries"] == 0


def test_semantic_search_falls_back_when_batch_embedding_fails(monkeypatch):
    class FailingBatchEmbedding(FakeEmbeddingFunction):
        async def aembed_documents(self, texts):
            raise RuntimeError("batch embedding failed")

    docs = [_doc(1, "alpha", 0.8)]

    def fake_filter(documents):
        return {"documents": list(documents), "threshold": 0.5}

    monkeypatch.setattr(toolkit_module, "filter_grade_threshold", fake_filter)
    store = FakeVectorStore(search_results=[(docs[0], 0.8)])
    store.embedding_func = FailingBatchEmbedding()
    toolkit = RetrievalToolkit(store)

    result = asyncio.run(
        toolkit._search_by_multi_queries_in_database(
            queries=["alpha beta", "gamma delta"],
            top_k=1,
        )
    )

    assert store.embedding_func.query_calls == ["alpha beta", "gamma delta"]
    assert len(store.search_calls) == 4
    assert result["total_hits"] == 1
    assert result["top_score"] == 0.8


def test_keyword_search_degrades_when_count_fails():
    class CountFailingMilvus(FakeAsyncMilvus):
        async def query(self, **kwargs):
            if kwargs.get("output_fields") == ["count(*)"]:
                raise RuntimeError("count failed")
            return await super().query(**kwargs)

    store = FakeVectorStore(rows=[_row(1, "alpha")], count=1)
    store.aclient = CountFailingMilvus(rows=store.aclient.rows, count=store.aclient.count)
    toolkit = RetrievalToolkit(store)

    result = asyncio.run(toolkit._search_by_grep(keywords=["alpha"], top_k=1))

    assert result["matched_total"] is None
    assert result["total_hits"] == 1
    assert result["truncated"] is False


def test_tool_schema_bounds_and_chunk_range_validation():
    with pytest.raises(ValidationError):
        KeywordSearchInput(keywords=[])
    with pytest.raises(ValidationError):
        SemanticSearchInput(queries=[])
    with pytest.raises(ValidationError):
        SemanticSearchInput(queries=["q"] * 11)
    with pytest.raises(ValidationError):
        SemanticSearchInput(queries=["q"], top_k=0)
    with pytest.raises(ValidationError):
        ReadFileChunksInput(document_id=1, start_chunk_index=5, end_chunk_index=4)
    with pytest.raises(ValidationError):
        ExpandContextInput(document_id=1, chunk_index=0, window_size=10)


def test_execute_tool_validates_special_cased_tools():
    store = FakeVectorStore(rows=[], count=0)
    toolkit = RetrievalToolkit(store)

    with pytest.raises(ValidationError):
        asyncio.run(toolkit.execute_tool("keyword_search", {"keywords": [], "top_k": 1}))
    with pytest.raises(ValidationError):
        asyncio.run(toolkit.execute_tool("semantic_search", {"queries": ["q"], "top_k": 0}))
