import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_mcp import server


def test_result_summary_includes_retrieval_signals():
    summary = server._result_summary({
        "knowledgeBaseId": 7,
        "totalHits": 2,
        "matchedTotal": 10,
        "keptBeforeTopK": 5,
        "topScore": 0.82,
        "scoreThreshold": 0.55,
        "truncated": True,
        "failedQueries": 1,
        "results": [
            {
                "chunkId": 1,
                "documentId": 2,
                "fileName": "a.md",
                "chunkIndex": 0,
                "totalChunks": 3,
                "score": 0.82,
                "scoreType": "vector",
                "content": "正文",
            }
        ],
    })

    assert summary["matchedTotal"] == 10
    assert summary["keptBeforeTopK"] == 5
    assert summary["topScore"] == 0.82
    assert summary["scoreThreshold"] == 0.55
    assert summary["truncated"] is True
    assert summary["failedQueries"] == 1
    assert summary["returnedCount"] == 1
    assert summary["items"][0]["chunkId"] == 1
