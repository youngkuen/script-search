from dataclasses import dataclass

from core.rag_pipeline import retrieve_and_rerank


@dataclass
class FakeChunk:
    text: str


class FakeVectorStore:
    def __init__(self, results):
        self._results = results

    def query(self, vector, top_k=5, where=None):
        return self._results[:top_k]


class FakeBM25Index:
    def __init__(self, results):
        self._results = results

    def search(self, query_text, top_k=5):
        return self._results[:top_k]


def fake_embed(text: str):
    return [0.0]


def test_composes_hybrid_search_then_rerank():
    # 하이브리드 검색은 a/b/c 순으로 나오지만, 재랭킹 fake는 순서를 뒤집는다.
    vector_store = FakeVectorStore([("a", 0.0), ("b", 0.5), ("c", 1.0)])
    bm25_index = FakeBM25Index([("a", 3.0), ("b", 2.0), ("c", 1.0)])
    chunk_by_id = {"a": FakeChunk("a-text"), "b": FakeChunk("b-text"), "c": FakeChunk("c-text")}

    def reverse_rerank_score_fn(query_text, texts):
        # 원래 순서를 뒤집는 점수를 부여 (마지막 텍스트가 가장 높은 점수)
        return list(range(len(texts)))

    result = retrieve_and_rerank(
        "쿼리", vector_store, bm25_index, chunk_by_id, alpha=1.0,
        hybrid_top_n=10, rerank_top_n=2, embed_fn=fake_embed, score_fn=reverse_rerank_score_fn,
    )

    assert len(result) == 2
    assert result[0][0] == "c"  # rerank fake가 마지막 후보에 가장 높은 점수를 줌


def test_empty_hybrid_search_short_circuits_without_reranking():
    vector_store = FakeVectorStore([])
    bm25_index = FakeBM25Index([])

    called = {"rerank": False}

    def spy_score_fn(query_text, texts):
        called["rerank"] = True
        return []

    result = retrieve_and_rerank(
        "쿼리", vector_store, bm25_index, {}, alpha=0.5,
        embed_fn=fake_embed, score_fn=spy_score_fn,
    )

    assert result == []
    assert called["rerank"] is False
