from core.scoring import execute_hybrid_search


class FakeVectorStore:
    def __init__(self, results):
        self._results = results

    def query(self, vector, top_k=5):
        return self._results[:top_k]


class FakeBM25Index:
    def __init__(self, results):
        self._results = results

    def search(self, query_text, top_k=5):
        return self._results[:top_k]


def fake_embed(text: str):
    return [0.0]  # FakeVectorStore ignores the vector value


def test_hybrid_score_formula_vector_only():
    vector_store = FakeVectorStore([("a", 0.0), ("b", 0.5), ("c", 1.0)])
    bm25_index = FakeBM25Index([("a", 2.0), ("b", 4.0), ("c", 6.0)])

    results = execute_hybrid_search(
        "쿼리", vector_store, bm25_index, alpha=1.0, top_n=10, threshold=0.0, embed_fn=fake_embed
    )

    assert results == [("a", 1.0), ("b", 0.5), ("c", 0.0)]


def test_threshold_boundary_is_inclusive():
    vector_store = FakeVectorStore([("a", 0.0), ("b", 0.5), ("c", 1.0)])
    bm25_index = FakeBM25Index([("a", 2.0), ("b", 4.0), ("c", 6.0)])

    results = execute_hybrid_search(
        "쿼리", vector_store, bm25_index, alpha=1.0, top_n=10, threshold=0.5, embed_fn=fake_embed
    )

    assert results == [("a", 1.0), ("b", 0.5)]


def test_all_below_threshold_returns_empty_list():
    vector_store = FakeVectorStore([("a", 0.0), ("b", 0.5), ("c", 1.0)])
    bm25_index = FakeBM25Index([("a", 2.0), ("b", 4.0), ("c", 6.0)])

    results = execute_hybrid_search(
        "쿼리", vector_store, bm25_index, alpha=1.0, top_n=10, threshold=2.0, embed_fn=fake_embed
    )

    assert results == []


def test_top_n_limits_result_count():
    vector_store = FakeVectorStore([("a", 0.0), ("b", 0.5), ("c", 1.0)])
    bm25_index = FakeBM25Index([("a", 2.0), ("b", 4.0), ("c", 6.0)])

    results = execute_hybrid_search(
        "쿼리", vector_store, bm25_index, alpha=1.0, top_n=1, threshold=0.0, embed_fn=fake_embed
    )

    assert results == [("a", 1.0)]


def test_candidate_missing_from_one_source_defaults_to_zero():
    vector_store = FakeVectorStore([("a", 0.0)])
    bm25_index = FakeBM25Index([("d", 5.0)])

    results = execute_hybrid_search(
        "쿼리", vector_store, bm25_index, alpha=0.5, top_n=10, threshold=0.0, embed_fn=fake_embed
    )

    result_ids = {r[0] for r in results}
    assert result_ids == {"a", "d"}
