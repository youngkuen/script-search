"""ExecuteHybridSearch — Vector·BM25 점수를 alpha로 가중합해 검색 결과를 만든다."""

from typing import Callable


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def execute_hybrid_search(
    query_text: str,
    vector_store,
    bm25_index,
    alpha: float,
    top_n: int = 5,
    threshold: float = 0.0,
    candidate_k: int = 20,
    embed_fn: Callable[[str], object] | None = None,
    week_filter: int | None = None,
    id_week_map: dict[str, int] | None = None,
) -> list[tuple[str, float]]:
    if embed_fn is None:
        # 질문(쿼리)은 query 프리픽스로 인코딩한다 (e5 비대칭 임베딩).
        from data.embedding_model import encode_query as embed_fn

    query_vector = embed_fn(query_text)

    # 주차 필터: 벡터 쪽은 chromadb where 조건으로, BM25 쪽은 후보를 주차로 걸러낸다.
    if week_filter is not None:
        vector_results = vector_store.query(
            query_vector, top_k=candidate_k, where={"week": {"$eq": week_filter}}
        )
        bm25_results = bm25_index.search(query_text, top_k=candidate_k * 10)
        if id_week_map is not None:
            bm25_results = [
                (cid, s) for cid, s in bm25_results if id_week_map.get(cid) == week_filter
            ][:candidate_k]
    else:
        vector_results = vector_store.query(query_vector, top_k=candidate_k)
        bm25_results = bm25_index.search(query_text, top_k=candidate_k)

    # chromadb의 query()는 distance(낮을수록 유사)를 반환한다.
    # 부호를 뒤집어 "높을수록 유사"로 맞춘 뒤 min-max 정규화한다.
    vector_similarity = {chunk_id: -dist for chunk_id, dist in vector_results}
    bm25_scores = {chunk_id: score for chunk_id, score in bm25_results}

    norm_vector = _min_max_normalize(vector_similarity)
    norm_bm25 = _min_max_normalize(bm25_scores)

    candidate_ids = set(norm_vector) | set(norm_bm25)
    hybrid_scores = {
        chunk_id: alpha * norm_vector.get(chunk_id, 0.0) + (1 - alpha) * norm_bm25.get(chunk_id, 0.0)
        for chunk_id in candidate_ids
    }

    ranked = sorted(hybrid_scores.items(), key=lambda pair: pair[1], reverse=True)
    filtered = [(chunk_id, score) for chunk_id, score in ranked if score >= threshold]
    return filtered[:top_n]
