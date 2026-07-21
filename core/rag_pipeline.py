"""RetrieveAndRerank — 하이브리드 검색(넓게) + Cross-Encoder 재랭킹(좁게) 2-stage 오케스트레이션.

Part 4-2의 "넓게 수집, 정밀 선발" 원칙 그대로: hybrid_top_n(기본 20)으로 넓게 후보를 모은 뒤
rerank_top_n(기본 5)으로 좁혀 LLM에 전달한다. core/scoring.py는 이 모듈에서 한 줄도 바뀌지 않는다.
"""

from typing import Callable

from core.reranking import rerank_candidates
from core.scoring import execute_hybrid_search


def retrieve_and_rerank(
    query_text: str,
    vector_store,
    bm25_index,
    chunk_by_id: dict,
    alpha: float,
    hybrid_top_n: int = 20,
    rerank_top_n: int = 5,
    threshold: float = 0.0,
    week_filter: int | None = None,
    id_week_map: dict[str, int] | None = None,
    embed_fn: Callable[[str], object] | None = None,
    score_fn: Callable[[str, list[str]], list[float]] | None = None,
) -> list[tuple[str, float]]:
    candidates = execute_hybrid_search(
        query_text,
        vector_store,
        bm25_index,
        alpha=alpha,
        top_n=hybrid_top_n,
        threshold=threshold,
        embed_fn=embed_fn,
        week_filter=week_filter,
        id_week_map=id_week_map,
    )
    if not candidates:
        return []

    return rerank_candidates(query_text, candidates, chunk_by_id, top_n=rerank_top_n, score_fn=score_fn)
