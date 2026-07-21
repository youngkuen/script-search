"""RerankCandidates — Cross-Encoder로 하이브리드 검색 후보를 정밀 재평가한다 (Part 4-2).

Bi-Encoder(하이브리드 검색)로 넓게 뽑은 후보를 Cross-Encoder로 좁혀 선발하는 2-stage 구조의
2단계다. 1단계(core/scoring.py)는 건드리지 않는다.
"""

from typing import Callable


def rerank_candidates(
    query_text: str,
    candidates: list[tuple[str, float]],
    chunk_by_id: dict,
    top_n: int,
    score_fn: Callable[[str, list[str]], list[float]] | None = None,
) -> list[tuple[str, float]]:
    if score_fn is None:
        from data.reranker import score_pairs as score_fn

    # chunk_by_id에 없는 후보(원본 청크 정보 유실)는 재랭킹 대상에서 제외한다.
    scoreable = [(chunk_id, chunk_by_id[chunk_id].text) for chunk_id, _ in candidates if chunk_id in chunk_by_id]
    if not scoreable:
        return []

    chunk_ids = [chunk_id for chunk_id, _ in scoreable]
    texts = [text for _, text in scoreable]
    rerank_scores = score_fn(query_text, texts)

    ranked = sorted(zip(chunk_ids, rerank_scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_n]
