from dataclasses import dataclass

from core.reranking import rerank_candidates


@dataclass
class FakeChunk:
    text: str


def fake_score_fn(query_text: str, texts: list[str]) -> list[float]:
    # 텍스트 길이를 점수로 삼는 결정론적 fake — 순서 검증에 사용
    return [float(len(t)) for t in texts]


def test_reranks_by_score_and_truncates_to_top_n():
    chunk_by_id = {"a": FakeChunk("짧음"), "b": FakeChunk("이건 더 긴 텍스트입니다"), "c": FakeChunk("중간정도길이텍스트")}
    candidates = [("a", 0.9), ("b", 0.1), ("c", 0.5)]  # 원래 하이브리드 점수는 무시되고 재랭킹 점수로 재정렬됨

    result = rerank_candidates("쿼리", candidates, chunk_by_id, top_n=2, score_fn=fake_score_fn)

    assert len(result) == 2
    assert result[0][0] == "b"  # 가장 긴 텍스트가 1위


def test_candidate_missing_from_chunk_by_id_is_skipped():
    chunk_by_id = {"a": FakeChunk("텍스트")}
    candidates = [("a", 0.9), ("missing", 0.5)]

    result = rerank_candidates("쿼리", candidates, chunk_by_id, top_n=5, score_fn=fake_score_fn)

    ids = {chunk_id for chunk_id, _ in result}
    assert "missing" not in ids
    assert "a" in ids


def test_empty_candidates_returns_empty_list():
    result = rerank_candidates("쿼리", [], {}, top_n=5, score_fn=fake_score_fn)
    assert result == []


def test_all_candidates_missing_returns_empty_list():
    result = rerank_candidates("쿼리", [("x", 0.9)], {}, top_n=5, score_fn=fake_score_fn)
    assert result == []


def test_score_fn_receives_query_and_ordered_texts():
    received = {}

    def recording_score_fn(query_text, texts):
        received["query"] = query_text
        received["texts"] = texts
        return [1.0] * len(texts)

    chunk_by_id = {"a": FakeChunk("텍스트A"), "b": FakeChunk("텍스트B")}
    rerank_candidates("내 질문", [("a", 0.1), ("b", 0.2)], chunk_by_id, top_n=5, score_fn=recording_score_fn)

    assert received["query"] == "내 질문"
    assert received["texts"] == ["텍스트A", "텍스트B"]
