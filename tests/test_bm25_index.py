from data.bm25_index import BM25Index


def test_search_ranks_exact_keyword_match_first():
    ids = ["a", "b", "c"]
    texts = [
        "쿠버네티스 파드는 컨테이너를 포함한다",
        "도커 이미지를 빌드하는 방법",
        "파이썬 비동기 프로그래밍 asyncio",
    ]
    index = BM25Index(ids, texts)

    results = index.search("쿠버네티스 파드", top_k=3)

    assert results[0][0] == "a"


def test_search_respects_top_k():
    ids = ["a", "b", "c"]
    texts = ["강의 내용 하나", "강의 내용 둘", "강의 내용 셋"]
    index = BM25Index(ids, texts)

    results = index.search("강의", top_k=2)

    assert len(results) == 2
