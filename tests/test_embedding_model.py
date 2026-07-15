from data.embedding_model import (
    encode,
    encode_batch,
    encode_passage,
    encode_query,
    token_len,
)


def test_encode_returns_vector():
    vector = encode("쿠버네티스 파드는 어떻게 작동하나요?")

    assert len(vector) > 0


def test_encode_batch_returns_consistent_dimension():
    vectors = encode_batch(["첫 번째 문장", "두 번째 문장"])

    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])


def test_query_and_passage_prefixes_produce_different_vectors():
    # e5 비대칭: 같은 텍스트라도 query/passage 프리픽스가 다르면 벡터가 달라야 한다.
    q = encode_query("연차 신청 방법")
    p = encode_passage("연차 신청 방법")

    assert len(q) == len(p)
    assert list(q) != list(p)


def test_token_len_is_positive():
    assert token_len("hello world") > 0
