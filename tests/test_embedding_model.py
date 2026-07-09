from data.embedding_model import encode, encode_batch


def test_encode_returns_vector():
    vector = encode("쿠버네티스 파드는 어떻게 작동하나요?")

    assert len(vector) > 0


def test_encode_batch_returns_consistent_dimension():
    vectors = encode_batch(["첫 번째 문장", "두 번째 문장"])

    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])
