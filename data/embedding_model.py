"""로컬 다국어 임베딩 모델 래퍼 (intfloat/multilingual-e5-small).

e5 계열은 비대칭(asymmetric) 임베딩 모델이라, 문서에는 "passage: ",
질문에는 "query: " 프리픽스를 붙여야 검색 성능이 제대로 나온다 (Part 3-1 비대칭 임베딩).
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode(text: str):
    return get_model().encode(text)


def encode_batch(texts: list[str]):
    return get_model().encode(texts)


def encode_query(text: str):
    """질문(쿼리) 인코딩 — e5 'query: ' 프리픽스 적용 (검색 타임)."""
    return get_model().encode(f"query: {text}")


def encode_passage(text: str):
    """문서(청크) 인코딩 — e5 'passage: ' 프리픽스 적용 (인덱싱 타임)."""
    return get_model().encode(f"passage: {text}")


def token_len(text: str) -> int:
    """모델 토크나이저 기준 토큰 수 (토큰 예산 청킹용)."""
    return len(get_model().tokenizer.encode(text, add_special_tokens=False))
