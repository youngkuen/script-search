"""로컬 다국어 임베딩 모델 래퍼 (intfloat/multilingual-e5-small)."""

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
