"""Cross-Encoder Re-ranking — 하이브리드 검색 후보를 쿼리와 함께 정밀 재평가한다 (Part 4-2).

강의 스크립트가 한국어라 영어 전용 MS MARCO 모델 대신 다국어 Cross-Encoder를 쓴다
(data/embedding_model.py가 다국어 e5 임베딩을 쓰는 것과 같은 이유).
"""

import torch
from sentence_transformers import CrossEncoder

CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

_model: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(CROSS_ENCODER_MODEL)
    return _model


def score_pairs(query: str, texts: list[str]) -> list[float]:
    """(query, text) 쌍마다 0~1로 정규화된 관련성 점수를 반환한다. 순서는 texts와 동일하다.

    Sigmoid 활성화(activation_fn)를 명시한다 — 기본값(원시 로짓)은 범위가 정해져 있지
    않아(음수~두 자릿수) core/confidence.py의 신뢰도 임계값(0.7/0.4)이 무의미해진다
    (4-2 강의: "activation_fn=Sigmoid를 추가하면 0~1로 정규화된다").
    """
    if not texts:
        return []
    pairs = [(query, text) for text in texts]
    scores = get_reranker().predict(pairs, activation_fn=torch.nn.Sigmoid())
    return [float(s) for s in scores]
