"""신뢰도 등급 — 검색/재랭킹 점수를 사용자 친화적인 등급으로 변환한다 (Part 5-2/6-1).

주의: 여기 임계값(0.7/0.4)은 출발점일 뿐이다. 실제 서비스에서는 사용하는 임베딩·재랭킹 모델과
도메인에 맞게 샘플 테스트로 조정해야 한다(6-1 강의에서 반복 강조된 원칙). 이 함수가 실제로
받는 score는 core/rag_chain.py를 거쳐 core/reranking.py의 재랭킹 결과, 즉
data/reranker.py::score_pairs()가 반환하는 값이다 — Sigmoid 활성화로 0~1 범위로 정규화되어
있으므로("높을수록 유사") 이 임계값을 그대로 적용할 수 있다. (2026-07-22: score_pairs가
원시 Cross-Encoder 로짓을 그대로 반환하던 시절엔 이 임계값이 잘못된 스케일에 적용되고
있었다 — 실제 CLI 실행으로 발견해 수정함, docs/adr.yaml ADR-008 참고.)
"""

HIGH_THRESHOLD = 0.7
MEDIUM_THRESHOLD = 0.4


def confidence_label(score: float, high: float = HIGH_THRESHOLD, medium: float = MEDIUM_THRESHOLD) -> str:
    if score >= high:
        return "높음"
    if score >= medium:
        return "보통"
    return "낮음"
