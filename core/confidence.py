"""신뢰도 등급 — 검색/재랭킹 점수를 사용자 친화적인 등급으로 변환한다 (Part 5-2/6-1).

주의: 여기 임계값(0.7/0.4)은 출발점일 뿐이다. 실제 서비스에서는 사용하는 임베딩·재랭킹 모델과
도메인에 맞게 샘플 테스트로 조정해야 한다(6-1 강의에서 반복 강조된 원칙). 이 프로젝트의
score는 core/scoring.py에서 이미 0~1로 정규화되고 "높을수록 유사"로 뒤집혀 있으므로(Chroma의
raw distance가 아님), 방향 혼동 없이 이 임계값을 그대로 적용할 수 있다.
"""

HIGH_THRESHOLD = 0.7
MEDIUM_THRESHOLD = 0.4


def confidence_label(score: float, high: float = HIGH_THRESHOLD, medium: float = MEDIUM_THRESHOLD) -> str:
    if score >= high:
        return "높음"
    if score >= medium:
        return "보통"
    return "낮음"
