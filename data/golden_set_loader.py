"""golden_set.yaml을 GoldenSetItem 리스트로 변환한다."""

from dataclasses import dataclass
from pathlib import Path

import yaml

_VALID_SCORES = {0, 1, 2, 3}


@dataclass
class GoldenSetItem:
    item_id: str
    question_text: str
    expected_chunks: list[str]  # 정답으로 인정할 chunk_id 목록. 같은 개념을 여러 소스(예: 이론+실습)가
    # 설명하는 경우가 있어 복수 허용한다(gs-23 사례, docs/EXPERIMENTS.md "gs-23 회귀 원인 심층 분석" 참고).
    relevance_score: int
    ground_truth: str | None = None  # RAGAS Context Recall용 정답 텍스트 (Part 6-1). 없으면 RAGAS 평가 대상에서 제외.


def load_golden_set(path: Path) -> list[GoldenSetItem]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    items = []
    for entry in data.get("golden_set", []):
        score = entry["relevance_score"]
        if score not in _VALID_SCORES:
            raise ValueError(
                f"invalid relevance_score {score!r} for item {entry.get('item_id')!r} "
                f"(must be one of {sorted(_VALID_SCORES)})"
            )
        items.append(
            GoldenSetItem(
                item_id=entry["item_id"],
                question_text=entry["question_text"],
                expected_chunks=list(entry["expected_chunks"]),
                relevance_score=score,
                ground_truth=entry.get("ground_truth"),
            )
        )
    return items
