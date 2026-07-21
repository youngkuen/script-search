from pathlib import Path

import pytest

from data.golden_set_loader import load_golden_set


def test_load_valid_golden_set(tmp_path: Path):
    f = tmp_path / "golden_set.yaml"
    f.write_text(
        """
golden_set:
  - item_id: "gs-01"
    question_text: "질문 하나"
    expected_chunk: "파일_0"
    relevance_score: 3
  - item_id: "gs-02"
    question_text: "질문 둘"
    expected_chunk: "파일_1"
    relevance_score: 2
""",
        encoding="utf-8",
    )

    items = load_golden_set(f)

    assert len(items) == 2
    assert items[0].item_id == "gs-01"
    assert items[0].relevance_score == 3
    assert items[1].expected_chunk == "파일_1"


def test_load_golden_set_rejects_out_of_range_score(tmp_path: Path):
    f = tmp_path / "golden_set.yaml"
    f.write_text(
        """
golden_set:
  - item_id: "gs-01"
    question_text: "잘못된 점수"
    expected_chunk: "파일_0"
    relevance_score: 5
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_golden_set(f)


def test_load_empty_golden_set(tmp_path: Path):
    f = tmp_path / "golden_set.yaml"
    f.write_text("golden_set: []\n", encoding="utf-8")

    assert load_golden_set(f) == []


def test_ground_truth_defaults_to_none_when_absent(tmp_path: Path):
    f = tmp_path / "golden_set.yaml"
    f.write_text(
        """
golden_set:
  - item_id: "gs-01"
    question_text: "질문"
    expected_chunk: "파일_0"
    relevance_score: 3
""",
        encoding="utf-8",
    )

    items = load_golden_set(f)

    assert items[0].ground_truth is None


def test_ground_truth_parses_when_present(tmp_path: Path):
    f = tmp_path / "golden_set.yaml"
    f.write_text(
        """
golden_set:
  - item_id: "gs-01"
    question_text: "질문"
    expected_chunk: "파일_0"
    relevance_score: 3
    ground_truth: "정답 텍스트입니다."
""",
        encoding="utf-8",
    )

    items = load_golden_set(f)

    assert items[0].ground_truth == "정답 텍스트입니다."
