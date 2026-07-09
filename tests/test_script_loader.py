from pathlib import Path

from data.script_loader import load_all, load_script_file


def test_valid_file(tmp_path: Path):
    f = tmp_path / "sample.txt"
    f.write_text("안녕하세요 강의 내용입니다", encoding="utf-8")

    result = load_script_file(f, week_number=1)

    assert result.status == "valid"
    assert result.raw_text == "안녕하세요 강의 내용입니다"
    assert result.week_number == 1


def test_empty_file(tmp_path: Path):
    f = tmp_path / "empty.txt"
    f.write_text("   \n  ", encoding="utf-8")

    result = load_script_file(f, week_number=2)

    assert result.status == "skipped_empty"
    assert result.raw_text is None


def test_encoding_error_file(tmp_path: Path):
    f = tmp_path / "broken.txt"
    f.write_bytes(b"\xff\xfe\x00\xff\x00\xfe")  # invalid utf-8

    result = load_script_file(f, week_number=3)

    assert result.status == "skipped_encoding_error"
    assert result.raw_text is None


def test_load_all_skips_invalid_and_continues(tmp_path: Path):
    week1_dir = tmp_path / "1주차"
    week1_dir.mkdir()
    (week1_dir / "a.txt").write_text("첫 번째 강의", encoding="utf-8")
    (week1_dir / "b.txt").write_text("", encoding="utf-8")

    week2_dir = tmp_path / "2주차"
    week2_dir.mkdir()
    (week2_dir / "c.txt").write_text("두 번째 강의", encoding="utf-8")

    results = load_all({1: week1_dir, 2: week2_dir})

    assert len(results) == 3
    statuses = sorted(r.status for r in results)
    assert statuses == ["skipped_empty", "valid", "valid"]
