from pathlib import Path

from core.chunking import chunk_script, chunk_text
from data.script_loader import ScriptFile


def test_chunk_text_splits_on_timestamps():
    raw = "[00:00] 첫 번째 문장.\n\n[00:18] 두 번째 문장.\n\n[01:05] 세 번째 문장."

    chunks = chunk_text(raw)

    assert len(chunks) == 3
    assert chunks[0] == ("00:00", "00:18", "[00:00] 첫 번째 문장.")
    assert chunks[1] == ("00:18", "01:05", "[00:18] 두 번째 문장.")


def test_chunk_text_last_block_runs_to_end():
    raw = "[00:00] 첫 문장.\n\n[00:10] 마지막 문장."

    chunks = chunk_text(raw)

    last_start, last_end, last_text = chunks[-1]
    assert last_start == "00:10"
    assert last_end == "00:10"  # 마지막 블록은 start=end로 표시
    assert "마지막 문장" in last_text


def test_chunk_text_no_timestamps_returns_empty():
    assert chunk_text("타임스탬프가 없는 그냥 텍스트") == []


def test_chunk_text_empty_string_returns_empty():
    assert chunk_text("") == []


def test_chunk_script_skips_invalid_files():
    invalid = ScriptFile(file_path=Path("broken.txt"), week_number=1, status="skipped_empty")

    assert chunk_script(invalid) == []


def test_chunk_script_assigns_sequential_chunk_ids():
    valid = ScriptFile(
        file_path=Path("1차시.txt"),
        week_number=1,
        status="valid",
        raw_text="[00:00] 첫 문장.\n\n[00:20] 둘째 문장.",
    )

    chunks = chunk_script(valid)

    assert [c.chunk_id for c in chunks] == ["1차시_0", "1차시_1"]
    assert chunks[0].script_file == Path("1차시.txt")
