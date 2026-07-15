from pathlib import Path

from core.chunking import (
    chunk_script,
    chunk_text,
    chunk_text_by_tokens,
    chunk_text_merged,
    normalize_text,
)
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


def test_normalize_text_strips_invisibles_and_nbsp():
    dirty = "연차​신청 방법﻿"  # ZWSP, NBSP, BOM 삽입
    clean = normalize_text(dirty)

    assert "​" not in clean and "﻿" not in clean
    assert " " not in clean  # NBSP는 일반 공백으로 치환
    assert clean == "연차신청 방법"


def test_chunk_text_merged_groups_blocks():
    raw = "[00:00] A.\n\n[00:10] B.\n\n[00:20] C.\n\n[00:30] D."

    merged = chunk_text_merged(raw, group_size=2)

    assert len(merged) == 2
    assert merged[0][0] == "00:00"  # 첫 그룹의 시작
    assert merged[0][1] == "00:20"  # 그룹 마지막 블록의 end
    assert "A." in merged[0][2] and "B." in merged[0][2]


def test_chunk_text_by_tokens_respects_budget():
    raw = "[00:00] a.\n\n[00:10] b.\n\n[00:20] c."
    # 블록마다 2토큰으로 세는 가짜 토크나이저 + 예산 3 → 블록이 하나씩만 담긴다
    merged = chunk_text_by_tokens(raw, max_tokens=3, token_len_fn=lambda t: 2)

    assert len(merged) == 3


def test_chunk_script_assigns_week_from_file():
    valid = ScriptFile(
        file_path=Path("10차시.txt"),
        week_number=4,
        status="valid",
        raw_text="[00:00] 첫 문장.\n\n[00:20] 둘째 문장.",
    )

    chunks = chunk_script(valid)

    assert all(c.week == 4 for c in chunks)
