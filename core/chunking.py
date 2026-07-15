"""ChunkScript — 스크립트를 ScriptChunk로 분할한다.

세 가지 청킹 전략을 지원한다 (Part 2-3 청킹 전략 실습):
- "block"  : [MM:SS] 타임스탬프 블록 하나 = 청크 하나 (기본, 기존 동작)
- "merged" : 연속한 블록을 group_size개씩 병합 (모드 A — 문맥 확대)
- "tokens" : 모델 토큰 예산(max_tokens)을 넘지 않게 블록을 이어붙여 분할 (모드 B — 토큰 기반)
"""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from data.script_loader import ScriptFile

_TIMESTAMP_RE = re.compile(r"^\[(\d{2}:\d{2})\]", re.MULTILINE)

# 보이지 않는 문자: 제로폭(ZWSP/ZWNJ/ZWJ), BOM, 소프트 하이픈, word joiner
_INVISIBLES = ("​", "‌", "‍", "﻿", "­", "⁠")


@dataclass
class ScriptChunk:
    chunk_id: str
    script_file: Path
    start_timestamp: str
    end_timestamp: str
    text: str
    week: int = 0
    embedding_id: str | None = None


def normalize_text(text: str) -> str:
    """유니코드 정규화(NFC) + 보이지 않는 문자 제거 (Part 2-2 전처리).

    비분리공백(U+00A0)은 일반 공백으로 바꾸고, 제로폭·BOM 등은 삭제한다.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ")
    for ch in _INVISIBLES:
        text = text.replace(ch, "")
    return text


def chunk_text(raw_text: str) -> list[tuple[str, str, str]]:
    """raw_text를 (start_timestamp, end_timestamp, text) 튜플 리스트로 분할한다."""
    matches = list(_TIMESTAMP_RE.finditer(raw_text))
    chunks = []
    for i, m in enumerate(matches):
        start_ts = m.group(1)
        start_pos = m.start()
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
            end_ts = matches[i + 1].group(1)
        else:
            end_pos = len(raw_text)
            end_ts = start_ts
        text = raw_text[start_pos:end_pos].strip()
        chunks.append((start_ts, end_ts, text))
    return chunks


def chunk_text_merged(raw_text: str, group_size: int = 3) -> list[tuple[str, str, str]]:
    """모드 A: 타임스탬프 블록을 group_size개씩 병합해 문맥을 넓힌다."""
    base = chunk_text(raw_text)
    merged = []
    for i in range(0, len(base), group_size):
        group = base[i : i + group_size]
        start_ts = group[0][0]
        end_ts = group[-1][1]
        text = "\n".join(g[2] for g in group)
        merged.append((start_ts, end_ts, text))
    return merged


def chunk_text_by_tokens(
    raw_text: str, max_tokens: int = 256, token_len_fn=None
) -> list[tuple[str, str, str]]:
    """모드 B: 블록을 이어붙이되 토큰 예산(max_tokens)을 넘지 않게 분할한다."""
    if token_len_fn is None:
        from data.embedding_model import token_len as token_len_fn

    base = chunk_text(raw_text)
    chunks = []
    cur_texts, cur_start, cur_end, cur_tokens = [], None, None, 0
    for start_ts, end_ts, text in base:
        n = token_len_fn(text)
        if cur_texts and cur_tokens + n > max_tokens:
            chunks.append((cur_start, cur_end, "\n".join(cur_texts)))
            cur_texts, cur_tokens = [], 0
        if not cur_texts:
            cur_start = start_ts
        cur_texts.append(text)
        cur_end = end_ts
        cur_tokens += n
    if cur_texts:
        chunks.append((cur_start, cur_end, "\n".join(cur_texts)))
    return chunks


def chunk_script(script_file: ScriptFile, strategy: str = "block", **kwargs) -> list[ScriptChunk]:
    if script_file.status != "valid" or not script_file.raw_text:
        return []

    raw = normalize_text(script_file.raw_text)
    if strategy == "merged":
        raw_chunks = chunk_text_merged(raw, **kwargs)
    elif strategy == "tokens":
        raw_chunks = chunk_text_by_tokens(raw, **kwargs)
    else:
        raw_chunks = chunk_text(raw)

    return [
        ScriptChunk(
            chunk_id=f"{script_file.file_path.stem}_{i}",
            script_file=script_file.file_path,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            text=text,
            week=script_file.week_number,
        )
        for i, (start_ts, end_ts, text) in enumerate(raw_chunks)
    ]
