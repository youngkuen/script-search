"""ChunkScript — [MM:SS] 타임스탬프 블록 단위로 스크립트를 ScriptChunk로 분할한다."""

import re
from dataclasses import dataclass
from pathlib import Path

from data.script_loader import ScriptFile

_TIMESTAMP_RE = re.compile(r"^\[(\d{2}:\d{2})\]", re.MULTILINE)


@dataclass
class ScriptChunk:
    chunk_id: str
    script_file: Path
    start_timestamp: str
    end_timestamp: str
    text: str
    embedding_id: str | None = None


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


def chunk_script(script_file: ScriptFile) -> list[ScriptChunk]:
    if script_file.status != "valid" or not script_file.raw_text:
        return []

    raw_chunks = chunk_text(script_file.raw_text)
    return [
        ScriptChunk(
            chunk_id=f"{script_file.file_path.stem}_{i}",
            script_file=script_file.file_path,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            text=text,
        )
        for i, (start_ts, end_ts, text) in enumerate(raw_chunks)
    ]
