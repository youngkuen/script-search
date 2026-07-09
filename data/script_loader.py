"""IngestScriptFile — 강의 스크립트 txt 파일을 읽어 ScriptFile로 변환한다."""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ScriptFile:
    file_path: Path
    week_number: int
    status: str  # "valid" | "skipped_encoding_error" | "skipped_empty"
    raw_text: str | None = None


def load_script_file(path: Path, week_number: int) -> ScriptFile:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("encoding error, skipping: %s", path)
        return ScriptFile(file_path=path, week_number=week_number, status="skipped_encoding_error")

    if not text.strip():
        logger.warning("empty file, skipping: %s", path)
        return ScriptFile(file_path=path, week_number=week_number, status="skipped_empty")

    return ScriptFile(file_path=path, week_number=week_number, status="valid", raw_text=text)


def load_all(week_dirs: dict[int, Path]) -> list[ScriptFile]:
    files = []
    for week, dir_path in week_dirs.items():
        for txt_path in sorted(dir_path.glob("*.txt")):
            files.append(load_script_file(txt_path, week))
    return files
