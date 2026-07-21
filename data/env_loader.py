""".env 파일에서 KEY=VALUE 줄을 읽어 os.environ에 채워 넣는다.

python-dotenv를 새로 추가하지 않고 표준 라이브러리만으로 구현했다(GEN-004 — 기존 의존성/최소
코드로 해결 가능하면 새 패키지를 도입하지 않는다). 이미 설정된 실제 환경변수는 덮어쓰지 않는다
— 셸에서 export한 값이 .env 파일보다 항상 우선한다(일반적인 dotenv 관례와 동일).
"""

import os
from pathlib import Path


def load_dotenv_if_present(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
