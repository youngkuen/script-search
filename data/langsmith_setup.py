"""LangSmith 온라인 모니터링 (Part 6-2) — 계정/키가 없으면 조용히 비활성 상태로 남긴다.

파이프라인 코드는 이 함수 호출 여부와 무관하게 항상 정상 동작한다(LangSmith는 옵션 기능).
"""

import os

DEFAULT_PROJECT = "script-search-rag"


def enable_tracing(project: str = DEFAULT_PROJECT) -> bool:
    """LANGCHAIN_API_KEY 또는 LANGSMITH_API_KEY가 설정되어 있으면 tracing을 켠다.

    반환값은 활성화 여부를 알려줄 뿐이다 — 호출부는 이 값을 체크해 실패 처리할 필요가 없다
    (키가 없으면 그냥 tracing 없이 계속 동작한다).
    """
    if not (os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY")):
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", project)
    return True
