"""Chat Model 클라이언트 — Anthropic Claude를 LangChain 인터페이스로 감싼다 (Part 5-1).

API 키는 langchain-anthropic이 ANTHROPIC_API_KEY 환경변수에서 직접 읽는다. 코드에서 키 문자열을
취급하지 않는다(GEN-005). 키가 없으면 모호한 SDK 예외 대신 명확한 에러 메시지로 즉시 실패한다.
"""

import os

from langchain_anthropic import ChatAnthropic

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def get_chat_model(temperature: float = 0.0, max_tokens: int | None = None) -> ChatAnthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다. "
            "https://console.anthropic.com 에서 발급받은 키를 `export ANTHROPIC_API_KEY=...`로 설정하세요."
        )
    model_name = os.environ.get("RAG_LLM_MODEL", DEFAULT_MODEL)
    kwargs = {"model": model_name, "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatAnthropic(**kwargs)
