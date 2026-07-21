"""GenerateAnswer — 재랭킹된 청크를 근거로 LangChain 기반 답변을 생성한다 (Part 5-1/5-2).

프롬프트는 5-1의 3원칙(문서 기반 답변·모르면 모른다·추측 금지) + 5-2의 인용 유도를 담는다.
Empty Retrieval(reranked가 빈 리스트)이면 LLM을 호출하지 않고 즉시 안내 메시지를 반환한다
(5-1 Error Handling — 불필요한 API 비용을 아낀다). 출처(sources)는 LLM 출력과 무관하게 코드에서
직접 구성하므로 신뢰도 표시가 항상 정확하다(5-2 설명가능성).

ChatPromptTemplate | chat_model | StrOutputParser()라는 LCEL 파이프 대신 각 단계를 명시적으로
.invoke()하는 이유: chat_model을 손으로 만든 Fake로 교체해 테스트하려면 Fake가 Runnable을
상속하거나 __call__을 구현할 필요 없이 .invoke()만 있으면 되기 때문이다(기존 테스트 컨벤션인
"모킹 라이브러리 없는 Fake 클래스"와의 일관성).
"""

from dataclasses import dataclass, field
from typing import Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.confidence import confidence_label

NO_RESULTS_MESSAGE = "관련 문서를 찾을 수 없습니다. 다른 방식으로 질문해 주세요."

SYSTEM_PROMPT = """당신은 강의 스크립트에 기반해 질문에 답하는 도우미입니다. 다음 규칙을 반드시 지키세요.

1. 반드시 <context> 안의 내용만을 근거로 답변하십시오.
2. 관련 정보가 없으면 "제공된 자료에서 해당 정보를 찾을 수 없습니다."라고 답하십시오.
3. <context>에 없는 내용을 임의로 추측하거나 보충하지 마십시오.
4. 답변에 사용한 내용의 출처(강의 파일명)를 자연스럽게 언급하십시오."""

HUMAN_TEMPLATE = "<context>\n{context}\n</context>\n\n질문: {question}"


@dataclass
class SourceCitation:
    chunk_id: str
    source_file: str
    start_timestamp: str
    end_timestamp: str
    score: float
    confidence: str


@dataclass
class RagAnswer:
    answer_text: str
    sources: list[SourceCitation] = field(default_factory=list)


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_TEMPLATE)]
    )


def build_context_text(reranked: list[tuple[str, float]], chunk_by_id: dict) -> str:
    """재랭킹된 청크 텍스트를 '---' 구분자로 이어붙인다 (format_docs 방식, Part 5-1)."""
    texts = [chunk_by_id[chunk_id].text for chunk_id, _ in reranked if chunk_id in chunk_by_id]
    return "\n---\n\n".join(texts)


def _build_sources(reranked: list[tuple[str, float]], chunk_by_id: dict) -> list[SourceCitation]:
    sources = []
    for chunk_id, score in reranked:
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None:
            continue
        sources.append(
            SourceCitation(
                chunk_id=chunk_id,
                source_file=str(chunk.script_file),
                start_timestamp=chunk.start_timestamp,
                end_timestamp=chunk.end_timestamp,
                score=score,
                confidence=confidence_label(score),
            )
        )
    return sources


def generate_answer(
    query_text: str,
    reranked: list[tuple[str, float]],
    chunk_by_id: dict,
    chat_model: Callable | None = None,
) -> RagAnswer:
    if not reranked:
        return RagAnswer(answer_text=NO_RESULTS_MESSAGE, sources=[])

    if chat_model is None:
        from data.llm_client import get_chat_model

        chat_model = get_chat_model()

    context = build_context_text(reranked, chunk_by_id)
    prompt_value = build_prompt().invoke({"context": context, "question": query_text})
    llm_result = chat_model.invoke(prompt_value)
    answer_text = StrOutputParser().invoke(llm_result)

    return RagAnswer(answer_text=answer_text, sources=_build_sources(reranked, chunk_by_id))
