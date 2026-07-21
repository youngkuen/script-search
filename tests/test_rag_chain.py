from dataclasses import dataclass

from core.rag_chain import NO_RESULTS_MESSAGE, build_context_text, generate_answer


@dataclass
class FakeChunk:
    text: str
    script_file: str
    start_timestamp: str
    end_timestamp: str


class FakeChatModel:
    def __init__(self, response_text: str = "가짜 답변"):
        self.response_text = response_text
        self.invoked = False
        self.last_input = None

    def invoke(self, prompt_value):
        self.invoked = True
        self.last_input = prompt_value
        return self.response_text


def _chunk_by_id():
    return {
        "a": FakeChunk(text="A의 내용", script_file="파일A.txt", start_timestamp="00:00", end_timestamp="00:10"),
        "b": FakeChunk(text="B의 내용", script_file="파일B.txt", start_timestamp="01:00", end_timestamp="01:10"),
    }


def test_empty_reranked_short_circuits_without_calling_llm():
    chat_model = FakeChatModel()

    answer = generate_answer("질문", [], _chunk_by_id(), chat_model=chat_model)

    assert answer.answer_text == NO_RESULTS_MESSAGE
    assert answer.sources == []
    assert chat_model.invoked is False


def test_generate_answer_returns_llm_text_and_sources_with_confidence():
    chat_model = FakeChatModel(response_text="정답은 이것입니다.")
    reranked = [("a", 0.8), ("b", 0.3)]

    answer = generate_answer("질문", reranked, _chunk_by_id(), chat_model=chat_model)

    assert answer.answer_text == "정답은 이것입니다."
    assert chat_model.invoked is True
    assert [s.chunk_id for s in answer.sources] == ["a", "b"]
    assert answer.sources[0].confidence == "높음"  # score 0.8 >= HIGH_THRESHOLD
    assert answer.sources[1].confidence == "낮음"  # score 0.3 < MEDIUM_THRESHOLD
    assert answer.sources[0].source_file == "파일A.txt"


def test_source_missing_from_chunk_by_id_is_skipped():
    chat_model = FakeChatModel()
    reranked = [("a", 0.9), ("missing", 0.5)]

    answer = generate_answer("질문", reranked, _chunk_by_id(), chat_model=chat_model)

    ids = {s.chunk_id for s in answer.sources}
    assert ids == {"a"}


def test_build_context_text_joins_with_separator():
    reranked = [("a", 0.8), ("b", 0.3)]

    context = build_context_text(reranked, _chunk_by_id())

    assert context == "A의 내용\n---\n\nB의 내용"
