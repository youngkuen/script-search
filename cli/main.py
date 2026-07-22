"""강의 스크립트 QA 시스템 — 인터랙티브 CLI (Part 5주차 RAG 확장).

질문을 입력하면 하이브리드 검색(4-1) + Cross-Encoder 재랭킹(4-2)으로 근거 청크를 좁히고,
LangChain 기반 답변 생성(5-1)을 거쳐 답변 + 출처 + 신뢰도(5-2)를 함께 보여준다.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = PROJECT_ROOT.parent / "강의 스크립트 크롤링"
sys.path.insert(0, str(PROJECT_ROOT))

from data.env_loader import load_dotenv_if_present  # noqa: E402
from data.script_loader import load_all  # noqa: E402
from data.vector_store import VectorStore, DEFAULT_PERSIST_DIR  # noqa: E402
from data.bm25_index import BM25Index  # noqa: E402
from data.llm_client import get_chat_model  # noqa: E402
from data.langsmith_setup import enable_tracing  # noqa: E402
from core.chunking import chunk_script  # noqa: E402
from core.indexing import index_chunks  # noqa: E402
from core.rag_pipeline import retrieve_and_rerank  # noqa: E402
from core.rag_chain import generate_answer  # noqa: E402

load_dotenv_if_present(PROJECT_ROOT / ".env")  # ANTHROPIC_API_KEY 등을 .env에서 읽어 os.environ에 채운다

ALPHA = 0.5  # golden_set 41문항 확장(2026-07-22) 후 재확정 — Hit Rate@5=0.976, MRR=0.859 (1,158청크)
HYBRID_TOP_N = 20  # 1단계: 하이브리드 검색으로 넓게 뽑는 후보 수 (Part 4-2)
RERANK_TOP_N = 5   # 2단계: Cross-Encoder 재랭킹 후 LLM에 전달할 최종 청크 수
THRESHOLD = 0.0

WEEK_DIRS = {
    1: CORPUS_ROOT / "1주차_스크립트",
    2: CORPUS_ROOT / "2주차_스크립트",
    3: CORPUS_ROOT / "3주차_스크립트",
    4: CORPUS_ROOT / "4주차_스크립트",
}

# 비대칭 임베딩(프리픽스)·코사인 거리공간·week 메타데이터가 추가되어 벡터 표현이 바뀌었으므로
# 기존 컬렉션과 섞이지 않도록 새 컬렉션 이름을 쓴다 (재색인은 자동으로 이루어진다).
COLLECTION_NAME = "script_chunks_v3"  # tokens 청킹 전환(2026-07-22) — 이전 컬렉션과 벡터 표현이 다름


def build_pipeline():
    script_files = load_all(WEEK_DIRS)
    valid_files = [f for f in script_files if f.status == "valid"]

    all_chunks = []
    for sf in valid_files:
        all_chunks.extend(chunk_script(sf))

    chunk_by_id = {c.chunk_id: c for c in all_chunks}
    id_week_map = {c.chunk_id: c.week for c in all_chunks}

    vector_store = VectorStore(persist_directory=str(DEFAULT_PERSIST_DIR), collection_name=COLLECTION_NAME)
    if vector_store.count() != len(all_chunks):
        print(f"색인 중... ({len(all_chunks)}개 청크, 시간이 걸립니다)")
        index_chunks(all_chunks, vector_store)

    bm25_index = BM25Index([c.chunk_id for c in all_chunks], [c.text for c in all_chunks])
    chat_model = get_chat_model()

    return vector_store, bm25_index, chunk_by_id, id_week_map, chat_model


def format_source(source) -> str:
    return (
        f"  [{Path(source.source_file).stem}] ({source.start_timestamp}~{source.end_timestamp}) "
        f"신뢰도={source.confidence} score={source.score:.3f}"
    )


def parse_week_filter(query: str) -> tuple[str, int | None]:
    """'/week N <질문>' 형식이면 (질문, N)을, 아니면 (원본, None)을 반환한다."""
    if query.startswith("/week "):
        parts = query.split(maxsplit=2)
        if len(parts) == 3 and parts[1].isdigit():
            return parts[2], int(parts[1])
    return query, None


def main() -> None:
    print("=== 강의 스크립트 QA 시스템 ===")
    print("질문을 입력하세요 (종료: exit 또는 quit)")
    print("특정 주차만 검색: '/week 4 <질문>'\n")

    vector_store, bm25_index, chunk_by_id, id_week_map, chat_model = build_pipeline()
    tracing_on = enable_tracing()
    print(f"준비 완료. (LangSmith 모니터링: {'켜짐' if tracing_on else '꺼짐'})\n")

    while True:
        try:
            query = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        query, week_filter = parse_week_filter(query)
        reranked = retrieve_and_rerank(
            query, vector_store, bm25_index, chunk_by_id, alpha=ALPHA,
            hybrid_top_n=HYBRID_TOP_N, rerank_top_n=RERANK_TOP_N, threshold=THRESHOLD,
            week_filter=week_filter, id_week_map=id_week_map,
        )
        answer = generate_answer(query, reranked, chunk_by_id, chat_model=chat_model)

        print(f"\n{answer.answer_text}\n")
        if answer.sources:
            print(f"출처 ({len(answer.sources)}개):")
            for source in answer.sources:
                print(format_source(source))
        print()


if __name__ == "__main__":
    main()
