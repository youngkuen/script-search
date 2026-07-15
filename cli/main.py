"""강의 스크립트 의미 검색기 — 인터랙티브 CLI."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = PROJECT_ROOT.parent / "강의 스크립트 크롤링"
sys.path.insert(0, str(PROJECT_ROOT))

from data.script_loader import load_all  # noqa: E402
from data.vector_store import VectorStore, DEFAULT_PERSIST_DIR  # noqa: E402
from data.bm25_index import BM25Index  # noqa: E402
from core.chunking import chunk_script  # noqa: E402
from core.indexing import index_chunks  # noqa: E402
from core.scoring import execute_hybrid_search  # noqa: E402

ALPHA = 0.5  # 1~4주차 코퍼스(5,120청크)·골든셋 26문항 재스윕으로 확정 — Hit Rate@5=0.885, MRR=0.782
TOP_N = 5
THRESHOLD = 0.0

WEEK_DIRS = {
    1: CORPUS_ROOT / "1주차_스크립트",
    2: CORPUS_ROOT / "2주차_스크립트",
    3: CORPUS_ROOT / "3주차_스크립트",
    4: CORPUS_ROOT / "4주차_스크립트",
}

# 비대칭 임베딩(프리픽스)·코사인 거리공간·week 메타데이터가 추가되어 벡터 표현이 바뀌었으므로
# 기존 컬렉션과 섞이지 않도록 새 컬렉션 이름을 쓴다 (재색인은 자동으로 이루어진다).
COLLECTION_NAME = "script_chunks_v2"


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

    return vector_store, bm25_index, chunk_by_id, id_week_map


def format_result(chunk_id: str, score: float, chunk_by_id: dict) -> str:
    chunk = chunk_by_id.get(chunk_id)
    if chunk is None:
        return f"  [{chunk_id}] score={score:.3f} (원본 청크 정보 없음)"
    snippet = chunk.text.replace("\n", " ")[:150]
    return (
        f"  [{chunk.script_file.stem}] ({chunk.start_timestamp}~{chunk.end_timestamp}) "
        f"score={score:.3f}\n    {snippet}"
    )


def parse_week_filter(query: str) -> tuple[str, int | None]:
    """'/week N <질문>' 형식이면 (질문, N)을, 아니면 (원본, None)을 반환한다."""
    if query.startswith("/week "):
        parts = query.split(maxsplit=2)
        if len(parts) == 3 and parts[1].isdigit():
            return parts[2], int(parts[1])
    return query, None


def main() -> None:
    print("=== 강의 스크립트 검색기 ===")
    print("질문을 입력하세요 (종료: exit 또는 quit)")
    print("특정 주차만 검색: '/week 4 <질문>'\n")

    vector_store, bm25_index, chunk_by_id, id_week_map = build_pipeline()
    print("준비 완료.\n")

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
        results = execute_hybrid_search(
            query, vector_store, bm25_index, alpha=ALPHA, top_n=TOP_N, threshold=THRESHOLD,
            week_filter=week_filter, id_week_map=id_week_map,
        )

        if not results:
            print("관련 내용을 찾지 못했습니다.\n")
            continue

        print(f"상위 {len(results)}개 결과:")
        for chunk_id, score in results:
            print(format_result(chunk_id, score, chunk_by_id))
        print()


if __name__ == "__main__":
    main()
