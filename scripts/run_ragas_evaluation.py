"""전체 RAG 파이프라인(검색+재랭킹+생성)을 실제로 돌려 RAGAS 4대 지표를 계산한다 (Part 6-1)."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = PROJECT_ROOT.parent / "강의 스크립트 크롤링"
sys.path.insert(0, str(PROJECT_ROOT))

from data.env_loader import load_dotenv_if_present  # noqa: E402
from data.script_loader import load_all  # noqa: E402
from data.golden_set_loader import load_golden_set  # noqa: E402
from data.vector_store import VectorStore, DEFAULT_PERSIST_DIR  # noqa: E402
from data.bm25_index import BM25Index  # noqa: E402
from core.chunking import chunk_script  # noqa: E402
from core.indexing import index_chunks  # noqa: E402
from core.rag_evaluation import collect_eval_samples, run_ragas_evaluation  # noqa: E402

load_dotenv_if_present(PROJECT_ROOT / ".env")  # ANTHROPIC_API_KEY 등을 .env에서 읽어 os.environ에 채운다

WEEK_DIRS = {
    1: CORPUS_ROOT / "1주차_스크립트",
    2: CORPUS_ROOT / "2주차_스크립트",
    3: CORPUS_ROOT / "3주차_스크립트",
    4: CORPUS_ROOT / "4주차_스크립트",
}
COLLECTION_NAME = "script_chunks_v3"  # tokens 청킹 전환(2026-07-22) — 이전 컬렉션과 벡터 표현이 다름
ALPHA = 0.4  # golden_set 복수 정답 전수 검토(2026-07-22) 후 재확정 — Hit Rate@5=0.962, MRR=0.872
MIN_RECOMMENDED_SAMPLES = 20  # Part 6-1: RAGAS는 통계적 유의미성을 위해 20개 이상 권장


def main() -> None:
    t0 = time.time()

    print("1. 스크립트 로딩 + 청킹...")
    script_files = load_all(WEEK_DIRS)
    valid_files = [f for f in script_files if f.status == "valid"]
    all_chunks = []
    for sf in valid_files:
        all_chunks.extend(chunk_script(sf))
    chunk_by_id = {c.chunk_id: c for c in all_chunks}
    print(f"   청크 {len(all_chunks)}개")

    print("2. 임베딩 + ChromaDB 색인...")
    vector_store = VectorStore(persist_directory=str(DEFAULT_PERSIST_DIR), collection_name=COLLECTION_NAME)
    if vector_store.count() != len(all_chunks):
        index_chunks(all_chunks, vector_store)
    bm25_index = BM25Index([c.chunk_id for c in all_chunks], [c.text for c in all_chunks])

    print("3. 골든셋 로딩 (ground_truth 있는 항목만 평가 대상)...")
    golden_set = load_golden_set(PROJECT_ROOT / "golden_set.yaml")
    eval_targets = [item for item in golden_set if item.ground_truth]
    print(f"   전체 {len(golden_set)}개 중 ground_truth 보유 {len(eval_targets)}개")
    if len(eval_targets) < MIN_RECOMMENDED_SAMPLES:
        print(
            f"   경고: RAGAS는 통계적으로 유의미한 결과를 위해 {MIN_RECOMMENDED_SAMPLES}개 이상을 "
            f"권장합니다 (현재 {len(eval_targets)}개). golden_set.yaml에 ground_truth를 더 추가해주세요."
        )
    if not eval_targets:
        print("   ground_truth가 채워진 항목이 없어 종료합니다.")
        return

    print("4. RAG 파이프라인 실행 (질문마다 검색+재랭킹+생성)...")
    samples = collect_eval_samples(eval_targets, vector_store, bm25_index, chunk_by_id, alpha=ALPHA)

    print("5. RAGAS 평가 실행 (LLM as a Judge, 시간이 걸립니다)...")
    result = run_ragas_evaluation(samples)

    print()
    print("=== RAGAS 평가 결과 ===")
    for metric_name, score in result["summary"].items():
        print(f"  {metric_name}: {score:.3f}")

    print()
    print(f"총 소요 시간: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
