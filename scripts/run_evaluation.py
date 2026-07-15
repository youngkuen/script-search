"""전체 파이프라인을 실제 corpus로 실행해 alpha를 확정하고 AC-007 기준을 검증한다."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = PROJECT_ROOT.parent / "강의 스크립트 크롤링"
sys.path.insert(0, str(PROJECT_ROOT))

from data.script_loader import load_all  # noqa: E402
from data.golden_set_loader import load_golden_set  # noqa: E402
from data.vector_store import VectorStore, DEFAULT_PERSIST_DIR  # noqa: E402
from data.bm25_index import BM25Index  # noqa: E402
from core.chunking import chunk_script  # noqa: E402
from core.indexing import index_chunks  # noqa: E402
from core.evaluation import sweep_alpha, check_thresholds  # noqa: E402

WEEK_DIRS = {
    1: CORPUS_ROOT / "1주차_스크립트",
    2: CORPUS_ROOT / "2주차_스크립트",
    3: CORPUS_ROOT / "3주차_스크립트",
    4: CORPUS_ROOT / "4주차_스크립트",
}
COLLECTION_NAME = "script_chunks_v2"


def main() -> None:
    t0 = time.time()

    print("1. 스크립트 로딩...")
    script_files = load_all(WEEK_DIRS)
    valid_files = [f for f in script_files if f.status == "valid"]
    print(f"   파일 {len(script_files)}개 (valid {len(valid_files)})")

    print("2. 청킹...")
    all_chunks = []
    for sf in valid_files:
        all_chunks.extend(chunk_script(sf))
    print(f"   청크 {len(all_chunks)}개")

    print("3. 임베딩 + ChromaDB 색인 (시간이 걸립니다)...")
    vector_store = VectorStore(persist_directory=str(DEFAULT_PERSIST_DIR), collection_name=COLLECTION_NAME)
    if vector_store.count() != len(all_chunks):
        index_chunks(all_chunks, vector_store)
    print(f"   색인 완료. 총 {vector_store.count()}개 벡터 (소요 {time.time() - t0:.1f}s)")

    print("4. BM25 인덱스 구축...")
    bm25_index = BM25Index([c.chunk_id for c in all_chunks], [c.text for c in all_chunks])

    print("5. 골든셋 로딩...")
    golden_set = load_golden_set(PROJECT_ROOT / "golden_set.yaml")
    print(f"   골든셋 {len(golden_set)}개 문항")

    print("6. alpha 스윕 (0.0~1.0, 0.1 간격)...")
    runs = sweep_alpha(golden_set, vector_store, bm25_index, top_n=5, threshold=0.0)

    print()
    print("=== alpha 스윕 결과 ===")
    for r in sorted(runs, key=lambda r: r.alpha):
        marker = " <= 선택됨" if r.is_selected else ""
        print(f"  alpha={r.alpha:.1f}  hit_rate@5={r.hit_rate_at_5:.3f}  mrr={r.mrr:.3f}  combined={r.combined_score:.3f}{marker}")

    selected = next(r for r in runs if r.is_selected)
    report = check_thresholds(selected)

    print()
    print("=== AC-007 최종 검증 ===")
    print(f"  확정된 alpha: {report['alpha']}")
    print(f"  Hit Rate@5: {report['hit_rate_at_5']:.3f} (목표 0.8) -> {'PASS' if report['hit_rate_pass'] else 'FAIL'}")
    print(f"  MRR:        {report['mrr']:.3f} (목표 0.5) -> {'PASS' if report['mrr_pass'] else 'FAIL'}")
    print()
    print(f"총 소요 시간: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
