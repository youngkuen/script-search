"""청킹 전략 3종(block / merged / tokens)의 검색 품질을 비교한다 (Part 2-3 실습).

전략마다 chunk_id 체계가 달라지므로, 정답을 chunk_id로 매칭하지 않고
"검색된 청크의 텍스트가 정답 블록의 텍스트를 포함하는가"로 판정한다(전략 독립적).
정답 블록 텍스트는 기본 block 청킹 기준 golden_set의 expected_chunks에서 가져온다(항목당 여러 개 가능).
"""

import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = PROJECT_ROOT.parent / "강의 스크립트 크롤링"
sys.path.insert(0, str(PROJECT_ROOT))

from data.script_loader import load_all  # noqa: E402
from data.golden_set_loader import load_golden_set  # noqa: E402
from data.vector_store import VectorStore  # noqa: E402
from data.bm25_index import BM25Index  # noqa: E402
from core.chunking import chunk_script  # noqa: E402
from core.indexing import index_chunks  # noqa: E402
from core.scoring import execute_hybrid_search  # noqa: E402

WEEK_DIRS = {
    1: CORPUS_ROOT / "1주차_스크립트",
    2: CORPUS_ROOT / "2주차_스크립트",
    3: CORPUS_ROOT / "3주차_스크립트",
    4: CORPUS_ROOT / "4주차_스크립트",
}

ALPHA = 0.5
TOP_N = 5

STRATEGIES = {
    "block  (기본: 블록 1개=청크)": ("block", {}),
    "merged (모드A: 블록 3개 병합)": ("merged", {"group_size": 3}),
    "tokens (모드B: 토큰 256 예산)": ("tokens", {"max_tokens": 256}),
}


def build_needles(valid_files, golden_set):
    """golden_set의 expected_chunks(block 기준)를 정답 블록 텍스트(needle) 목록으로 변환한다.

    항목당 정답이 여러 개일 수 있다(gs-23처럼 같은 개념을 이론+실습이 각각 설명하는 경우) —
    그중 하나라도 검색되면 hit로 판정한다.
    """
    block_text_by_id = {}
    for sf in valid_files:
        for chunk in chunk_script(sf, strategy="block"):
            block_text_by_id[chunk.chunk_id] = chunk.text
    needles = []
    for item in golden_set:
        texts = [block_text_by_id[cid] for cid in item.expected_chunks if cid in block_text_by_id]
        if texts:
            needles.append((item.question_text, texts))
    return needles


def evaluate(strategy, kwargs, valid_files, needles):
    all_chunks = []
    for sf in valid_files:
        all_chunks.extend(chunk_script(sf, strategy=strategy, **kwargs))
    text_by_id = {c.chunk_id: c.text for c in all_chunks}

    tmp_dir = tempfile.mkdtemp(prefix="chunk-cmp-")
    store = VectorStore(persist_directory=tmp_dir, collection_name="cmp")
    index_chunks(all_chunks, store)
    bm25 = BM25Index([c.chunk_id for c in all_chunks], [c.text for c in all_chunks])

    hits, reciprocal_ranks = 0, []
    for question, needle_texts in needles:
        results = execute_hybrid_search(question, store, bm25, alpha=ALPHA, top_n=TOP_N)
        rank = next(
            (
                i + 1
                for i, (cid, _) in enumerate(results)
                if any(needle in text_by_id.get(cid, "") for needle in needle_texts)
            ),
            None,
        )
        if rank:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

    n = len(needles)
    return {
        "n_chunks": len(all_chunks),
        "hit_rate_at_5": hits / n if n else 0.0,
        "mrr": sum(reciprocal_ranks) / n if n else 0.0,
    }


def main() -> None:
    print("스크립트 로딩·정답 needle 구성...")
    valid_files = [f for f in load_all(WEEK_DIRS) if f.status == "valid"]
    golden_set = load_golden_set(PROJECT_ROOT / "golden_set.yaml")
    needles = build_needles(valid_files, golden_set)
    print(f"  파일 {len(valid_files)}개, 평가 문항 {len(needles)}개\n")

    print(f"{'전략':<30} {'청크수':>7} {'HitRate@5':>10} {'MRR':>8}")
    print("-" * 60)
    for label, (strategy, kwargs) in STRATEGIES.items():
        t0 = time.time()
        r = evaluate(strategy, kwargs, valid_files, needles)
        print(
            f"{label:<30} {r['n_chunks']:>7} {r['hit_rate_at_5']:>10.3f} {r['mrr']:>8.3f}"
            f"   ({time.time() - t0:.0f}s)"
        )


if __name__ == "__main__":
    main()
