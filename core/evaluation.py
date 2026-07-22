"""Hit Rate@K, MRR, 조화평균 — 순수 평가 함수 + TuneAlpha(알파 스윕)."""

from dataclasses import dataclass

from core.scoring import execute_hybrid_search


@dataclass
class EvaluationRun:
    run_id: str
    alpha: float
    hit_rate_at_5: float
    mrr: float
    combined_score: float
    is_selected: bool = False


def sweep_alpha(
    golden_set,
    vector_store,
    bm25_index,
    alphas: list[float] | None = None,
    top_n: int = 5,
    threshold: float = 0.0,
    embed_fn=None,
) -> list[EvaluationRun]:
    if alphas is None:
        alphas = [round(i * 0.1, 1) for i in range(11)]  # 0.0, 0.1, ..., 1.0

    runs = []
    for alpha in alphas:
        results = []
        relevant = []
        for item in golden_set:
            search_results = execute_hybrid_search(
                item.question_text,
                vector_store,
                bm25_index,
                alpha=alpha,
                top_n=top_n,
                threshold=threshold,
                embed_fn=embed_fn,
            )
            results.append([chunk_id for chunk_id, _ in search_results])
            relevant.append(set(item.expected_chunks))

        hr = hit_rate_at_k(results, relevant, k=top_n)
        m = mrr(results, relevant)
        runs.append(
            EvaluationRun(
                run_id=f"run-alpha-{alpha}",
                alpha=alpha,
                hit_rate_at_5=hr,
                mrr=m,
                combined_score=harmonic_mean(hr, m),
            )
        )

    best = max(runs, key=lambda r: r.combined_score)
    best.is_selected = True
    return runs


def hit_rate_at_k(results: list[list[str]], relevant: list[set[str]], k: int) -> float:
    if not results:
        return 0.0
    hits = sum(1 for r, rel in zip(results, relevant) if set(r[:k]) & rel)
    return hits / len(results)


def mrr(results: list[list[str]], relevant: list[set[str]]) -> float:
    if not results:
        return 0.0
    reciprocal_ranks = []
    for r, rel in zip(results, relevant):
        rank = next((i + 1 for i, doc_id in enumerate(r) if doc_id in rel), None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def harmonic_mean(a: float, b: float) -> float:
    if a + b == 0:
        return 0.0
    return 2 * a * b / (a + b)


def check_thresholds(
    run: EvaluationRun, hit_rate_threshold: float = 0.8, mrr_threshold: float = 0.5
) -> dict:
    return {
        "alpha": run.alpha,
        "hit_rate_at_5": run.hit_rate_at_5,
        "mrr": run.mrr,
        "hit_rate_pass": run.hit_rate_at_5 >= hit_rate_threshold,
        "mrr_pass": run.mrr >= mrr_threshold,
    }
