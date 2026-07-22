from data.golden_set_loader import GoldenSetItem

from core.evaluation import check_thresholds, sweep_alpha


class FakeVectorStore:
    def query(self, vector, top_k=5):
        return [("a", 0.0), ("b", 1.0)]  # a가 항상 더 가까움


class FakeBM25Index:
    def search(self, query_text, top_k=5):
        return [("b", 5.0), ("a", 1.0)]  # b가 항상 raw score 더 높음


def _fake_embed(text: str):
    return [0.0]


def test_sweep_alpha_selects_exactly_one_run():
    golden_set = [GoldenSetItem(item_id="gs-1", question_text="질문", expected_chunks=["a"], relevance_score=3)]

    runs = sweep_alpha(
        golden_set,
        FakeVectorStore(),
        FakeBM25Index(),
        alphas=[0.0, 0.3, 0.5, 0.7, 1.0],
        top_n=5,
        threshold=0.0,
        embed_fn=_fake_embed,
    )

    selected = [r for r in runs if r.is_selected]
    assert len(selected) == 1
    assert selected[0].combined_score == max(r.combined_score for r in runs)


def test_sweep_alpha_default_produces_11_candidates():
    golden_set = [GoldenSetItem(item_id="gs-1", question_text="질문", expected_chunks=["a"], relevance_score=3)]

    runs = sweep_alpha(golden_set, FakeVectorStore(), FakeBM25Index(), top_n=5, threshold=0.0, embed_fn=_fake_embed)

    assert len(runs) == 11
    assert [r.alpha for r in runs] == [round(i * 0.1, 1) for i in range(11)]


def test_sweep_alpha_favors_vector_when_vector_is_more_accurate():
    # 검색 결과에서 a가 정답인데, bm25는 b를 더 높게 치고 vector는 a를 정확히 1위로 찾음
    golden_set = [GoldenSetItem(item_id="gs-1", question_text="질문", expected_chunks=["a"], relevance_score=3)]

    runs = sweep_alpha(golden_set, FakeVectorStore(), FakeBM25Index(), top_n=5, threshold=0.0, embed_fn=_fake_embed)
    pure_vector = next(r for r in runs if r.alpha == 1.0)
    pure_bm25 = next(r for r in runs if r.alpha == 0.0)

    assert pure_vector.mrr == 1.0  # a가 1위
    assert pure_bm25.mrr == 0.5  # a가 2위


def test_sweep_alpha_credits_hit_when_any_expected_chunk_matches():
    # b가 bm25에서 raw score가 더 높아 pure-bm25(alpha=0.0)에서는 b가 1위로 뽑힌다.
    # expected_chunks에 a와 b를 둘 다 정답으로 인정하면, b가 1위여도 mrr=1.0이어야 한다
    # (같은 개념을 여러 청크가 답할 수 있는 gs-23류 상황을 검증).
    golden_set = [
        GoldenSetItem(item_id="gs-1", question_text="질문", expected_chunks=["a", "b"], relevance_score=3)
    ]

    runs = sweep_alpha(golden_set, FakeVectorStore(), FakeBM25Index(), top_n=5, threshold=0.0, embed_fn=_fake_embed)
    pure_bm25 = next(r for r in runs if r.alpha == 0.0)

    assert pure_bm25.mrr == 1.0


def test_check_thresholds_reports_pass_fail():
    from core.evaluation import EvaluationRun

    passing = EvaluationRun(run_id="r1", alpha=0.5, hit_rate_at_5=0.9, mrr=0.6, combined_score=0.72)
    failing = EvaluationRun(run_id="r2", alpha=0.5, hit_rate_at_5=0.5, mrr=0.3, combined_score=0.375)

    assert check_thresholds(passing) == {
        "alpha": 0.5,
        "hit_rate_at_5": 0.9,
        "mrr": 0.6,
        "hit_rate_pass": True,
        "mrr_pass": True,
    }
    result = check_thresholds(failing)
    assert result["hit_rate_pass"] is False
    assert result["mrr_pass"] is False
