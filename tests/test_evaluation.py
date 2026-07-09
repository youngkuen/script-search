from core.evaluation import harmonic_mean, hit_rate_at_k, mrr


def test_hit_rate_at_k_all_hits():
    results = [["a", "b", "c"], ["x", "y", "z"]]
    relevant = [{"a"}, {"z"}]

    assert hit_rate_at_k(results, relevant, k=3) == 1.0


def test_hit_rate_at_k_respects_k():
    results = [["a", "b", "c"]]
    relevant = [{"c"}]

    assert hit_rate_at_k(results, relevant, k=1) == 0.0
    assert hit_rate_at_k(results, relevant, k=3) == 1.0


def test_mrr_rank_1_scores_1_0():
    results = [["a", "b", "c"]]
    relevant = [{"a"}]

    assert mrr(results, relevant) == 1.0


def test_mrr_rank_5_scores_0_2():
    results = [["w", "x", "y", "z", "a"]]
    relevant = [{"a"}]

    assert mrr(results, relevant) == 0.2


def test_mrr_no_match_scores_0():
    results = [["a", "b", "c"]]
    relevant = [{"z"}]

    assert mrr(results, relevant) == 0.0


def test_harmonic_mean_known_values():
    assert harmonic_mean(0.8, 0.5) == 2 * 0.8 * 0.5 / (0.8 + 0.5)


def test_harmonic_mean_zero_both():
    assert harmonic_mean(0.0, 0.0) == 0.0
