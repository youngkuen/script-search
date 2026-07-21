from core.confidence import confidence_label


def test_score_above_high_threshold_is_high():
    assert confidence_label(0.71) == "높음"


def test_score_exactly_at_high_threshold_is_high():
    assert confidence_label(0.7) == "높음"


def test_score_between_thresholds_is_medium():
    assert confidence_label(0.5) == "보통"


def test_score_exactly_at_medium_threshold_is_medium():
    assert confidence_label(0.4) == "보통"


def test_score_below_medium_threshold_is_low():
    assert confidence_label(0.39) == "낮음"


def test_custom_thresholds_are_respected():
    assert confidence_label(0.6, high=0.9, medium=0.5) == "보통"
