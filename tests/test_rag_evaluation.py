from dataclasses import dataclass

import pytest

from core.rag_evaluation import collect_eval_samples, run_ragas_evaluation


@dataclass
class FakeGoldenItem:
    item_id: str
    question_text: str
    expected_chunk: str
    relevance_score: int
    ground_truth: str | None = None


@dataclass
class FakeChunk:
    text: str


@dataclass
class FakeAnswer:
    answer_text: str


def fake_retrieve_fn(query_text, vector_store, bm25_index, chunk_by_id, alpha, hybrid_top_n=20, rerank_top_n=5):
    return [("a", 0.9)]


def fake_generate_fn(query_text, reranked, chunk_by_id):
    return FakeAnswer(answer_text=f"답변: {query_text}")


def test_items_without_ground_truth_are_skipped():
    golden_set = [
        FakeGoldenItem("gs-01", "질문1", "chunk_0", 3, ground_truth="정답1"),
        FakeGoldenItem("gs-02", "질문2", "chunk_1", 3, ground_truth=None),
    ]
    chunk_by_id = {"a": FakeChunk("a의 텍스트")}

    samples = collect_eval_samples(
        golden_set, None, None, chunk_by_id, alpha=0.5,
        retrieve_fn=fake_retrieve_fn, generate_fn=fake_generate_fn,
    )

    assert len(samples) == 1
    assert samples[0]["user_input"] == "질문1"


def test_sample_shape_matches_ragas_input_fields():
    golden_set = [FakeGoldenItem("gs-01", "질문1", "chunk_0", 3, ground_truth="정답1")]
    chunk_by_id = {"a": FakeChunk("a의 텍스트")}

    samples = collect_eval_samples(
        golden_set, None, None, chunk_by_id, alpha=0.5,
        retrieve_fn=fake_retrieve_fn, generate_fn=fake_generate_fn,
    )

    sample = samples[0]
    assert sample["user_input"] == "질문1"
    assert sample["response"] == "답변: 질문1"
    assert sample["retrieved_contexts"] == ["a의 텍스트"]
    assert sample["reference"] == "정답1"


def test_no_items_with_ground_truth_yields_empty_samples():
    golden_set = [FakeGoldenItem("gs-01", "질문1", "chunk_0", 3, ground_truth=None)]

    samples = collect_eval_samples(
        golden_set, None, None, {}, alpha=0.5,
        retrieve_fn=fake_retrieve_fn, generate_fn=fake_generate_fn,
    )

    assert samples == []


def test_run_ragas_evaluation_raises_on_empty_samples():
    with pytest.raises(ValueError):
        run_ragas_evaluation([])
