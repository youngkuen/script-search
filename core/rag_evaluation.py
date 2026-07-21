"""RAGAS 평가 (Part 6-1) — ground_truth가 있는 골든셋 항목만으로 Faithfulness/Answer Relevancy/
Context Precision/Context Recall을 계산한다.

RAGAS는 LLM as a Judge 방식이라 실제 LLM 호출 비용이 든다(질문마다 여러 번의 판단 호출).
`collect_eval_samples`는 순수 조립 로직이라 fake로 단위테스트하고, `run_ragas_evaluation`은
ragas/LLM을 실제로 호출하는 통합 영역이라 수동 검증 대상이다(README의 Data Layer 통합테스트 방침과 동일).
"""

from typing import Callable

from core.rag_chain import generate_answer
from core.rag_pipeline import retrieve_and_rerank

_METRIC_COLUMNS = [
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall",
]


def collect_eval_samples(
    golden_set,
    vector_store,
    bm25_index,
    chunk_by_id: dict,
    alpha: float,
    hybrid_top_n: int = 20,
    rerank_top_n: int = 5,
    retrieve_fn: Callable | None = None,
    generate_fn: Callable | None = None,
) -> list[dict]:
    """ground_truth가 있는 항목만 실제 파이프라인으로 돌려 RAGAS 입력 형식으로 조립한다."""
    if retrieve_fn is None:
        retrieve_fn = retrieve_and_rerank
    if generate_fn is None:
        generate_fn = generate_answer

    samples = []
    for item in golden_set:
        if not item.ground_truth:
            continue
        reranked = retrieve_fn(
            item.question_text,
            vector_store,
            bm25_index,
            chunk_by_id,
            alpha,
            hybrid_top_n=hybrid_top_n,
            rerank_top_n=rerank_top_n,
        )
        answer = generate_fn(item.question_text, reranked, chunk_by_id)
        contexts = [chunk_by_id[chunk_id].text for chunk_id, _ in reranked if chunk_id in chunk_by_id]
        samples.append(
            {
                "user_input": item.question_text,
                "response": answer.answer_text,
                "retrieved_contexts": contexts,
                "reference": item.ground_truth,
            }
        )
    return samples


def run_ragas_evaluation(samples: list[dict], chat_model=None, embeddings=None) -> dict:
    """samples를 RAGAS EvaluationDataset으로 감싸 4대 지표를 계산한다.

    반환값: {"summary": {지표명: 평균점수}, "dataframe": 질문별 점수 pandas.DataFrame}
    """
    if not samples:
        raise ValueError("평가할 샘플이 없습니다 — 골든셋에 ground_truth가 채워진 항목이 없습니다.")

    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        ContextRecall,
        Faithfulness,
        LLMContextPrecisionWithReference,
    )

    if chat_model is None:
        from data.llm_client import get_chat_model

        chat_model = get_chat_model()
    if embeddings is None:
        from data.rag_embeddings import E5LangchainEmbeddings

        embeddings = E5LangchainEmbeddings()

    dataset = EvaluationDataset(samples=[SingleTurnSample(**s) for s in samples])
    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            LLMContextPrecisionWithReference(),
            ContextRecall(),
        ],
        llm=LangchainLLMWrapper(chat_model),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
    )

    df = result.to_pandas()
    summary = {col: float(df[col].mean()) for col in _METRIC_COLUMNS if col in df.columns}
    return {"summary": summary, "dataframe": df}
