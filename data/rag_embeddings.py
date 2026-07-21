"""RAGAS Answer Relevancy(Part 6-1)가 요구하는 langchain_core.embeddings.Embeddings 인터페이스를
기존 e5 임베딩 모델(data/embedding_model.py)로 어댑팅한다. 새 임베딩 모델을 추가하지 않고 기존
것을 감싸기만 한다(GEN-004).
"""

from langchain_core.embeddings import Embeddings

from data.embedding_model import encode_passage, encode_query


class E5LangchainEmbeddings(Embeddings):
    """e5 비대칭 임베딩(쿼리/문서 프리픽스 구분)을 LangChain Embeddings 인터페이스로 노출한다."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in encode_passage(t)] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in encode_query(text)]
