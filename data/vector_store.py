"""ChromaDB persist client 래핑 — EmbedAndIndexChunk의 저장/조회 담당."""

from pathlib import Path

import chromadb

# 프로젝트 경로("메타코드/...")에 한글이 포함되어 있으면 chromadb의 Rust 바인딩이
# 저장된 HNSW 인덱스를 재오픈할 때 실패한다 (Windows에서 실측 확인됨:
# "Error loading hnsw index"). ASCII 전용 경로를 기본 저장 위치로 쓴다.
DEFAULT_PERSIST_DIR = Path.home() / ".cache" / "script-search" / "chroma"


class VectorStore:
    def __init__(self, persist_directory: str, collection_name: str = "script_chunks"):
        self._client = chromadb.PersistentClient(path=persist_directory)
        # Part 3-2: 거리공간을 코사인으로 명시한다. 지정하지 않으면 기본이 제곱 L2다.
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunk_id: str, vector, metadata: dict | None = None) -> None:
        # chromadb rejects an empty metadata dict ({}) but accepts None.
        # numpy.float32 원소가 남아있으면 "list of floats" 검증에서 거부되므로
        # float()으로 순수 Python float로 변환한다 (list(numpy_array)만으로는 불충분).
        # add 대신 upsert를 써서 같은 id를 다시 넣어도(재색인·임베딩 변경) 안전하게 덮어쓴다.
        self._collection.upsert(
            ids=[chunk_id],
            embeddings=[[float(x) for x in vector]],
            metadatas=[metadata] if metadata else None,
        )

    def query(self, vector, top_k: int = 5, where: dict | None = None) -> list[tuple[str, float]]:
        """상위 top_k개를 (chunk_id, distance) 튜플로 반환한다. distance는 낮을수록 유사하다.

        where가 주어지면 메타데이터 조건(예: {"week": {"$eq": 4}})으로 검색을 좁힌다.
        """
        kwargs = {"where": where} if where else {}
        result = self._collection.query(
            query_embeddings=[[float(x) for x in vector]], n_results=top_k, **kwargs
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        return list(zip(ids, distances))

    def count(self) -> int:
        return self._collection.count()
