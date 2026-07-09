from pathlib import Path

from data.vector_store import VectorStore


def test_add_and_query_roundtrip(tmp_path: Path):
    store = VectorStore(persist_directory=str(tmp_path))

    store.add("a", [1.0, 0.0, 0.0])
    store.add("b", [0.0, 1.0, 0.0])
    store.add("c", [0.0, 0.0, 1.0])

    results = store.query([1.0, 0.0, 0.0], top_k=3)

    assert results[0][0] == "a"  # 자기 자신과 가장 가까움
    assert len(results) == 3


def test_count_reflects_added_vectors(tmp_path: Path):
    store = VectorStore(persist_directory=str(tmp_path))

    store.add("x", [1.0, 0.0])
    store.add("y", [0.0, 1.0])

    assert store.count() == 2


def test_metadata_is_stored(tmp_path: Path):
    store = VectorStore(persist_directory=str(tmp_path))

    store.add("a", [1.0, 0.0], metadata={"week": 1})

    result = store._collection.get(ids=["a"], include=["metadatas"])
    assert result["metadatas"][0]["week"] == 1
