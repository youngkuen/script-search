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


def test_query_where_filters_by_metadata(tmp_path: Path):
    store = VectorStore(persist_directory=str(tmp_path))

    # 같은 벡터라도 week 메타데이터가 다르면 where로 걸러져야 한다.
    store.add("w1", [1.0, 0.0], metadata={"week": 1})
    store.add("w4", [1.0, 0.0], metadata={"week": 4})

    results = store.query([1.0, 0.0], top_k=5, where={"week": {"$eq": 4}})

    assert [cid for cid, _ in results] == ["w4"]
