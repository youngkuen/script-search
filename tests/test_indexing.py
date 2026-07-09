from pathlib import Path

from core.chunking import ScriptChunk
from core.indexing import index_chunks


class FakeStore:
    def __init__(self):
        self.added = []

    def add(self, chunk_id, vector, metadata=None):
        self.added.append((chunk_id, vector, metadata))


def fake_encode(text: str):
    return [len(text), 0.0]


def test_index_chunks_calls_encode_and_add_once_per_chunk():
    chunks = [
        ScriptChunk(chunk_id="a_0", script_file=Path("a.txt"), start_timestamp="00:00", end_timestamp="00:10", text="첫 청크"),
        ScriptChunk(chunk_id="a_1", script_file=Path("a.txt"), start_timestamp="00:10", end_timestamp="00:20", text="둘째 청크"),
    ]
    store = FakeStore()

    index_chunks(chunks, store, encode_fn=fake_encode)

    assert len(store.added) == 2
    assert store.added[0][0] == "a_0"
    assert store.added[0][1] == fake_encode("첫 청크")


def test_index_chunks_sets_embedding_id():
    chunks = [
        ScriptChunk(chunk_id="a_0", script_file=Path("a.txt"), start_timestamp="00:00", end_timestamp="00:10", text="첫 청크"),
    ]
    store = FakeStore()

    index_chunks(chunks, store, encode_fn=fake_encode)

    assert chunks[0].embedding_id == "a_0"
