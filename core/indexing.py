"""EmbedAndIndexChunk — ScriptChunk를 임베딩해 벡터 저장소에 색인한다."""

from typing import Callable

from core.chunking import ScriptChunk


def index_chunks(
    chunks: list[ScriptChunk],
    store,
    encode_fn: Callable[[str], object] | None = None,
) -> None:
    if encode_fn is None:
        from data.embedding_model import encode as encode_fn

    for chunk in chunks:
        vector = encode_fn(chunk.text)
        store.add(
            chunk.chunk_id,
            vector,
            metadata={
                "start_timestamp": chunk.start_timestamp,
                "end_timestamp": chunk.end_timestamp,
                "source_file": str(chunk.script_file),
            },
        )
        chunk.embedding_id = chunk.chunk_id
