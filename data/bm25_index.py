"""BuildBM25Index — 키워드 기반 검색 인덱스.

data/ 레이어는 core/의 ScriptChunk 타입을 몰라야 하므로 (Data→Logic import 금지),
청크 대신 id·text 원시 리스트를 입력으로 받는다. ScriptChunk → id/text 변환은
호출하는 쪽(core/)의 책임이다.
"""

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Index:
    def __init__(self, ids: list[str], texts: list[str]):
        self._chunk_ids = ids
        corpus_tokens = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(corpus_tokens)

    def search(self, query_text: str, top_k: int = 5) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(_tokenize(query_text))
        ranked = sorted(zip(self._chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]
