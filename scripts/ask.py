"""로컬 LLM(Ollama)로 '근거 기반 답변'까지 하는 미니 RAG.

기존 검색기(R: 임베딩+BM25 하이브리드)에 생성(G)을 붙였다.
  질문 → 관련 청크 top-k 검색 → 프롬프트 조립(A) → 로컬 LLM 호출(G) → 근거 기반 답변
새 패키지 없이 파이썬 기본 urllib로 Ollama REST API(localhost:11434)를 호출한다.

실행: uv run python scripts/ask.py   (또는 .venv\\Scripts\\python scripts\\ask.py)
사전: Ollama가 실행 중이고 아래 MODEL 모델이 받아져 있어야 함 (ollama list로 확인).
"""

import sys
import json
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = PROJECT_ROOT.parent / "강의 스크립트 크롤링"
sys.path.insert(0, str(PROJECT_ROOT))

from data.script_loader import load_all  # noqa: E402
from data.vector_store import VectorStore, DEFAULT_PERSIST_DIR  # noqa: E402
from data.bm25_index import BM25Index  # noqa: E402
from core.chunking import chunk_script  # noqa: E402
from core.scoring import execute_hybrid_search  # noqa: E402

# ── 설정 (바꾸고 싶으면 여기만) ──────────────────────────────
MODEL = "gemma4:e2b"        # 답 생성용 로컬 LLM
OLLAMA_URL = "http://localhost:11434/api/generate"
ALPHA = 0.5                # 하이브리드 가중치(검색기와 동일)
TOP_K = 4                  # LLM에 근거로 넘길 청크 수

# 주: 무관 질문을 프로그램적으로 차단하는 '관련성 게이트'는 넣지 않았다.
# 거리 임계값·LLM 판정·리랭커를 모두 시도했으나 이 잡음 많은 구어체
# 스크립트 코퍼스에선 관련/무관이 안 갈렸다(docs/EXPERIMENTS.md 참고).
# 대신 build_prompt의 "자료에 없으면 '자료에 없음'만" 규칙에 의존한다.

WEEK_DIRS = {i: CORPUS_ROOT / f"{i}주차_스크립트" for i in (1, 2, 3, 4)}


def build_pipeline():
    """스크립트를 청킹하고 (영속화된) 벡터 인덱스 + BM25 인덱스를 준비한다."""
    chunks = []
    for sf in [f for f in load_all(WEEK_DIRS) if f.status == "valid"]:
        chunks.extend(chunk_script(sf))
    chunk_by_id = {c.chunk_id: c for c in chunks}
    vector_store = VectorStore(persist_directory=str(DEFAULT_PERSIST_DIR), collection_name="script_chunks_v2")
    bm25_index = BM25Index([c.chunk_id for c in chunks], [c.text for c in chunks])
    return vector_store, bm25_index, chunk_by_id


def build_context(hits, chunk_by_id):
    """검색된 청크를 번호 매긴 [자료] 문자열로 만든다."""
    blocks = []
    for i, (chunk_id, _score) in enumerate(hits, 1):
        c = chunk_by_id[chunk_id]
        blocks.append(f"[{i}] ({c.script_file.stem}, {c.start_timestamp})\n{c.text}")
    return "\n\n".join(blocks)


def build_prompt(question, context):
    """검색된 [자료]로 근거 기반 답변을 유도하는 프롬프트(A 단계)."""
    return (
        "당신은 강의 스크립트 도우미입니다. 반드시 아래 [자료]에 있는 내용만 근거로 [질문]에 한국어로 답하세요.\n"
        "규칙: [자료]에 답이 없으면 당신의 다른 지식을 절대 쓰지 말고 정확히 '자료에 없음'이라고만 쓰세요. 추측·창작 금지.\n"
        "답 끝에 근거로 쓴 자료 번호를 [1], [2] 형태로 붙이세요.\n\n"
        f"[자료]\n{context}\n\n[질문] {question}\n\n[답변]"
    )


def ask_ollama(prompt):
    """Ollama 로컬 서버에 프롬프트를 보내 답변 텍스트를 받는다(G 단계)."""
    payload = json.dumps({"model": MODEL, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8")).get("response", "").strip()
    except urllib.error.URLError as e:
        return (f"[오류] Ollama 호출 실패: {e}\n"
                f"  - Ollama가 실행 중인가요? (터미널에서 `ollama list` 확인)\n"
                f"  - 모델명이 맞나요? 현재 MODEL = '{MODEL}'")


def main():
    print(f"=== 로컬 RAG (검색 + {MODEL} 생성) ===")
    print("질문을 입력하세요 (종료: exit)\n")
    vector_store, bm25_index, chunk_by_id = build_pipeline()
    print("준비 완료.\n")

    while True:
        try:
            question = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        hits = execute_hybrid_search(question, vector_store, bm25_index, alpha=ALPHA, top_n=TOP_K, threshold=0.0)
        if not hits:
            print("관련 자료를 찾지 못했습니다.\n")
            continue

        print("\n[검색된 근거]")
        for i, (chunk_id, score) in enumerate(hits, 1):
            c = chunk_by_id[chunk_id]
            print(f"  [{i}] {c.script_file.stem} ({c.start_timestamp}) score={score:.3f}")

        context = build_context(hits, chunk_by_id)

        print("[답변 생성 중... 로컬 모델이라 첫 답은 조금 느릴 수 있어요]")
        answer = ask_ollama(build_prompt(question, context))
        print("\n[답변]\n" + answer + "\n")


if __name__ == "__main__":
    main()
