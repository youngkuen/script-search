# 미니 프로젝트: 강의 스크립트 의미 검색기 → QA 시스템

> Phase 0~6(아래)은 1~3주차까지 배운 것만으로 완결시킨 v1 기록입니다.
> Phase 7+는 5주차(Re-ranking·LangChain·RAGAS·LangSmith) 학습 완료 후 QA 시스템으로
> 확장한 v2입니다 (`.harness/ouroboros/seeds/seed-v2.yaml`). FastAPI 서빙(11주차)·
> 배포(12주차)·Streamlit UI(Part 7)는 여전히 범위 밖입니다.

## 최종 결과물

터미널에 질문을 입력하면, 1~3주차 강의 스크립트(`강의 스크립트 크롤링/`) 중 관련된 부분을
순위대로 찾아서 보여주는 CLI. **답을 생성하지 않고, 찾아주기만 하는 검색기**입니다.

## 확정된 범위

| 항목 | 결정 |
|---|---|
| 검색 대상 데이터 | 1~3주차 강의 스크립트 txt 파일 (`강의 스크립트 크롤링/1주차_스크립트`, `2주차_스크립트`, `3주차_스크립트`) |
| 임베딩 모델 | 로컬 오픈소스 다국어 모델 (`intfloat/multilingual-e5-small` 등) — API 키·비용 불필요 |
| 인터페이스 | 터미널 CLI (FastAPI 없음) |

## 진행 방식

각 단계는 **직접 코드를 쓰고 실행**하고, Claude Code(하네스 적용)가 설계 리뷰·디버깅·개념 설명을 돕는 방식으로 진행합니다.

---

## Phase 0. 프로젝트 뼈대 세우기 — *1주차 도구*
- **씀**: uv 환경 세팅, Claude Code `/init` & CLAUDE.md, 하네스(harness) 시스템
- **할 일**: `uv init`, `chromadb`·`sentence-transformers`·`rank_bm25` 설치, CLAUDE.md에 프로젝트 목적 기록
- **체크포인트**: 세 패키지 import 에러 없이 통과

## Phase 1. 데이터 청킹 — *3주차 후반(청킹 전략)*
- **씀**: 스크립트의 `[00:00]` 타임스탬프 블록을 검색 단위(chunk)로 사용
- **할 일**: txt 파일들을 파싱해서 `{text, 파일명, 주차, 차시}` 형태의 청크 리스트로 변환
- **체크포인트**: 전체 청크 개수 출력

## Phase 2. 임베딩 생성 — *3주차 1-1~1-4 (임베딩 원리, API vs 오픈소스)*
- **씀**: `intfloat/multilingual-e5-small`
- **할 일**: 모델 로드 → 전체 청크 인코딩
- **체크포인트**: 벡터 배열 shape = (청크 수, 차원수) 확인

## Phase 3. Vector DB 구축 — *3주차 "ChromaDB 로컬 벡터 저장소"*
- **할 일**: Chroma collection 생성, 청크+벡터+메타데이터 저장 및 persist
- **체크포인트**: `collection.count()`가 청크 수와 일치, 재실행해도 유지되는지 확인

## Phase 4. 하이브리드 검색 결합 — *3주차 "BM25+Vector 하이브리드 검색"*
- **할 일**: BM25 인덱스 구축 → 두 점수를 정규화 후 가중합(alpha)으로 결합
- **체크포인트**: 쿼리 하나 넣어서 상위 5개 결과가 그럴듯한지 확인

## Phase 5. 골든셋 제작 + 정량 평가 — *3주차 "Hit Rate & MRR"*
- **할 일**: (질문, 정답 청크) 쌍 15~20개 직접 작성, Hit Rate@k·MRR 측정
- 골든셋은 대신 만들어주지 않고 직접 작성 — "느낌이 아니라 숫자로 증명"하는 3주차 원칙을 실제로 적용하는 단계
- **체크포인트**: alpha 값(BM25 비중 vs Vector 비중)을 바꿔가며 성능 비교

## Phase 6. 검색 CLI — *1주차(Claude Code로 코드 작성·디버깅 연습)*
- **할 일**: 질문 입력 → 상위 N개 결과(파일명·시간·스니펫) 출력하는 루프
- **체크포인트**: 실제 질문으로 원하는 강의 부분이 나오는지 체감

---

## Phase 7. 평가 데이터셋 먼저 — *Part 7 종합 프로젝트의 순서 원칙*
- **씀**: Part 7 강의가 강조한 "시스템을 만들기 전에 평가 질문·정답부터 작성하라"는 원칙
- **할 일**: `golden_set.yaml` 기존 항목에 `ground_truth`(정답 텍스트) 최소 20개 이상 직접 작성 — RAGAS Context Recall에 필수
- **체크포인트**: `ground_truth`가 채워진 항목 수 출력

## Phase 8. Re-ranking — *4-2 Cross-Encoder Re-ranking*
- **할 일**: `data/reranker.py`(다국어 CrossEncoder) + `core/reranking.py` + `core/rag_pipeline.py`(하이브리드 검색 20개 → 재랭킹 5개 2-stage)
- **체크포인트**: 같은 질문에 대해 재랭킹 전/후 상위 결과 순서가 달라지는지 확인

## Phase 9. RAG 체인 — *5-1 LangChain·5-2 근거추적*
- **할 일**: `data/llm_client.py`(Anthropic Claude Haiku) + `core/rag_chain.py`(4원칙 프롬프트, 출처·신뢰도 반환) + `cli/main.py` QA 출력 전환
- **체크포인트**: 실제 질문 → 답변+출처+신뢰도 확인, 범위 밖 질문 → Empty Retrieval 안내 메시지 확인

## Phase 10. RAGAS 평가 — *6-1*
- **할 일**: `core/rag_evaluation.py` + `scripts/run_ragas_evaluation.py`
- **체크포인트**: Faithfulness/Answer Relevancy/Context Precision/Context Recall 4개 지표 출력 확인

## Phase 11. LangSmith 연동 (옵션) — *6-2*
- **할 일**: `data/langsmith_setup.py`, `LANGCHAIN_API_KEY` 설정 시 자동 활성화
- **체크포인트**: 키 없이도 QA가 정상 동작하는지, 키 설정 후 smith.langchain.com에 trace가 찍히는지 확인

## 범위 밖 (다음 예고)
- API로 서빙 → FastAPI (11주차)
- 배포 (12주차)
- Streamlit 웹 UI (Part 7 종합 프로젝트 — 이 프로젝트는 CLI 유지로 결정)
