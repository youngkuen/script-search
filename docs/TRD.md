# Technical Requirements Document — 강의 스크립트 의미 검색기

> Source: `.harness/ouroboros/seeds/seed-v1.yaml` (§1~7), `.harness/ouroboros/seeds/seed-v2.yaml` (§8)

## 1. Overview

- **목표 요약**: 1~3주차 강의 스크립트를 로컬 임베딩 + ChromaDB + BM25 하이브리드 검색으로 찾아주는 개인용 CLI. 답을 생성하지 않고 순위별 검색 결과만 제공하며, 직접 만든 골든셋으로 Hit Rate@5·MRR을 측정해 성능을 정량적으로 검증한다.
- **아키텍처 패턴**: 3-tier layered (Presentation `cli/` → Logic `core/` → Data `data/`)
- **v2 갱신 (5주차)**: §8 참고 — 검색기에서 QA 시스템으로 확장(Re-ranking, RAG 답변 생성, RAGAS 평가, LangSmith 모니터링). §2~4는 seed-v1 시점 그대로 두고, v2에서 바뀐 부분만 §8에 별도로 기술한다.

## 2. Layer Design

### 2.1 Presentation Layer (`cli/`)

- **책임**: 질문 입력 수집, 검색 호출, 결과 출력 포맷팅, "관련 내용을 찾지 못했습니다" 메시지 처리
- **모듈**: `cli/main.py` — 인터랙티브 루프 (질문 입력 → 결과 출력 → 다음 질문)
- **입력**: 사용자 질문(문자열)
- **출력 포맷**: 파일명 · 시간 범위(`start_timestamp`~`end_timestamp`) · 텍스트 스니펫, `rank` 순
- **금지**: `chromadb`, `rank_bm25` 직접 import (반드시 `core/`를 경유)

### 2.2 Logic Layer (`core/`)

- **모듈**:
  - `core/chunking.py` — `ChunkScript` (스크립트 텍스트 → `ScriptChunk` 리스트, 타임스탬프 블록 단위)
  - `core/scoring.py` — `ExecuteHybridSearch` (vector_score/bm25_score 계산 → `hybrid_score = alpha * vector_score + (1-alpha) * bm25_score` → threshold 필터 → top_n)
  - `core/evaluation.py` — `TuneAlpha` (골든셋 기준 Hit Rate@5·MRR 계산, alpha 후보 스윕, 조화평균 최대값 확정)
- **비즈니스 규칙**:
  - `hybrid_score < threshold`인 결과는 절대 노출하지 않는다 (AC-004)
  - alpha 확정 기준은 Hit Rate@5·MRR의 **조화평균**(harmonic mean) 최대화 — 산술평균 대신 채택한 이유는 seed의 tech_decisions 참고 (AC-006)
  - `EvaluationRun.is_selected=true`는 항상 정확히 1개
- **트랜잭션 경계**: 없음 (단일 프로세스, DB 트랜잭션 불필요 — ChromaDB/BM25는 append-only 인덱싱)

### 2.3 Data Layer (`data/`)

- **모듈**:
  - `data/script_loader.py` — `IngestScriptFile` (파일 read, 인코딩 오류·빈 파일 감지 → `ScriptFile.status` 결정, skip 시 로그만 남기고 예외를 Logic으로 전파하지 않음)
  - `data/vector_store.py` — `EmbedAndIndexChunk` 호출 지점, ChromaDB persist client 래핑 (add/query)
  - `data/bm25_index.py` — `BuildBM25Index`, BM25 인덱스 구축/조회 래핑 (rank_bm25)
- **엔티티 영속 대상**: `ScriptFile`, `ScriptChunk`(embedding_id로 ChromaDB와 연결), `GoldenSetItem`(YAML 파일), `EvaluationRun`
- **비영속(ephemeral)**: `SearchQuery`, `SearchResult` — 저장하지 않는다 (ARCHITECTURE_INVARIANTS.md #4)
- **외부 연동**: 없음 (로컬 파일 + 로컬 ChromaDB persist 디렉토리 + 로컬 BM25 인덱스뿐, 네트워크 호출 없음)

## 3. Layer Communication

- **Presentation → Logic**: direct call — `cli/main.py`가 `core.scoring.execute_hybrid_search(query)`를 직접 호출
- **Logic → Data**: repository-style 함수 호출 — `core/`는 `data.vector_store.query(...)`, `data.bm25_index.search(...)`, `data.script_loader.load_all(...)`만 호출하고 ChromaDB/BM25 API를 직접 만지지 않음
- **데이터 전달 형식**: `dataclass` (DTO 역할, ORM 없음) — 예: `ScriptChunk`, `SearchResult`, `SearchQuery`를 각각 `@dataclass`로 정의

## 4. Directory Structure

```
미니프로젝트_강의스크립트검색기/
├── cli/
│   ├── __init__.py
│   └── main.py                 # 인터랙티브 검색 루프
├── core/
│   ├── __init__.py
│   ├── chunking.py              # ChunkScript
│   ├── scoring.py               # ExecuteHybridSearch, hybrid_score 계산
│   └── evaluation.py            # TuneAlpha, Hit Rate@5 / MRR
├── data/
│   ├── __init__.py
│   ├── script_loader.py         # IngestScriptFile
│   ├── vector_store.py          # ChromaDB 래퍼 (EmbedAndIndexChunk)
│   └── bm25_index.py            # BM25 래퍼 (BuildBM25Index)
├── tests/
│   ├── test_chunking.py
│   ├── test_scoring.py
│   └── test_evaluation.py
├── golden_set.yaml               # GoldenSetItem 15~20개
├── .chroma/                       # ChromaDB persist 디렉토리 (.gitignore 대상)
└── pyproject.toml                 # uv 프로젝트 설정
```

## 5. Test Strategy

- **프레임워크**: pytest
- **Logic Layer** (`core/`): 단위 테스트 최우선
  - `test_chunking.py`: 타임스탬프 경계, 빈 스크립트, 마지막 블록 처리
  - `test_scoring.py`: `hybrid_score` 계산 공식, threshold 필터링(경계값 포함), 결과 없음 케이스
  - `test_evaluation.py`: Hit Rate@5, MRR, 조화평균 계산의 정확성 — 알고 있는 입력/출력으로 검증
- **Data Layer** (`data/`): 통합 테스트 — 실제 ChromaDB(로컬 persist)·BM25 인덱스에 넣고 빼보는 것까지 확인, 인코딩 오류/빈 파일 skip 동작 검증
- **Presentation Layer** (`cli/`): 수동 검증 (AC-008) — 실제 질문을 입력해 결과가 그럴듯한지 확인. 개인 CLI 도구라 E2E 자동화는 생략
- **원칙**: 구현과 테스트를 같은 타이밍에 작성 (일괄 작성 금지)
- **골든셋/평가 자체가 시스템 레벨 테스트 역할을 겸함** (AC-006, AC-007) — 별도 E2E 테스트 스위트를 만들지 않음

## 6. Decisions & Trade-offs

이번 TRD 논의에서 추가로 확정한 것 (seed의 tech_decisions에 없던 것):

| 결정 | 이유 |
|---|---|
| 테스트 프레임워크: **pytest** | Python 생태계 표준, uv와 통합 쉬움 |
| 골든셋 파일 형식: **YAML** | 사람이 직접 읽고 쓰기 편함, 인터뷰·시드와 관리 방식 일관성 |
| alpha 스윕 범위: **0.0~1.0, 0.1 간격 (11개 후보)** | 이 프로젝트 규모에 적절한 촘촘함 — 너무 거칠면 최적값을 놓치고, 너무 세밀하면 계산 낭비 |
| `.harness/gates/rules/boundaries.yaml`에 `cli/`·`core/`·`data/` 전용 규칙 추가 | 범용 템플릿(src/services 등)은 이 프로젝트 디렉토리와 안 맞아 게이트가 무력화되어 있었음 |
| `ARCHITECTURE_INVARIANTS.md` #4 채움 (비영속 SearchQuery/SearchResult) | seed ontology에서 나온 핵심 invariant를 팀 문서(개인 프로젝트지만 미래의 나)에 고정 |

## 7. Implementation Order

Data → Logic → Presentation 순서, 각 단계 직후 해당 테스트 작성:

1. `data/script_loader.py` (IngestScriptFile) + `test_data` 수동 확인 — AC-001 선행 조건
2. `core/chunking.py` (ChunkScript) + `test_chunking.py` — AC-001
3. `data/vector_store.py` (ChromaDB 래핑) — AC-002
4. `data/bm25_index.py` (BM25 래핑) — AC-003
5. `core/scoring.py` (ExecuteHybridSearch) + `test_scoring.py` — AC-004
6. `golden_set.yaml` 직접 작성 — AC-005 (사용자 수동 작업)
7. `core/evaluation.py` (TuneAlpha) + `test_evaluation.py` — AC-006, AC-007
8. `cli/main.py` — AC-008

## 8. v2 확장 — RAG 답변 생성 (5주차, seed-v2.yaml)

### 8.1 Layer Design 추가분

**Data (`data/`)** — 외부 SDK는 전부 여기서만 만짐:
- `data/llm_client.py` — `get_chat_model()`: `ChatAnthropic`. API 키는 `ANTHROPIC_API_KEY` 환경변수를 langchain-anthropic이 직접 읽음(코드에서 키를 취급하지 않음). **키 미설정 시 즉시 `RuntimeError`로 실패** — 모호한 SDK 스택트레이스 대신 "키를 설정하라"는 명확한 메시지.
- `data/reranker.py` — `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (다국어) lazy singleton + `score_pairs()`.
- `data/rag_embeddings.py` — RAGAS Answer Relevancy가 요구하는 `langchain_core.embeddings.Embeddings` 인터페이스를 기존 e5 임베딩(`data/embedding_model.py`)으로 어댑팅.
- `data/langsmith_setup.py` — `enable_tracing()`: 키 없으면 조용히 `False`, 파이프라인은 계속 동작.

**Logic (`core/`)**:
- `core/reranking.py` — `rerank_candidates()`: Cross-Encoder 2단계 재랭킹.
- `core/rag_pipeline.py` — `retrieve_and_rerank()`: `core/scoring.py::execute_hybrid_search`(넓게) → `rerank_candidates`(좁게) 조합. **`core/scoring.py`는 한 줄도 수정하지 않음.**
- `core/confidence.py` — `confidence_label()`: 점수 → 높음/보통/낮음.
- `core/rag_chain.py` — `generate_answer()`: LangChain `ChatPromptTemplate`+`StrOutputParser`를 각각 `.invoke()`로 명시적 체이닝(LCEL `|` 대신 — 이유는 8.4 참고). Empty Retrieval이면 LLM 미호출.
- `core/rag_evaluation.py` — `collect_eval_samples()`(순수 조립)+`run_ragas_evaluation()`(ragas 호출).

**Data 스키마**: `data/golden_set_loader.py::GoldenSetItem`에 `ground_truth: str | None = None` 추가(하위호환).

**Presentation (`cli/main.py`)**: 검색 결과 나열 → 답변+출처+신뢰도 출력으로 전환. `cli/`는 여전히 `langchain_anthropic`/`ragas`/`sentence_transformers`를 직접 import하지 않음(전부 `core.*` 경유) — `.harness/gates/rules/boundaries.yaml`의 `cli` 규칙에 세 패키지를 추가해 게이트로 강제.

### 8.2 외부 연동 (seed-v1의 "외부 연동: 없음"에서 변경됨)

- **Anthropic API** (`ANTHROPIC_API_KEY` 필요, **유료**) — `cli/main.py` 실행 시마다 질문당 1회 이상 호출. RAGAS 평가(`scripts/run_ragas_evaluation.py`)는 질문당 여러 번의 LLM 판정 호출이 추가로 발생해 비용이 더 든다.
- **LangSmith** (`LANGCHAIN_API_KEY`/`LANGSMITH_API_KEY`, **옵션**) — 없으면 트레이싱 없이 정상 동작. 있으면 매 요청이 smith.langchain.com으로 전송됨.
- API 키가 없을 때는 `data/llm_client.py`에서 즉시 명확한 에러 메시지로 실패한다(GEN-003 — 처리 가능한 지점에서 예외를 잡는다).

### 8.3 의존성 버전 고정 (실제로 겪은 문제)

`ragas`는 최신(0.4.3 등)이 `langchain-community` 최신과 조합될 때 `ragas/llms/base.py`가 `langchain_community.chat_models.vertexai`를 무조건 import하다 `ImportError`가 난다(해당 서브모듈이 최신 langchain-community에서 제거됨). 그래서 `pyproject.toml`을 다음 범위로 고정했다:

```
langchain-core>=0.3,<1.0
langchain-anthropic>=0.3,<0.4
ragas>=0.2,<0.3
```

이 조합에서만 `langchain-community`가 0.3.x대로 함께 해석되어 정상 동작함을 실제로 확인했다(자세한 경위는 `docs/adr.yaml`, `seed-v2.yaml`의 tech_decisions 참고).

### 8.4 Test Strategy 추가분

- `core/rag_chain.py`의 `generate_answer()`는 `prompt | chat_model | StrOutputParser()`라는 LCEL 파이프 대신 `prompt.invoke()` → `chat_model.invoke()` → `StrOutputParser().invoke()`를 명시적으로 호출한다. 이유: 기존 프로젝트의 "모킹 라이브러리 없이 손으로 만든 Fake" 테스트 컨벤션을 유지하려면 Fake `chat_model`이 `.invoke()`만 구현하면 되게 만드는 편이, `Runnable` 프로토콜(`__or__` 코어시션 대상이 되려면 `Runnable` 상속 또는 `__call__` 구현 필요)을 흉내 내게 만드는 것보다 훨씬 단순하다.
- `data/llm_client.py`·`data/reranker.py`·`data/rag_embeddings.py`·`data/langsmith_setup.py`는 외부 SDK/모델 통합이라 단위테스트 대신 수동 검증 — 기존 Data Layer 방침(README: "통합 테스트 위주")과 동일.
- `core/rag_evaluation.py::run_ragas_evaluation()`(실제 `ragas.evaluate()` 호출)도 같은 이유로 수동 검증. `collect_eval_samples()`(순수 조립 로직)만 단위테스트.
