# Technical Requirements Document — 강의 스크립트 의미 검색기

> Source: `.harness/ouroboros/seeds/seed-v1.yaml`

## 1. Overview

- **목표 요약**: 1~3주차 강의 스크립트를 로컬 임베딩 + ChromaDB + BM25 하이브리드 검색으로 찾아주는 개인용 CLI. 답을 생성하지 않고 순위별 검색 결과만 제공하며, 직접 만든 골든셋으로 Hit Rate@5·MRR을 측정해 성능을 정량적으로 검증한다.
- **아키텍처 패턴**: 3-tier layered (Presentation `cli/` → Logic `core/` → Data `data/`)

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
