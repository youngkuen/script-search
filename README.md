# 강의 스크립트 QA 시스템

메타코드 AI 엔지니어 부트캠프 1~5주차 수업(패키지 관리, LLM/Tool Use, 임베딩·Vector DB·정량 평가,
4주차 RAG 데이터 파이프라인, 그리고 5주차 검색 최적화·LangChain·RAG 평가/모니터링)을 직접 손으로
복습하기 위해 만든 개인 미니 프로젝트입니다. 강의에서 배운 개념을 하나씩 실제로 구현해보고,
"느낌이 아니라 숫자로 증명한다"는 원칙을 골든셋 기반 정량 평가(Hit Rate/MRR, RAGAS)로 직접
적용했습니다.

원래는 "답을 생성하지 않고 찾아주기만 하는 검색기"로 시작했지만(1~4주차, 아래 v1 성능 참고),
5주차 학습을 마친 뒤 Cross-Encoder 재랭킹·LangChain RAG 체인·근거추적·RAGAS 평가·LangSmith
모니터링을 반영해 **질문 → 답변 + 출처 + 신뢰도**를 돌려주는 QA 시스템으로 확장했습니다
(`.harness/ouroboros/seeds/seed-v2.yaml` 참고). FastAPI 서빙과 배포는 여전히 이후 주차 범위라 제외했습니다.

```
질문> 직원 출장 시 숙박비 한도는 얼마야?

국내 출장의 경우 1박당 숙박비 한도는 15만 원입니다. 해외 출장은 지역별 기준이 다르므로
별표 3을 확인하시기 바랍니다. (출처: 임직원 출장 규정.pdf)

출처 (3개):
  [임직원_출장_규정] (00:12~01:04) 신뢰도=높음 score=0.842
  [임직원_출장_규정] (02:30~03:10) 신뢰도=보통 score=0.512
  ...
```

## 데이터에 대한 안내 (저작권)

검색 대상인 강의 스크립트 원문과, 거기서 파생된 골든셋(`golden_set.yaml`)은 메타코드의
저작권 콘텐츠이므로 이 리포지토리에는 포함하지 않았습니다(`.gitignore` 처리). 대신:

- 스크립트는 [셀레니움](https://www.selenium.dev/)으로 강의 사이트의 자막 패널을 스크롤하며
  직접 크롤링한 개인 학습용 데이터입니다(코드는 이 리포 밖에 별도 보관).
- 골든셋 형식은 `golden_set.example.yaml`에 예시 데이터로만 남겨두었습니다.
- 코드(`cli/`, `core/`, `data/`)는 강의 스크립트 없이도 그대로 읽을 수 있고, 자신의 텍스트
  코퍼스로 바로 재사용할 수 있습니다.

## 아키텍처

### 데이터 흐름 (파이프라인)

색인은 한 번만 실행되는 오프라인 단계이고, 질의응답은 질문마다 실행되는 온라인 단계입니다.
임베딩이 색인 시점과 검색 시점에 **각각 다른 프리픽스로 두 번** 쓰이는 것(비대칭 임베딩,
4주차)과, 벡터DB에 저장만 해서는 부족하고 검색·재랭킹을 거쳐야 LLM에 넘길 근거가 만들어지는
것이 이 파이프라인에서 가장 중요한 두 지점입니다.

```
[오프라인] 색인 — 문서가 바뀔 때만 재실행
  데이터(원본 스크립트)
    ──▶ 청킹 (전처리: 유니코드 정규화 포함 + 토큰 256개 예산으로 블록 병합, chunking.py)
    ──▶ 임베딩 · passage 프리픽스 (embedding_model.py, indexing.py)
    ──▶ 벡터DB 저장 (ChromaDB, vector_store.py)

[온라인] 질의응답 — 질문마다 실행
  질문
    ──▶ 임베딩 · query 프리픽스 (embedding_model.py)          ← 임베딩이 색인 때와 별도로 한 번 더
    ──▶ 검색 · BM25+벡터 하이브리드, 20개 후보 (scoring.py)
    ──▶ 재랭킹 · Cross-Encoder로 5개로 압축 (reranker.py, reranking.py, rag_pipeline.py)
    ──▶ LLM 답변 생성 · Claude Haiku, 4원칙 프롬프트 (llm_client.py, rag_chain.py)
    ──▶ 답변 + 출처 + 신뢰도 (confidence.py)

[오프라인, 옵션] 평가 · 모니터링
  RAGAS 평가 — 위 온라인 파이프라인 전체를 배치로 재실행해 품질 측정 (rag_evaluation.py)
  LangSmith — 온라인 단계 실행 중 각 호출을 옆에서 기록만 함, 파이프라인 흐름엔 안 끼어듦 (옵션, langsmith_setup.py)
```

### 3-Tier 레이어

3-tier 레이어 분리를 강제하며 진행했습니다 (레이어를 건너뛰는 import는 커밋 전 게이트가 차단).
위 파이프라인의 각 단계는 아래 세 디렉토리 중 하나에 속합니다:

| 레이어 | 디렉토리 | 책임 |
|---|---|---|
| Presentation | `cli/main.py` | 질문 입력, 답변+출처+신뢰도 출력, `.env` 로드, LangSmith tracing 활성화 |
| Logic | `core/` | 청킹·색인 조율·하이브리드 스코어링·골든셋 평가(1~4주차: `chunking.py`/`indexing.py`/`scoring.py`/`evaluation.py`) + 재랭킹·RAG 체인·RAGAS 평가(5주차: `reranking.py`/`rag_pipeline.py`/`confidence.py`/`rag_chain.py`/`rag_evaluation.py`) |
| Data | `data/` | ChromaDB·BM25·로컬 임베딩·파일 로딩(1~4주차: `vector_store.py`/`bm25_index.py`/`embedding_model.py`/`script_loader.py`) + Anthropic·Cross-Encoder·LangSmith·`.env`(5주차: `llm_client.py`/`reranker.py`/`rag_embeddings.py`/`langsmith_setup.py`/`env_loader.py`) |

`cli/`는 여전히 `chromadb`·`rank_bm25`·`langchain_anthropic`·`ragas`·`sentence_transformers` 같은
외부 SDK를 직접 import하지 않고 전부 `core.*`/`data.*`를 경유합니다(`.harness/gates/rules/boundaries.yaml`로
자동 차단). `SearchQuery`/`SearchResult`는 DB에 영속화하지 않는 비영속(ephemeral) 엔티티로 설계했습니다 —
검색은 상태를 남길 필요 없는 1회성 조회이기 때문입니다. 자세한 온톨로지와 결정 근거는
[`seed-v1.yaml`](.harness/ouroboros/seeds/seed-v1.yaml)(검색기 v1)과
[`seed-v2.yaml`](.harness/ouroboros/seeds/seed-v2.yaml)(QA 시스템 v2)에 있습니다.

## 강의 개념 → 구현 매핑

| Phase | 배운 개념 (몇 주차) | 실제 구현 |
|---|---|---|
| 0 | uv 패키지 관리, Claude Code 하네스 (1주차) | `uv init`, 이 리포의 `.harness/`·`CLAUDE.md` |
| 1 | 청킹 전략 (3주차) | 스크립트의 `[00:00]` 타임스탬프 블록을 검색 단위로 사용 — `core/chunking.py` |
| 2 | 임베딩 원리, API vs 오픈소스 (3주차) | 로컬 다국어 모델(`intfloat/multilingual-e5-small`)로 전체 청크 인코딩 — `data/embedding_model.py` |
| 3 | ChromaDB 로컬 벡터 저장소 (3주차) | 청크+벡터+메타데이터 persist — `data/vector_store.py` |
| 4 | BM25 + Vector 하이브리드 검색 (3주차) | 두 점수를 alpha로 정규화·가중합 — `core/scoring.py` |
| 5 | Hit Rate@k, MRR 정량 평가 (3주차) | 골든셋 직접 작성 + alpha 스윕으로 최적값 확정 — `core/evaluation.py`, `scripts/run_evaluation.py` |
| 6 | 디버깅/코드 작성 연습 (1주차) | 질문 입력 → 하이브리드 검색 → 결과 출력 루프 — `cli/main.py` |
| 7 | 비대칭 임베딩 (4주차) | e5 계열 요구 프리픽스(query:/passage:)를 인덱싱·검색에 분리 적용 — `data/embedding_model.py` |
| 8 | 거리공간·메타데이터 필터 (4주차) | `hnsw:space=cosine` 명시 + `/week N` 주차 필터 검색 — `data/vector_store.py`, `core/scoring.py`, `cli/main.py` |
| 9 | 유니코드 전처리 (4주차) | NFC 정규화 + 보이지 않는 문자(제로폭·BOM·NBSP) 제거 — `core/chunking.py` |
| 10 | 청킹 전략 비교 (4주차) | block / merged(블록 병합) / tokens(토큰 예산) 3종의 Hit Rate·MRR 비교 후 **tokens 채택**(2026-07-22) — `scripts/compare_chunking.py`, `core/chunking.py` |
| 11 | Cross-Encoder Re-ranking (4-2) | 다국어 Cross-Encoder로 하이브리드 후보를 넓게(20)→좁게(5) 2-stage 재랭킹 — `data/reranker.py`, `core/reranking.py`, `core/rag_pipeline.py` |
| 12 | LangChain·LCEL 파이프라인 (5-1) | ChatPromptTemplate + Anthropic Claude Haiku로 답변 생성, 4원칙 프롬프트(문서 근거만/모르면 모른다/추측 금지/출처 언급) — `data/llm_client.py`, `core/rag_chain.py` |
| 13 | 응답 근거 추적·설명 가능성 (5-2) | 출처 청크(파일명·타임스탬프)와 신뢰도 등급(높음/보통/낮음)을 답변과 함께 반환 — `core/confidence.py`, `core/rag_chain.py::SourceCitation` |
| 14 | RAGAS 평가 (6-1) | golden_set의 ground_truth로 Faithfulness/Answer Relevancy/Context Precision/Context Recall 계산 — `core/rag_evaluation.py`, `scripts/run_ragas_evaluation.py` |
| 15 | LangSmith 모니터링 (6-2) | 계정·키가 있으면 환경변수 3줄로 tracing 활성화, 없으면 조용히 비활성 — `data/langsmith_setup.py` |

## 실제 성능

**청킹 전략은 tokens(토큰 256 예산)를 채택했습니다** (2026-07-22, 아래 비교 실험 참고). 88개
파일을 **1,158개 청크**로 색인하고, 골든셋(**41문항** — 1~3주차 20 + 4주차 6 + 커버리지 확장
15, 아래 "골든셋 커버리지 확장" 참고)으로 alpha를 0.0~1.0까지 재스윕한 뒤 확정했습니다.

| 지표 | 목표 | 실제 (alpha=0.5, tokens + golden_set 41문항) |
|---|---|---|
| Hit Rate@5 | ≥ 0.8 | **0.976** |
| MRR | ≥ 0.5 | **0.859** |

alpha 확정 기준은 Hit Rate@5와 MRR의 산술평균이 아니라 **조화평균**을 사용했습니다 — 한쪽
지표만 높아도 점수가 높게 나오는 것을 방지하기 위한 선택입니다.

### 청킹 전략 비교 및 채택 (`scripts/compare_chunking.py`)

1~3주차만 색인했을 때는 Hit Rate@5 1.000이었으나, 4주차 문항을 추가하자 원래 쓰던 block
전략(타임스탬프 블록 1개=청크 1개)으로는 0.885로 내려갔습니다. 4주차 개념 질문의 정답이 짧은
블록 여러 개에 걸쳐 흩어져 있어(예: "근사 최근접 이웃 검색... 영어로는?"의 한국어 소개와 영어
번역이 인접한 두 블록으로 분리) 정확한 블록을 top-5에 올리기 어려운 경우가 있었기 때문입니다.
같은 코퍼스(88파일)·같은 골든셋(26문항)으로 세 청킹 전략을 비교했습니다:

| 전략 | 청크 수 | Hit Rate@5 | MRR | 색인 시간 |
|---|---|---|---|---|
| block (블록 1개=청크) | 5,120 | 0.885 | 0.782 | ~330s |
| merged (블록 3개 병합) | 1,735 | 0.923 | 0.761 | ~165s |
| **tokens (토큰 256 예산) — 채택** | **1,158** | **0.923** | **0.782** | ~118s |

**tokens를 채택**했습니다 — Hit Rate·MRR 모두 최상이면서 청크 수가 가장 적어 색인·검색 비용도
가장 낮았습니다. 4주차 골든셋 검토(gs-24, "근사 최근접 이웃 검색을 영어로?")에서 이 전략이
실제로 블록 경계 절단 문제를 고치는지 재검증했고(miss → rank 1 hit), golden_set 26개 항목
전부를 새 chunk_id로 재매핑한 뒤 alpha를 0.5→0.4로 재확정했습니다. 이 과정에서 gs-23이
새로 top-5를 놓치는 트레이드오프가 있었는데, 원인을 깊이 파보니 검색 결함이 아니라 같은
개념(asymmetric embedding)을 이론 강의와 실습이 각각 설명하고 있어 **정답이 원래 두 곳에
있던 것**으로 밝혀졌습니다. 그래서 `golden_set`의 `expected_chunk`(단일)를
`expected_chunks`(복수 허용)로 스키마를 확장했고, 그 결과 Hit Rate@5 0.923→0.962로 개선됐습니다.
이어서 나머지 25개 항목도 실제 검색 결과와 원문을 대조해 **전수 검토**했더니, 7개 항목에서
같은 패턴(정답이 여러 소스에 있음)을 추가로 발견해 반영했습니다 — 특히 gs-05는 원래 정답이
top-5 중 5위(가장 약한 매칭)였는데 1~4위가 전부 같은 사실을 더 명확하게 설명하고 있었습니다.
반면 표면적으로만 비슷하고 맥락이 다른 경우(예: gs-18의 "함수 환각" vs 다른 강의의 "tool
환각")는 의도적으로 제외해 골든셋이 무분별하게 관대해지지 않도록 경계를 지켰습니다. 최종
MRR은 0.788→**0.872**로 개선됐습니다. 전체 과정과 근본 원인 분석은
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)에 정직하게 기록했습니다.

### 골든셋 커버리지 확장 (26 → 41문항)

"데이터 품질에 더 투자하면 성능이 더 좋아질까?"를 점검하다가, 검색 대상 88개 파일 중
golden_set이 실제로 정답 청크로 참조하는 파일은 24개뿐이라는 걸 확인했습니다. 나머지 64개
파일(Transformer 구조, Tokenizer, KV Cache, BLEU/ROUGE, 문서 파싱, 텍스트 전처리 등)은
문항이 하나도 없어 검증이 안 되고 있었던 겁니다. 4개 주차에 고르게 분산한 15개 파일을 골라
문항을 추가했고(gs-27~gs-41), 검증 과정에서 gs-28/29/33/35처럼 정답이 인접 청크나 다른
차시에 나뉘어 있는 경우를 발견해 복수 정답으로 등록했습니다. gs-34는 1단계 하이브리드 검색
기준 top-5를 놓치는 실제 약점을 억지로 고치지 않고 그대로 남겼습니다.

41문항으로 재스윕한 결과 최적 alpha가 0.4에서 **0.5**로 이동했습니다(Hit Rate@5
0.962→0.976, MRR 0.872→0.859). alpha 이동으로 기존 항목 gs-26이 1단계 검색에서 새로
top-5를 놓치게 됐지만, Cross-Encoder 재랭킹까지 포함한 실제 파이프라인으로 확인해 보니
gs-26과 gs-34 둘 다 재랭킹 후에는 1위로 복구되어 실사용자에게는 영향이 없었습니다 — Hit
Rate/MRR 스크립트가 재랭킹 이전 1단계만 측정하기 때문에 생기는 차이입니다. 상세 근거는
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) "golden_set 15문항 확장" 항목 참고.

### RAGAS 평가 (v2 — 검색+생성 전체 품질)

Hit Rate/MRR은 검색 단계만 측정하므로, 답변 생성(5-1)까지 포함한 품질은 별도로 측정해야 했습니다.
아래 결과는 골든셋이 26문항이던 시점에 그 전부에 `ground_truth`(정답 텍스트)를 채운 뒤
`scripts/run_ragas_evaluation.py`로 전체 파이프라인(검색+재랭킹+생성)을 실제로 돌려 계산한
것입니다(Anthropic Claude Haiku 기준). 이후 골든셋 커버리지 확장으로 추가된 15문항(gs-27~41)도
`ground_truth`를 함께 작성해 두었지만, 아직 이 표의 RAGAS 재측정에는 반영하지 않았습니다 —
재측정하려면 스크립트를 다시 실행하면 됩니다(LLM 호출 비용 발생). 프롬프트 개선(Answer
Relevancy 목표)과 청킹 전략 전환(tokens 채택), 두 차례 변경 전후를 비교했습니다 — 전체 실험
과정은 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) 참고:

| 지표 | 단계 | 최초(block) | 프롬프트 개선 후(block) | **tokens 채택 후(최종)** |
|---|---|---|---|---|
| Faithfulness (충실도) | 생성 | 0.877 | 0.878 | **0.958** |
| Answer Relevancy (답변 관련성) | 생성 | 0.784 | 0.790 | **0.859** |
| Context Precision (검색 정밀도) | 검색 | 0.844 | 0.844 | **0.910** |
| Context Recall (검색 재현율) | 검색 | 0.923 | 0.962 | **1.000** |

청킹 전략을 tokens로 바꾸자 4개 지표 전부 뚜렷하게 개선됐습니다 — 청크당 더 완전한 문맥을
담게 되면서 검색뿐 아니라 LLM 생성 품질에도 도움이 된 것으로 해석됩니다. 다만 두 가지는
정직하게 밝혀둡니다: (1) 26문항은 강의 권장 최소치(20개)는 넘겼지만 통계적으로 넉넉한 표본은
아니고, (2) 이 재평가 중 104개 판정 작업 중 1개가 `LLMDidNotFinishException`(max_tokens 부족)으로
실패해, Context Recall=1.000처럼 극단적인 수치는 "완벽"이 아니라 "강한 개선 신호" 정도로
해석하는 것이 정확합니다.

## 스펙 우선 개발 (Ouroboros)

Claude Code로 코드를 작성하되, 곧바로 구현에 들어가지 않고
[`studioKjm/ai-harness-template`](https://github.com/studioKjm/ai-harness-template)을 적용해
스펙을 먼저 확정하는 방식으로 진행했습니다:

```
/interview → /seed → /trd → /decompose → /run → /evaluate
 (요구사항 인터뷰)  (불변 스펙)  (기술설계)  (원자적 태스크 분해)  (구현)   (3단계 검증)
```

- `.harness/ouroboros/seeds/seed-v1.yaml` — 목표·제약·8개 Acceptance Criteria·온톨로지를 담은 불변 스펙
- `.harness/ouroboros/seeds/seed-v2.yaml` — 5주차 확장(AC-009~AC-013: 재랭킹·RAG생성·근거추적·RAGAS·LangSmith) 스펙, seed-v1을 대체하지 않고 버전만 올림
- `docs/TRD.md` — 레이어별 기술 설계 (§8이 v2 확장분)
- `.harness/ouroboros/decompositions/` — AC를 원자적 태스크로 분해한 실행 순서 (decompose-v1, decompose-v2)
- `.harness/ouroboros/evaluations/eval-v1-2026-07-09.yaml` — v1 기계적/의미적 검증 결과 (8/8 AC 충족, 테스트 38/38 통과)
- 커밋 전 게이트(`.harness/gates/`) — 시크릿 유출, 레이어 경계 위반, 3-tier 분리 위반을 자동 차단

구현·평가 과정에서 하네스 템플릿 자체의 버그도 몇 개 발견해 고쳤습니다: 게이트 스크립트들이
`.venv/`를 제외하지 않아 발생한 오탐, `check-layers.sh`가 `cli/`·`data/` 디렉토리를 레이어로
인식하지 못한 문제, 경계 규칙의 정규식이 `cli.`를 `client`와 잘못 매칭한 문제 등입니다. 또한
ChromaDB의 Rust 바인딩이 한글 경로에 저장된 HNSW 인덱스를 재오픈하지 못하는 버그를 발견해
persist 경로를 ASCII 경로(`~/.cache/script-search/chroma`)로 우회했습니다.

## 실행 방법

QA 시스템(v2)은 답변 생성에 Anthropic Claude API를 쓰므로 **`ANTHROPIC_API_KEY`가
필수**입니다(검색·임베딩 자체는 여전히 로컬·무료). [console.anthropic.com](https://console.anthropic.com)에서
키를 발급받아 `.env.example`을 `.env`로 복사한 뒤 값을 채워 넣으세요(`.env`는 `.gitignore`에
등록되어 있어 커밋되지 않습니다). `python-dotenv` 없이 `data/env_loader.py`가 직접
읽어서 적용하며, 셸에서 이미 export한 값이 있으면 그쪽이 항상 우선합니다.

```bash
cp .env.example .env && vi .env            # ANTHROPIC_API_KEY 값 채우기
uv sync                                    # 의존성 설치
uv run pytest -q                           # 테스트 (76개)
uv run python -m cli.main                  # QA CLI 실행 (질문 → 답변+출처+신뢰도)
uv run python scripts/run_evaluation.py    # 재색인 + alpha 스윕/검증 (Hit Rate·MRR, 검색 단계만)
uv run python scripts/run_ragas_evaluation.py  # RAGAS 4대 지표 평가 (검색+생성 전체, LLM 호출 비용 발생)
uv run python scripts/compare_chunking.py  # 청킹 전략 3종 비교 실험
```

LangSmith 모니터링은 옵션입니다 — `LANGCHAIN_API_KEY`(또는 `LANGSMITH_API_KEY`)를 추가로 설정하면
자동으로 켜지고, 없으면 그냥 켜지지 않을 뿐 QA 기능은 그대로 동작합니다.

CLI는 상위 폴더의 `../강의 스크립트 크롤링/{1,2,3,4}주차_스크립트/`에서 텍스트 파일을 찾습니다.
자신의 텍스트 코퍼스로 써보려면 `cli/main.py`의 `WEEK_DIRS`만 원하는 디렉토리로 바꾸면 됩니다.

특정 주차만 검색하려면 질문 앞에 `/week N`을 붙입니다 (4주차 메타데이터 필터 실습):

```
질문> /week 4 벡터 인덱싱 알고리즘 두 가지는?
```

RAGAS 평가를 실행하려면 `golden_set.yaml`의 항목에 `ground_truth`(정답 텍스트)를 직접
채워야 합니다 — 자동 생성되지 않습니다(`golden_set.example.yaml` 참고, 6-1 강의 원칙: "Ground
Truth는 도메인 전문가가 검토해야 한다").

## 앞으로 계획

이 프로젝트는 부트캠프 진행에 맞춰 계속 보강할 예정입니다:

- [x] 4주차 강의 스크립트를 색인에 추가(upsert)하고 alpha 재검증 (완료 — alpha=0.5, 88파일/5,120청크)
- [x] 4주차 기법 적용: 비대칭 임베딩, 코사인 거리공간, 주차 메타데이터 필터, 유니코드 전처리, 청킹 전략 비교
- [x] 5주차 기법 적용: Cross-Encoder 재랭킹, LangChain RAG 체인, 근거추적, RAGAS 평가, LangSmith 옵션 연동 (완료 — `seed-v2.yaml`)
- [x] golden_set.yaml에 RAGAS용 ground_truth 26개 전부 채우고 실제 RAGAS 평가 실행 (완료 — Faithfulness 0.877 등, 위 결과 참고)
- [x] 4주차 골든 6문항 검토 완료 — 데이터는 전부 정확, 다만 2문항(gs-24/26)이 실제 검색에서 top-5 miss(원인 진단은 `docs/EXPERIMENTS.md`)
- [x] gs-24 사례(블록 경계 절단) 검증 기준으로 청킹 전략 채택 여부 결정 — **tokens 채택** (Hit Rate@5 0.885→0.923, RAGAS 4개 지표 전부 개선, `docs/adr.yaml` ADR-007)
- [x] Answer Relevancy(0.784, 4대 지표 중 최저) 개선 시도 — 2개 버전 비교 후 "질문 주제 되풀이" 채택(0.790), 상세 기록은 `docs/EXPERIMENTS.md`
- [ ] RAGAS 자체의 run-to-run 변동성을 먼저 정량화(동일 프롬프트 반복 실행)해 프롬프트 개선이 노이즈인지 재검증
- [x] gs-23 회귀 원인 심층 분석 완료 — 검색 실패가 아니라 같은 개념(asymmetric embedding)이 이론(10차시)·실습(13차시) 두 곳에 설명돼 있어 정답이 여러 개였던 것으로 확인(`docs/EXPERIMENTS.md`). golden_set의 단일 `expected_chunk` 스키마 한계로 결론
- [x] golden_set 스키마를 `expected_chunk`(단일) → `expected_chunks`(복수 허용)로 확장 완료 — gs-23에 정답 3개 반영 (`docs/adr.yaml` ADR-009)
- [x] 나머지 25개 항목도 전수 검토 완료 — 7개 항목(gs-03/05/14/16/17/20/21)에 복수 정답 추가 반영, 최종 Hit Rate@5=0.962·MRR=0.872 (`docs/EXPERIMENTS.md` "golden_set 26개 전수 검토")
- [x] hybrid_top_n(20)까지 넓혀 6~20위권(재랭킹 이전) 숨은 정답 후보 26개 항목 전체 확인 완료 — gs-03에 정답 1개 추가, gs-01은 강의마다 다른 벤치마크 수치(사실 상충)로 판단해 반영 보류(`docs/EXPERIMENTS.md` "hybrid_top_n 6~20위권 확장 검토")
- [x] golden_set 커버리지 확장 완료 — 미커버 64개 파일 중 15개 파일에 문항 추가(26→41), alpha 0.4→0.5 재확정, Hit Rate@5=0.976·MRR=0.859 (`docs/EXPERIMENTS.md` "golden_set 15문항 확장", `docs/adr.yaml` ADR-010)
- [ ] gs-26 유형("여러 강의에 흩어진 사실")은 청킹과 무관 — 쿼리 확장 등 별도 접근 검토
- [ ] 새로 추가된 15문항(gs-27~41)의 ground_truth로 RAGAS 재측정 — 아직 26문항 시점 결과만 반영됨
- [ ] (범위 밖으로 남겨둔) FastAPI 서빙·배포·Streamlit UI는 해당 주차 학습 후 별도 프로젝트로 확장 검토

## 테스트

```bash
uv run pytest -q
# 76 passed
```
