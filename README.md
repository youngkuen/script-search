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

3-tier 레이어 분리를 강제하며 진행했습니다 (레이어를 건너뛰는 import는 커밋 전 게이트가 차단).
5주차 확장으로 검색 파이프라인 뒤에 재랭킹·답변 생성·평가·모니터링이 추가됐습니다:

```
질문
  │
  ▼
Presentation (cli/main.py)
  │  질문 입력 · 답변+출처+신뢰도 출력 · .env 로드 · LangSmith tracing 켜기
  ▼
Logic (core/)
  │  하이브리드 검색(scoring.py) ──▶ Cross-Encoder 재랭킹(reranking.py, rag_pipeline.py)
  │        ▼
  │  RAG 답변 생성(rag_chain.py) — 4원칙 프롬프트 + 신뢰도(confidence.py)
  │        ▼
  │  RAGAS 평가(rag_evaluation.py, 오프라인 배치)
  ▼
Data (data/)
     ChromaDB · BM25 · 로컬 임베딩 · Anthropic Claude(llm_client.py) ·
     다국어 Cross-Encoder(reranker.py) · LangSmith(langsmith_setup.py, 옵션)
```

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
| 10 | 청킹 전략 비교 (4주차) | block / merged(블록 병합) / tokens(토큰 예산) 3종의 Hit Rate·MRR 비교 — `scripts/compare_chunking.py` |
| 11 | Cross-Encoder Re-ranking (4-2) | 다국어 Cross-Encoder로 하이브리드 후보를 넓게(20)→좁게(5) 2-stage 재랭킹 — `data/reranker.py`, `core/reranking.py`, `core/rag_pipeline.py` |
| 12 | LangChain·LCEL 파이프라인 (5-1) | ChatPromptTemplate + Anthropic Claude Haiku로 답변 생성, 4원칙 프롬프트(문서 근거만/모르면 모른다/추측 금지/출처 언급) — `data/llm_client.py`, `core/rag_chain.py` |
| 13 | 응답 근거 추적·설명 가능성 (5-2) | 출처 청크(파일명·타임스탬프)와 신뢰도 등급(높음/보통/낮음)을 답변과 함께 반환 — `core/confidence.py`, `core/rag_chain.py::SourceCitation` |
| 14 | RAGAS 평가 (6-1) | golden_set의 ground_truth로 Faithfulness/Answer Relevancy/Context Precision/Context Recall 계산 — `core/rag_evaluation.py`, `scripts/run_ragas_evaluation.py` |
| 15 | LangSmith 모니터링 (6-2) | 계정·키가 있으면 환경변수 3줄로 tracing 활성화, 없으면 조용히 비활성 — `data/langsmith_setup.py` |

## 실제 성능

4주차 스크립트를 추가해 **88개 파일, 5,120개 청크** 전체를 색인하고, 골든셋(**26문항** — 1~3주차 20
+ 4주차 6)으로 alpha를 0.0~1.0까지 스윕한 뒤 확정했습니다.

| 지표 | 목표 | 실제 (alpha=0.5) |
|---|---|---|
| Hit Rate@5 | ≥ 0.8 | **0.885** |
| MRR | ≥ 0.5 | **0.782** |

alpha 확정 기준은 Hit Rate@5와 MRR의 산술평균이 아니라 **조화평균**을 사용했습니다 — 한쪽
지표만 높아도 점수가 높게 나오는 것을 방지하기 위한 선택입니다.

> 참고: 1~3주차만 색인했을 때는 Hit Rate@5 1.000이었으나, 4주차 문항을 추가하자 0.885로 내려갔습니다.
> 4주차 개념 질문의 정답이 짧은 타임스탬프 블록 여러 개에 걸쳐 흩어져 있어(예: "HNSW ... Navigable
> Small World"가 블록 경계에서 잘림) 정확한 블록을 top-5에 올리기 어려운 경우가 있기 때문입니다.
> 이 관찰이 아래 청킹 전략 비교(block vs merged vs tokens)의 동기가 되었습니다. 4주차 골든 6문항은
> 초안이므로 직접 검토·보정하는 것을 권장합니다.

### 청킹 전략 비교 (실험 — `scripts/compare_chunking.py`)

위 관찰(블록이 잘게 쪼개져 정답 블록을 특정하기 어려움)을 검증하려고, 같은 코퍼스(88파일)·같은
골든셋(26문항)으로 세 청킹 전략을 비교했습니다. 전략마다 chunk_id가 달라지므로 "검색된 청크
텍스트가 정답 블록 텍스트를 포함하는가"로 판정했습니다 (alpha=0.5).

| 전략 | 청크 수 | Hit Rate@5 | MRR | 색인 시간 |
|---|---|---|---|---|
| block (기본: 타임스탬프 블록 1개) | 5,120 | 0.885 | 0.782 | ~330s |
| merged (모드 A: 블록 3개 병합) | 1,735 | **0.923** | 0.761 | ~165s |
| tokens (모드 B: 토큰 256 예산) | 1,158 | **0.923** | **0.782** | ~118s |

- 블록을 병합/토큰 예산으로 묶어 문맥을 넓히자 **Hit Rate@5가 0.885 → 0.923으로 개선**됐고,
  청크 수가 크게 줄어(5,120 → 1,158) 색인·검색 비용도 함께 감소했습니다.
- 특히 **tokens(모드 B)** 가 Hit Rate·MRR 모두 최상이면서 청크 수가 가장 적어 가장 유리했습니다.
- 기본 파이프라인은 아직 block을 씁니다. tokens로 바꾸려면 재색인 + alpha 재스윕 + 골든셋
  expected_chunk 재산정이 필요하므로, 채택 여부는 별도 결정으로 남겨두었습니다.

### RAGAS 평가 (v2 — 검색+생성 전체 품질)

Hit Rate/MRR은 검색 단계만 측정하므로, 답변 생성(5-1)까지 포함한 품질은 별도로 측정해야 했습니다.
골든셋 26문항 전부에 `ground_truth`(정답 텍스트)를 채운 뒤 `scripts/run_ragas_evaluation.py`로
전체 파이프라인(검색+재랭킹+생성)을 실제로 돌려 RAGAS 4대 지표를 계산했습니다(Anthropic Claude
Haiku 기준). 첫 실행 후 Answer Relevancy가 가장 낮게 나와, 프롬프트를 두 차례 바꿔가며 개선을
시도했습니다(세 버전 전체 비교와 실패 사례는 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) 참고):

| 지표 | 단계 | 원본 | 채택된 버전 |
|---|---|---|---|
| Faithfulness (충실도) | 생성 | 0.877 | **0.878** |
| Answer Relevancy (답변 관련성) | 생성 | 0.784 | **0.790** |
| Context Precision (검색 정밀도) | 검색 | 0.844 | **0.844** |
| Context Recall (검색 재현율) | 검색 | 0.923 | **0.962** |

프롬프트에 "질문에서 묻는 대상(핵심 주제·표현)을 답변 문장에 되풀이하며 답하라"는 규칙을
추가해(`core/rag_chain.py`) Answer Relevancy를 소폭 개선했습니다. 다만 개선폭(+0.006)이 RAGAS
자체의 run-to-run 변동성(같은 파이프라인인데도 Context Recall이 ±0.04 흔들림)보다 작아서,
"확실한 개선"이라기보다 "퇴보 없이 나아진 최선의 시도"로 보는 게 정확합니다. 26문항은 강의 권장
최소치(20개)는 넘겼지만 통계적으로 넉넉한 표본은 아니라서, 이 수치들 전반을 확정 성능이 아니라
첫 벤치마크로 보시면 됩니다.

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
uv run pytest -q                           # 테스트 (74개)
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
- [ ] 위 gs-24 사례(블록 경계 절단) 검증 기준으로 청킹 전략(merged/tokens) 채택 여부 결정
- [x] Answer Relevancy(0.784, 4대 지표 중 최저) 개선 시도 — 2개 버전 비교 후 "질문 주제 되풀이" 채택(0.790), 상세 기록은 `docs/EXPERIMENTS.md`
- [ ] RAGAS 자체의 run-to-run 변동성을 먼저 정량화(동일 프롬프트 반복 실행)해 위 개선이 노이즈인지 재검증
- [ ] (범위 밖으로 남겨둔) FastAPI 서빙·배포·Streamlit UI는 해당 주차 학습 후 별도 프로젝트로 확장 검토

## 테스트

```bash
uv run pytest -q
# 74 passed
```
