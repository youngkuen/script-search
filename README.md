# 강의 스크립트 의미 검색기

메타코드 AI 엔지니어 부트캠프 1~4주차 수업(패키지 관리, LLM/Tool Use, 임베딩·Vector DB·정량 평가,
그리고 4주차 RAG 데이터 파이프라인)을 직접 손으로 복습하기 위해 만든 개인 미니 프로젝트입니다.
강의에서 배운 개념을 하나씩 실제로 구현해보고, "느낌이 아니라 숫자로 증명한다"는 원칙을
골든셋 기반 정량 평가로 직접 적용했습니다. 4주차 확장에서는 비대칭 임베딩(e5 query/passage 프리픽스),
코사인 거리공간 명시, 주차별 메타데이터 필터 검색, 유니코드 전처리, 청킹 전략 비교(block/merged/tokens)를
추가했습니다.

터미널에 질문을 입력하면 강의 스크립트 중 관련된 구간을 순위대로 찾아 보여주는 CLI입니다.
**답을 생성하지 않고, 찾아주기만 하는 검색기**입니다 — RAG(답변 생성), FastAPI 서빙, 배포는
아직 배우지 않은 범위라 의도적으로 제외했습니다.

```
질문> 벡터 인덱싱 알고리즘 두 가지는?
상위 5개 결과:
  [23차시_...HNSW vs IVF 알고리즘] (00:12~01:04) score=0.842
    실무에서 가장 많이 쓰이는 두 가지 벡터 인덱싱 알고리즘은 HNSW와 IVF입니다 ...
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

```
Presentation (cli/)  ──→  Logic (core/)  ──→  Data (data/)
   질문 입출력               청킹/스코어링         ChromaDB, BM25, 파일 I/O
                             /alpha 튜닝
```

| 레이어 | 디렉토리 | 책임 |
|---|---|---|
| Presentation | `cli/main.py` | 질문 입력, 결과(파일명·타임스탬프·스니펫) 출력 |
| Logic | `core/` | 청킹(`chunking.py`), 색인 조율(`indexing.py`), 하이브리드 스코어링(`scoring.py`), 골든셋 평가(`evaluation.py`) |
| Data | `data/` | ChromaDB(`vector_store.py`), BM25(`bm25_index.py`), 임베딩 모델 로딩(`embedding_model.py`), 스크립트/골든셋 파일 로딩 |

`SearchQuery`/`SearchResult`는 DB에 영속화하지 않는 비영속(ephemeral) 엔티티로 설계했습니다 —
검색은 상태를 남길 필요 없는 1회성 조회이기 때문입니다. 자세한 온톨로지와 결정 근거는
[`.harness/ouroboros/seeds/seed-v1.yaml`](.harness/ouroboros/seeds/seed-v1.yaml)에 있습니다.

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

## 스펙 우선 개발 (Ouroboros)

Claude Code로 코드를 작성하되, 곧바로 구현에 들어가지 않고
[`studioKjm/ai-harness-template`](https://github.com/studioKjm/ai-harness-template)을 적용해
스펙을 먼저 확정하는 방식으로 진행했습니다:

```
/interview → /seed → /trd → /decompose → /run → /evaluate
 (요구사항 인터뷰)  (불변 스펙)  (기술설계)  (원자적 태스크 분해)  (구현)   (3단계 검증)
```

- `.harness/ouroboros/seeds/seed-v1.yaml` — 목표·제약·8개 Acceptance Criteria·온톨로지를 담은 불변 스펙
- `docs/TRD.md` — 레이어별 기술 설계
- `.harness/ouroboros/decompositions/` — AC를 원자적 태스크로 분해한 실행 순서
- `.harness/ouroboros/evaluations/eval-v1-2026-07-09.yaml` — 기계적/의미적 검증 결과 (8/8 AC 충족, 테스트 38/38 통과)
- 커밋 전 게이트(`.harness/gates/`) — 시크릿 유출, 레이어 경계 위반, 3-tier 분리 위반을 자동 차단

구현·평가 과정에서 하네스 템플릿 자체의 버그도 몇 개 발견해 고쳤습니다: 게이트 스크립트들이
`.venv/`를 제외하지 않아 발생한 오탐, `check-layers.sh`가 `cli/`·`data/` 디렉토리를 레이어로
인식하지 못한 문제, 경계 규칙의 정규식이 `cli.`를 `client`와 잘못 매칭한 문제 등입니다. 또한
ChromaDB의 Rust 바인딩이 한글 경로에 저장된 HNSW 인덱스를 재오픈하지 못하는 버그를 발견해
persist 경로를 ASCII 경로(`~/.cache/script-search/chroma`)로 우회했습니다.

## 실행 방법

```bash
uv sync                                  # 의존성 설치
uv run pytest -q                         # 테스트 (46개)
uv run python -m cli.main                # 검색 CLI 실행
uv run python scripts/run_evaluation.py  # 재색인 + alpha 스윕/검증
uv run python scripts/compare_chunking.py  # 청킹 전략 3종 비교 실험
```

CLI는 상위 폴더의 `../강의 스크립트 크롤링/{1,2,3,4}주차_스크립트/`에서 텍스트 파일을 찾습니다.
자신의 텍스트 코퍼스로 써보려면 `cli/main.py`의 `WEEK_DIRS`만 원하는 디렉토리로 바꾸면 됩니다.

특정 주차만 검색하려면 질문 앞에 `/week N`을 붙입니다 (4주차 메타데이터 필터 실습):

```
질문> /week 4 벡터 인덱싱 알고리즘 두 가지는?
```

## 앞으로 계획

이 프로젝트는 부트캠프 진행에 맞춰 계속 보강할 예정입니다:

- [x] 4주차 강의 스크립트를 색인에 추가(upsert)하고 alpha 재검증 (완료 — alpha=0.5, 88파일/5,120청크)
- [x] 4주차 기법 적용: 비대칭 임베딩, 코사인 거리공간, 주차 메타데이터 필터, 유니코드 전처리, 청킹 전략 비교
- [ ] 4주차 골든 6문항 초안 검토·보정, 필요 시 청킹 전략(merged/tokens) 채택 여부 결정
- [ ] (범위 밖으로 남겨둔) RAG·FastAPI 서빙·배포는 해당 주차 학습 후 별도 프로젝트로 확장 검토

## 테스트

```bash
uv run pytest -q
# 46 passed
```
