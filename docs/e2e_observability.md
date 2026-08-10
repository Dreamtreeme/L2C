---
title: "E2E 관측 환경"
type: guide
area: observability
status: active
updated: 2026-08-11
tags:
  - l2c
  - docs/observability
---

# E2E 관측 환경

L2C는 로컬 `*.summary.json`을 원본 실행 기록으로 유지하고, LangSmith를 trace 탐색과 추세 시각화에 사용합니다. 별도의 LLM 평가자는 두지 않습니다. 답변의 의미 적합성은 사용자 테스트로 검증하고, 자동 지표는 실행 성공 여부와 수집 결과처럼 코드로 판정 가능한 값만 기록합니다.

## 설정

`.env`에 다음 값을 설정합니다.

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<발급받은 키>
LANGSMITH_PROJECT=l2c-e2e
LANGSMITH_HIDE_INPUTS=true
LANGSMITH_HIDE_OUTPUTS=true
L2C_LANGSMITH_E2E_FEEDBACK=1
L2C_E2E_EXPERIMENT=reflex-regression
VISION_RECIPE_VERSION=roi-v1
```

입출력 숨김은 채용공고 OCR 원문이나 사용자 질문이 외부 trace에 노출되지 않게 하는 기본 권장값입니다. 추적이나 feedback 전송이 실패해도 실제 수집과 로컬 요약 저장은 계속됩니다.

## 모델 가격표

기본 가격표는 `config/model_pricing.json`입니다. 현재 사용하는 Gemini 모델의
Standard 유료 단가만 기록하며, 출력 단가는 공개 답변과 내부 사고 토큰을 모두
포함합니다. 공식 가격표를 다시 확인한 날짜와 원문 URL도 파일에 함께 보존합니다.

모델을 바꾸거나 공급사가 단가를 변경하면 가격표를 먼저 갱신해야 합니다. 실행에
사용된 모델 ID가 없으면 비용을 임의로 추정하지 않고 `unpriced_models`에 남깁니다.
별도 가격표를 비교할 때만 `LLM_PRICING_FILE`로 경로를 재정의합니다.

## 실행

비교할 실행에는 같은 `scenario-id`와 `experiment-name`을 사용합니다. 저장된 경험 없이 화면마다 판단하는 실행은 `autonomous`, 검증된 레시피를 우선 활용하고 불일치 구간만 다시 판단하는 실행은 `experience_guided`로 구분합니다.

```powershell
python -m benchmark.run_realtime_e2e `
  --site wanted `
  --search-keyword "iOS 개발자" `
  --original-query "ios 개발자 공고 2개" `
  --target-count 2 `
  --count-mode explicit `
  --scenario-id wanted-ios-2 `
  --experiment-name reflex-regression `
  --execution-mode experience_guided `
  --recipe-version roi-v1 `
  --log logs/e2e_wanted_ios2.log
```

실행 결과는 로그 옆의 `.summary.json`에도 남습니다. `git_commit`, 변경 파일 유무, 설정 fingerprint, 모델, 레시피 버전을 저장하므로 성능 변화가 코드와 설정 중 어디에서 발생했는지 비교할 수 있습니다.

`experience_guided_preconditions.performance_comparable=true`인 실행만 경험 기반
탐색 성능 표본으로 사용합니다. `active_roi_recipe_missing`처럼 사전조건이 실패한
실행은 수집에 성공해도 자율 탐색 폴백으로 분류합니다. Reflex와 작업 목록 재생이
모두 0회인 기록도 같은 기준으로 비교 대상에서 제외합니다.

성능 비교의 단일 원본은 `.summary.json`입니다. 텍스트 로그는 화면·행동 원인을 조사할 때만 사용하며, 정규식으로 OCR·추론·Reflex 횟수나 시간을 다시 계산하지 않습니다.

```powershell
python -m benchmark.profile_reflex_trace logs/e2e_wanted_ios2.summary.json
```

기존 실행 기록 전체를 재사용할 때는 먼저 출처 수준과 비교 단위를 감사합니다.

```powershell
python -m benchmark.audit_e2e_history logs `
  --json-output logs/e2e_history_audit.json `
  --markdown-output benchmark/e2e_history_audit.md `
  --minimum-group-size 2
```

감사 도구는 `git_commit`, 설정 fingerprint, 시나리오, 사이트, 실행 모드,
질의와 목표 수가 같은 기록만 한 그룹으로 묶습니다. `git_dirty=true` 실행은
코드 상태를 완전히 재현할 수 없으므로 최종 성능 기준이 아니라 개발 중 회귀와
트러블슈팅 증거로 분리합니다. 표본이 작은 그룹에는 p95를 붙이지 않고 성공
건수와 실행시간 최소·중앙·최대값을 그대로 표시합니다.
현재 분류 결과는 [`benchmark/e2e_history_audit.md`](../benchmark/e2e_history_audit.md)에
보관합니다.

### 자율 탐색과 경험 기반 탐색 회귀

`benchmark.run_regression_matrix`는 격리 DB에서 같은 작업의 자율 탐색과 경험 기반 탐색을 순서대로 실행합니다.

```powershell
python -m benchmark.run_regression_matrix `
  --scenario wanted-ios-autonomous `
  --scenario wanted-ios-experience-guided
```

자율 탐색 프로세스에서는 자동승격을 끕니다. 수집이 끝나면 부모 프로세스가 실제 서비스와 같은 `RecipePromotionWorker`를 사용해 해당 후보만 재시도하며 검토합니다. 프로세스 종료, 저장 품질, 목표 수, 실제 레시피 승격을 모두 통과한 경우에만 짝이 되는 경험 기반 탐색을 실행합니다. 하나라도 실패하면 `paired_autonomous_promotion_failed`로 건너뛰므로, 승격되지 않은 실행을 경험 기반 탐색 성능으로 잘못 집계하지 않습니다.

수집과 승격 비용은 다음 필드로 분리합니다.

| 범위 | 시간·토큰·비용 필드 |
|---|---|
| 수집 실행 | `execution_time_sec`, `total_tokens`, `estimated_cost` |
| Critic 승격 | `promotion_time_sec`, `promotion_total_tokens`, `promotion_estimated_cost` |
| 자율 탐색과 승격 전체 | `workflow_total_tokens`, `workflow_estimated_cost` |

Critic 호출은 별도 실행 문맥과 LangSmith trace를 사용합니다. 재시도가 발생하면 모든 시도의 시간, 토큰, 비용을 합산하며, 단가를 알 수 없는 모델은 기존 원칙대로 비용을 임의 추정하지 않습니다.

### 자연어 제품 회귀

`benchmark.run_product_chat_matrix`는 실제 `/api/chat` SSE 경로를 같은 FastAPI
수명에서 실행합니다. 원본 DB는 수정하지 않고 SQLite `backup`으로 만든 테스트
DB를 사용합니다.

```powershell
python -m benchmark.run_product_chat_matrix `
  --source-db data/jobs.db `
  --db-path logs/product_matrix.db `
  --log logs/product_matrix.log `
  --summary logs/product_matrix.summary.json
```

기본 행렬은 다음 세 계약을 검사합니다.

| 시나리오 | 자동 판정 |
|---|---|
| DB 전용 비교 | 수집 이벤트 없음, DB 행 변화 없음, 최소 인용 수와 `job_id` 무결성 |
| 범위가 없는 요청 | `waiting_input`, 수집 이벤트 없음, 구조화된 객관식 선택지 |
| 실제 사이트 수집 | 수집 시작·완료 이벤트, 완료 답변, 저장 DB에 존재하는 `job_id` 인용 |

답변 문구나 특정 회사명을 문자열로 맞추지 않습니다. 상태, 이벤트, DB 변화,
인용처럼 코드로 결정할 수 있는 제품 계약만 자동 판정하고 답변의 의미 품질은
사람이 검토합니다. 전체 SSE와 런타임 로그는 로그 파일에만 기록하며 콘솔에는
요약을 출력합니다. 화면 진행을 같이 볼 때만 `--verbose`를 사용합니다.

지휘자의 요청 이해만 빠르게 회귀할 때는 실제 12개 자연어 질문을 구조화 출력으로
평가합니다.

```powershell
python -m benchmark.profile_investigation_planner `
  --summary `
  --failures-only `
  --max-concurrency 3
```

## Trace 구조

- `l2c.e2e`: 한 E2E 실행의 root trace
- `l2c.recipe-promotion`: 후보별 Critic 검토와 재시도 trace
- `worker_prepare_screen`, `worker_graph`, `worker_review`, `job_persistence`: 수집 생명주기
- `ocr_request`: 실제 PaddleOCR worker 요청과 timeout
- LangGraph 노드와 LLM 호출: 판단 흐름과 모델 토큰
- Classic Gemini 정제 호출: `classic_extraction` LLM 사용량

각 단계에는 `stage`, `component`, 성공 여부와 실패 코드가 붙습니다. `graph:reflex`의 `action_source=reflex`와 `graph:selection`의 `action_source=job_card_queue`가 각각 Reflex와 공고 카드 큐 hit의 기준입니다. 중간 실패 후 복구된 실행은 최종 성공으로 집계하고, 실패 이력은 `recovered_failure_count`와 `internal_failure_codes`에 남깁니다.

## 작업자 실행 경로 조회

`worker_submissions`에는 행동 순번별 화면 관찰, 실행 피드백, 다음 화면 전환이 함께 저장됩니다. 다음 명령은 가장 최근 제출물을 관찰 → 행동 → 다음 관찰 순서로 출력합니다.

```powershell
python scripts/inspect_worker_trace.py
```

특정 실행이나 정확한 제출물을 조회할 수도 있습니다.

```powershell
python scripts/inspect_worker_trace.py --run-id worker-20260723192058-c0cd94e8
python scripts/inspect_worker_trace.py --submission-id worker-20260723192058-c0cd94e8 --json
```

텍스트 출력은 긴 실행 ID에서 `observation:0001`처럼 관찰 순번만 줄여 보여 줍니다. `--json` 출력은 실행 ID, 전체 관찰 ID, 스크린샷 경로를 그대로 유지하므로 실패 경로 분석이나 별도 시각화 입력으로 사용할 수 있습니다.

## 대시보드

LangSmith에서 root run 이름 `l2c.e2e`를 기준으로 다음 지표를 구성합니다.

| 목적 | feedback 또는 metadata | 표시 방식 |
|---|---|---|
| 성공률 | `e2e_success` | 평균, site/scenario별 분리 |
| 실패 구간 | `terminal_failure_stage`, `terminal_failure_code` | 범주별 실행 수 |
| 실행시간 | `execution_time_sec` | p50, p95, 시계열 |
| 토큰 소비 | `total_tokens`, `tokens_per_persisted_item` | 합계와 공고당 평균 |
| 비용 | `estimated_cost_usd`, `cost_per_persisted_item_usd` | 단가 파일이 있을 때만 |
| OCR 안정성 | `ocr_timeout_count`, `recovered_failure_count` | 합계와 성공 실행 비교 |
| 경험 기반 탐색 성과 | `reflex_hits`, `queue_replay_hits` | 경험 기반 탐색 실행의 평균 |
| 변경 영향 | `git_commit`, `git_dirty`, `config_fingerprint`, `recipe_version` | 필터와 그룹 |

비교할 때는 `scenario_id`, `site`, `target_count`, `execution_mode`가 같은 실행만 묶습니다. 서로 다른 검색 난이도나 수집 개수를 한 그래프에 섞으면 실행시간과 토큰 변화의 원인을 판단할 수 없습니다.
