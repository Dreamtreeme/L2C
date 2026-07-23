---
title: "E2E 관측 환경"
type: guide
area: observability
status: active
updated: 2026-07-23
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

## 실행

비교할 실행에는 같은 `scenario-id`와 `experiment-name`을 사용하고, 자율탐색은 `cold`, 레시피 반복탐색은 `warm`으로 구분합니다.

```powershell
python -m benchmark.run_realtime_e2e `
  --site wanted `
  --query "ios 개발자 공고 2개" `
  --target-count 2 `
  --count-mode explicit `
  --scenario-id wanted-ios-2 `
  --experiment-name reflex-regression `
  --run-mode warm `
  --recipe-version roi-v1 `
  --log logs/e2e_wanted_ios2.log
```

실행 결과는 로그 옆의 `.summary.json`에도 남습니다. `git_commit`, 변경 파일 유무, 설정 fingerprint, 모델, 레시피 버전을 저장하므로 성능 변화가 코드와 설정 중 어디에서 발생했는지 비교할 수 있습니다.

성능 비교의 단일 원본은 `.summary.json`입니다. 텍스트 로그는 화면·행동 원인을 조사할 때만 사용하며, 정규식으로 OCR·추론·Reflex 횟수나 시간을 다시 계산하지 않습니다.

```powershell
python -m benchmark.profile_reflex_trace logs/e2e_wanted_ios2.summary.json
```

## Trace 구조

- `l2c.e2e`: 한 E2E 실행의 root trace
- `worker_prepare_screen`, `worker_graph`, `worker_review`, `job_persistence`: 수집 생명주기
- `ocr_request`: 실제 PaddleOCR worker 요청과 timeout
- LangGraph 노드와 LLM 호출: 판단 흐름과 모델 토큰
- 직접 호출한 OpenAI/Ollama 모델: 별도의 LLM child trace

각 단계에는 `stage`, `component`, 성공 여부와 실패 코드가 붙습니다. `graph:reflex`의 `action_source=reflex`와 `graph:selection`의 `action_source=card_queue`가 각각 Reflex와 카드 큐 hit의 기준입니다. 중간 실패 후 복구된 실행은 최종 성공으로 집계하고, 실패 이력은 `recovered_failure_count`와 `internal_failure_codes`에 남깁니다.

## 작업자 실행 경로 조회

`worker_submissions`에는 행동 순번별 판단 캡처, 실행 피드백, 다음 화면 전환이 함께 저장됩니다. 다음 명령은 가장 최근 제출물을 캡처 → 행동 → 다음 캡처 순서로 출력합니다.

```powershell
python scripts/inspect_worker_trace.py
```

특정 실행의 최신 검토 시도나 정확한 제출물을 조회할 수도 있습니다.

```powershell
python scripts/inspect_worker_trace.py --run-id worker-20260723192058-c0cd94e8
python scripts/inspect_worker_trace.py --submission-id worker-20260723192058-c0cd94e8:0 --json
```

텍스트 출력은 긴 실행 ID에서 `capture:0001`처럼 캡처 순번만 줄여 보여 줍니다. `--json` 출력은 실행 ID, 검토 시도, 전체 캡처 ID, 스크린샷 경로를 그대로 유지하므로 실패 경로 분석이나 별도 시각화 입력으로 사용할 수 있습니다.

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
| 반복탐색 성과 | `reflex_hits`, `queue_replay_hits` | warm 실행의 평균 |
| 변경 영향 | `git_commit`, `git_dirty`, `config_fingerprint`, `recipe_version` | 필터와 그룹 |

비교할 때는 `scenario_id`, `site`, `target_count`, `run_mode`가 같은 실행만 묶습니다. 서로 다른 검색 난이도나 수집 개수를 한 그래프에 섞으면 실행시간과 토큰 변화의 원인을 판단할 수 없습니다.
