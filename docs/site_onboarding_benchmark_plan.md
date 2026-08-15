---
title: "Codex 보조 신규 사이트 적용 공수 검증 계획"
type: plan
area: evaluation
status: planned
updated: 2026-08-16
tags:
  - l2c
  - docs/evaluation
  - benchmark
---

# Codex 보조 신규 사이트 적용 공수 검증 계획

## 검증 질문

공통 실행기가 준비된 L2C에서 Codex가 홈페이지 주소와 같은 수집 계약만 받았을 때,
신규 사이트를 지원하는 추가 공수가 Classic DOM 방식과 Vision 방식에서 얼마나 다른지
측정한다.

측정 결과는 다음 범위에서만 사용한다.

> Codex를 이용해 현재 L2C에 신규 사이트를 추가할 때의 증분 개발 공수

Vision 런타임 전체를 처음부터 개발한 비용과 DOM 자동화 기술 전체의 개발비는 이
실험으로 비교하지 않는다. Classic 공통 수집 실행기를 준비하는 비용은 별도 지표로
공개하고 사이트별 증분 공수와 합치지 않는다.

## 비교 대상

| 사이트 | 공식 홈페이지 | 선택 이유 |
|---|---|---|
| 인크루트 | `https://job.incruit.com/` | 메뉴와 정보 밀도가 높은 전통적인 채용 목록 UI |
| 랠릿 | `https://www.rallit.com/` | 카드·필터 중심 UI와 텍스트형 상세 공고 |

두 사이트 모두 기준 커밋의 Classic 어댑터와 Vision 프로필에 없어야 한다. 사이트
선정 단계의 공개 접근 가능 여부 확인은 허용하지만 DOM selector, 카드 위치와 상세 URL은
사전에 기록하지 않는다.

## 동일 수집 계약

Classic과 Vision은 다음 정보만 입력받는다.

```json
{
  "homepage": "사이트 공식 홈페이지",
  "development_query": "백엔드 개발자",
  "target_count": 2,
    "required_fields": [
      "company_name",
      "position",
      "main_tasks",
      "requirements",
      "url"
  ],
  "time_limit_minutes": 90
}
```

공고 URL, 회사명, 제목, 카드 순번, 좌표, selector와 검색 방법은 제공하지 않는다.
두 방식은 같은 `CollectionIntent`, `JobPosting`과 격리된 SQLite DB를 사용한다.

## 0단계: 비교 기반 준비

- [x] 대상 사이트와 무관한 합성 채용 페이지를 준비한다.
- [x] Classic에 `홈페이지 -> 검색 -> 결과 URL 수집 -> 상세 추출 -> 정제 -> 저장`을
      연결하는 공통 실행기를 만든다.
- [x] Classic 공통 실행기는 합성 페이지와 기존 지원 사이트만 사용해 검증한다.
- [x] ~~Vision은 현재 공통 물리 도구로 같은 합성 수집 계약을 통과하는지 확인한다.~~
      Vision의 기존 `CollectionResult`와 DB `JobPosting`을 공통 품질 검사기가 읽는
      계약 테스트로 대체한다. 실제 화면 실행은 두 후보 사이트의 Vision 세션에서 한다.
- [x] 두 방식의 결과를 같은 품질 검사기로 판정한다.
- [x] Classic 공통 기반 준비시간과 변경 줄 수를 별도 기록한다.
- [x] 제품 코드에 `incruit`, `rallit`, `인크루트`, `랠릿` 전용 구현이 없는지 확인한다.
- [x] 작업 트리를 정리하고 비교 기준 커밋 SHA를 확정한다.

이 단계가 끝난 커밋을 `T0`로 사용한다. `T0` 이전 비용은 사이트별 적용시간에 포함하지
않지만 결과 보고서에서 숨기지 않는다.

## Codex 실행 조건

| 조건 | 기준 |
|---|---|
| 모델 | 네 작업에서 같은 Codex 모델과 추론 설정 사용 |
| 시작 코드 | 모두 같은 `T0` 커밋 |
| 컨텍스트 | 방식·사이트마다 새 Codex 작업 사용 |
| 시간 제한 | 작업당 90분 |
| 사용자 입력 | 최초 지시와 권한 승인만 허용 |
| 추가 힌트 | 제공 시 문장과 횟수를 개입 기록에 남김 |
| 정보 격리 | 다른 방식의 diff, 로그, selector와 프로필 열람 금지 |
| 완료 조건 | 미사용 검색어 검증과 품질 판정 통과 |

다음 네 개의 독립 worktree와 브랜치를 사용한다.

| 작업 | 브랜치 |
|---|---|
| 인크루트 Classic | `codex/benchmark-classic-incruit` |
| 인크루트 Vision | `codex/benchmark-vision-incruit` |
| 랠릿 Classic | `codex/benchmark-classic-rallit` |
| 랠릿 Vision | `codex/benchmark-vision-rallit` |

인크루트는 Classic부터 실행하고 랠릿은 Vision부터 실행한다. 사이트별 첫 구현에서 얻은
정보가 같은 사이트의 다른 방식에 전달되지 않도록 새 작업과 worktree를 유지한다.

## 방식별 허용 범위

### Classic

- Playwright와 DOM을 사용한다.
- role, label, text locator를 우선하며 CSS는 필요한 경우에만 사용한다.
- 사이트별 검색·목록·상세 어댑터를 작성할 수 있다.
- Playwright의 자동 대기와 재시도를 사용한다.
- OCR, 스크린샷 인식, 고정 공고 URL과 공고 ID를 사용하지 않는다.
- 고정 `sleep`과 결과 카드 순번에만 의존하는 구현은 허용하지 않는다.

### Vision

- 기존 화면 캡처, OCR, 물리 입력과 자율탐색을 사용한다.
- 사이트 프로필, 공식 주소와 일반적인 사이트 이용 지침을 작성할 수 있다.
- DOM, Playwright selector와 사이트별 Python 행동 코드를 사용하지 않는다.
- 고정 공고 URL, 공고 ID와 절대좌표를 사용하지 않는다.
- 공통 런타임 수정은 허용하지만 변경 줄 수와 이유를 별도 기록한다.

## Codex 공통 지시문

사이트명과 방식별 허용 범위만 바꾸고 다음 본문은 동일하게 사용한다.

```text
공식 홈페이지에서 시작해 주어진 검색어로 채용공고를 검색하고 관련 공고 2건의
회사명, 제목, 주요 업무, 자격요건과 출처 URL을 격리 DB에 저장하십시오.

특정 공고 URL, 회사명, 제목, 카드 순번을 하드코딩하지 마십시오. 허용된 방식 안에서
구현하고 실제 실행으로 검증하십시오. 개발 검색어 성공 후 미사용 검색어 검증과 품질
검사를 모두 통과하면 작업을 종료하십시오. 제한시간은 90분입니다.
```

실행 전에 이 문구를 별도 프롬프트 파일로 고정하고 해시를 증거 manifest에 기록한다.

## 시간 측정

| 시점 | 정의 |
|---|---|
| 시작 | 최초 Codex 지시가 전송된 시각 |
| 최초 성공 | 개발 검색어로 유효 공고 2건이 처음 저장된 시각 |
| 최종 성공 | 미사용 검색어와 품질 검사가 모두 통과한 시각 |
| 실패 | 90분 안에 최종 성공하지 못한 상태 |

Codex 응답 대기, 사이트 확인, 코드 작성, 테스트, 브라우저 실행과 오류 수정 시간을 모두
포함한다. 사용자가 작업을 중단하거나 다른 업무를 수행한 세션은 무효 처리하고 새
worktree에서 다시 시작한다.

## 미사용 검색어 검증

개발에는 `백엔드 개발자`만 사용한다. 구현 완료 후 다음 검색어를 순서대로 한 번씩
실행한다.

1. `프론트엔드 개발자`
2. `데이터 엔지니어`
3. `안드로이드 개발자`

각 실행은 서로 다른 공고 2건을 저장해야 한다. 실행 전에 해당 사이트에 관련 결과가
2건 이상 보이는지만 확인한다. 결과 부족 시 실패로 계산하지 않고 미리 정한 예비 검색어
`QA 엔지니어`, `DevOps 엔지니어` 순서로 대체하고 변경 이유를 manifest에 기록한다.

## 품질 판정

방식 이름을 숨긴 수집 결과를 같은 기준으로 검토한다.

- [ ] 검색어가 공고 제목 또는 주요 업무의 중심 직무와 일치한다.
- [ ] 회사명과 공고 제목이 상세 페이지와 일치한다.
- [ ] 주요 업무와 자격요건에 상세 본문의 근거가 있다.
- [ ] 출처 URL이 허용 도메인에 속하고 서로 중복되지 않는다.
- [ ] 목표 개수와 `JobPosting` 스키마를 충족한다.
- [ ] 특정 공고 URL, 제목, ID와 카드 순번 하드코딩이 없다.

구조 검사는 자동화하고 직무 관련성과 상세 근거는 방식 정보를 가린 결과로 한 번
확인한다. 최종 검토 결과와 근거 텍스트를 evidence JSON에 남긴다.

## 측정 지표

| 지표 | 단위 | 의미 |
|---|---:|---|
| `foundation_preparation_sec` | 초 | Classic 공통 기반 준비시간 |
| `prompt_to_first_success_sec` | 초 | 최초 수집까지 걸린 시간 |
| `prompt_to_acceptance_sec` | 초 | 최종 품질 통과까지 걸린 시간 |
| `human_intervention_count` | 회 | 권한 승인 외 사용자 입력 횟수 |
| `fix_iteration_count` | 회 | 실패 후 코드 또는 설정 수정 횟수 |
| `site_specific_changed_loc` | 줄 | 사이트 전용 제품 코드 변경량 |
| `common_runtime_changed_loc` | 줄 | 공통 런타임 변경량 |
| `modified_file_count` | 개 | 변경된 제품 파일 수 |
| `locator_count` | 개 | Classic의 사이트 결합 지점 수 |
| `profile_line_count` | 줄 | Vision 사이트 프로필 규모 |
| `acceptance_success_count` | 회 | 통과한 미사용 검색어 수 |
| `runtime_sec` | 초 | 최종 검증 실행시간 |
| `llm_tokens` | 토큰 | Vision 실행 모델 사용량 |
| `llm_cost` | USD | 등록 가격표 기준 Vision 실행비용 |

시간과 코드량을 하나의 임의 점수로 합치지 않는다. 개발 공수, 결과 품질과 반복 실행
비용을 분리해서 제시한다.

## 증거 보존

```text
docs/evidence/site_onboarding/
  baseline_manifest.json
  prompt_contract.md
  classic_incruit_session.json
  classic_incruit.patch
  classic_incruit_acceptance.json
  vision_incruit_session.json
  vision_incruit.patch
  vision_incruit_acceptance.json
  classic_rallit_session.json
  classic_rallit.patch
  classic_rallit_acceptance.json
  vision_rallit_session.json
  vision_rallit.patch
  vision_rallit_acceptance.json
  comparison_report.json
```

각 session에는 기준 SHA, 시작·종료 시각, Codex 작업 식별자, 모델 설정, 사용자 개입,
실패 반복과 결과 커밋 SHA를 기록한다. 구조화 실행 summary는 [[e2e_observability]]의
계약을 재사용한다. 집계는 기존 `benchmark/site_adaptation_eval.py`를 확장해 수행한다.

## 사전 판정 규칙

| 판정 | 조건 |
|---|---|
| 지지 | 두 사이트에서 양쪽 모두 품질을 통과하고 Vision의 최종 시간과 사이트 전용 코드량이 모두 적음 |
| 혼합 | 품질은 통과했지만 사이트 또는 지표에 따라 우위가 다름 |
| 기각 | Vision이 실패하거나 두 사이트에서 Classic보다 시간과 코드량이 모두 큼 |
| 비교 불가 | 외부 장애 또는 양쪽에 공통으로 검색 결과가 부족해 계약을 실행할 수 없음 |

결과 문구에는 사이트 수와 실험 조건을 함께 쓴다. 두 사이트 결과를 전체 웹사이트에
일반화하지 않는다.

## 실행 체크리스트

- [x] Classic 공통 기반 준비 및 비용 기록
- [x] 공통 계약·프롬프트·품질 검사기 고정
- [ ] `T0` 커밋과 네 개 worktree 생성
- [ ] 인크루트 Classic 실행
- [ ] 인크루트 Vision 실행
- [ ] 랠릿 Vision 실행
- [ ] 랠릿 Classic 실행
- [ ] 방식 정보를 가린 품질 검토
- [ ] 비교 보고서 생성
- [ ] [[design_decisions]]과 루트 README에 측정된 범위만 반영

## 계획 변경 기록

실행 중 계약이나 판정 기준을 바꾸면 기존 문장을 삭제하지 않고 취소선으로 남긴 뒤,
바로 아래에 변경 날짜와 이유를 기록한다. 결과를 본 뒤 유리한 기준으로 바꾸지 않는다.

- 2026-08-16: 최초 초안의 필드명 `responsibilities`, `source_url`을 각각 공통
  `JobPosting` 계약의 `main_tasks`, `url`로 수정했다. ~~별도 벤치마크 필드명을
  사용한다.~~ 두 실행 방식과 DB가 이미 공유하는 정규 필드명을 그대로 사용한다.
- 2026-08-16: 로컬 합성 사이트는 HTTP이고 Vision `SiteProfile`은 공식 HTTPS 주소를
  요구한다. 합성 프로필·인증서 예외를 제품 기준선에 추가하면 비교 대상과 무관한
  변경이 생긴다. 따라서 합성 Vision E2E는 공통 결과·DB 입력 계약 테스트로 바꾸고,
  실제 Vision 실행은 인크루트와 랠릿에서 검증한다.
