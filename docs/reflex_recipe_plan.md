---
title: "Reflex Recipe 구현 기준"
type: plan
area: reflex
status: active
updated: 2026-07-29
tags:
  - l2c
  - docs/reflex
---

# L2C 반사 레시피(Reflex Recipe) 구현 기준

> 네이밍 기준은 `docs/naming_conventions.md`를 따른다. 새 코드의 active recipe는 `state_key`나 코드가 분류한 화면 역할이 아니라 전체 안정 경로의 `recipe_key`, `site`, `task_category`, URL 범위와 단계별 화면 서명을 기준으로 동작한다.

## 목표

- 새 화면은 Vision/ReAct 자율 탐색이 판단한다.
- 반복되는 안정 UI 조작만 Reflex가 reasoning 없이 재생한다.
- DOM selector, Playwright selector, 절대좌표는 Vision/Realtme 경로에 저장하지 않는다.
- 공고 카드 선택, 본문 수집 완료 판단, 예외 복구처럼 실행마다 달라지는 판단은 LLM 또는 카드 큐 정책에 남긴다.

## 현재 저장 흐름

```text
자율 탐색 action
-> recorded_step 저장
-> worker_submission 저장
-> recipe_candidate를 pending_review로 저장
-> 사용자 답변 경로는 계속 진행
-> 백엔드 승격 작업자가 후보를 선점
-> Critic review/promote
-> active_recipe 저장
-> 다음 경험 기반 탐색에서 Reflex replay
```

승격 검토는 현재 답변의 전제 조건이 아니다. FastAPI 수명주기의 단일 작업자가 SQLite 대기열을 처리하므로 요청과 E2E는 Critic 완료를 기다리지 않는다. 전송 오류는 `pending_review`로 재시도하고, 재시도 한도를 넘기면 의미상 거절인 `revise`와 구분해 `review_failed`로 남긴다.

## Active Recipe 기준

활성 레시피는 한 성공 실행에서 연속해서 검증된 단계를 순서가 보존된
`stable_recipe_path` 하나로 저장한다. 첫 단계는 반드시 ROI로 다시 찾을 수 있는
클릭/입력이어야 한다. 이후에는 ROI 클릭/입력과 전환 계약이 있는 키 입력·뒤로가기·
탭 전환을 같은 경로에 포함할 수 있다.

- `site`: 사이트 식별자.
- `task_category`: 검색, 로그인, 결제, 사이트 탐색 같은 작업 분류.
- `page_role`: home, search, job_detail, popup 등 행동 당시의 설명용 화면 역할. 재생 차단 조건으로 사용하지 않는다.
- `roi_signature`: target 주변 crop pHash와 crop 비율.
- `screen_context_signature`: 키 입력·뒤로가기처럼 타깃 좌표가 없는 행동 직전의 전체 화면 pHash와 캡처 크기.
- `target.center_ratio` 또는 `target.bbox_ratio`: 현재 OCR marker 재탐색용 비율 좌표.
- `replay_mode`: Critic이 `fixed` 또는 `parameterized`로 승인한 단계.

추론 단계, 실패 단계, 폐기한 분기, 검증되지 않은 행동은 경로의 경계다. 앞뒤에
승인된 단계가 있더라도 그 경계를 건너 하나의 경로로 합치지 않는다. 경로 키는
전체 단계의 순서와 의미를 포함하므로 `A-B-C`와 `A-D-C`는 별도 레시피다.
새 후보를 승격할 때도 단계가 겹친다는 이유로 다른 후보의 경로를 삭제하지 않는다.
동일한 전체 경로를 여러 후보가 검증한 경우 레시피 행은 합치고
`recipe_sources`에 후보별 근거를 따로 연결한다.

`state_key`, Jaccard anchor 유사도와 코드 기반 화면 역할 분류는 active replay의
기본 조회 기준이 아니다. 전체 화면 pHash는 타깃 좌표가 없는 단계의 행동 직전
문맥을 확인할 때만 `screen_context_signature`로 사용한다.

전체 경로 키는 `path4#` 버전을 사용한다. 이전 원자 단계 키와 실행 의미가 달라
기존 활성 행은 마이그레이션 때 폐기하지만 `recipe_candidates`는 보존한다.
기존 승인 후보를 다시 쓰려면 `benchmark/rebuild_active_recipes.py --apply`로 현재
승격 정책을 재적용한다.

## Replay 순서

1. `perception_node`가 OCR/SoM marker와 현재 스크린샷을 만든다.
2. `RecipeStore.get_site_recipes(site, task_category)`로 전체 경로 후보를 가져온다.
3. 각 경로의 첫 단계 URL 범위와 ROI pHash를 현재 화면과 비교한다.
4. 화면 역할 이름은 관측 로그에 남기되 재생 허용 여부에는 사용하지 않는다.
5. ROI가 맞으면 저장된 target 비율에 가까운 현재 OCR marker를 찾는다.
6. 첫 단계가 통과한 경로 하나를 선택하고 `active_reflex_recipe`에 경로 키와 다음 단계 번호를 저장한다.
7. 현재 캡처에서 검증한 물리 행동 하나만 `ActionRequest`로 실행한다.
8. 다음 캡처에서 직전 전환 계약과 현재 경로 단계의 화면 조건을 검증한다.
9. 조건이 맞으면 후보를 다시 고르지 않고 같은 경로의 다음 행동을 실행한다. ROI 대상 행동은 매 단계 ROI와 현재 marker를 다시 검사하고, 좌표 없는 행동은 자율탐색 때 저장한 `screen_context_signature`와 현재 화면 pHash를 대조한다.
10. 마지막 단계가 성공하면 활성 경로 상태를 지운다. 중간 전환이나 다음 단계 검증이 실패하면 경로 전체를 폐기하고 reasoning으로 폴백하며 같은 run 안에서 해당 `recipe_key`를 차단한다.

## 승격 정책

Critic은 의미 판단을 담당한다. 코드는 후보를 포장하고 필수 구조만 검사한다.

- `fixed`: 같은 UI 조작이 여러 실행에서 그대로 유효한 단계.
- `parameterized`: UI 조작은 같고 입력 슬롯만 바뀌는 단계.
- `reasoning`: 현재 화면, 현재 결과, 방문 여부, 목표 개수에 따라 판단해야 하는 단계.

공고 제목 클릭은 기본적으로 `reasoning`이다. 검색 열기, 검색어 입력, 검색 제출처럼 반복 증거가 있는 컨트롤만 active recipe 후보가 된다. 상세 펼치기 자동 클릭은 현재 사이트 `page_guidance.reveal_controls`에 선언된 OCR 라벨과 정확히 일치할 때만 허용한다.

LLM이 한 번에 여러 행동을 생성하는 것은 허용하지 않는다. 전체 경로를 저장하는
것과 여러 행동을 검증 없이 한꺼번에 실행하는 것은 다르다. `ActionRequest`는
항상 현재 캡처로 검증한 도구 호출 하나만 담고, 경로의 다음 단계는 행동 후 새
캡처와 전환 판정을 거쳐 실행한다.

## 실패 처리

- ROI pHash가 맞지 않으면 즉시 reasoning fallback.
- 좌표 없는 행동의 `screen_context_signature`가 없거나 현재 화면 pHash와 다르면 즉시 reasoning fallback.
- target marker 비율 매칭이 실패하면 즉시 reasoning fallback.
- Reflex action 뒤 전환 계약이 확인되지 않으면 해당 `recipe_key`는 같은 run 안에서 재시도하지 않는다.

이 기준은 Reflex가 자율 탐색 전체를 대체하지 않고, 고정 가능한 부모 경로만 빠르게 재생하도록 제한한다.
