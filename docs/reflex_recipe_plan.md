---
title: "Reflex Recipe 구현 기준"
type: plan
area: reflex
status: active
updated: 2026-07-23
tags:
  - l2c
  - docs/reflex
---

# L2C 반사 레시피(Reflex Recipe) 구현 기준

> 네이밍 기준은 `docs/naming_conventions.md`를 따른다. 새 코드의 active recipe는 `state_key`가 아니라 `recipe_key`, `site`, `task_category`, `page_role`, `roi_signature` 기준으로 동작한다.

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

활성 레시피는 다음 조건을 만족하는 클릭/입력 단계만 저장한다.

- `site`: 사이트 식별자.
- `task_category`: 검색, 로그인, 결제, 사이트 탐색 같은 작업 분류.
- `page_role`: home, search, job_detail, popup 등 행동이 기록된 화면 역할.
- `roi_signature`: target 주변 crop pHash와 crop 비율.
- `target.center_ratio` 또는 `target.bbox_ratio`: 현재 OCR marker 재탐색용 비율 좌표.
- `replay_mode`: Critic이 `fixed` 또는 `parameterized`로 승인한 단계.

`state_key`, 전체 화면 pHash, Jaccard anchor 유사도는 active replay의 기본 조회 기준이 아니다.

## Replay 순서

1. `perception_node`가 OCR/SoM marker와 현재 스크린샷을 만든다.
2. 현재 URL/OCR 텍스트에서 `current_page_role`을 보수적으로 분류한다.
3. `RecipeStore.get_site_recipes(site, task_category)`로 후보를 가져온다.
4. 각 후보 step에 대해 `page_role`이 현재 화면과 맞는지 확인한다.
5. 저장된 `roi_signature.crop_rect_ratio`로 현재 스크린샷의 같은 ROI를 crop하고 pHash 거리를 검사한다.
6. ROI가 맞으면 저장된 target 비율에 가까운 현재 OCR marker를 찾는다.
7. 통과하면 `click_marker` 또는 `type_in_marker` 도구 호출을 만들어 `execution_node`가 기존 실행 경로로 처리한다.
8. 전환 계약이 `unknown`이면 reasoning으로 폴백하고 같은 run 안에서 해당 `recipe_key`를 차단한다.

## 승격 정책

Critic은 의미 판단을 담당한다. 코드는 후보를 포장하고 필수 구조만 검사한다.

- `fixed`: 같은 UI 조작이 여러 실행에서 그대로 유효한 단계.
- `parameterized`: UI 조작은 같고 입력 슬롯만 바뀌는 단계.
- `reasoning`: 현재 화면, 현재 결과, 방문 여부, 목표 개수에 따라 판단해야 하는 단계.

공고 제목 클릭은 기본적으로 `reasoning`이다. 검색 열기, 검색어 입력, 검색 제출처럼 반복 증거가 있는 컨트롤만 active recipe 후보가 된다. 상세 펼치기 자동 클릭은 현재 사이트 `page_guidance.reveal_controls`에 선언된 OCR 라벨과 정확히 일치할 때만 허용한다.

## 실패 처리

- `page_role`이 없거나 현재 화면과 다르면 즉시 reasoning fallback.
- ROI pHash가 맞지 않으면 즉시 reasoning fallback.
- target marker 비율 매칭이 실패하면 즉시 reasoning fallback.
- Reflex action 뒤 전환 계약이 확인되지 않으면 해당 `recipe_key`는 같은 run 안에서 재시도하지 않는다.

이 기준은 Reflex가 자율 탐색 전체를 대체하지 않고, 고정 가능한 부모 경로만 빠르게 재생하도록 제한한다.
