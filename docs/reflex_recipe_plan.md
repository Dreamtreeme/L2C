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
-> Critic이 후보 단계 유지/제거 판정
-> active_recipe 저장
-> 다음 경험 기반 탐색에서 Reflex replay
```

승격 검토는 현재 답변의 전제 조건이 아니다. FastAPI 수명주기의 단일 작업자가 SQLite 대기열을 처리하므로 요청과 E2E는 Critic 완료를 기다리지 않는다. 전송 오류는 `pending_review`로 재시도하고, 재시도 한도를 넘기면 의미상 거절인 `revise`와 구분해 `review_failed`로 남긴다.

## Active Recipe 기준

활성 레시피는 한 성공 실행을 `시작 상태 + 상태 전이 목록 + 완료 상태`인
`recipe_path` 하나로 저장한다. 각 `recipe_transition`은 `before + actions +
after`를 가지며, 첫 전이의 첫 행동은 반드시 ROI로 다시 찾을 수 있는
클릭/입력이어야 한다. 이후에는 ROI 클릭/입력과 화면 문맥이 있는 키 입력·
뒤로가기·탭 전환을 같은 경로에 포함할 수 있다.

- `site`: 사이트 식별자.
- `task_category`: 검색, 로그인, 결제, 사이트 탐색 같은 작업 분류.
- `page_role`: home, search, job_detail, popup 등 행동 당시의 설명용 화면 역할. 재생 차단 조건으로 사용하지 않는다.
- `roi_signature`: target 주변 crop pHash와 crop 비율.
- `screen_context_signature`: 키 입력·뒤로가기처럼 타깃 좌표가 없는 행동 직전의 전체 화면 pHash와 캡처 크기.
- `target.center_ratio` 또는 `target.bbox_ratio`: 현재 OCR marker 재탐색용 비율 좌표.
- `replay_mode`: 자율탐색이 `fixed` 또는 `parameterized`로 제안하고 Critic이 유지한 단계.
- `before` / `after`: 행동 묶음 전후의 검증 가능한 화면 체크포인트.
- `actions`: 중간 화면 관찰 없이 실행해도 되는 행동 묶음. 현재는 단일 행동 또는 `검색어 입력 + Enter`만 허용한다.

Critic이 중간 행동을 제거했을 때 삭제 전후 체크포인트가 같다는 근거가 없으면
그 지점에서 경로를 끝낸다. 뒤쪽 행동을 별도 레시피로 자동 생성하지 않는다.
경로 키는
전체 단계의 순서와 의미를 포함하므로 `A-B-C`와 `A-D-C`는 별도 레시피다.
새 후보를 승격할 때도 단계가 겹친다는 이유로 다른 후보의 경로를 삭제하지 않는다.
동일한 전체 경로를 여러 후보가 검증한 경우 레시피 행은 합치고
`recipe_sources`에 후보별 근거를 따로 연결한다.

`state_key`, Jaccard anchor 유사도와 코드 기반 화면 역할 분류는 active replay의
기본 조회 기준이 아니다. 전체 화면 pHash는 타깃 좌표가 없는 단계의 행동 직전
문맥을 확인할 때만 `screen_context_signature`로 사용한다.

전체 경로 키는 `path6#` 버전을 사용한다. 단일 행동 목록을 저장하던 이전 활성
행은 스키마 전환 때 폐기하지만 `recipe_candidates`는 보존한다.
기존 승인 후보를 다시 쓰려면 `benchmark/rebuild_active_recipes.py --apply`로 현재
승격 정책을 재적용한다.

## Replay 순서

1. `perception_node`가 OCR/SoM marker와 현재 스크린샷을 만든다.
2. `RecipeStore.get_site_recipes(site, task_category)`로 전체 경로 후보를 가져온다.
3. 각 경로의 첫 전이 URL 범위와 ROI pHash를 현재 화면과 비교한다.
4. 화면 역할 이름은 관측 로그에 남기되 재생 허용 여부에는 사용하지 않는다.
5. ROI가 맞으면 저장된 target 비율에 가까운 현재 OCR marker를 찾는다.
6. 첫 전이가 통과한 경로 하나를 선택하고 `active_reflex_recipe`에 현재 전이 번호를 저장한다. 이 시점에는 번호를 증가시키지 않는다.
7. 현재 캡처에서 검증한 단일 행동 또는 `입력 + Enter` 행동 묶음을 `ActionRequest`로 실행한다.
8. OpenCV 연속 프레임 비교로 화면 변화 시작과 렌더링 안정화를 기다린 뒤 OCR을 실행한다.
9. 저장된 `after`의 ROI 앵커 또는 화면 문맥이 현재 화면과 일치할 때만 다음 전이 번호로 이동한다.
10. 다음 전이의 ROI 대상은 현재 marker로 다시 찾고, 타깃 없는 행동은 저장한 `screen_context_signature`를 확인한다.
11. 마지막 전이의 도착 상태가 검증되면 활성 경로 상태를 지운다. 검증이 실패하면 경로 전체를 폐기하고 reasoning으로 폴백하며 같은 run 안에서 해당 `recipe_key`를 차단한다.

## 승격 정책

자율탐색은 행동을 선택할 때 재사용 방식을 함께 제안한다. Critic은 해당 제안을
수정하지 않고 잘못된 대상, 무효 행동, 폐기된 복구 경로와 불안정한 단계만
제거한다. 코드는 화면 증거와 필수 구조를 독립적으로 검사한다.

- `fixed`: 같은 UI 조작이 여러 실행에서 그대로 유효한 단계.
- `parameterized`: UI 조작은 같고 입력 슬롯만 바뀌는 단계.
- `reasoning`: 현재 화면, 현재 결과, 방문 여부, 목표 개수에 따라 판단해야 하는 단계.

공고 제목 클릭은 기본적으로 `reasoning`이다. 검색 열기, 검색어 입력, 검색 제출처럼 반복 증거가 있는 컨트롤만 active recipe 후보가 된다. 상세 펼치기 자동 클릭은 현재 사이트 `page_guidance.reveal_controls`에 선언된 OCR 라벨과 정확히 일치할 때만 허용한다.

LLM이 한 번에 여러 행동을 생성하는 것은 허용하지 않는다. 다만 자율탐색 기록에서
`type_in_marker` 직후 `Enter`가 하나의 화면 전환을 만든 것이 확인되면 활성
레시피 승격 시 두 행동을 같은 전이로 묶는다. 서로 다른 타깃 클릭처럼 중간 화면을
다시 봐야 하는 행동은 묶지 않는다.

## 실패 처리

- ROI pHash가 맞지 않으면 즉시 reasoning fallback.
- 좌표 없는 행동의 `screen_context_signature`가 없거나 현재 화면 pHash와 다르면 즉시 reasoning fallback.
- target marker 비율 매칭이 실패하면 즉시 reasoning fallback.
- OpenCV 프레임 비교에서 화면 변화가 없거나 저장된 도착 체크포인트가 맞지 않으면 해당 `recipe_key`는 같은 run 안에서 재시도하지 않는다.

이 기준은 Reflex가 자율 탐색 전체를 대체하지 않고, 고정 가능한 부모 경로만 빠르게 재생하도록 제한한다.
