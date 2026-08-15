---
title: "Reflex Recipe 구현 기준"
type: plan
area: reflex
status: active
updated: 2026-08-15
tags:
  - l2c
  - docs/reflex
---

# L2C 반사 레시피(Reflex Recipe) 구현 기준

> 네이밍 기준은 `docs/naming_conventions.md`를 따른다. active recipe는 `site + task_category`로 식별하고, 선택된 경로의 URL 범위와 단계별 화면 서명으로 현재 재생 가능 여부를 확인한다.

## 목표

- 새 화면은 Vision/ReAct 자율 탐색이 판단한다.
- 반복되는 안정 UI 조작만 Reflex가 reasoning 없이 재생한다.
- DOM selector, Playwright selector, 절대좌표는 Vision/Realtme 경로에 저장하지 않는다.
- 공고 카드 선택, 본문 수집 완료 판단, 예외 복구처럼 실행마다 달라지는 판단은 LLM 또는 카드 큐 정책에 남긴다.

## 현재 저장 흐름

```text
자율 탐색 ActionRequest
-> 직전 ScreenCheckpoint와 실행할 PhysicalAction 묶음을 기록
-> 묶음 실행 뒤 한 번 캡처해 도착 화면과 증거를 붙인 ExperienceTransition 완성
-> worker_submission 저장
-> recipe_candidate를 recorded로 저장
-> 사용자 답변 경로는 계속 진행
-> 자동 승격이 켜진 실행은 후보를 pending_review 대기열에 등록
-> 백엔드 승격 작업자가 후보를 선점
-> 실행 계약이 완성된 전이만 Critic에 전달
-> 시작 화면과 각 전이 후 화면을 실행 순서대로 Critic에 전달
-> Critic이 전이 묶음 유지/제거 판정
-> 남은 전이가 하나의 연속 경로일 때만 내용을 바꾸지 않고 active_recipe 저장
-> 다음 경험 기반 탐색에서 Reflex replay
```

승격 검토는 현재 답변의 전제 조건이 아니다. FastAPI 수명주기의 단일 작업자가 SQLite 대기열을 처리하므로 요청과 E2E는 Critic 완료를 기다리지 않는다. 전송 오류는 `pending_review`로 재시도하고, 재시도 한도를 넘기면 `review_failed`로 남긴다. Critic의 내용 판정은 `accept` 또는 `reject`다.

## Active Recipe 기준

활성 레시피는 한 성공 실행의 `ExperienceTransition` 목록인
`ExperiencePath` 하나로 저장한다. 각 전이는 `before + actions + after`를
가지며, 시작 화면과 완료 화면은 첫 전이의 `before`와 마지막 전이의 `after`에서
계산한다. 각 전이의 첫 행동은 ROI로 다시 찾을 수 있는 클릭/입력이다.
입력과 Enter 또는 입력과 화면에 이미 보이는 제출 클릭을 중간 관찰 없이 실행한
경우에는 두 행동이 처음부터 같은 전이로 기록된다.
입력 뒤 새 관찰이 필요했던 경우에는 입력 전이를 따로 유지한다. 입력 직후 화면이
거의 변하지 않았더라도 바로 다음 성공 전이의 시작 화면과 이어지면 해당 입력은
`preparation transition`으로 검토한다. 승격 단계에서 두 전이를 다시 합치지는 않는다.

- `site`: 사이트 식별자.
- `task_category`: 검색, 로그인, 결제, 사이트 탐색 같은 작업 분류.
- `page_role`: home, search, job_detail, popup 등 행동 당시의 설명과 기록 완전성 확인에 쓰는 화면 역할. 현재 화면과의 재생 일치 판정에는 사용하지 않는다.
- `roi_signature`: target 주변 crop pHash와 crop 비율.
- `target.center_ratio` 또는 `target.bbox_ratio`: 현재 OCR marker 재탐색용 비율 좌표.
- `replay_mode`: 실행된 도구와 입력 슬롯 계약에서 `fixed` 또는 `parameterized`로 계산된 행동.
- `before` / `after`: `ScreenCheckpoint`로 표현한 행동 묶음 전후 화면.
- `actions`: 중간 화면 관찰 없이 실행한 행동 묶음. 현재는 단일 클릭·입력, `입력 + Enter`, `입력 + 제출 클릭`을 허용한다.
- `evidence`: 실행 결과, 전환 상태, OpenCV 변화율과 OCR 문맥을 담은 자율탐색 근거. 활성 경로 저장 시에는 제거한다.

Critic이 중간 전이를 제거했을 때 삭제 전후 체크포인트가 같은 관찰 ID이거나,
URL·화면 역할이 같고 화면 pHash까지 정확히 같을 때만 경로를 다시 연결한다.
근사 pHash만으로는 두 상태를 같은 화면으로 취급하지 않는다. 남은 전이가 하나의
연속 경로를 만들지 못하면 후보 전체를 승격하지 않는다.
활성 레시피의 동일성은 `site + task_category`로 정한다. 검색어, 마커 이름,
화면 좌표와 행동 경로 표현은 레시피 키에 넣지 않는다. 같은 목적의 새 성공 경로가
들어오면 기존 행을 최신 경로로 갱신한다. 경로가 바뀌면 이전 경로의 재생 성공·실패
횟수와 후보 근거를 초기화해 서로 다른 경로의 성과가 섞이지 않게 한다. 동일한 경로를
다른 후보가 다시 검증한 경우에만 `recipe_sources`에 근거를 추가한다.

`state_key`, Jaccard anchor 유사도와 코드 기반 화면 역할 분류는 active replay의
기본 조회 기준이 아니다.

목적 기반 키는 `experience9#` 버전을 사용한다. 조회와 재생은 이 버전의 활성
경로만 대상으로 한다. 후보 테이블도 `contract_version=3`만 검토한다. 이전 경로
내용 기반 키는 중복된 목적을 보존할 수 있으므로 저장소 초기화 시 삭제한다.

## Replay 순서

1. `capture`가 안정된 현재 화면을 저장하고 `ocr`이 OCR/SoM marker와 화면 서명을 만든다.
2. `RecipeStore.get_site_recipes(site, task_category)`로 전체 경로 후보를 가져온다.
3. 각 경로의 첫 전이 URL 범위와 ROI pHash를 현재 화면과 비교한다.
4. 화면 역할 이름은 관측 로그에 남기되 재생 허용 여부에는 사용하지 않는다.
5. ROI가 맞으면 저장된 target 비율에 가까운 현재 OCR marker를 찾는다.
6. 첫 전이가 통과한 경로 하나를 선택하고 `ReplaySession`에 경로 키, 현재 전이 번호와 검증 대기 번호를 저장한다. 이 시점에는 번호를 증가시키지 않는다.
7. 현재 캡처에서 검증한 단일 클릭·입력 또는 `입력 + Enter` 행동 묶음을 `ActionRequest`로 실행한다.
8. OpenCV 연속 프레임 비교로 화면 변화 시작과 렌더링 안정화를 기다린다. 변화 뒤에는 설정된 정숙 시간 동안 추가 변화가 없어야 준비 완료로 판정하고 OCR을 실행한다.
9. 저장된 `after`의 ROI 앵커 또는 화면 문맥이 현재 화면과 일치할 때만 다음 전이 번호로 이동한다.
10. 다음 전이의 ROI 대상도 현재 marker로 다시 찾는다.
11. 마지막 전이의 도착 상태가 검증되면 `ReplaySession`을 끝낸다. 검증이 실패하면 세션을 종료하고 reasoning으로 폴백하며 같은 run 안에서 해당 `recipe_key`를 차단한다.

## 승격 정책

자율탐색의 실제 도구 계약에서 재사용 방식을 계산한다. 실행 직후 코드는 전이의
결과와 화면 증거를 한 번 확정한다. Critic은 잘못된 대상, 무효 행동, 폐기된 복구
경로와 불안정한 전이 묶음을 제거하며 실행 내용을 수정하지 않는다.

- `fixed`: 같은 UI 조작이 여러 실행에서 그대로 유효한 단계.
- `parameterized`: UI 조작은 같고 입력 슬롯만 바뀌는 단계.
- `reasoning`: 현재 화면, 현재 결과, 방문 여부, 목표 개수에 따라 판단해야 하는 단계.

공고 제목 클릭은 기본적으로 `reasoning`이다. 검색 열기, 검색어 입력, 검색 제출처럼 반복 증거가 있는 컨트롤만 active recipe 후보가 된다. 상세 펼치기 자동 클릭은 현재 사이트 `page_guidance.reveal_controls`에 선언된 OCR 라벨과 정확히 일치할 때만 허용한다.

LLM은 화면에 입력칸과 제출 수단이 함께 명확히 보일 때 `type_in_marker` 뒤
`Enter` 또는 제출 클릭을 한 요청으로 만들 수 있다. 실행기는 호출 순서대로 수행한
뒤 한 번만 화면을 관찰해 행동 묶음 전이를 완성한다. 중간 화면을 다시 봐야 하는
행동은 서로 다른 요청과 전이로 기록한다. 독립 입력 전이는 바로 뒤의 성공 행동을
준비한 기록일 때만 Critic 검토 대상으로 남긴다. 승격 단계에서 행동을 다시 합치지
않는다.

## 실패 처리

- ROI pHash가 맞지 않으면 즉시 reasoning fallback.
- target marker 비율 매칭이 실패하면 즉시 reasoning fallback.
- OpenCV 프레임 비교에서 화면 변화가 없거나 저장된 도착 체크포인트가 맞지 않으면 해당 `recipe_key`는 같은 run 안에서 재시도하지 않는다.

이 기준은 Reflex가 자율 탐색 전체를 대체하지 않고, 고정 가능한 부모 경로만 빠르게 재생하도록 제한한다.

## 계약 전환 검증

2026-08-12에 원티드 `iOS 개발자 1건`을 격리 DB에서
`자율탐색 -> Critic 승격 -> 경험 기반 탐색` 순서로 실행했다.

| 지표 | 자율탐색 | 경험 기반 탐색 |
|---|---:|---:|
| 실행시간 | 54.90초 | 42.16초 |
| 추론 호출 | 8회 | 5회 |
| 전체 토큰 | 49,233 | 31,331 |
| Reflex 전이 적중 | 0회 | 4회 |
| Reflex 경로 완주 | 0회 | 1/1회 |

두 실행 모두 서로 다른 공고 1건을 저장했고 품질 계약을 통과했다. 경험 기반
탐색은 홈 검색 열기, 검색어 입력, 직무 결과 선택과 포지션 탭 선택으로 이어지는
4단계 경로를 중간 실패 없이 완료했다.
