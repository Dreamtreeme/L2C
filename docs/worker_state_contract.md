---
title: "작업자 상태 계약"
type: reference
area: architecture
status: active
updated: 2026-07-31
tags:
  - l2c
  - docs/architecture
---

# 작업자 상태 계약

작업자 그래프는 `GraphState` 하나를 공유한다. 각 필드는 아래 노드가 생성하고
정해진 노드만 갱신한다. 이름 규칙은 [[naming_conventions]]을 따른다.

## 주요 상태 소유권

| 상태 | 생성·초기화 | 갱신 | 소비 |
|---|---|---|---|
| `current_capture_id`, `current_screenshot` | `create_worker_state()` | `capture_node()` | 전환·OCR·실행·기록 |
| `ocr_capture_id`, `ocr_complete` | `create_worker_state()` | `capture_node()`, `ocr_node()`, 검증된 관찰 재사용 | 수집·선택·Reflex·추론 |
| `current_markers`, `screen_signature` | `create_worker_state()` | `ocr_node()`, 검증된 목록 관찰 재사용 | 선택·Reflex·실행 |
| `pending_action` | `create_worker_state()` | 선택·Reflex·추론 노드 | 실행 노드 |
| `last_action_result`, `execution_records` | `create_worker_state()` | 실행 노드 | 기록 노드 |
| `transition_request` | `create_worker_state()` | 실행 노드 | 캡처·전환 노드 |
| `transition_result`, `transition_records` | `create_worker_state()` | 전환 노드 | 선택·기록·관측 |
| `active_reflex_recipe` | `create_worker_state()` | Reflex·전환 노드 | 선택·Reflex |
| `job_card_queue`, `active_job_card` | `create_worker_state()` | 카드 선택·실행 효과 | 선택·수집 |
| `job_detail_buffer`, `job_detail_coverage` | `create_worker_state()` | 수집·상세 실행 효과 | 상세 정책·제출 |

## 불변조건

1. `ocr_complete=true`이면 `ocr_capture_id`와 `current_capture_id`가 같다.
2. `capture_node()`가 새 캡처를 만들면 기존 OCR·마커·화면 서명을 함께 비운다.
3. 이전 OCR은 현재 프레임이 행동 전 프레임과 같다고 검증된 경우에만 새 캡처에 연결한다.
4. `pending_action.metadata.decision_capture_id`는 행동을 선택한 `current_capture_id`다.
5. Reflex 전이 번호는 `0 <= current_transition_index < transition_count`를 만족한다.
6. Reflex의 다음 전이는 저장된 도착 상태 검증이 성공한 뒤에만 활성화한다.
7. 뒤로가기 후 목록 마커는 목록 pHash 검증이 성공한 경우에만 현재 캡처에 연결한다.

`current_observation_errors()`는 캡처·OCR 결합과 활성 Reflex 범위 위반을
단위 테스트와 디버깅에서 검사한다.
