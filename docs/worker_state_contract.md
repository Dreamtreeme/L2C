---
title: "작업자 상태 계약"
type: reference
area: architecture
status: active
updated: 2026-08-27
tags:
  - l2c
  - docs/architecture
---

# 작업자 상태 계약

Vision Worker LangGraph는 `agent/runtime/worker_contracts.py`의 `WorkerState`를 공유한다. `WorkerState`와 그 여덟 구역은 실행 중 항상 모든 필드를 갖는 완성 상태다. 노드는 `WorkerStateUpdate` 안의 `WorkerRequestPatch`, `ObservationPatch` 같은 부분 갱신 타입을 반환한다. 각 구역의 LangGraph reducer가 기존 값과 패치를 병합한다. 이름 규칙은 [L2C 네이밍 규칙](naming_conventions.md)을 따른다.

## 상태 구역

| 구역 | 대표 필드 | 주 갱신 주체 |
|---|---|---|
| `request` | `goal`, `worker_run_id`, `collection_intent`, 행동 권한 계약 | 작업자 실행 진입점 |
| `observation` | 캡처 ID, 화면 파일, URL, OCR 마커, 화면 서명, 페이지 역할 | `capture_node()`, `ocr_node()` |
| `decision` | `pending_action`, 추론 호출 상태 | `decision_node` 내부 선택·Reflex·추론 정책 |
| `transition` | 행동 이벤트, 오류 수, 전환 요청과 판정 결과 | `execution_node`, `observation_node` |
| `replay` | Reflex trace, 활성 경로, 차단 경로 | `decision_node`, `observation_node` |
| `collection` | 검토 완료 공고·화면 근거, 카드 큐, 상세 OCR 버퍼, 검토 대기 초안과 최근 검토 결과 | `decision_node`, `execution_node`, `review_node` |
| `progress` | `stage` | 선택·실행·검토 정책 |
| `lifecycle` | `is_finished` | 실행 효과와 종료 정책 |

`request`는 한 작업자 실행의 입력 계약이다. Reflex의 가변 검색어도 별도 상태로
복사하지 않고 `collection_intent.search_keyword`에서 직접 가져온다. 나머지
구역은 그래프가 현재 캡처를 처리하면서 갱신하는 실행 상태다.

상세 읽기가 끝났다고 판단한 실행 노드는 누적 OCR과 대표 상세 스크린샷을 `JobDraft`로 만들고 `collection.pending_job_draft`에 둔다. `worker_review` 노드는 OCR 내용과 화면의 공간 배치를 함께 검토해 `needs_more`, `complete`, `source_incomplete`, `invalid_target` 중 하나를 반환한다. `complete`일 때만 `collection.job_captures`와 `collection.collected_jobs`에 각각 화면 근거와 구조화된 공고를 추가한다.

`collection.job_card_queue`는 선택한 공고의 제목과 처리 상태를 보관한다. 상세 수집 뒤에는 결정된 복귀 행동을 수행하고, 현재 목록 OCR에서 다음 제목을 찾아 클릭한다. 최초 큐가 만들어진 뒤에는 카드 선택 모델을 다시 호출하지 않는다. 큐를 소진하고도 목표가 남으면 일반 화면 추론이 현재 목록을 스크롤하거나 필터를 조정하고 새 미방문 카드를 큐에 추가한다.

## 노드 갱신 계약

```python
return {
    "observation": {
        "observation_id": observation_id,
        "current_screenshot": image_path,
        "ocr_complete": False,
    }
}
```

위 반환값은 전체 상태가 아니다. `ObservationPatch`이며 LangGraph가 기존 `observation`에 세 필드를 병합하고 다른 일곱 구역은 유지한다. 노드 내부에서 여러 부분 갱신을 연속 계산할 때는 `apply_worker_state_update()`를 사용한다. 완성 상태 타입과 패치 타입을 분리하므로 필수 상태 누락은 정적 검사에서 잡고, 노드가 일부 필드만 반환하는 것은 허용한다.

`WorkerExecutionContext`는 행동 요청 하나를 실행하는 동안 검증된 행동 요청, 런타임 의존성, 작업 상태 사본과 후속 행동을 직접 보관한다. 여러 원자 행동이 같은 문맥에서 순서대로 상태를 바꾸므로 실행 단위가 끝나면 완성된 `WorkerState`를 반환한다. 다른 그래프 노드는 필요한 구역만 `WorkerStateUpdate`로 반환한다. 실행 필드를 별도 상태 객체와 동기화하는 변환 표는 없다.

## 런타임 의존성

`WorkerState`에는 OCR 객체, 모델 클라이언트, 브라우저 도구와 잠금을 저장하지 않는다. `WorkerExecutionService`가 그래프 실행 시 `WorkerDependencies`를 LangGraph `context`로 전달한다.

```python
app.stream(
    initial_state,
    context=WorkerDependencies(vision=worker_runtime, data=data_services),
)
```

캡처, OCR, 추론과 실행 노드는 `Runtime[WorkerDependencies]`를 인자로 받고 `runtime.context.vision`에서 의존성을 사용한다. 이 방식은 실행 상태의 체크포인트 직렬화와 프로세스 자원 수명주기를 분리한다.

## 불변조건

1. `observation_id`는 현재 화면 파일, OCR 마커, 화면 서명과 페이지 역할을 하나의 관찰로 묶는다.
2. `capture_node()`가 새 관찰을 만들면 이전 OCR 마커, 화면 서명과 페이지 역할을 비우고 `ocr_complete=false`로 설정한다.
3. `decision.pending_action.observation_id`는 행동이 근거로 삼은 관찰을 가리킨다.
4. 화면 변경 행동은 명시적인 입력 문자열, 대상 마커 ID와 도착 화면을 가진 `transition.transition_request`를 만들고 다음 캡처가 이를 판정한다. 대기 중인 전환이 없으면 값은 `None`이다.
5. 전환은 `before_observation_id`와 `after_observation_id`로 행동 전후 관찰을 연결하고 같은 행동 순번의 `transition.action_events`에 저장된다.
6. 활성 Reflex 전이 번호는 저장된 도착 상태 검증이 성공한 뒤에만 증가한다.
7. 저장 좌표는 현재 ROI pHash 검증이 끝난 경우에만 경험 경로 행동에 사용한다. 검증 실패 시 저장 좌표를 강행하지 않고 자율 판단으로 복귀한다.
8. `request`에는 직렬화 가능한 값만 저장하고 런타임 객체는 LangGraph 문맥으로 전달한다.
9. 수집 완료 공고는 `CollectedJob`으로 생성한 뒤 필드 이름이나 타입을 다시 변환하지 않는다.
10. 큐 카드 클릭은 도구 호출의 `queue_id`가 실제 대기 항목과 일치할 때만 해당 항목을 활성화한다.
11. `JobReview.status=complete`인 공고만 수집 수에 포함한다. `needs_more`는 상세 OCR 버퍼와 활성 카드를 유지하고, `source_incomplete`와 `invalid_target`은 현재 카드를 제외한다.

`current_observation_ready()`가 현재 관찰의 OCR 완료 여부를 검사한다. Reflex 전이 범위와 도착 상태는 `agent/recipe/replay_runtime.py`, 일반 화면 변화와 OCR 재사용은 `agent/runtime/transition_runtime.py`가 담당한다. `worker_transition.py`의 `TransitionDecision`은 이 판정 결과를 상태 패치와 실행 기록에 한 번 반영한다.
