---
title: "작업자 상태 계약"
type: reference
area: architecture
status: active
updated: 2026-08-08
tags:
  - l2c
  - docs/architecture
---

# 작업자 상태 계약

Vision Worker LangGraph는 `agent/runtime/worker_contracts.py`의 `WorkerState`를 공유한다. 상태는 책임별 구역으로 나뉘며 노드는 변경한 구역만 `WorkerStateUpdate`로 반환한다. 각 구역의 LangGraph reducer가 기존 값과 부분 갱신을 병합한다. 이름 규칙은 [[naming_conventions]]을 따른다.

## 상태 구역

| 구역 | 대표 필드 | 주 갱신 주체 |
|---|---|---|
| `request` | `goal`, `worker_run_id`, `recipe_params`, `job_collection_contract` | 작업자 실행 진입점 |
| `observation` | 캡처 ID, 화면 파일, URL, OCR 마커, 화면 서명, 페이지 역할 | `capture_node()`, `ocr_node()` |
| `decision` | `pending_action`, 카드 선택 trace | 선택·Reflex·추론 노드 |
| `transition` | 행동 이벤트, 오류 수, 전환 요청과 판정 결과 | 실행·전환 노드 |
| `replay` | Reflex trace, 활성 경로, 차단 경로 | Reflex·전환 노드 |
| `collection` | 추출 공고, 카드 큐, 목록 기억, 상세 OCR 버퍼와 판독 범위 | OCR·선택·실행 효과 |
| `lifecycle` | `is_finished` | 실행 효과와 종료 정책 |
| `safety` | 행동 권한, 사용자 승인 대기 | 실행 전 안전 검증 |

`request`는 한 작업자 실행의 입력 계약이다. 나머지 구역은 그래프가 현재 캡처를 처리하면서 갱신하는 실행 상태다.

## 노드 갱신 계약

```python
return {
    "observation": {
        "current_capture_id": capture_id,
        "current_screenshot": image_path,
        "ocr_complete": False,
    }
}
```

위 반환값은 전체 상태가 아니다. LangGraph가 기존 `observation`에 세 필드를 병합하고 다른 일곱 구역은 유지한다. 노드 내부에서 여러 부분 갱신을 연속 계산할 때는 `apply_worker_state_update()`를 사용한다.

`WorkerExecutionContext`는 행동 요청 하나를 실행하는 동안 입력과 결과를 분리한다. `ActionExecutionInput`은 최초 상태, 검증된 행동 요청과 런타임 의존성을 보관한다. `ActionExecutionResult`는 작업 상태 사본, 행동 결과, 이벤트와 후속 행동을 보관한다. 실행 코드는 책임 구역을 직접 갱신하며, 마지막에 최초 상태와 달라진 구역만 반환한다. 실행 필드를 그래프 상태로 다시 복사하는 동기화 표는 없다.

## 런타임 의존성

`WorkerState`에는 OCR 객체, 모델 클라이언트, 브라우저 도구와 잠금을 저장하지 않는다. `WorkerExecutionService`가 그래프 실행 시 `WorkerDependencies`를 LangGraph `context`로 전달한다.

```python
app.stream(
    initial_state,
    context=WorkerDependencies(vision=worker_runtime),
)
```

캡처, OCR, 추론과 실행 노드는 `Runtime[WorkerDependencies]`를 인자로 받고 `runtime.context.vision`에서 의존성을 사용한다. 이 방식은 실행 상태의 체크포인트 직렬화와 프로세스 자원 수명주기를 분리한다.

## 불변조건

1. `observation.ocr_complete=true`이면 `ocr_capture_id`와 `current_capture_id`가 같다.
2. `capture_node()`가 새 캡처를 만들면 이전 OCR 마커, 화면 서명과 페이지 역할을 비운다.
3. `decision.pending_action.metadata.decision_capture_id`는 행동을 선택한 캡처 ID다.
4. 화면 변경 행동은 `transition.transition_request`를 만들고 다음 캡처가 이를 판정한다.
5. 전환 판정은 같은 행동 순번의 `transition.action_events`에 연결된다.
6. 활성 Reflex 전이 번호는 저장된 도착 상태 검증이 성공한 뒤에만 증가한다.
7. 뒤로가기 후 목록 마커는 목록 화면 검증이 끝난 경우에만 다음 카드 선택에 사용한다.
8. `request`에는 직렬화 가능한 값만 저장하고 런타임 객체는 LangGraph 문맥으로 전달한다.

`current_observation_matches_capture()`가 현재 캡처와 OCR 결합을 검사한다. Reflex 전이 범위와 도착 상태는 `agent/recipe/replay_runtime.py`와 `worker_transition.py`에서 검증한다.
