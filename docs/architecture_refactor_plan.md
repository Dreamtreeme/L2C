---
title: "아키텍처 리팩터링 계획"
type: plan
area: architecture
status: active
updated: 2026-07-29
tags:
  - l2c
  - docs/architecture
---

# 아키텍처 리팩터링 계획

기준일: 2026-07-29

## 진행 현황

| 단계 | 핵심 결과 | 상태 |
|---|---|---|
| 0. 런타임 기준 고정 | Python 3.13, 앱·OCR 환경 분리, CUDA·전체 테스트 검증 | 완료 |
| 1. 조사 저장·재개 | LangGraph SQLite 체크포인터와 `interrupt()` 전환 | 완료 |
| 2. 런타임 소유권 | FastAPI `lifespan`과 `ApplicationRuntime` 통합 | 완료 |
| 3. 행동 계약 | `AIMessage` 어댑터 제거, `ActionRequest` 통일 | 완료 |
| 4. 작업자 그래프 분할 | 관찰·전환·선택·원자 실행 경계 분리 | 완료 |
| 5. 설정·사이트 프로필 | 타입 설정과 단일 사이트 프로필 계약 | 완료 |
| 6. 관측 경로 | LangGraph 이벤트, SSE, 로컬 지표, LangSmith 연결 정리 | 완료 |
| 7. 회귀 검증 | 다중 사이트 자율·반복 E2E와 기존 코드 제거 | 완료 |

## 목표

L2C의 강점인 비전 기반 물리 조작, ROI Reflex, 결과 카드 큐, 전환 검증은 유지한다. 반면 LangGraph와 FastAPI가 이미 제공하는 실행 상태 저장, 재개, 수명주기, 이벤트 스트리밍을 애플리케이션 코드가 중복 구현한 부분은 대체한다.

리팩터링 원칙은 다음과 같다.

1. 실행 상태는 LangGraph 체크포인트가 소유한다.
2. 채용공고와 레시피 같은 업무 데이터만 애플리케이션 DB가 소유한다.
3. LLM 출력과 결정론적 정책은 같은 `ActionRequest` 계약을 사용한다.
4. 한 그래프 노드는 한 번의 재시도 또는 한 번의 외부 부작용 경계만 소유한다.
5. 설정, 진행 이벤트, 시간, 토큰 사용량은 각각 한 경로에서만 생성한다.
6. 기존 코드를 보존하기 위한 어댑터는 전환 단계가 끝나면 삭제한다.

## 현재 진단

| 영역 | 현재 상태 | 문제 |
|---|---|---|
| 조사 재개 | 업무 DB와 분리된 LangGraph SQLite 체크포인트 사용 | 사용자 확인이 필요한 조사 그래프만 체크포인트 재개 |
| 사용자 확인 | `interrupt()` 중단 후 같은 `thread_id`에 `Command(resume=...)` 전달 | 완료 |
| 작업자 상태 | 직렬화 가능한 평면 `GraphState`를 단일 계약으로 사용 | 책임별 `worker_*` 모듈이 자기 필드만 갱신하며 중첩 호환 상태는 제거 |
| 행동 계약 | `pending_action: ActionRequest`와 `last_action_result: ActionResult`로 분리 | 완료 |
| 노드 책임 | 관찰·전환·수집·선택·추론·실행·기록과 실행 검증·전달·상태 효과를 분리 | 완료 |
| 런타임 | `ApplicationRuntime`과 `VisionWorkerRuntime`이 자원 수명을 소유 | 완료 |
| 계획 진행 | 장식용 작업 계획 상태와 도구 제거 | 완료 |
| 관측 | `RunContext`와 구조화 그래프 이벤트를 단일 집계 원본으로 사용 | 완료 |
| 설정 | 타입 설정 모듈과 외부 OCR 부트스트랩만 환경 변수를 직접 읽음 | 완료 |
| 사이트 정보 | 사이트마다 검증된 `profile.json` 하나를 사용 | 완료 |
| 레시피 학습 | Critic 검토와 결정론적 활성 승격을 별도 모듈로 분리 | 완료 |

## 실행 순서

### 0단계. 런타임 기준 고정 - 완료

- Python 3.13.14로 전환한다.
- 앱과 PaddleOCR 환경을 `.venv-app`, `.venv-ocr`로 분리한다.
- 현재 사용 패키지를 Windows와 CUDA 13 조합에서 exact pin한다.
- PaddleOCR 3.x `predict()` 계약으로 전환한다.
- 호환성 검사와 전체 단위 테스트를 설치 절차에 포함한다.

완료 기준:

- 두 환경 `pip check` 통과
- PyTorch, Paddle CUDA 연산 통과
- OmniParser와 실제 한국어 OCR 통과
- `agent/tests` 전체 통과

### 1단계. 조사 그래프의 저장·중단·재개 정상화

상태: 완료 (2026-07-23)

도입:

- 로컬 앱에 `langgraph-checkpoint-sqlite`의 SQLite 체크포인터를 사용한다.
- `investigation_id`를 LangGraph `thread_id`로 사용한다.
- 조사 그래프의 확인 질문은 `interrupt()`로 중단하고 `Command(resume=...)`로 재개한다.
- 노드가 다시 실행될 수 있으므로 `interrupt()` 앞의 외부 부작용을 제거하거나 멱등 처리한다.

삭제·대체:

- `InvestigationWorkflow._save()`와 각 노드의 수동 저장을 제거한다.
- `run()`에서 저장 JSON을 읽어 새 초기 상태를 만드는 재개 코드를 제거한다.
- `InvestigationStore` 코드와 `investigation_sessions.state_json` 의존성을 제거한다.
- API의 `resume_mode="restart_from_request"`를 실제 체크포인트 재개로 바꾼다.

주요 파일:

- `agent/graph/investigation_workflow.py`
- `agent/application/investigation_store.py` (삭제)
- `agent/application/chat_service.py`
- `agent/web_server.py`

완료 기준:

- 확인 질문 전에는 DB 조회와 브라우저 실행이 일어나지 않는다.
- 프로세스를 재시작한 뒤 같은 `thread_id`로 정확한 다음 단계부터 이어진다.
- 확인 질문 재개 시 이전 LLM 호출, DB 검사, 수집 부작용이 중복 실행되지 않는다.

적용 결과:

- 업무 DB와 분리된 `*.investigation_checkpoints.db`에 그래프 상태를 저장한다.
- `investigation_id`를 `thread_id`로 사용하고 확인 질문은 `interrupt()`에서 멈춘다.
- 확인 답변은 `Command(resume=...)`로 전달하며 그래프의 `START`와 요청 분석을 다시 실행하지 않는다.
- `InvestigationStore`와 노드별 `_save()` 호출을 제거했다. 기존 업무 DB의 과거 `investigation_sessions` 테이블은 데이터 파괴를 피하기 위해 자동 삭제하지 않지만 새 코드에서는 읽거나 생성하지 않는다.
- 확인 질문 전에 DB 조회와 수집이 실행되지 않는 테스트, 새 프로세스에 해당하는 워크플로 재생성 후 재개 테스트, 이전 분석 LLM 비중복 테스트를 추가했다.

범위 경계:

- 채용공고 수집 작업자는 사이트 프로필이 허용한 안전한 조회 행동만 실행한다. 예상하지 못한 민감 행동은 실행 전에 중단하지만 작업자 내부 재개는 제공하지 않는다. 로그인·지원·결제처럼 민감 행동을 실제 서비스 범위에 넣을 때 별도의 체크포인트 작업자로 확장한다.

### 2단계. 애플리케이션 런타임 소유권 통합

상태: 완료 (2026-07-23)

도입:

- `ApplicationRuntime`이 설정, 체크포인터, 컴파일된 그래프, 비전 작업자, 승격 작업자를 소유한다.
- FastAPI `lifespan`에서 런타임을 만들고 `app.state`에 보관하며 종료 시 모두 닫는다.
- OCR과 모델 객체는 요청 간 재사용하되, DB 답변만 필요한 요청에서는 로드하지 않는다.
- 수집 요청이 들어오면 현재 방식처럼 OCR 준비와 브라우저 열기를 병렬로 시작한다.
- 물리 입력은 하나의 작업자 잠금으로 직렬화하되 잠금도 런타임 인스턴스가 소유한다.

삭제·대체:

- `_action_tools`, `_perception`, `_chat_service`, 승격 작업자 싱글턴 같은 모듈 전역을 제거한다.
- 요청 경로에서 그래프를 매번 `compile()`하지 않는다.
- 종료 시 브라우저만 닫고 OCR 프로세스의 소유자는 남는 현재의 분산 정리 코드를 통합한다.

완료 기준:

- 연속 두 수집 요청에서 OCR PID와 모델 인스턴스가 유지된다.
- 요청 종료 시 브라우저 창은 닫히고, 앱 종료 시 OCR 하위 프로세스와 GPU 자원이 해제된다.
- DB 전용 질문은 비전 런타임을 초기화하지 않는다.

적용 결과:

- `ApplicationRuntime`이 조사 체크포인터, 컴파일된 조사 그래프, `ChatService`, `VisionWorkerRuntime`, Reflex 승격 작업자를 소유한다.
- FastAPI는 `lifespan`에서 런타임을 한 번 만들고 `app.state.runtime`으로 요청에 제공한다. CLI와 E2E 실행기도 명시적인 런타임 수명주기를 사용한다.
- `VisionWorkerRuntime`은 `PerceptionEngine`, `ActionTools`, 컴파일된 작업자 그래프, UI 판단 모델 캐시와 물리 입력 잠금을 지연 생성한다.
- 수집 요청 종료에는 브라우저만 닫고 PaddleOCR 하위 프로세스와 화면 캡처 자원은 유지한다. 애플리케이션 종료에만 OCR, 캡처, 모델 캐시와 체크포인트 연결을 닫는다.
- `_perception`, `_action_tools`, `_ui_llm_with_tools`, `_chat_service`, 전역 실행 잠금, 전역 승격 작업자 싱글턴을 제거했다.
- 런타임을 만들거나 DB 질문을 처리하는 것만으로는 비전 자원이 초기화되지 않는다. 첫 수집 요청에서 OCR 준비와 브라우저 열기를 병렬 실행하는 기존 시작 장벽은 유지한다.

검증 결과:

- 연속 실행 세션에서 같은 `PerceptionEngine`, `ActionTools`, OCR PID를 재사용하고 요청 종료의 브라우저 정리로 OCR이 닫히지 않는 단위 테스트를 추가했다.
- FastAPI 시작·종료가 하나의 `ApplicationRuntime.start()`와 `close()`를 호출하는 테스트를 추가했다.
- 작업자 잠금이 같은 런타임의 동시 물리 입력을 직렬화하는 테스트를 유지했다.

### 3단계. 행동 계약에서 채팅 메시지 제거

상태: 완료 (2026-07-23)

도입:

- 그래프 상태의 다음 행동을 `ActionRequest | None`으로 고정한다.
- LLM 응답은 추론 노드 경계에서 한 번만 `ActionRequest`로 변환한다.
- Reflex, 카드 큐, 상세 페이지 정책도 `ActionRequest`를 직접 반환한다.
- 실행 결과는 별도 `ActionResult` 계약으로 기록한다.

삭제·대체:

- `ActionRequest.to_ai_message()`와 `build_action_message()`를 제거한다.
- `last_action_result: AIMessage`와 `hasattr(..., "tool_calls")` 검사를 제거한다.
- 결정론적 정책 테스트가 LangChain 메시지 내부 구조를 검사하지 않게 바꾼다.

완료 기준:

- `agent/runtime`에서 `AIMessage` import가 0개다.
- 같은 실행기가 LLM, Reflex, 큐에서 생성된 행동을 모두 처리한다.
- 허용되지 않은 도구명과 잘못된 인자는 실행 전 Pydantic 검증에서 거절된다.

적용 결과:

- 작업자 상태에서 실행 전 명령은 `pending_action`, 실행 후 결과는 `last_action_result`로 분리했다.
- `ActionRequest`는 출처, 설명, 검증된 `ToolCallRequest` 목록을 보유한다. 큐 식별자와 전환 출처 같은 실행 추적 정보는 도구 인자가 아닌 호출 `metadata`에 저장한다.
- LLM 응답은 `reasoning_node`에서 한 번만 `ActionRequest`로 변환한다. 카드 선택기, 결과 카드 큐, 중복 공고 정책, Reflex도 같은 객체를 직접 만든다.
- `execution_node`는 요청 출처와 무관하게 하나의 실행 경로를 사용하고 결과를 `ActionResult`로 기록한다.
- `ActionRequest.to_ai_message()`, `build_action_message()`, 작업자 상태의 `AIMessage`, `hasattr(..., "tool_calls")` 호환 코드를 삭제했다.

검증 결과:

- `agent/runtime`과 작업자 그래프에서 `AIMessage` import가 0개임을 확인했다.
- LLM, Reflex, 공고 카드 큐 출처가 같은 `execution_node`와 물리 디스패처를 사용하는 테스트를 추가했다.
- 알 수 없는 도구, 필수 인자 누락, 사이트 허용 목록 밖 도구가 실행 전에 거절되는 테스트를 추가했다.
- 전체 `agent/tests` 386건을 통과했다.

### 4단계. 작업자 그래프와 거대 노드 분할

상태: 완료 (2026-07-28)

목표 흐름:

```text
관찰 -> 전환 판정 -> 행동 선택 -> 실행 단위 수행 -> 다시 관찰
                         |                |
                   큐/Reflex/LLM      실행 기록
```

분할:

- 관찰: 화면 준비 대기, 캡처, SoM/OCR 실행만 담당한다.
- 전환 판정: OpenCV 연속 프레임으로 변화 시작과 안정화를 기다리고, 저장된
  도착 ROI 또는 화면 문맥으로 전이 완료를 확인한다.
- 행동 선택: 큐, 상세 정책, Reflex, LLM 순서로 하나의 `ActionRequest`를 만든다.
- 행동 실행: 한 캡처에서 검증한 물리 행동 하나를 실행한다. 입력과 Enter처럼
  중간 재관찰이 필요 없는 결정론적 조합만 한 전이의 행동 묶음으로 허용한다.
  Reflex는 현재 전이 번호를 상태에 보관하고, 저장된 도착 상태를 검증한 뒤에만
  다음 전이로 이동한다.
- 수집 상태 반영: 상세 OCR 병합, 카드 완료, 종료 조건을 순수 상태 전이로 처리한다.
- 기록: 피드백 episode와 레시피 후보 기록을 실행 결과에서 생성한다.

상태 정리:

- 관찰, 전환, 수집, 재생, 제어 상태는 하나의 직렬화 가능한 평면 계약으로 유지하되 책임별 모듈만 자기 필드를 갱신한다.
- `queue_replay_hit`, `page_policy_hit`, `reflex_hit` 같은 병렬 플래그 대신 `pending_action.source` 하나를 사용한다.
- `last_action_screen_changed`는 전환 결과로 통합한다.
- 경로 객체는 체크포인트 직렬화를 위해 문자열로 저장한다.

삭제·대체:

- `update_plan_progress`, `plan`, `current_plan_step`과 관련 프롬프트를 제거한다.
- 그래프 상태의 `step_durations`를 제거하고 관측 계층에서만 시간을 기록한다.
- 분할 후 사용되지 않는 `nodes.py` 헬퍼와 호환 어댑터를 삭제한다.

완료 기준:

- 각 그래프 노드는 한 가지 외부 부작용 또는 순수 상태 전이만 수행한다.
- 화면 변경 행동 뒤에는 항상 관찰 노드가 한 번만 실행된다.
- Wanted 기준 결과 품질, Reflex hit, 카드 큐 재생 수가 리팩터링 전 기준보다 낮아지지 않는다.

적용 결과:

- `nodes.py`를 삭제하고 `worker_observation`, `worker_transition`, `worker_collection`, `worker_selection`, `worker_reasoning`, `worker_execution`, `worker_recording`으로 소유권을 분리했다.
- `worker_execution.py`는 실행 진입점만 남기고 상태 조립, 원자 도구 전달, 행동별 후속 처리를 각각 `worker_execution_context`, `worker_execution_dispatch`, `worker_execution_handlers`로 분리했다.
- `worker_execution_handlers.py`는 검증 → 실행 → 효과 → 기록 순서만 조율한다. 실행 전 안전 규칙은 `worker_action_guard.py`, 카드·상세 완료 후 상태 반영은 `worker_action_effects.py`가 담당한다.
- UI·상태·종료 행동 분류는 `runtime/worker_actions.py`의 단일 계약을 사용하며 피드백 기록에 있던 중복 행동 집합을 제거했다.
- `worker_reasoning.py`에는 카드 선택기, 모델 호출과 응답 검증만 남기고 화면·수집·전환 프롬프트 조립은 `worker_reasoning_prompt.py`로 분리했다. 호출되지 않던 동일 대상 재시도 판정 함수도 삭제했다.
- 중첩 상태와 평면 호환 필드를 함께 갱신하던 시도는 상태 복제와 갱신 누락을 만들기 때문에 제거했다. `create_worker_state()`가 평면 상태의 유일한 생성 지점이다.
- 화면 변경 뒤에는 캡처와 전환 판정을 거쳐 필요한 OCR을 먼저 수행한다. 상세 후속 추론이 새 화면 OCR을 앞지르던 순서 오류도 수정했다.
- 마커 조회와 타깃의 픽셀·비율 좌표 생성은 `vision/target_snapshot.py`로 통합하고 피드백, 실행 기록, 카드 큐에 있던 중복 구현을 제거했다.
- 삭제된 거대 노드의 로컬 백업 파일도 제거했다. 현재 서비스가 실행하지 않는 민감 행동의 중단 후 재개는 작업자 그래프 분할 완료 조건에서 제외한다.

### 지휘자 그래프 조립과 업무 노드 분리

- 1,395줄이던 `investigation_workflow.py`에는 그래프 연결, 체크포인트 중단·재개, 실행 진입만 남겼다.
- 조사 상태와 모델 계약은 `investigation_context.py`, 결정론적 근거 판정은 `investigation_evidence_policy.py`가 소유한다.
- 요청 해석·확인 질문, 근거 검사·계획, 수집 실행, 문서 조회·답변은 각각 `investigation_*_nodes.py`로 분리했다.
- 각 노드 묶음은 필요한 모델·서비스만 생성자로 받고, 서로를 참조하지 않는다. 최상위 workflow만 노드 묶음을 조립한다.

### 수집 실행과 애플리케이션 책임 분리

- `InvestigationCollectionNodes`가 `CollectionService.collect`를 직접 호출한다.
- 검색 의도·사이트 프로필·작업자 목표 생성은 `collection_request_builder.py`가 담당한다.
- 화면 준비부터 단일 Worker 실행, 제출물 생성과 재귀 한도 보고는 `collection_worker_runner.py`가 담당한다.
- Critic 검토, 승인 데이터 저장, 완결된 실행의 레시피 후보 등록은 `collection_submission_service.py`가 담당한다.
- 구조화 인자를 다시 조립하던 미사용 LangChain 도구 래퍼는 삭제했다.

### 레시피 검토와 승격 책임 분리

- `candidate_reviewer.py`는 Critic 입력 구성, LLM 호출, 응답 계약 검사와 후보 상태 갱신만 담당한다.
- `candidate_promotion.py`는 승인된 단계 주석, ROI 단계와 문맥 후속 전략 생성, 활성 레시피 DB 반영을 담당한다.
- 기록·검토·승격 모듈에서 중복 정의하던 행동 종류는 `replay_actions.py`의 단일 계약을 사용한다.
- 저장된 Critic 판정을 재적용하는 벤치마크도 검토기가 아니라 승격 모듈을 직접 호출한다.

### 5단계. 설정과 사이트 프로필을 타입 계약으로 통합

상태: 완료 (2026-07-23)

설정:

- `pydantic-settings`의 `BaseSettings`로 경로, 모델, OCR, 브라우저, Reflex, 관측 설정을 나눈다.
- 기존 환경 변수 이름은 validation alias로 읽되 코드에서는 타입 필드만 사용한다.
- 허용 범위가 있는 timeout, 재시도, 출력 토큰 수는 시작 시 검증한다.
- 라이브러리 모듈의 직접 `os.getenv()` 호출을 제거한다.

사이트 프로필:

- 사이트별 identity, domains, 시작 URL, page roles, 도구 정책, 화면 안내를 하나의 타입 계약으로 검증한다.
- `registry.json`, `manual.json`, `tools.json`의 중복 필드는 단일 프로필로 합친다.
- `SKILL.md`는 구조화 필드와 겹치지 않는 판단 지침만 남기거나 프로필에서 렌더링한다.
- 서버 시작 시 모든 사이트 프로필을 읽어 잘못된 URL 패턴, 알 수 없는 도구, 중복 도메인을 실패 처리한다.

완료 기준:

- 애플리케이션 코드의 직접 환경 변수 조회는 설정 모듈과 외부 라이브러리 부트스트랩에만 남는다.
- 새 사이트는 검증된 프로필 하나와 범용 E2E 시나리오만 추가하면 등록된다.

적용 결과:

- `agent/config/settings.py`의 타입 설정이 앱, 모델, OCR, 브라우저, Reflex, 관측 설정을 검증한다.
- 사이트별 `registry.json`, `manual.json`, `tools.json`, 중복 `SKILL.md`를 삭제하고 `profile.json` 하나로 통합했다.
- 사람인과 고용24의 공식 홈 리다이렉트 경로도 프로필의 화면 역할 증거로 선언해 Reflex 조회와 화면 판정이 같은 기준을 사용한다.
- 직접 환경 변수 조회는 Windows 설치 경로 탐색과 격리된 PaddleOCR 프로세스 부트스트랩에만 남겼다.

### 6단계. 관측 경로 단일화

상태: 완료 (2026-07-23)

- 그래프는 LangGraph `custom` stream으로 진행 이벤트를 낸다.
- 백엔드는 이 이벤트를 SSE로 변환하고, 같은 이벤트를 로컬 지표와 structlog에 전달한다.
- LangSmith는 그래프·LLM trace와 평가 피드백을 담당하고 로컬 실행은 LangSmith 없이도 완결된다.
- 토큰 사용량은 제공자 응답 메타데이터를 정규화한 한 집계기에서만 합산한다.
- 실행 시간은 단계별 시작·종료 이벤트 한 쌍으로 계산한다.
- 로그 후처리 정규식으로 다시 계산하는 지표는 제거한다.

완료 기준:

- SSE, JSON 로그, E2E 요약, LangSmith trace가 같은 `run_id`, 단계명, 성공 여부를 사용한다.
- 총 토큰 수가 개별 LLM 호출 합계와 일치한다.
- 실패 E2E에서 마지막 성공 단계와 실패 코드가 자동으로 보인다.

적용 결과:

- 그래프 단계는 시작·종료 구조화 이벤트에 `component`, `duration_sec`, `action_source`, 성공 여부를 기록한다.
- Reflex와 카드 큐 hit는 로그 문구가 아니라 `action_source`로 집계한다.
- `step_durations`와 텍스트 로그 정규식 재계산을 삭제했다. `profile_reflex_trace.py`는 `.summary.json`만 입력으로 받는다.
- SSE, 로컬 JSON 요약과 LangSmith가 같은 `run_id`와 단계 이름을 사용한다.

### 7단계. 회귀 검증과 기존 코드 제거

상태: 완료 (2026-07-29)

단위·통합 테스트:

- 체크포인트 중단/재개와 부작용 중복 방지
- 런타임 재사용과 종료
- 세 행동 원천(LLM, Reflex, 큐)의 동일 실행 계약
- 전환 성공, 무변화, 로딩, timeout 경로
- 사이트 프로필 전체 계약 검사
- 토큰·시간 집계 일치

E2E 행렬:

| 축 | 최소 범위 |
|---|---|
| 사이트 | Wanted, 사람인, 고용24와 신규 사이트 1개 |
| 실행 | 첫 자율 탐색, 같은 작업의 경험 기반 탐색 |
| 요청 | 명확한 개수, 화면 전체, 모호한 질문 후 확인 |
| 결과 | 저장 품질, 중복 처리, 부분 완료, 최종 DB 답변 |
| 성능 | 실행 시간, OCR p50/p95, 추론 횟수·시간, 토큰·비용, Reflex/큐 hit |

최종 검증 결과:

- 실행기와 추론 문맥 분리 후 Python 3.13 전체 단위 테스트 230건을 통과했다.
- 모든 수치는 텍스트 로그 재계산이 아니라 각 실행의 `.summary.json`을 기준으로 확인했다.

| 시나리오 | 실행시간 | 품질 | 주요 관측 |
|---|---:|---|---|
| 원티드 자율 탐색, iOS 2건 | 70.46초 | 2건 저장, 통과 | 승격 가능한 ROI 단계 2개 |
| 원티드 경험 기반 탐색, iOS 2건 | 63.38초 | 2건 저장, 통과 | 추론 11회→8회, Reflex 2회 |
| 사람인 자율 탐색, AI 1건 | 44.81초 | 1건 저장, 통과 | 승격 가능한 ROI 단계 2개 |
| 사람인 경험 기반 탐색, AI 1건 | 48.16초 | 1건 저장, 통과 | 추론 5회→3회, Reflex 2회 |
| 로켓펀치 자율 탐색, 백엔드 1건 | 43.91초 | 1건 저장, 통과 | SPA 측면 상세 URL 증거 보완 후 재검증 |
| 고용24 자율 탐색, 데이터 분석 1건 | 38.94초 | 1건 저장, 통과 | 후보 저장 후 Critic 승격 성공 |
| 고용24 경험 기반 탐색, 동일 요청 | 35.89초 | 1건 저장, 통과 | Reflex 1회, 후속 전략 1회 |
| 원티드 경험 기반 탐색, 백엔드 1건 | 39.31초 | 1건 저장, 통과 | iOS 레시피의 검색어·개수 일반화 |
| 원티드 기존 DB, iOS 2건 | 20.33초 | 기존 2건 확인, 통과 | 상세 진입 0회, 화면 추론 0회 |

로켓펀치는 목록 URL의 `selectedJobId` 쿼리로 상세 패널을 여는데 기존 프로필은 전용 `/jobs/{id}`만 상세 URL로 선언했다. 상세 OCR 버퍼가 비어 `finish_detail_reading`을 41회 반복한 실패 실행은 128.42초·0건·394,516토큰이었다. 프로필에 SPA 상세 URL 증거를 선언한 뒤 43.91초·1건·65,399토큰으로 정상 종료했다.

고용24 경험 기반 탐색은 Reflex 기능과 품질은 통과했지만 자율 탐색보다 빨라지지는 않았다. 따라서 레시피 적중은 추론 경로를 줄이는 기능 증거로만 사용하며, 한 번의 실행시간 개선을 일반화하지 않는다.

기존에는 DB 중복 카드를 `skipped`로 표시하면서 목표 수에는 이번 실행에서 상세
수집한 `done` 카드만 계산했다. 이 불일치 때문에 해결된 카드만 있는 큐가 종료되지
않았다. 현재는 `done`과 `skipped + job_id`만 해결 수로 계산하며, 근거 없는
스킵은 제외한다. 실제 중복 E2E에서 DB 공고 2건을 확인한 직후 정상 종료했다.

로그인·지원·결제 같은 민감 행동은 현재 서비스 범위가 아니다. 이를 제품 범위에 추가할 때만 작업자 체크포인트 중단·재개를 별도 단계로 설계한다.

## 유지할 핵심 구현

- DOM 없이 화면·OCR·물리 입력으로 조작하는 Realtime 경계
- PaddleOCR 하위 프로세스 재사용과 장애 시에만 재시작하는 정책
- ROI pHash와 OCR 위치 검증을 결합한 Reflex
- 목록 복귀 후 결과 카드 큐 재생
- 화면 변경 시간을 확인하는 OpenCV 연속 프레임 비교와 저장 상태 확인용 ROI pHash
- 자율 탐색 결과를 후보로 저장한 뒤 별도 Critic이 승격하는 흐름
- 채용공고 원문, 구조화 데이터, 근거를 분리하는 저장 경계

## 공식 기준

- LangGraph 상태 저장: <https://docs.langchain.com/oss/python/langgraph/persistence>
- LangGraph 중단과 재개: <https://docs.langchain.com/oss/python/langgraph/interrupts>
- LangGraph 이벤트 스트리밍: <https://docs.langchain.com/oss/python/langgraph/streaming>
- FastAPI 수명주기: <https://fastapi.tiangolo.com/advanced/events/>
- Pydantic Settings: <https://docs.pydantic.dev/latest/concepts/pydantic_settings/>
- structlog contextvars: <https://www.structlog.org/en/stable/api.html#structlog.contextvars.bind_contextvars>
- 런타임·GPU·OCR 버전 기준: [runtime_compatibility.md](runtime_compatibility.md)
