---
title: "L2C 운영 트러블슈팅"
type: reference
area: troubleshooting
status: active
updated: 2026-08-16
tags:
  - l2c
  - docs/troubleshooting
---

# L2C 운영 트러블슈팅

현재 코드에서 다시 발생할 수 있는 문제와 해결 계약을 정리한다. 폐기된 로컬 모델, 구버전 PaddleOCR, 초기 pHash 실험은 [초기 비전 자동화 실험 기록](docs/history/legacy_vision_experiments.md)에 보관했다. 구현과 문서가 다르면 [시스템 아키텍처](ARCHITECTURE.md)와 테스트를 우선 확인한다.

## 빠른 진단

| 현상 | 먼저 볼 지표·상태 | 관련 절 |
|---|---|---|
| OCR 한 번이 수십 초 걸림 | OCR 작업자 세대, 요청별 추론 시간, timeout·재시작 로그 | 1 |
| 회색·흰 화면을 OCR함 | `wait_reason`, `low_information`, 프레임 변화율 | 2 |
| 화면은 비슷한데 Reflex가 없음 | 후보 수, URL 범위, ROI 거리, 차단 사유 | 3 |
| 후보는 있는데 활성 레시피가 없음 | `recipe_candidates.status`, Critic 재시도 횟수 | 4 |
| 상세 페이지를 검색 목록으로 오인 | 현재 URL, 사이트 프로필의 화면 역할 규칙 | 5 |
| 노드 단위 테스트만 통과하고 실제 그래프 실패 | LangGraph `runtime.context`, 완성 상태와 patch 계약 | 6 |
| 작업 종료가 다른 브라우저에 영향 | 바인딩된 창 ID, 탭 전환 뒤 URL 동기화 | 7 |
| 저장됐는데 답변 근거가 비어 있음 | 정규화 URL, 출처 식별자, 공고 ID 연결 | 8 |
| 경험 기반 실행이 한 번 더 느림 | 추론 순감소량, 경로 완주, 폴백, 결과 품질 | 9 |

## 1. OCR 작업자의 간헐적 지연

### 현상

대부분의 OCR은 짧게 끝나지만 일부 전체 화면 요청이 timeout까지 반환되지 않았다. 같은 화면과 같은 작업자에서도 빠른 요청과 느린 요청이 섞여 나타났다.

### 원인 판별

IPC 왕복, 이미지 직렬화, 작업자 시작과 Paddle 추론을 따로 계측했다. 지연 구간은 Paddle 추론 호출 내부였다. 작업자 세대 기록에서는 느린 요청 전후에도 같은 프로세스가 유지됐다.

초기 Python·PaddleOCR 조합에는 다음 문제가 함께 있었다.

- PaddleOCR 2.x와 3.x API 계약이 혼재했다.
- 앱의 PyTorch·OpenCV와 Paddle 의존성이 한 환경에서 같은 DLL·`cv2` 이름을 공유했다.
- 설치 문서의 버전과 실제 실행 프로세스 버전이 달랐다.

timeout은 멈춘 요청을 회수한다. 원인 판별에는 단계별 추론 시간과 작업자 세대를 사용한다.

### 현재 처리

- 앱은 `.venv-app`, PaddleOCR 작업자는 `.venv-ocr`에서 실행한다.
- PaddleOCR subprocess는 애플리케이션 수명 동안 재사용한다.
- 요청 횟수로 재시작하지 않고 실제 timeout이나 프로세스 종료 때만 복구한다.
- 시작 시 Python, Paddle, CUDA와 cuDNN 조합을 검사한다.
- 브라우저 화면에 필요 없는 문서 방향·왜곡·텍스트 줄 방향 분류를 끈다.

검증된 버전과 설치 명령은 [런타임 호환 기준](docs/runtime_compatibility.md)에 한 번만 기록한다.

### 재발 확인

```powershell
.\.venv-ocr\Scripts\python.exe scripts\check_runtime_compat.py --profile ocr
.\scripts\test.cmd agent\tests\regression\test_paddle_runtime.py -q
```

로그에서는 평균만 보지 말고 요청별 OCR 시간, 작업자 세대와 재시작 이유를 함께 확인한다. 재시작 없이 두 번째 요청이 빨라지면 모델 초기화는 정상적으로 재사용된 것이다.

## 2. 로딩 화면과 관찰 상태 불일치

### 현상

- 회색 배경, 흰 화면, skeleton을 정상 화면으로 저장해 OCR과 추론을 낭비했다.
- 클릭 뒤 화면이 변하는 도중의 캡처를 다음 행동 근거로 사용했다.
- 여러 계층이 각자 대기해 동일 전환을 두 번 이상 기다렸다.
- 주소창 URL, 스크린샷과 OCR이 서로 다른 시점의 상태를 가리켰다.

### 원인

화면 변화 감지, 로딩 완료, pHash 동일성 검증을 같은 문제로 다뤘다. pHash는 두 정지 화면의 유사도를 비교할 수 있지만 렌더링이 시작됐는지, 움직임이 끝났는지는 알려주지 않는다. 캡처·OCR·판단을 식별자 없이 갱신하면 이전 OCR이 새 화면에 남는다.

### 현재 처리

`LoadingWait`가 저해상도 OpenCV 프레임을 메모리에서 반복 비교한다.

1. 행동 전 기준 프레임과 달라져 화면 변화가 시작됐는지 본다.
2. 마지막 움직임 뒤 정숙 시간이 지났는지 본다.
3. 회색·흰 화면처럼 정보량이 낮은 프레임은 준비 완료로 인정하지 않는다.
4. 준비된 최종 프레임만 파일로 저장한다.

`capture_node()`는 새 `observation_id`를 만들고 이전 OCR·서명·화면 역할을 비운다. `ocr_node()`는 그 관찰에만 마커를 붙인다. 판단과 행동 요청도 같은 `observation_id`를 참조한다. 화면이 바뀌면 새 관찰을 만들기 전까지 이전 마커를 재사용하지 않는다.

pHash는 경험 경로의 화면·대상 동일성 확인에 사용한다. 로딩 완료 판단은 OpenCV 프레임 비교가 담당한다.

### 재발 확인

- [로딩 대기 테스트](agent/tests/core/test_loading_wait.py)
- [작업자 그래프 경계 테스트](agent/tests/core/test_worker_graph_boundaries.py)
- [작업자 상태 계약](docs/worker_state_contract.md)

## 3. 자율탐색 기록을 연속 경험 경로로 재생

### 현상

초기 Reflex는 단계 적중 수가 있어도 전체 경로를 끝내지 못했다. 같은 목적의 레시피가 원자 행동별로 중복됐고, 검색어 입력과 제출이 끊겼다. OCR 마커 ID를 저장해 다음 실행에서 잘못된 위치를 누르기도 했다.

### 근본 원인

재사용 단위를 클릭 하나로 두면 행동 전후 문맥이 사라진다. OCR 마커 ID는 한 OCR 결과 안에서만 유효하며 다음 캡처의 UI 식별자가 아니다. 전체 화면 pHash는 배너와 공고 목록 내용 변화에 민감했고, 문자열 유사도는 OCR 순서 변화에 흔들렸다.

### 현재 기록 계약

자율탐색은 행동 전 화면, 연속 행동과 도착 화면을 하나의 전이로 기록한다.

```text
ScreenCheckpoint(before)
  + 중간 관찰 없이 실행한 actions
  + ScreenCheckpoint(after)
  + 실행 결과 evidence
```

- 입력과 `Enter`, 입력과 제출 클릭처럼 함께 있어야 효과가 나는 행동은 같은 전이에 기록할 수 있다.
- 행동 전후 화면은 `before_observation_id`, `after_observation_id`로 연결한다.
- 클릭·입력 대상은 저장 당시 bbox·중심 비율과 대상 주변 ROI pHash를 가진다.
- 한 성공 실행의 의미 노드를 `ExperienceRuleNode`로, 노드 내부 화면 전이를 `ExperienceRuleStep`으로 저장한다.
- 활성 경로의 목적 키는 `site + task_category`다. 같은 목적의 새 경로는 중복 행을 만들지 않고 현재 경로를 갱신한다.

### 현재 재생 계약

1. `site + task_category`로 활성 경로 후보를 조회한다.
2. 현재 URL 범위와 첫 전이의 시작 조건을 검사한다.
3. 저장된 crop 비율로 현재 화면의 같은 ROI를 잘라 pHash를 비교한다.
4. ROI가 맞으면 저장된 대상 비율을 현재 화면 좌표로 복원해 행동을 실행한다.
5. OpenCV 대기 뒤 새 화면을 한 번 관찰한다.
6. 저장된 도착 체크포인트와 맞을 때만 다음 전이로 이동한다.
7. 어느 단계든 불일치하면 경로를 중단하고 같은 실행에서 그 경로를 차단한 뒤 자율 판단으로 복귀한다.

ROI 일치는 클릭 성공을 보장하지 않는다. 현재 구현은 “같은 영역의 시각 상태가 충분히 비슷하면 저장 좌표가 다시 유효하다”는 제한된 가정을 사용한다. 동적 목록 카드, 합쳐진 OCR bbox와 실행마다 달라지는 상세 대상은 경험 경로에 고정하지 않는다.

### Critic의 권한

그래프 구성 모델은 자율탐색 원본 이벤트를 삭제하지 않고 목적 노드와 `next`, `branch`, `recovery`, `feedback` 간선으로 구조화한다. Critic은 완성된 그래프를 본 뒤 실행 가능한 의미 노드에 대해 `keep` 또는 `drop`만 선택한다. 새로운 행동, 좌표, 화면 조건이나 실행 순서를 합성하지 않는다. 규칙 생성기는 선택된 노드를 다시 묶지 않고 각 원본 화면 전이를 물리 단계로 보존한다. 한 노드에 정상 경로와 실패 경로가 섞였거나 컴파일할 수 없는 행동이 있으면 후보 전체를 거부한다.

### 재발 확인

- [경험 경로 구현 기준](docs/reflex_recipe_plan.md)
- [기록 계약 테스트](agent/tests/regression/test_recipe_recording_contracts.py)
- [재생 계약 테스트](agent/tests/core/test_recipe_replay_contracts.py)
- [경로 계측 테스트](agent/tests/regression/test_reflex_path_observability.py)

## 4. Critic 검토와 자동 승격이 요청을 막음

### 현상

공고 수집과 DB 저장은 끝났지만 API 또는 E2E가 종료되지 않았다. Critic 요청이 504로 실패한 다음 실행에서는 경험 기반 탐색을 요청했어도 활성 레시피가 없어 모든 단계가 자율 판단으로 진행됐다.

### 원인

후보 저장, Critic 검토와 활성 레시피 승격을 사용자 요청의 동기식 후처리로 묶었다. timeout 뒤 후보 상태가 애매하게 남았고, 벤치마크가 승격 완료를 확인하지 않은 채 다음 실행을 시작했다.

### 현재 처리

```text
수집 성공
-> worker_submission 저장
-> recipe_candidate recorded
-> pending_review 등록
-> 사용자 응답 계속

별도 RecipePromotionWorker
-> pending_review 선점
-> reviewing
-> 원본 로그 실행 그래프 구성
-> Critic 노드 keep/drop
-> accepted + active recipe 또는 rejected
```

- SQLite 상태 전이로 후보 처리 소유권을 한 작업자에게 준다.
- 프로세스 시작 시 남은 `reviewing` 후보를 `pending_review`로 되돌린다.
- 전송 오류는 제한 횟수만 재시도하고 초과하면 `review_failed`로 남긴다.
- 경험 기반 E2E는 후보 생성 성공이 아니라 활성 경로 승격 완료를 확인한 뒤 시작한다.
- 수집 결과와 사용자 답변은 Critic 성공 여부에 의존하지 않는다.

### 재발 확인

- [후보 저장소](agent/recipe/candidate_store.py)
- [승격 작업자](agent/application/recipe_promotion_worker.py)
- [후보 검토 서비스](agent/application/recipe_candidate_review_service.py)
- [승격 계약 테스트](agent/tests/regression/test_recipe_promotion_contracts.py)

## 5. SPA와 사이트별 화면 상태

### 현상

- URL이 거의 변하지 않는 측면 상세 패널을 검색 결과 화면으로 오인했다.
- 외부 채용 중계 페이지에서 실제 공고 링크를 따라가지 않고 수집을 끝냈다.
- 새 탭이 열린 뒤 이전 탭 URL을 현재 화면 URL로 사용했다.
- 검색 결과가 부족한데도 모델이 작업을 완료했다.

### 원인

URL만으로 화면 상태를 판별하거나 특정 사이트의 버튼 이름과 카드 순서를 공통 실행기에 넣었다. SPA는 쿼리·fragment·선택 패널로 상태를 바꾸며, 같은 화면 역할도 사이트마다 보이는 단서가 다르다.

### 현재 처리

- 사이트별 공식 주소, URL 패턴, 화면 역할 단서와 페이지 이용 정보는 `agent/sites/<site>/profile.json`에 둔다.
- 공통 실행기는 사이트 이름에 따른 클릭 분기를 갖지 않고 프로필을 참고 정보로 추론에 전달한다.
- 주소창을 읽은 뒤 포커스를 페이지로 돌리고, 탭 전환·새 탭 뒤 URL을 다시 동기화한다.
- 상세 누적 버퍼가 바뀌면 `JobReview`가 `needs_more`, `complete`, `source_incomplete`, `invalid_target`을 결정한다.
- 목표 수가 남아 있으면 모델이 임의로 `finish_task`를 선택하지 못한다. 화면에서 더 수집할 후보가 없다는 근거가 있을 때만 부분 완료를 허용한다.

새 사이트 지원은 프로필에 관찰 가능한 단서를 추가하는 방식으로 시작한다. 사이트 전용 결정론적 정책이 필요하면 공통 코드의 조건문이 아니라 해당 프로필 또는 분리된 어댑터가 소유한다.

### 재발 확인

- [사이트 프로필 안내](agent/sites/README.md)
- [사이트 등록 테스트](agent/tests/regression/test_site_registry.py)
- [Rallit 프로필 회귀](agent/tests/regression/test_rallit_site_profile.py)

## 6. LangGraph 상태와 도구 계약 불일치

### 현상

- 노드를 직접 호출한 테스트는 통과했지만 컴파일된 그래프에서는 런타임 의존성이 주입되지 않았다.
- `query`와 `search_keyword`가 섞여 활성 레시피가 `missing_required_inputs`로 탈락했다.
- 경량 모델이 `scroll.amount="down"`을 반환해 도구 검증에서 실패했다.
- 상세 검토 버퍼를 `dict(...)`로 바꾸는 과정에서 타입 정보와 필드가 사라졌다.

### 원인

상태, patch, LLM 도구 인자와 저장 스키마가 같은 개념에 다른 이름을 사용했다. LangGraph가 인식하는 `runtime` 인자 규칙도 직접 호출 테스트에서는 드러나지 않았다. 느슨한 `dict[str, Any]` 변환이 경계마다 들어가면서 오류가 실행 후반까지 전달됐다.

### 현재 처리

- `WorkerState`는 `request`, `observation`, `decision`, `transition`, `replay`, `collection`, `lifecycle` 구역으로 고정한다.
- 노드는 전체 상태 대신 구역별 `*Patch`를 반환하며 reducer가 병합한다.
- OCR 객체, 모델, 브라우저 도구는 상태에 넣지 않고 `Runtime[WorkerDependencies]`로 주입한다.
- 검색어 필드는 `collection_intent.search_keyword` 하나만 사용한다.
- LLM 응답은 Pydantic `ActionRequest`와 실제 도구 스키마로 한 번 검증한다. `scroll.direction`과 `scroll.amount`의 허용값을 분리한다.
- `CollectedJob`, `JobCapture`, 상세 버퍼는 단계 사이에서 임의의 dict로 재구성하지 않는다.
- 정적 타입 검사는 `agent/application`, `agent/graph`, `agent/recipe`, `agent/runtime` 경계를 포함한다.

### 재발 확인

```powershell
.\scripts\test.cmd agent\tests\core\test_worker_graph_boundaries.py -q
.\scripts\test.cmd agent\tests\core\test_runtime_safety_contracts.py -q
.\.venv-app\Scripts\python.exe -m mypy
```

상태 필드의 소유권과 불변조건은 [작업자 상태 계약](docs/worker_state_contract.md)에 정리돼 있다.

## 7. 브라우저 창 소유권과 Windows 렌더링

### 현상

- 자동화 창이 이미 닫힌 뒤 정리 코드가 사용자의 다른 Chrome 창을 닫았다.
- 주소창 URL 복사 후 키 입력이 웹 페이지가 아니라 주소창에 들어갔다.
- Chrome 창이 가려졌을 때 캡처가 흰 화면으로 유지됐다.
- 뒤로가기가 가능한데 탭부터 닫아 검색 목록 문맥을 잃었다.

### 원인

브라우저 종류나 활성 창 제목만으로 자동화 대상을 다시 찾았다. Windows Chrome은 가려진 창의 렌더링을 줄일 수 있고, 주소창 복사는 GUI 포커스를 바꾼다. 탭 종료를 일반 복구 동작으로 허용하면 사용자의 탭과 작업 상태를 구분하기 어렵다.

### 현재 처리

- 수집을 시작할 때 새 브라우저 창 ID를 바인딩하고 해당 ID만 조작·종료한다.
- 바인딩된 창이 사라지면 다른 창을 대신 닫지 않는다.
- 주소창 복사 뒤 `Esc`와 콘텐츠 포커스 이동을 수행한다.
- Chrome 실행 인자에 `--disable-backgrounding-occluded-windows`를 적용한다.
- 흰 화면을 깨우기 위한 임의 클릭은 사용하지 않는다.
- 상세에서 먼저 `go_back`을 시도하고 화면 변화가 없다고 확인된 경우에만 현재 탭 닫기를 허용한다.
- 수집 요청이 끝나면 브라우저 창은 닫지만 OCR 작업자와 컴파일된 그래프는 재사용한다.

로그인과 CAPTCHA는 자동화하지 않는다. 사용자의 계정·민감 입력이 필요한 화면에서는 자동 실행을 중단한다.

### 재발 확인

- [브라우저·백엔드 경계 테스트](agent/tests/core/test_backend_boundaries.py)
- [Perception 회귀 테스트](agent/tests/regression/test_perception.py)

## 8. 수집 결과와 답변 근거 연결

### 현상

공고가 DB에 저장됐는데도 최종 답변 인용이 비거나 다른 사이트 공고로 분류됐다. 자동 평가와 수동 판정에서 같은 공고 URL의 일치 결과가 달랐고, query에 공고 ID를 두는 사이트의 서로 다른 공고를 같은 URL로 판정하기도 했다.

### 원인

- 사이트별 `source_platform` 표기가 서로 달랐다.
- URL 정규화가 모든 query를 제거해 query 안의 공고 ID까지 잃었다.
- 자동 품질 평가와 수동 평가가 서로 다른 URL 정규화 함수를 사용했다.
- 목록 카드의 제목만으로 상세 공고를 확정했다.

### 현재 처리

- 출처 식별자는 사이트 프로필의 `source_platform`으로 정규화한다.
- URL 정규화는 추적 파라미터만 제거하고 공고를 식별하는 path와 query 값은 보존한다.
- 자동 품질 평가와 수동 판정은 같은 `normalize_job_url()` 계약을 사용한다.
- DB는 검토가 끝난 원본 상세 URL을 고유키로 보존한다. 목록 단계의 중복 회피는 같은 사이트의 회사명·공고 제목 정체성을 별도로 비교한다.
- 상세 OCR 누적 결과를 `JobReview`가 검토하고 `complete`인 공고만 `CollectedJob`으로 만든다.
- 화면 근거 `JobCapture`와 구조화 공고를 같은 수집 단위로 전달한다.
- 답변은 DB에서 조회한 공고 ID와 필드 근거를 연결하며, 인용 검증에 실패한 문장은 제거한다.
- 기존 DB에 같은 공고가 있으면 새 카드 수집 없이도 요청 근거로 사용할 수 있다.

### 재발 확인

- [DB 영속성 테스트](agent/tests/core/test_db_persistence.py)
- [공고 검토 테스트](agent/tests/core/test_job_review_service.py)
- [답변 경계 테스트](agent/tests/core/test_backend_boundaries.py)
- [제품 데모 및 검증](docs/product_demo.md)

## 9. 비결정론적 GUI 실행의 평가

### 문제

같은 검색어를 다시 실행해도 공고 순서, 광고, 로그인 상태, 네트워크, 모델 판단과 OCR 결과가 달라진다. 자율탐색은 이미 방문한 공고를 중복 회피하므로 첫 실행과 똑같은 상세 대상을 재현하기도 어렵다. 자율탐색 1회와 경험 기반 탐색 1회의 총 실행시간 차이를 그대로 성능 개선으로 해석할 수 없다.

### 현재 평가 계약

| 질문 | 측정값 |
|---|---|
| 결과가 맞는가 | 목표 수, 고유 URL, 필수 필드, 출처 무결성, 고정 대상 정확 일치 |
| 경험 경로가 실제로 작동했는가 | 경로 시작·완주·중간 실패·폴백 횟수 |
| 추론을 줄였는가 | 자율 판단 호출 순감소량, 생략된 단계와 화면 |
| 비용을 줄였는가 | 모델별 입력·출력 토큰과 가격표 기반 비용, Critic·보조 호출 포함 |
| 운영이 안정적인가 | OCR timeout, 화면 준비 실패, 잘못된 창 종료, 부분 완료 사유 |

고정 URL 또는 고정 결과 계약을 만들 수 있는 경우에는 자율탐색과 경험 기반 탐색의 결과 품질을 먼저 맞춘다. 그다음 경험 경로가 대체한 추론 구간을 비교한다. 고정할 수 없는 실행은 성공 사례로 남길 수는 있지만 인과적인 속도 비교 근거로 사용하지 않는다.

보고서에는 저장된 행동 사용 횟수와 함께 `reflex_path_completed_count`, `reflex_path_failed_count`, `reflex_path_fallback_count`와 대체한 LLM 판단 수를 기록한다. 총 실행시간은 OCR·네트워크·모델 지연을 포함한 운영 지표로 남기고, 개선 원인은 고정 대상 계약과 단계별 계측으로 판별한다.

### 변경 규칙

반복 회귀의 원인은 세 가지였다.

1. 새 책임을 추가하면서 이전 실행 경로를 제거하지 않았다.
2. 제품 계약, E2E 계약, 테스트와 문서를 한 변경에서 함께 갱신하지 않았다.
3. 원인을 계측하기 전에 예외 조건과 재시도를 늘렸다.

새 정책은 한 계층에만 소유권을 둔다. 이전 정책은 같은 변경에서 삭제하고, 실패를 재현하는 최소 계약 테스트와 실제 E2E 한 건으로 확인한다. 실행 결과는 텍스트 로그가 아니라 `.summary.json`을 집계 원본으로 사용한다.

측정 필드와 해석 기준은 [E2E 관측 환경](docs/e2e_observability.md), 재현 명령은 [제품 데모 및 검증](docs/product_demo.md)에 있다.
